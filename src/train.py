"""
train.py - Training-Script für BERT-Modelle.

Unterstützt:
- Training eines einzelnen Modells
- 5-Fold Cross Validation
- Verschiedene Hyperparameter-Konfigurationen
- Checkpointing & Resume

Verwendung:
    python -m src.train --model GBERT --epochs 3 --batch_size 32
    python -m src.train --model all --cv
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from src.config import (
    LABEL_NAMES,
    MAX_LENGTH,
    MODELS,
    MODELS_DIR,
    N_FOLDS,
    NUM_LABELS,
    SEED,
    TENSORBOARD_DIR,
    TRAINING_CONFIG,
)
from src.data_loader import (
    create_cv_splits,
    create_hf_dataset,
    load_data,
    sample_data,
)
from src.evaluate import (
    collect_cv_results,
    compute_metrics,
    evaluate_predictions,
    save_results,
    save_results_csv,
)
from src.models import check_gpu, free_model_memory, load_model_and_tokenizer
from src.utils import set_seed, timer, TrainingTimer, format_time


def train_single_fold(
    model_key: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    fold: int = 0,
    config: Optional[Dict] = None,
    output_dir: Optional[Path] = None,
) -> Tuple[Dict, float]:
    """
    Trainiert ein Modell auf einem einzelnen Fold.

    Args:
        model_key: Schlüssel des Modells (z.B. "GBERT").
        train_df: Trainings-DataFrame.
        val_df: Validierungs-DataFrame.
        fold: Fold-Nummer.
        config: Trainings-Konfiguration (überschreibt Defaults).
        output_dir: Ausgabeverzeichnis.

    Returns:
        (eval_results, training_time_seconds)
    """
    cfg = {**TRAINING_CONFIG}
    if config:
        cfg.update(config)

    if output_dir is None:
        output_dir = MODELS_DIR / model_key.lower() / f"fold_{fold}"

    # Modell & Tokenizer laden
    model, tokenizer = load_model_and_tokenizer(model_key)

    # Datasets erstellen
    print(f"  📊 Train: {len(train_df)} | Val: {len(val_df)}")
    train_dataset = create_hf_dataset(train_df, tokenizer, MAX_LENGTH)
    val_dataset = create_hf_dataset(val_df, tokenizer, MAX_LENGTH)

    # Training Arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
        num_train_epochs=cfg["num_train_epochs"],
        learning_rate=cfg["learning_rate"],
        fp16=cfg["fp16"] and torch.cuda.is_available(),
        warmup_steps=cfg["warmup_steps"],
        weight_decay=cfg["weight_decay"],
        logging_steps=cfg["logging_steps"],
            evaluation_strategy=cfg.get("eval_strategy", cfg.get("evaluation_strategy", "epoch")),
        save_strategy=cfg["save_strategy"],
        load_best_model_at_end=cfg["load_best_model_at_end"],
        metric_for_best_model=cfg["metric_for_best_model"],
        save_total_limit=cfg.get("save_total_limit", 2),
        logging_dir=str(TENSORBOARD_DIR / f"{model_key}_fold{fold}"),
        report_to=cfg.get("report_to", "tensorboard"),
        seed=SEED,
        dataloader_num_workers=cfg.get("dataloader_num_workers", 0),
    )

    # Trainer erstellen
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # Training
    print(f"\n🚀 Training {model_key} (Fold {fold}) ...")
    start_time = time.time()
    trainer.train()
    train_time = time.time() - start_time
    print(f"  ⏱ Trainingszeit: {format_time(train_time)}")

    # Evaluation
    eval_results = trainer.evaluate()
    print(f"  📈 F1: {eval_results['eval_f1']:.4f} | Acc: {eval_results['eval_accuracy']:.4f}")

    # Bestes Modell speichern
    best_model_dir = MODELS_DIR / f"{model_key.lower()}_best"
    trainer.save_model(str(best_model_dir / f"fold_{fold}"))
    tokenizer.save_pretrained(str(best_model_dir / f"fold_{fold}"))

    # Vorhersagen für detaillierte Analyse
    predictions = trainer.predict(val_dataset)
    preds = np.argmax(predictions.predictions, axis=1)
    labels = predictions.label_ids

    detailed_results = evaluate_predictions(
        labels, preds, model_name=f"{model_key}_fold{fold}", verbose=True
    )
    detailed_results["train_time_sec"] = train_time
    detailed_results["fold"] = fold

    # Speicher freigeben
    free_model_memory(model)

    return detailed_results, train_time


def train_with_cv(
    model_key: str,
    train_df: pd.DataFrame,
    n_folds: int = N_FOLDS,
    config: Optional[Dict] = None,
) -> Dict:
    """
    Trainiert ein Modell mit K-Fold Cross Validation.

    Args:
        model_key: Schlüssel des Modells.
        train_df: Vollständiger Trainings-DataFrame.
        n_folds: Anzahl Folds.
        config: Trainings-Config.

    Returns:
        Aggregiertes Ergebnis-Dict.
    """
    print(f"\n{'#'*60}")
    print(f"  {model_key} - {n_folds}-Fold Cross Validation")
    print(f"{'#'*60}")

    cv_splits = create_cv_splits(train_df, n_folds=n_folds, save=True)
    fold_results = []
    total_time = 0

    for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
        print(f"\n{'='*40}")
        print(f"  Fold {fold_idx + 1}/{n_folds}")
        print(f"{'='*40}")

        fold_train = train_df.iloc[train_idx].reset_index(drop=True)
        fold_val = train_df.iloc[val_idx].reset_index(drop=True)

        results, train_time = train_single_fold(
            model_key=model_key,
            train_df=fold_train,
            val_df=fold_val,
            fold=fold_idx,
            config=config,
        )

        fold_results.append(results)
        total_time += train_time

    # Aggregieren
    aggregated = collect_cv_results(fold_results, model_key)
    aggregated["total_train_time_sec"] = total_time
    aggregated["fold_results"] = fold_results

    print(f"\n{'='*60}")
    print(f"  {model_key} - CV Zusammenfassung")
    print(f"{'='*60}")
    print(f"  F1 (Macro):  {aggregated['f1_macro_mean']:.4f} ± {aggregated['f1_macro_std']:.4f}")
    print(f"  Accuracy:    {aggregated['accuracy_mean']:.4f} ± {aggregated['accuracy_std']:.4f}")
    print(f"  Gesamtzeit:  {format_time(total_time)}")

    # Ergebnisse speichern
    save_results(aggregated, f"{model_key}_cv_results.json")

    return aggregated


def train_all_models(
    train_df: pd.DataFrame,
    model_keys: Optional[List[str]] = None,
    n_folds: int = N_FOLDS,
    config: Optional[Dict] = None,
) -> List[Dict]:
    """
    Trainiert alle Modelle mit CV.

    Args:
        train_df: Trainings-DataFrame.
        model_keys: Liste der zu trainierenden Modelle (None = alle).
        n_folds: Anzahl Folds.
        config: Trainings-Config.

    Returns:
        Liste der aggregierten Ergebnisse.
    """
    if model_keys is None:
        model_keys = list(MODELS.keys())

    all_results = []
    training_timer = TrainingTimer()

    for model_key in model_keys:
        training_timer.start(model_key)
        results = train_with_cv(model_key, train_df, n_folds, config)
        training_timer.stop()
        all_results.append(results)

    # Zusammenfassung
    print(training_timer.summary())
    training_timer.save(MODELS_DIR / "training_times.json")

    # CSV mit allen Ergebnissen
    rows = []
    for r in all_results:
        for fold_r in r.get("fold_results", []):
            rows.append({
                "model": fold_r.get("model", "").split("_fold")[0],
                "fold": fold_r.get("fold", 0),
                "f1_macro": fold_r.get("f1_macro", 0),
                "precision_macro": fold_r.get("precision_macro", 0),
                "recall_macro": fold_r.get("recall_macro", 0),
                "accuracy": fold_r.get("accuracy", 0),
                "train_time_sec": fold_r.get("train_time_sec", 0),
            })
    if rows:
        save_results_csv(rows, "all_results.csv")

    return all_results


# ============================================================
# CLI
# ============================================================

def main():
    """
    Hauptfunktion für das Training-Skript.
    
    Parst Kommandozeilenargumente und startet den Trainingsprozess
    (Single Run oder Cross-Validation).
    """
    parser = argparse.ArgumentParser(description="BERT Fine-Tuning für Hate Speech Detection")
    parser.add_argument(
        "--model", type=str, default="GBERT",
        choices=list(MODELS.keys()) + ["all"],
        help="Modell zum Trainieren (oder 'all')",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Anzahl Epochen")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch Size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning Rate")
    parser.add_argument("--cv", action="store_true", help="5-Fold Cross Validation")
    parser.add_argument("--folds", type=int, default=5, help="Anzahl CV-Folds")
    parser.add_argument(
        "--preprocessing", type=str, default="original",
        help="Preprocessing-Variante",
    )
    parser.add_argument("--data_fraction", type=float, default=1.0, help="Datenanteil (0-1)")
    parser.add_argument("--seed", type=int, default=42, help="Random Seed")

    args = parser.parse_args()

    # Setup
    set_seed(args.seed)
    check_gpu()

    # Config
    config = {
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "learning_rate": args.lr,
    }

    # Daten laden
    print("\n📂 Lade Daten ...")
    train_df, test_df = load_data(preprocessing_variant=args.preprocessing)
    print(f"  Train: {len(train_df)} | Test: {len(test_df)}")

    # Daten-Sampling
    if args.data_fraction < 1.0:
        train_df = sample_data(train_df, fraction=args.data_fraction)
        print(f"  Sampling: {args.data_fraction*100:.0f}% → {len(train_df)} Trainings-Samples")

    # Training
    if args.model == "all":
        results = train_all_models(train_df, n_folds=args.folds, config=config)
    elif args.cv:
        results = train_with_cv(args.model, train_df, n_folds=args.folds, config=config)
    else:
        # Einfaches Train/Test Split
        from sklearn.model_selection import train_test_split
        t_df, v_df = train_test_split(
            train_df, test_size=0.2, random_state=args.seed, stratify=train_df["label"]
        )
        results, _ = train_single_fold(args.model, t_df, v_df, config=config)

    print("\n✅ Training abgeschlossen!")


if __name__ == "__main__":
    main()
