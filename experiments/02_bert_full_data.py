"""
Experiment 2: BERT Fine-Tuning mit 100% Daten

Trainiert alle drei BERT-Modelle (mBERT, GBERT, HateBERT) mit
5-Fold Cross Validation auf dem vollständigen GermEval 2018 Datensatz.

Verwendung:
    python -m experiments.02_bert_full_data
    python -m experiments.02_bert_full_data --model GBERT
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import MODELS, N_FOLDS, SEED, TRAINING_CONFIG
from src.data_loader import load_data
from src.evaluate import print_comparison_table, save_results
from src.models import check_gpu
from src.train import train_all_models, train_with_cv
from src.utils import set_seed, timer


def run_bert_full_data(model_keys=None, n_folds=N_FOLDS, config=None):
    """
    Trainiert BERT-Modelle mit Cross Validation.

    Args:
        model_keys: Liste der Modelle (None = alle).
        n_folds: Anzahl CV-Folds.
        config: Trainings-Konfiguration.
    """
    print("=" * 70)
    print("  Experiment 2: BERT Fine-Tuning (100% Data)")
    print("=" * 70)

    set_seed(SEED)
    has_gpu = check_gpu()

    cfg = {**TRAINING_CONFIG}
    if config:
        cfg.update(config)

    # Falls keine GPU: Batch-Sizes reduzieren, fp16 aus
    if not has_gpu:
        cfg["per_device_train_batch_size"] = 8
        cfg["per_device_eval_batch_size"] = 16
        cfg["fp16"] = False
        print("⚠ Keine GPU – reduzierte Konfiguration aktiv.")

    # Daten laden
    print("\n📂 Lade GermEval 2018 Daten ...")
    train_df, test_df = load_data(preprocessing_variant="full_preprocessing")
    print(f"  Train: {len(train_df)} Samples")
    print(f"  Test:  {len(test_df)} Samples")
    print(f"  Labels: {dict(train_df['label'].value_counts())}")

    # Training
    if model_keys is None:
        model_keys = list(MODELS.keys())

    all_results = []
    for model_key in model_keys:
        with timer(f"Training {model_key}"):
            result = train_with_cv(
                model_key=model_key,
                train_df=train_df,
                n_folds=n_folds,
                config=cfg,
            )
            all_results.append(result)

    # Vergleich
    print_comparison_table(all_results)

    # Finale Evaluation auf Test-Set (mit bestem Fold-Modell)
    print("\n📊 Finale Evaluation auf Test-Set:")
    # Hier könnte man das beste Modell laden und auf dem Test-Set evaluieren
    # Das wird im evaluate.py Script gemacht

    # Ergebnisse speichern
    save_results(
        {"experiment": "bert_full_data", "models": {r["model"]: r for r in all_results}},
        "bert_full_data_results.json",
    )

    print("\n✅ Experiment 2 abgeschlossen!")
    return all_results


def main():
    parser = argparse.ArgumentParser(description="BERT Fine-Tuning Experiment")
    parser.add_argument(
        "--model", type=str, default=None,
        choices=list(MODELS.keys()) + [None],
        help="Einzelnes Modell (default: alle)",
    )
    parser.add_argument("--folds", type=int, default=N_FOLDS, help="Anzahl CV-Folds")
    parser.add_argument("--epochs", type=int, default=3, help="Anzahl Epochen")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch Size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning Rate")

    args = parser.parse_args()

    config = {}
    if args.epochs:
        config["num_train_epochs"] = args.epochs
    if args.batch_size:
        config["per_device_train_batch_size"] = args.batch_size
    if args.lr:
        config["learning_rate"] = args.lr

    model_keys = [args.model] if args.model else None
    run_bert_full_data(model_keys=model_keys, n_folds=args.folds, config=config)


if __name__ == "__main__":
    main()
