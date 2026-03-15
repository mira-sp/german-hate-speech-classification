"""
evaluate.py - Evaluation-Funktionen und Metriken.

Berechnet: F1 (Macro/Micro), Precision, Recall, Accuracy,
           Confusion Matrix, Per-Class Metrics.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)

from src.config import LABEL_NAMES, METRICS_DIR


def compute_metrics(eval_pred) -> Dict[str, float]:
    """
    Compute-Metrics-Funktion für den HuggingFace Trainer.

    Args:
        eval_pred: (predictions, labels) Tupel vom Trainer.

    Returns:
        Dict mit Metriken.
    """
    predictions, labels = eval_pred
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    preds = np.argmax(predictions, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    acc = accuracy_score(labels, preds)

    # Per-Class Metriken
    precision_per_class, recall_per_class, f1_per_class, support = (
        precision_recall_fscore_support(labels, preds, average=None, zero_division=0)
    )

    metrics = {
        "accuracy": float(acc),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "f1_micro": float(f1_score(labels, preds, average="micro")),
    }

    # Per-Class
    for i, name in enumerate(LABEL_NAMES):
        metrics[f"f1_{name}"] = float(f1_per_class[i])
        metrics[f"precision_{name}"] = float(precision_per_class[i])
        metrics[f"recall_{name}"] = float(recall_per_class[i])

    return metrics


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "",
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Umfassende Evaluation von Vorhersagen.

    Args:
        y_true: Wahre Labels.
        y_pred: Vorhergesagte Labels.
        model_name: Name des Modells (für Ausgabe).
        verbose: Ausführliche Ausgabe.

    Returns:
        Dict mit allen Metriken.
    """
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)

    # Per-Class
    precision_pc, recall_pc, f1_pc, support_pc = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)

    results = {
        "model": model_name,
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "f1_micro": float(f1_micro),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "confusion_matrix": cm.tolist(),
    }

    for i, name in enumerate(LABEL_NAMES):
        results[f"f1_{name}"] = float(f1_pc[i])
        results[f"precision_{name}"] = float(precision_pc[i])
        results[f"recall_{name}"] = float(recall_pc[i])
        results[f"support_{name}"] = int(support_pc[i])

    if verbose:
        print(f"\n{'='*60}")
        if model_name:
            print(f"  Evaluation: {model_name}")
        print(f"{'='*60}")
        print(f"  Accuracy:         {acc:.4f}")
        print(f"  F1 (Macro):       {f1_macro:.4f}")
        print(f"  F1 (Micro):       {f1_micro:.4f}")
        print(f"  Precision (Macro): {precision_macro:.4f}")
        print(f"  Recall (Macro):    {recall_macro:.4f}")
        print(f"\n  Per-Class:")
        for i, name in enumerate(LABEL_NAMES):
            print(f"    {name:>10}: P={precision_pc[i]:.4f}  R={recall_pc[i]:.4f}  F1={f1_pc[i]:.4f}  (n={support_pc[i]})")
        print(f"\n  Confusion Matrix:")
        print(f"    {cm}")
        print(f"{'='*60}")

    return results


def collect_cv_results(
    fold_results: List[Dict],
    model_name: str,
) -> Dict:
    """
    Aggregiert Ergebnisse über alle CV-Folds.

    Args:
        fold_results: Liste von Ergebnis-Dicts pro Fold.
        model_name: Name des Modells.

    Returns:
        Aggregiertes Ergebnis-Dict mit Mean und Std.
    """
    metric_keys = ["accuracy", "f1_macro", "f1_micro", "precision_macro", "recall_macro"]
    for name in LABEL_NAMES:
        metric_keys.extend([f"f1_{name}", f"precision_{name}", f"recall_{name}"])

    aggregated = {"model": model_name}

    for key in metric_keys:
        values = [r[key] for r in fold_results if key in r]
        if values:
            aggregated[f"{key}_mean"] = float(np.mean(values))
            aggregated[f"{key}_std"] = float(np.std(values))
            aggregated[f"{key}_values"] = values

    return aggregated


def save_results(
    results: Dict,
    filename: str,
    output_dir: Path = METRICS_DIR,
):
    """Speichert Ergebnisse als JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / filename
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Ergebnisse gespeichert: {filepath}")


def save_results_csv(
    all_results: List[Dict],
    filename: str = "all_results.csv",
    output_dir: Path = METRICS_DIR,
):
    """Speichert alle Ergebnisse als CSV-Tabelle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_results)
    filepath = output_dir / filename
    df.to_csv(filepath, index=False)
    print(f"CSV gespeichert: {filepath}")


def print_comparison_table(results_list: List[Dict]):
    """Druckt eine Vergleichstabelle aller Modelle."""
    print(f"\n{'='*80}")
    print(f"{'Modell':>15} | {'F1 (Macro)':>12} | {'Precision':>12} | {'Recall':>12} | {'Accuracy':>12}")
    print(f"{'-'*80}")

    for r in sorted(results_list, key=lambda x: x.get("f1_macro_mean", x.get("f1_macro", 0)), reverse=True):
        name = r.get("model", "?")
        if "f1_macro_mean" in r:
            f1 = f"{r['f1_macro_mean']:.4f}±{r['f1_macro_std']:.4f}"
            prec = f"{r['precision_macro_mean']:.4f}±{r['precision_macro_std']:.4f}"
            rec = f"{r['recall_macro_mean']:.4f}±{r['recall_macro_std']:.4f}"
            acc = f"{r['accuracy_mean']:.4f}±{r['accuracy_std']:.4f}"
        else:
            f1 = f"{r.get('f1_macro', 0):.4f}"
            prec = f"{r.get('precision_macro', 0):.4f}"
            rec = f"{r.get('recall_macro', 0):.4f}"
            acc = f"{r.get('accuracy', 0):.4f}"

        print(f"{name:>15} | {f1:>12} | {prec:>12} | {rec:>12} | {acc:>12}")

    print(f"{'='*80}")
