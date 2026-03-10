"""
Experiment 3: Data Size Variation (Learning Curve)

Trainiert das beste Modell (GBERT) mit verschiedenen Trainings-Datenmengen
(25%, 50%, 75%, 100%), um eine Learning Curve zu erstellen.

Verwendung:
    python -m experiments.03_data_size_variation
    python -m experiments.03_data_size_variation --model GBERT --sizes 0.1 0.25 0.5 0.75 1.0
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import DATA_SIZES, MODELS, N_FOLDS, PLOTS_DIR, SEED, TRAINING_CONFIG
from src.data_loader import load_data, sample_data
from src.evaluate import save_results, save_results_csv
from src.models import check_gpu
from src.train import train_with_cv
from src.utils import set_seed, timer, TrainingTimer


def plot_learning_curve(
    results: Dict[str, Dict],
    model_name: str,
    output_path: Path = PLOTS_DIR / "learning_curve.png",
):
    """
    Erstellt einen Learning-Curve Plot.

    Args:
        results: Dict von {size_key: aggregated_results}
        model_name: Name des Modells.
        output_path: Speicherpfad.
    """
    sizes_pct = []
    train_sizes = []
    f1_means = []
    f1_stds = []
    acc_means = []
    acc_stds = []

    for size_key in sorted(results.keys(), key=lambda x: float(x.strip("%")) / 100):
        r = results[size_key]
        pct = float(size_key.strip("%"))
        sizes_pct.append(pct)
        train_sizes.append(r.get("train_size", 0))
        f1_means.append(r["f1_macro_mean"])
        f1_stds.append(r["f1_macro_std"])
        acc_means.append(r["accuracy_mean"])
        acc_stds.append(r["accuracy_std"])

    f1_means = np.array(f1_means)
    f1_stds = np.array(f1_stds)
    acc_means = np.array(acc_means)
    acc_stds = np.array(acc_stds)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # F1 Score
    ax1.plot(sizes_pct, f1_means, "o-", linewidth=2, markersize=8, color="#2196F3", label="F1 (Macro)")
    ax1.fill_between(sizes_pct, f1_means - f1_stds, f1_means + f1_stds, alpha=0.2, color="#2196F3")
    ax1.set_xlabel("Training Data Size (%)", fontsize=12)
    ax1.set_ylabel("F1-Score (Macro)", fontsize=12)
    ax1.set_title(f"Learning Curve: {model_name} – F1 Score", fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    ax1.set_xticks(sizes_pct)

    # Accuracy
    ax2.plot(sizes_pct, acc_means, "s-", linewidth=2, markersize=8, color="#4CAF50", label="Accuracy")
    ax2.fill_between(sizes_pct, acc_means - acc_stds, acc_means + acc_stds, alpha=0.2, color="#4CAF50")
    ax2.set_xlabel("Training Data Size (%)", fontsize=12)
    ax2.set_ylabel("Accuracy", fontsize=12)
    ax2.set_title(f"Learning Curve: {model_name} – Accuracy", fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    ax2.set_xticks(sizes_pct)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"📊 Learning Curve gespeichert: {output_path}")


def run_data_size_variation(
    model_key: str = "GBERT",
    data_sizes: List[float] = None,
    n_folds: int = N_FOLDS,
):
    """
    Trainiert ein Modell mit verschiedenen Datenmengen.

    Args:
        model_key: Modell-Schlüssel.
        data_sizes: Liste von Datenanteil-Werten (0.0 - 1.0).
        n_folds: Anzahl CV-Folds.
    """
    if data_sizes is None:
        data_sizes = DATA_SIZES

    print("=" * 70)
    print(f"  Experiment 3: Data Size Variation ({model_key})")
    print("=" * 70)

    set_seed(SEED)
    check_gpu()

    # Daten laden
    print("\n📂 Lade Daten ...")
    train_df, test_df = load_data(preprocessing_variant="full_preprocessing")
    print(f"  Vollständiger Trainingssatz: {len(train_df)} Samples")

    results_by_size = {}
    training_timer = TrainingTimer()
    # Bestehende Ergebnisse laden und übernehmen
    import json
    from src.config import METRICS_DIR
    results_path = METRICS_DIR / "data_size_variation_results.json"
    if results_path.exists():
        with open(results_path, "r", encoding="utf-8") as f:
            try:
                existing = json.load(f)
                if "results" in existing:
                    results_by_size.update(existing["results"])
                    print(f"  📁 Bestehende Ergebnisse geladen: {len(existing['results'])} Datengrößen")
            except Exception as e:
                print(f"  ⚠️ Konnte bestehende Ergebnisse nicht laden: {e}")
                pass

    for size in data_sizes:
        size_pct = f"{int(size * 100)}%"
        print(f"\n{'#' * 60}")
        print(f"  Trainingsgröße: {size_pct}")
        print(f"{'#' * 60}")

        # Daten sampeln
        sampled_df = sample_data(train_df, fraction=size, seed=SEED, stratify=True)
        print(f"  Samples: {len(sampled_df)} ({size_pct} von {len(train_df)})")

        training_timer.start(f"{model_key}_{size_pct}")

        with timer(f"{model_key} @ {size_pct}"):
            result = train_with_cv(
                model_key=model_key,
                train_df=sampled_df,
                n_folds=n_folds,
            )

        training_timer.stop()

        result["train_size"] = len(sampled_df)
        result["data_fraction"] = size
        results_by_size[size_pct] = result

    # Zusammenfassung
    print(training_timer.summary())

    print(f"\n{'=' * 70}")
    print(f"  Data Size Variation – Zusammenfassung")
    print(f"{'=' * 70}")
    print(f"{'Size':>8} | {'Train N':>8} | {'F1 (Macro)':>15} | {'Accuracy':>15}")
    print(f"{'-' * 60}")
    for size_key in sorted(results_by_size.keys(), key=lambda x: float(x.strip("%"))):
        r = results_by_size[size_key]
        f1_str = f"{r['f1_macro_mean']:.4f} ± {r['f1_macro_std']:.4f}"
        acc_str = f"{r['accuracy_mean']:.4f} ± {r['accuracy_std']:.4f}"
        print(f"{size_key:>8} | {r['train_size']:>8} | {f1_str:>15} | {acc_str:>15}")

    # Plot
    plot_learning_curve(results_by_size, model_key)

    # Ergebnisse speichern
    save_results(
        {"experiment": "data_size_variation", "model": model_key, "results": results_by_size},
        "data_size_variation_results.json",
    )

    # CSV
    rows = []
    for size_key, r in results_by_size.items():
        for fold_r in r.get("fold_results", []):
            rows.append({
                "model": model_key,
                "data_size": size_key,
                "train_n": r["train_size"],
                "fold": fold_r.get("fold", 0),
                "f1_macro": fold_r.get("f1_macro", 0),
                "accuracy": fold_r.get("accuracy", 0),
                "train_time_sec": fold_r.get("train_time_sec", 0),
            })
    save_results_csv(rows, "data_size_variation_results.csv")

    print("\n✅ Experiment 3 abgeschlossen!")
    return results_by_size


def main():
    parser = argparse.ArgumentParser(description="Data Size Variation Experiment")
    parser.add_argument(
        "--model", type=str, default="GBERT",
        choices=list(MODELS.keys()),
        help="Modell (default: GBERT)",
    )
    parser.add_argument(
        "--sizes", nargs="+", type=float, default=DATA_SIZES,
        help="Datenanteil-Werte (z.B. 0.25 0.5 0.75 1.0)",
    )
    parser.add_argument("--folds", type=int, default=N_FOLDS, help="Anzahl CV-Folds")

    args = parser.parse_args()
    run_data_size_variation(
        model_key=args.model,
        data_sizes=args.sizes,
        n_folds=args.folds,
    )


if __name__ == "__main__":
    main()
