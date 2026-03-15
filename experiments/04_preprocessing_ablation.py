"""
Experiment 4: Preprocessing Ablation Study

Testet verschiedene Text-Preprocessing-Varianten mit dem besten Modell (GBERT),
um den Einfluss von Preprocessing auf die Klassifikations-Performance zu messen.

Varianten:
- original: Kein Preprocessing
- remove_urls: URLs durch [URL] ersetzen
- normalize_usernames: @mentions durch @USER ersetzen
- remove_emojis: Emojis entfernen
- lowercase: Alles kleinschreiben
- full_preprocessing: URLs + Usernames + Emojis + Hashtags + Whitespace
- full_preprocessing_lowercase: full_preprocessing + Lowercase

Verwendung:
    python -m experiments.04_preprocessing_ablation
    python -m experiments.04_preprocessing_ablation --model GBERT --folds 5
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import METRICS_DIR, MODELS, N_FOLDS, PLOTS_DIR, SEED, TRAINING_CONFIG
from src.data_loader import load_data
from src.evaluate import save_results, save_results_csv
from src.models import check_gpu
from src.preprocessing import PREPROCESSING_VARIANTS
from src.train import train_with_cv
from src.utils import set_seed, timer, TrainingTimer


def plot_preprocessing_comparison(
    results: Dict[str, Dict],
    model_name: str,
    output_path: Path = PLOTS_DIR / "preprocessing_ablation.png",
):
    """
    Erstellt einen Balkendiagramm-Vergleich der Preprocessing-Varianten.
    """
    variants = []
    f1_means = []
    f1_stds = []

    # Sortiere nach F1 (absteigend)
    sorted_items = sorted(
        results.items(),
        key=lambda x: x[1].get("f1_macro_mean", 0),
        reverse=True,
    )

    for variant_name, r in sorted_items:
        variants.append(variant_name)
        f1_means.append(r["f1_macro_mean"])
        f1_stds.append(r["f1_macro_std"])

    f1_means = np.array(f1_means)
    f1_stds = np.array(f1_stds)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(variants)))

    bars = ax.barh(
        range(len(variants)), f1_means,
        xerr=f1_stds,
        color=colors,
        edgecolor="gray",
        capsize=4,
        height=0.6,
    )

    ax.set_yticks(range(len(variants)))
    ax.set_yticklabels(variants, fontsize=11)
    ax.set_xlabel("F1-Score (Macro)", fontsize=12)
    ax.set_title(f"Preprocessing Ablation Study: {model_name}", fontsize=14)
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.3)

    # Werte an den Balken
    for i, (mean, std) in enumerate(zip(f1_means, f1_stds)):
        ax.text(mean + std + 0.005, i, f"{mean:.4f}", va="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Preprocessing Ablation Plot gespeichert: {output_path}")


def run_preprocessing_ablation(
    model_key: str = "GBERT",
    variants: Optional[List[str]] = None,
    n_folds: int = N_FOLDS,
):
    """
    Führt die Preprocessing Ablation Study durch.

    Args:
        model_key: Modell-Schlüssel.
        variants: Liste der zu testenden Varianten (None = alle).
        n_folds: Anzahl CV-Folds.
    """
    if variants is None:
        variants = list(PREPROCESSING_VARIANTS.keys())

    print("=" * 70)
    print(f"  Experiment 4: Preprocessing Ablation ({model_key})")
    print("=" * 70)

    set_seed(SEED)
    check_gpu()

    # Lade vorhandene Ergebnisse (falls vorhanden)
    results_json_path = METRICS_DIR / "preprocessing_ablation_results.json"
    results_by_variant = {}
    if results_json_path.exists():
        import json
        with open(results_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if "results" in data:
                results_by_variant = data["results"]
                print(f"Lade vorhandene Ergebnisse: {len(results_by_variant)} Varianten")

    training_timer = TrainingTimer()

    for variant in variants:
        # Überspringe bereits trainierte Varianten
        if variant in results_by_variant:
            print(f"\nVariante '{variant}' bereits trainiert - überspringe")
            continue

        print(f"\n{'#' * 60}")
        print(f"  Preprocessing: {variant}")
        print(f"{'#' * 60}")

        # Daten mit dieser Variante laden
        train_df, test_df = load_data(preprocessing_variant=variant)
        print(f"  Train: {len(train_df)} | Test: {len(test_df)}")

        # Zeige Beispiel
        print(f"  Beispiel-Text: {train_df['text'].iloc[0][:100]}...")

        training_timer.start(f"{model_key}_{variant}")

        with timer(f"{model_key} ({variant})"):
            result = train_with_cv(
                model_key=model_key,
                train_df=train_df,
                n_folds=n_folds,
            )

        training_timer.stop()

        result["preprocessing"] = variant
        results_by_variant[variant] = result

        # SPEICHERE NACH JEDER VARIANTE (incremental save)
        save_results(
            {"experiment": "preprocessing_ablation", "model": model_key, "results": results_by_variant},
            "preprocessing_ablation_results.json",
        )
        print(f"  Zwischenergebnis gespeichert ({len(results_by_variant)}/{len(variants)} Varianten)")

    # Zusammenfassung
    print(training_timer.summary())

    print(f"\n{'=' * 70}")
    print(f"  Preprocessing Ablation – Zusammenfassung")
    print(f"{'=' * 70}")
    print(f"{'Variante':>30} | {'F1 (Macro)':>15} | {'Accuracy':>15}")
    print(f"{'-' * 65}")

    sorted_variants = sorted(
        results_by_variant.items(),
        key=lambda x: x[1]["f1_macro_mean"],
        reverse=True,
    )
    for variant_name, r in sorted_variants:
        f1_str = f"{r['f1_macro_mean']:.4f} ± {r['f1_macro_std']:.4f}"
        acc_str = f"{r['accuracy_mean']:.4f} ± {r['accuracy_std']:.4f}"
        print(f"{variant_name:>30} | {f1_str:>15} | {acc_str:>15}")

    # Plot
    plot_preprocessing_comparison(results_by_variant, model_key)

    # Ergebnisse speichern
    save_results(
        {"experiment": "preprocessing_ablation", "model": model_key, "results": results_by_variant},
        "preprocessing_ablation_results.json",
    )

    # CSV
    rows = []
    for variant_name, r in results_by_variant.items():
        for fold_r in r.get("fold_results", []):
            rows.append({
                "model": model_key,
                "preprocessing": variant_name,
                "fold": fold_r.get("fold", 0),
                "f1_macro": fold_r.get("f1_macro", 0),
                "precision_macro": fold_r.get("precision_macro", 0),
                "recall_macro": fold_r.get("recall_macro", 0),
                "accuracy": fold_r.get("accuracy", 0),
                "train_time_sec": fold_r.get("train_time_sec", 0),
            })
    save_results_csv(rows, "preprocessing_ablation_results.csv")

    print("\nExperiment 4 abgeschlossen!")
    return results_by_variant


def main():
    parser = argparse.ArgumentParser(description="Preprocessing Ablation Study")
    parser.add_argument(
        "--model", type=str, default="GBERT",
        choices=list(MODELS.keys()),
        help="Modell (default: GBERT)",
    )
    parser.add_argument(
        "--variants", nargs="+", type=str, default=None,
        choices=list(PREPROCESSING_VARIANTS.keys()),
        help="Preprocessing-Varianten (default: alle)",
    )
    parser.add_argument("--folds", type=int, default=N_FOLDS, help="Anzahl CV-Folds")

    args = parser.parse_args()
    run_preprocessing_ablation(
        model_key=args.model,
        variants=args.variants,
        n_folds=args.folds,
    )


if __name__ == "__main__":
    main()
