"""
Generate all visualizations for paper/presentation.

Creates all plots in results/plots/:
- Model comparison barplot
- Per-class metrics
- Confusion matrices (for all models)
- Learning curve (Experiment 3)
- Preprocessing ablation (Experiment 4)
- Training time comparison

Run after all experiments are complete.

Usage:
    python scripts/generate_all_plots.py
"""

import sys
from pathlib import Path

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import METRICS_DIR, PLOTS_DIR, LABEL_NAMES


def plot_learning_curve():
    """Plot data size variation results (Experiment 3)."""
    print("\n📈 Generiere Learning Curve...")
    
    data_path = METRICS_DIR / "data_size_variation_results.json"
    if not data_path.exists():
        print("  ⚠️  Keine Daten gefunden - Experiment 3 noch nicht gelaufen")
        return
    
    with open(data_path, "r") as f:
        data = json.load(f)
    
    # Extract results dict
    results = data.get("results", data)
    
    # Parse
    sizes = []
    f1_means = []
    f1_stds = []
    acc_means = []
    acc_stds = []
    
    for size_key in sorted(results.keys(), key=lambda x: float(x.strip("%"))):
        size_data = results[size_key]
        sizes.append(int(size_key.strip("%")))
        f1_means.append(size_data["f1_macro_mean"])
        f1_stds.append(size_data["f1_macro_std"])
        acc_means.append(size_data["accuracy_mean"])
        acc_stds.append(size_data["accuracy_std"])
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.errorbar(sizes, f1_means, yerr=f1_stds, marker="o", linewidth=2.5,
                markersize=9, capsize=5, label="F1-Score (Macro)", color="#3498db")
    ax.errorbar(sizes, acc_means, yerr=acc_stds, marker="s", linewidth=2.5,
                markersize=9, capsize=5, label="Accuracy", color="#e74c3c")
    
    ax.set_xticks(sizes)  # Force x-ticks to be exactly at data points [10, 25, 50, 75, 100]
    ax.set_xlabel("Trainingsdatengröße (%)", fontsize=13)
    ax.set_ylabel("Score", fontsize=13)
    ax.set_title("Learning Curve: GBERT auf GermEval 2018", fontsize=15, fontweight="bold")
    ax.legend(fontsize=12, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 0.9)
    
    plt.tight_layout()
    output_path = PLOTS_DIR / "learning_curve.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✅ {output_path}")


def plot_preprocessing_comparison():
    """Plot preprocessing ablation results (Experiment 4)."""
    print("\n🔧 Generiere Preprocessing Ablation Plot...")
    
    data_path = METRICS_DIR / "preprocessing_ablation_results.json"
    if not data_path.exists():
        print("  ⚠️  Keine Daten gefunden - Experiment 4 noch nicht gelaufen")
        return
    
    with open(data_path, "r") as f:
        data = json.load(f)
    
    results = data.get("results", {})
    
    # Sort by F1
    sorted_variants = sorted(
        results.items(),
        key=lambda x: x[1]["f1_macro_mean"],
        reverse=True
    )
    
    variants = [v[0] for v in sorted_variants]
    f1_means = [v[1]["f1_macro_mean"] for v in sorted_variants]
    f1_stds = [v[1]["f1_macro_std"] for v in sorted_variants]
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = ["#2ecc71" if i == 0 else "#95a5a6" for i in range(len(variants))]
    bars = ax.barh(range(len(variants)), f1_means, xerr=f1_stds,
                   color=colors, capsize=4, edgecolor="gray", height=0.6)
    
    ax.set_yticks(range(len(variants)))
    ax.set_yticklabels(variants, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("F1-Score (Macro)", fontsize=13)
    ax.set_title("Preprocessing Ablation Study: GBERT", fontsize=15, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    
    # Values at bars
    for i, (mean, std) in enumerate(zip(f1_means, f1_stds)):
        ax.text(mean + std + 0.005, i, f"{mean:.4f}", va="center", fontsize=10)
    
    plt.tight_layout()
    output_path = PLOTS_DIR / "preprocessing_ablation.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✅ {output_path}")


def plot_all_confusion_matrices():
    """Generate confusion matrices for all BERT models."""
    print("\n🔲 Generiere Confusion Matrices...")
    
    cm_dir = PLOTS_DIR / "confusion_matrices"
    cm_dir.mkdir(parents=True, exist_ok=True)
    
    models = ["GBERT", "mBERT", "HateBERT"]
    
    for model_name in models:
        result_path = METRICS_DIR / f"{model_name}_cv_results.json"
        if not result_path.exists():
            print(f"  ⚠️  {model_name}: Keine Daten gefunden")
            continue
        
        with open(result_path, "r") as f:
            data = json.load(f)
        
        # Aggregate confusion matrix across folds
        fold_results = data.get("fold_results", [])
        if not fold_results:
            continue
        
        # Sum confusion matrices
        cm_sum = np.zeros((2, 2))
        for fold in fold_results:
            cm = np.array(fold["confusion_matrix"])
            cm_sum += cm
        
        # Normalize
        cm_normalized = cm_sum / cm_sum.sum(axis=1, keepdims=True)
        
        # Plot
        fig, ax = plt.subplots(figsize=(7, 6))
        
        sns.heatmap(cm_normalized, annot=True, fmt=".2%", cmap="Blues",
                    xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES,
                    cbar_kws={"label": "Proportion"}, ax=ax,
                    annot_kws={"fontsize": 13})
        
        ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
        ax.set_ylabel("True Label", fontsize=12, fontweight="bold")
        ax.set_title(f"Confusion Matrix: {model_name}", fontsize=14, fontweight="bold")
        
        plt.tight_layout()
        output_path = cm_dir / f"cm_{model_name.lower()}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  ✅ {output_path.name}")


def plot_per_class_comparison():
    """Plot per-class F1 scores for all models."""
    print("\n📊 Generiere Per-Class Metrics...")
    
    models = ["GBERT", "mBERT", "HateBERT"]
    results = {}
    
    for model_name in models:
        result_path = METRICS_DIR / f"{model_name}_cv_results.json"
        if result_path.exists():
            with open(result_path, "r") as f:
                results[model_name] = json.load(f)
    
    if not results:
        print("  ⚠️  Keine BERT-Ergebnisse gefunden")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for i, label_name in enumerate(LABEL_NAMES):
        ax = axes[i]
        
        model_names = []
        f1_means = []
        f1_stds = []
        
        for model in models:
            if model in results:
                model_names.append(model)
                f1_means.append(results[model].get(f"f1_{label_name}_mean", 0))
                f1_stds.append(results[model].get(f"f1_{label_name}_std", 0))
        
        colors = ["#2ecc71" if m == "GBERT" else "#3498db" for m in model_names]
        bars = ax.bar(range(len(model_names)), f1_means, yerr=f1_stds,
                      color=colors, capsize=5, edgecolor="gray", width=0.6)
        
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, fontsize=11)
        ax.set_ylabel("F1-Score", fontsize=12)
        ax.set_title(f"Klasse: {label_name}", fontsize=13, fontweight="bold")
        ax.set_ylim(0, 1)
        ax.grid(True, axis="y", alpha=0.3)
        
        # Values above bars
        for bar, mean in zip(bars, f1_means):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                    f"{mean:.3f}", ha="center", fontsize=10)
    
    plt.tight_layout()
    output_path = PLOTS_DIR / "per_class_metrics.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✅ {output_path}")


def plot_overall_comparison():
    """Plot overall model comparison (all models including baselines)."""
    print("\n🏆 Generiere Model Comparison...")
    
    # Load baseline results
    baseline_path = METRICS_DIR / "baseline_results.json"
    baselines = {}
    if baseline_path.exists():
        with open(baseline_path, "r") as f:
            baseline_data = json.load(f)
            # Convert list to dict keyed by model name
            if "baselines" in baseline_data:
                for item in baseline_data["baselines"]:
                    baselines[item["model"]] = item
            else:
                baselines = baseline_data
    
    # Load BERT results
    models = ["GBERT", "mBERT", "HateBERT"]
    bert_results = {}
    for model_name in models:
        result_path = METRICS_DIR / f"{model_name}_cv_results.json"
        if result_path.exists():
            with open(result_path, "r") as f:
                bert_results[model_name] = json.load(f)
    
    # Prepare data
    all_models = []
    f1_scores = []
    colors = []
    
    # Baselines
    for name, data in baselines.items():
        all_models.append(name)
        f1_scores.append(data["f1_macro"])
        colors.append("#95a5a6")  # Gray
    
    # BERT
    for name, data in bert_results.items():
        all_models.append(name)
        f1_scores.append(data["f1_macro_mean"])
        if name == "GBERT":
            colors.append("#2ecc71")  # Green for best
        else:
            colors.append("#3498db")  # Blue
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars = ax.bar(range(len(all_models)), f1_scores, color=colors, edgecolor="gray", width=0.7)
    ax.set_xticks(range(len(all_models)))
    ax.set_xticklabels(all_models, rotation=0, fontsize=11)
    ax.set_ylabel("F1-Score (Macro)", fontsize=13)
    ax.set_title("Model Comparison: Hate Speech Detection", fontsize=15, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, max(f1_scores) * 1.1)
    
    # Values above bars
    for bar, score in zip(bars, f1_scores):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f"{score:.3f}", ha="center", fontsize=10, fontweight="bold")
    
    plt.tight_layout()
    output_path = PLOTS_DIR / "model_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✅ {output_path}")


def main():
    print("="*70)
    print("  Generiere alle Visualisierungen")
    print("="*70)
    
    # Ensure plot directories exist
    (PLOTS_DIR / "confusion_matrices").mkdir(parents=True, exist_ok=True)
    (PLOTS_DIR / "error_analysis").mkdir(parents=True, exist_ok=True)
    
    # Generate all plots
    plot_overall_comparison()
    plot_per_class_comparison()
    plot_all_confusion_matrices()
    plot_learning_curve()
    plot_preprocessing_comparison()
    
    print("\n" + "="*70)
    print("  ✅ Alle Plots generiert!")
    print(f"  📁 Gespeichert in: {PLOTS_DIR}")
    print("="*70)


if __name__ == "__main__":
    main()
