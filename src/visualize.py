"""
visualize.py - Plotting-Funktionen für Paper-Quality Visualisierungen.

Erstellt: Model Comparison, Confusion Matrices, Learning Curves, Error Analysis.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.config import LABEL_NAMES, METRICS_DIR, PLOTS_DIR

# Globales Styling
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def plot_model_comparison(
    results: List[Dict],
    metric: str = "f1_macro",
    output_path: Optional[Path] = None,
    title: str = "Model Comparison",
):
    """
    Erstellt einen Barplot zum Vergleich aller Modelle.

    Args:
        results: Liste von Ergebnis-Dicts (mit _mean und _std Feldern).
        metric: Zu vergleichende Metrik.
        output_path: Speicherpfad.
        title: Plot-Titel.
    """
    if output_path is None:
        output_path = PLOTS_DIR / "model_comparison.png"

    models = []
    means = []
    stds = []
    colors = []

    # Farbpalette
    color_map = {
        "Majority": "#9E9E9E",
        "Lexikon": "#FF9800",
        "RandomForest+TF-IDF": "#FF5722",
        "LogReg+TF-IDF": "#E91E63",
        "SVM+TF-IDF": "#9C27B0",
        "mBERT": "#2196F3",
        "GBERT": "#4CAF50",
        "HateBERT": "#F44336",
    }

    for r in sorted(results, key=lambda x: x.get(f"{metric}_mean", x.get(metric, 0))):
        name = r.get("model", "?")
        models.append(name)

        if f"{metric}_mean" in r:
            means.append(r[f"{metric}_mean"])
            stds.append(r[f"{metric}_std"])
        else:
            means.append(r.get(metric, 0))
            stds.append(0)

        colors.append(color_map.get(name, "#607D8B"))

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.barh(
        range(len(models)), means,
        xerr=stds if any(s > 0 for s in stds) else None,
        color=colors,
        edgecolor="gray",
        capsize=4,
        height=0.6,
    )

    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    ax.set_xlabel("F1-Score (Macro)")
    ax.set_title(title)
    ax.set_xlim(0, 1.0)
    ax.grid(True, axis="x", alpha=0.3)

    # Werte an Balken
    for i, (m, s) in enumerate(zip(means, stds)):
        label = f"{m:.3f}" + (f"±{s:.3f}" if s > 0 else "")
        ax.text(m + (s if s > 0 else 0) + 0.01, i, label, va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"📊 Model Comparison gespeichert: {output_path}")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    label_names: List[str] = LABEL_NAMES,
    output_path: Optional[Path] = None,
    normalize: bool = True,
):
    """
    Erstellt eine Confusion-Matrix-Heatmap.

    Args:
        y_true: Wahre Labels.
        y_pred: Vorhergesagte Labels.
        model_name: Modellname.
        label_names: Label-Bezeichnungen.
        output_path: Speicherpfad.
        normalize: Ob die Matrix normalisiert werden soll.
    """
    if output_path is None:
        output_path = PLOTS_DIR / "confusion_matrices" / f"cm_{model_name.lower()}.png"

    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm_display = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        fmt = ".2f"
        title = f"Confusion Matrix (normalized): {model_name}"
    else:
        cm_display = cm
        fmt = "d"
        title = f"Confusion Matrix: {model_name}"

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
        ax=ax,
        square=True,
        cbar_kws={"label": "Proportion" if normalize else "Count"},
    )

    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(title, fontsize=14)

    # Raw counts als Annotation unterhalb
    if normalize:
        for i in range(len(label_names)):
            for j in range(len(label_names)):
                ax.text(
                    j + 0.5, i + 0.75,
                    f"(n={cm[i, j]})",
                    ha="center", va="center",
                    fontsize=8, color="gray",
                )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"📊 Confusion Matrix gespeichert: {output_path}")


def plot_metrics_per_class(
    results: List[Dict],
    output_path: Optional[Path] = None,
):
    """
    Erstellt einen gruppierten Barplot mit Per-Class Metriken.
    """
    if output_path is None:
        output_path = PLOTS_DIR / "per_class_metrics.png"

    models = [r["model"] for r in results]
    x = np.arange(len(models))
    width = 0.2

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, label_name in enumerate(LABEL_NAMES):
        ax = axes[idx]
        f1_vals = [r.get(f"f1_{label_name}_mean", r.get(f"f1_{label_name}", 0)) for r in results]
        prec_vals = [r.get(f"precision_{label_name}_mean", r.get(f"precision_{label_name}", 0)) for r in results]
        rec_vals = [r.get(f"recall_{label_name}_mean", r.get(f"recall_{label_name}", 0)) for r in results]

        ax.bar(x - width, f1_vals, width, label="F1", color="#2196F3")
        ax.bar(x, prec_vals, width, label="Precision", color="#4CAF50")
        ax.bar(x + width, rec_vals, width, label="Recall", color="#FF9800")

        ax.set_xlabel("Model")
        ax.set_ylabel("Score")
        ax.set_title(f"Per-Class Metrics: {label_name}")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha="right")
        ax.legend()
        ax.set_ylim(0, 1)
        ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"📊 Per-Class Metrics gespeichert: {output_path}")


def plot_error_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    texts: List[str],
    model_name: str,
    output_path: Optional[Path] = None,
):
    """
    Analysiert und visualisiert die Fehlerverteilung nach Textlänge.
    """
    if output_path is None:
        output_path = PLOTS_DIR / "error_analysis" / f"error_dist_{model_name.lower()}.png"

    correct = y_true == y_pred
    text_lengths = [len(t.split()) for t in texts]

    df = pd.DataFrame({
        "text_length": text_lengths,
        "correct": correct,
        "error_type": [
            "correct" if c else ("FP" if p == 1 else "FN")
            for c, p in zip(correct, y_pred)
        ],
    })

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Fehlerrate nach Textlänge
    ax1 = axes[0]
    bins = pd.cut(df["text_length"], bins=10)
    error_rate = df.groupby(bins)["correct"].apply(lambda x: 1 - x.mean())
    error_rate.plot(kind="bar", ax=ax1, color="#F44336", edgecolor="gray")
    ax1.set_xlabel("Text Length (words)")
    ax1.set_ylabel("Error Rate")
    ax1.set_title(f"Error Rate by Text Length: {model_name}")
    ax1.tick_params(axis="x", rotation=45)
    ax1.grid(True, axis="y", alpha=0.3)

    # 2. FP vs FN Verteilung
    ax2 = axes[1]
    error_counts = df[df["error_type"] != "correct"]["error_type"].value_counts()
    colors_pie = {"FP": "#FF9800", "FN": "#F44336"}
    if len(error_counts) > 0:
        error_counts.plot(
            kind="pie", ax=ax2, autopct="%1.1f%%",
            colors=[colors_pie.get(x, "#9E9E9E") for x in error_counts.index],
            startangle=90,
        )
        ax2.set_ylabel("")
        ax2.set_title(f"Error Type Distribution: {model_name}")
    else:
        ax2.text(0.5, 0.5, "No Errors", ha="center", va="center")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"📊 Error Distribution gespeichert: {output_path}")


def plot_training_times(
    results: List[Dict],
    output_path: Optional[Path] = None,
):
    """Visualisiert die Trainingszeiten pro Modell."""
    if output_path is None:
        output_path = PLOTS_DIR / "training_times.png"

    models = [r["model"] for r in results]
    times_min = [r.get("total_train_time_sec", 0) / 60 for r in results]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(models, times_min, color="#607D8B", edgecolor="gray", height=0.5)

    for i, t in enumerate(times_min):
        ax.text(t + 0.5, i, f"{t:.1f} min", va="center", fontsize=10)

    ax.set_xlabel("Training Time (minutes)")
    ax.set_title("Training Time Comparison (5-Fold CV)")
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"📊 Training Times gespeichert: {output_path}")


def create_results_latex_table(
    results: List[Dict],
    output_path: Optional[Path] = None,
) -> str:
    """
    Erstellt eine LaTeX-Tabelle der Ergebnisse für die Hausarbeit.

    Returns:
        LaTeX-String.
    """
    if output_path is None:
        output_path = METRICS_DIR / "results_table.tex"

    header = r"""
\begin{table}[htbp]
\centering
\caption{Vergleich der Modell-Performance auf GermEval 2018 (5-Fold CV)}
\label{tab:results}
\begin{tabular}{lcccc}
\toprule
\textbf{Modell} & \textbf{F1 (Macro)} & \textbf{Precision} & \textbf{Recall} & \textbf{Accuracy} \\
\midrule
"""

    rows = []
    for r in sorted(results, key=lambda x: x.get("f1_macro_mean", x.get("f1_macro", 0)), reverse=True):
        name = r.get("model", "?")
        if "f1_macro_mean" in r:
            f1 = f"${r['f1_macro_mean']:.3f} \\pm {r['f1_macro_std']:.3f}$"
            prec = f"${r['precision_macro_mean']:.3f} \\pm {r['precision_macro_std']:.3f}$"
            rec = f"${r['recall_macro_mean']:.3f} \\pm {r['recall_macro_std']:.3f}$"
            acc = f"${r['accuracy_mean']:.3f} \\pm {r['accuracy_std']:.3f}$"
        else:
            f1 = f"${r.get('f1_macro', 0):.3f}$"
            prec = f"${r.get('precision_macro', 0):.3f}$"
            rec = f"${r.get('recall_macro', 0):.3f}$"
            acc = f"${r.get('accuracy', 0):.3f}$"

        rows.append(f"{name} & {f1} & {prec} & {rec} & {acc} \\\\")

    footer = r"""
\bottomrule
\end{tabular}
\end{table}
"""

    table = header + "\n".join(rows) + "\n" + footer

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(table)
    print(f"📊 LaTeX-Tabelle gespeichert: {output_path}")

    return table
