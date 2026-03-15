"""
Experiment 1: Baseline Models

Implementiert und evaluiert drei Baseline-Modelle:
1. Majority Baseline (immer "No Hate Speech")
2. Lexikon-basiert (deutsche Schimpfwort-Liste)
3. Random Forest + TF-IDF

Verwendung:
    python -m experiments.01_baselines
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# Projekt-Root zum Pfad hinzufügen
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import HATE_LEXICON, LABEL_NAMES, SEED
from src.data_loader import load_data, create_cv_splits
from src.evaluate import (
    evaluate_predictions,
    save_results,
    save_results_csv,
    print_comparison_table,
    collect_cv_results,
)
from src.utils import set_seed, timer, TrainingTimer


# ============================================================
# 1. Majority Baseline
# ============================================================

def majority_baseline(y_test: np.ndarray) -> np.ndarray:
    """Sagt immer die Mehrheitsklasse vorher (OTHER = 0)."""
    return np.zeros(len(y_test), dtype=int)


# ============================================================
# 2. Lexikon-basierte Baseline
# ============================================================

def lexicon_baseline(texts: list, lexicon: list = HATE_LEXICON) -> np.ndarray:
    """
    Klassifiziert basierend auf Vorhandensein von Schimpfwörtern.

    Args:
        texts: Liste von Texten.
        lexicon: Liste von Hate-Wörtern.

    Returns:
        Array von Vorhersagen (0 oder 1).
    """
    predictions = []
    for text in texts:
        text_lower = text.lower()
        has_hate_word = any(word in text_lower for word in lexicon)
        predictions.append(1 if has_hate_word else 0)
    return np.array(predictions)


# ============================================================
# 3. TF-IDF + Machine Learning Baselines
# ============================================================

def tfidf_baseline(
    X_train: list,
    y_train: np.ndarray,
    X_test: list,
    classifier_name: str = "random_forest",
    max_features: int = 10000,
    ngram_range: tuple = (1, 2),
) -> np.ndarray:
    """
    TF-IDF Vektorisierung + klassischer ML-Klassifikator.

    Args:
        classifier_name: "random_forest", "logistic_regression", oder "svm"

    Returns:
        Array von Vorhersagen.
    """
    # Vektorisierung
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Klassifikator
    classifiers = {
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=50, random_state=SEED, n_jobs=-1
        ),
        "logistic_regression": LogisticRegression(
            max_iter=1000, random_state=SEED, C=1.0
        ),
        "svm": LinearSVC(
            max_iter=5000, random_state=SEED, C=1.0
        ),
    }

    clf = classifiers[classifier_name]
    clf.fit(X_train_vec, y_train)
    predictions = clf.predict(X_test_vec)

    return predictions


# ============================================================
# Hauptexperiment
# ============================================================

def run_baselines():
    """Führt alle Baseline-Experimente durch."""

    print("=" * 70)
    print("  Experiment 1: Baseline Models")
    print("=" * 70)

    set_seed(SEED)

    # Daten laden
    print("\nLade GermEval 2018 Daten ...")
    train_df, test_df = load_data(preprocessing_variant="original")
    print(f"  Train: {len(train_df)} | Test: {len(test_df)}")
    print(f"  Label-Verteilung (Train): {dict(train_df['label'].value_counts())}")
    print(f"  Label-Verteilung (Test):  {dict(test_df['label'].value_counts())}")

    X_train = train_df["text"].tolist()
    y_train = train_df["label"].values
    X_test = test_df["text"].tolist()
    y_test = test_df["label"].values

    all_results = []
    training_timer = TrainingTimer()

    # ---- 1. Majority Baseline ----
    print("\n" + "-" * 50)
    print("  1. Majority Baseline")
    print("-" * 50)
    training_timer.start("Majority Baseline")
    preds_majority = majority_baseline(y_test)
    training_timer.stop()
    results_majority = evaluate_predictions(y_test, preds_majority, "Majority")
    all_results.append(results_majority)

    # ---- 2. Lexikon Baseline ----
    print("\n" + "-" * 50)
    print("  2. Lexikon-basierte Baseline")
    print("-" * 50)
    training_timer.start("Lexikon Baseline")
    preds_lexicon = lexicon_baseline(X_test)
    training_timer.stop()
    results_lexicon = evaluate_predictions(y_test, preds_lexicon, "Lexikon")
    all_results.append(results_lexicon)

    # ---- 3. Random Forest + TF-IDF ----
    print("\n" + "-" * 50)
    print("  3. Random Forest + TF-IDF")
    print("-" * 50)
    training_timer.start("Random Forest")
    preds_rf = tfidf_baseline(X_train, y_train, X_test, "random_forest")
    training_timer.stop()
    results_rf = evaluate_predictions(y_test, preds_rf, "RandomForest+TF-IDF")
    all_results.append(results_rf)

    # ---- 4. Logistic Regression + TF-IDF ----
    print("\n" + "-" * 50)
    print("  4. Logistic Regression + TF-IDF")
    print("-" * 50)
    training_timer.start("Logistic Regression")
    preds_lr = tfidf_baseline(X_train, y_train, X_test, "logistic_regression")
    training_timer.stop()
    results_lr = evaluate_predictions(y_test, preds_lr, "LogReg+TF-IDF")
    all_results.append(results_lr)

    # ---- 5. SVM + TF-IDF ----
    print("\n" + "-" * 50)
    print("  5. SVM + TF-IDF")
    print("-" * 50)
    training_timer.start("SVM")
    preds_svm = tfidf_baseline(X_train, y_train, X_test, "svm")
    training_timer.stop()
    results_svm = evaluate_predictions(y_test, preds_svm, "SVM+TF-IDF")
    all_results.append(results_svm)

    # ---- Vergleich ----
    print_comparison_table(all_results)
    print(training_timer.summary())

    # ---- Ergebnisse speichern ----
    save_results(
        {"baselines": all_results},
        "baseline_results.json",
    )

    # CSV
    rows = []
    for r in all_results:
        rows.append({
            "model": r["model"],
            "f1_macro": r["f1_macro"],
            "precision_macro": r["precision_macro"],
            "recall_macro": r["recall_macro"],
            "accuracy": r["accuracy"],
        })
    save_results_csv(rows, "baseline_results.csv")

    print("\nExperiment 1 abgeschlossen!")


if __name__ == "__main__":
    run_baselines()
