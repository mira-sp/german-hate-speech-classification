"""
data_loader.py - Datenladen, Splitting und Dataset-Erstellung.

Unterstützt:
- Laden der GermEval 2018 Daten (Roh oder vorverarbeitet)
- Stratified K-Fold Cross Validation Splits
- HuggingFace Dataset-Erstellung mit Tokenisierung
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset as TorchDataset

from src.config import (
    LABEL_MAP,
    MAX_LENGTH,
    N_FOLDS,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    SEED,
    SPLITS_DIR,
    TRAIN_FILE,
    TEST_FILE,
)
from src.preprocessing import (
    apply_preprocessing,
    load_germeval_file,
    prepare_binary_labels,
)


# ============================================================
# Custom PyTorch Dataset
# ============================================================

class HateSpeechDataset(TorchDataset):
    """PyTorch Dataset für tokenisierte Texte."""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)


# ============================================================
# Daten laden
# ============================================================

def load_data(
    use_processed: bool = False,
    preprocessing_variant: str = "original",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Lädt Trainings- und Testdaten.

    Args:
        use_processed: Wenn True, lade aus data/processed/ (CSV).
        preprocessing_variant: Preprocessing-Variante für Rohdaten.

    Returns:
        (train_df, test_df) mit Spalten: text, coarse_label, fine_label, label
    """
    if use_processed:
        train_path = PROCESSED_DATA_DIR / "train.csv"
        test_path = PROCESSED_DATA_DIR / "test.csv"
        if train_path.exists() and test_path.exists():
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            return train_df, test_df
        else:
            print("⚠ Vorverarbeitete Daten nicht gefunden, lade Rohdaten...")

    # Rohdaten laden
    train_df = load_germeval_file(TRAIN_FILE)
    test_df = load_germeval_file(TEST_FILE)

    # Labels vorbereiten
    train_df = prepare_binary_labels(train_df)
    test_df = prepare_binary_labels(test_df)

    # Preprocessing anwenden
    train_df = apply_preprocessing(train_df, variant=preprocessing_variant)
    test_df = apply_preprocessing(test_df, variant=preprocessing_variant)

    return train_df, test_df


def load_all_data(
    preprocessing_variant: str = "original",
) -> pd.DataFrame:
    """Lädt und kombiniert Trainings- und Testdaten zu einem DataFrame."""
    train_df, test_df = load_data(preprocessing_variant=preprocessing_variant)
    train_df["split"] = "train"
    test_df["split"] = "test"
    all_df = pd.concat([train_df, test_df], ignore_index=True)
    return all_df


# ============================================================
# Cross-Validation Splits
# ============================================================

def create_cv_splits(
    df: pd.DataFrame,
    n_folds: int = N_FOLDS,
    seed: int = SEED,
    save: bool = True,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Erstellt stratifizierte K-Fold Cross-Validation Splits.

    Args:
        df: DataFrame mit 'label'-Spalte.
        n_folds: Anzahl der Folds.
        seed: Random Seed.
        save: Splits als JSON speichern.

    Returns:
        Liste von (train_indices, val_indices) Tupeln.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    splits = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(df, df["label"])):
        splits.append((train_idx, val_idx))

        if save:
            split_info = {
                "fold": fold_idx,
                "train_indices": train_idx.tolist(),
                "val_indices": val_idx.tolist(),
                "train_size": len(train_idx),
                "val_size": len(val_idx),
            }
            split_path = SPLITS_DIR / f"fold_{fold_idx}.json"
            with open(split_path, "w") as f:
                json.dump(split_info, f, indent=2)

    if save:
        print(f"✅ {n_folds} CV-Splits gespeichert in {SPLITS_DIR}")

    return splits


def load_cv_splits(n_folds: int = N_FOLDS) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Lädt gespeicherte CV-Splits."""
    splits = []
    for fold_idx in range(n_folds):
        split_path = SPLITS_DIR / f"fold_{fold_idx}.json"
        with open(split_path, "r") as f:
            info = json.load(f)
        splits.append((
            np.array(info["train_indices"]),
            np.array(info["val_indices"]),
        ))
    return splits


# ============================================================
# Tokenisierung & HuggingFace Dataset
# ============================================================

def tokenize_data(
    texts: List[str],
    tokenizer,
    max_length: int = MAX_LENGTH,
) -> dict:
    """
    Tokenisiert eine Liste von Texten mit einem HuggingFace-Tokenizer.

    Returns:
        Dict mit input_ids, attention_mask, etc. als Tensoren.
    """
    encodings = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return encodings


def create_torch_dataset(
    df: pd.DataFrame,
    tokenizer,
    max_length: int = MAX_LENGTH,
) -> HateSpeechDataset:
    """
    Erstellt ein PyTorch-Dataset aus einem DataFrame.

    Args:
        df: DataFrame mit 'text' und 'label' Spalten.
        tokenizer: HuggingFace Tokenizer.
        max_length: Maximale Token-Länge.

    Returns:
        HateSpeechDataset
    """
    texts = df["text"].tolist()
    labels = df["label"].tolist()
    encodings = tokenize_data(texts, tokenizer, max_length)
    return HateSpeechDataset(encodings, labels)


def create_hf_dataset(
    df: pd.DataFrame,
    tokenizer,
    max_length: int = MAX_LENGTH,
):
    """
    Erstellt ein HuggingFace Dataset (für den Trainer).

    Returns:
        datasets.Dataset
    """
    from datasets import Dataset as HFDataset

    dataset = HFDataset.from_pandas(df[["text", "label"]])

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    dataset = dataset.map(tokenize_fn, batched=True)
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    return dataset


# ============================================================
# Hilfsfunktionen
# ============================================================

def get_data_stats(df: pd.DataFrame) -> Dict:
    """Berechnet Statistiken über den Datensatz."""
    stats = {
        "total_samples": len(df),
        "label_distribution": df["label"].value_counts().to_dict(),
        "label_percentages": (df["label"].value_counts(normalize=True) * 100).round(1).to_dict(),
        "avg_text_length": df["text"].str.len().mean(),
        "median_text_length": df["text"].str.len().median(),
        "max_text_length": df["text"].str.len().max(),
        "min_text_length": df["text"].str.len().min(),
        "avg_word_count": df["text"].str.split().str.len().mean(),
    }
    return stats


def sample_data(
    df: pd.DataFrame,
    fraction: float = 1.0,
    seed: int = SEED,
    stratify: bool = True,
) -> pd.DataFrame:
    """
    Samplet einen Bruchteil der Daten (für Data-Size-Experiment).

    Args:
        df: Eingabe-DataFrame.
        fraction: Anteil der Daten (0.0 - 1.0).
        seed: Random Seed.
        stratify: Ob stratifiziert nach Label gesampelt werden soll.

    Returns:
        Gesampleter DataFrame.
    """
    if fraction >= 1.0:
        return df.copy()

    if stratify:
        # Stratifiziertes Sampling
        sampled = df.groupby("label", group_keys=False).apply(
            lambda x: x.sample(frac=fraction, random_state=seed)
        )
        return sampled.reset_index(drop=True)
    else:
        return df.sample(frac=fraction, random_state=seed).reset_index(drop=True)
