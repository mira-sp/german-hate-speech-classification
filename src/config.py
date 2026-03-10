"""
config.py - Zentrale Konfiguration für das Hate Speech Classification Projekt.

Enthält alle Hyperparameter, Pfade und Modell-Definitionen.
"""

import os
from pathlib import Path

# ============================================================
# Projektpfade
# ============================================================
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "germeval_2018"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
PLOTS_DIR = RESULTS_DIR / "plots"
MODELS_DIR = RESULTS_DIR / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
TENSORBOARD_DIR = LOGS_DIR / "tensorboard"

# Erstelle alle Verzeichnisse, falls nicht vorhanden
for d in [PROCESSED_DATA_DIR, SPLITS_DIR, METRICS_DIR, PLOTS_DIR, MODELS_DIR,
          LOGS_DIR, TENSORBOARD_DIR,
          PLOTS_DIR / "confusion_matrices", PLOTS_DIR / "error_analysis"]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
# Daten-Dateien
# ============================================================
TRAIN_FILE = RAW_DATA_DIR / "germeval2018.training.txt"
TEST_FILE = RAW_DATA_DIR / "germeval2018.test.txt"

# ============================================================
# Modell-Definitionen
# ============================================================
MODELS = {
    "mBERT": "bert-base-multilingual-cased",
    "GBERT": "deepset/gbert-base",
    "HateBERT": "GroNLP/hateBERT",
}

# ============================================================
# Label-Mapping  (GermEval 2018 Coarse Labels → binary)
# ============================================================
LABEL_MAP = {
    "OTHER": 0,
    "OFFENSE": 1,
}
NUM_LABELS = 2
LABEL_NAMES = ["OTHER", "OFFENSE"]


# Training-Hyperparameter  (optimiert für RTX 3060, 12 GB VRAM)

TRAINING_CONFIG = {
    "per_device_train_batch_size": 32,
    "per_device_eval_batch_size": 64,
    "num_train_epochs": 3,
    "learning_rate": 2e-5,
    "fp16": True,
    "dataloader_num_workers": 0,  # Windows Fix: 0=kein Multiprocessing
    "gradient_accumulation_steps": 1,
    "warmup_steps": 100,
    "weight_decay": 0.01,
    "logging_steps": 50,
    "eval_strategy": "epoch",
    "save_strategy": "epoch",
    "load_best_model_at_end": True,
    "metric_for_best_model": "f1",
    "save_total_limit": 2,
    "report_to": "tensorboard",
}

# Tokenizer-Einstellungen
MAX_LENGTH = 128

# ============================================================
# Cross-Validation
# ============================================================
N_FOLDS = 5

# ============================================================
# Hyperparameter-Tuning Suchraum
# ============================================================
HP_SEARCH = {
    "learning_rate": [1e-5, 2e-5, 3e-5, 5e-5],
    "batch_size": [16, 32],
}

# ============================================================
# Random Seed für Reproduzierbarkeit
# ============================================================
SEED = 42

# ============================================================
# Datengrößen-Experiment
# ============================================================
DATA_SIZES = [0.25, 0.50, 0.75, 1.0]

# ============================================================
# Hate-Lexikon für Lexikon-Baseline
# ============================================================
HATE_LEXICON = [
    "scheiß", "scheiss", "dumm", "idiot", "arschloch",
    "fick", "hurensohn", "wichser", "penner", "missgeburt",
    "bastard", "spast", "behindert", "schwuchtel", "fotze",
    "hure", "drecks", "assi", "vollidiot", "depp",
    "trottel", "spacken", "wixxer", "mongo", "opfer",
]
