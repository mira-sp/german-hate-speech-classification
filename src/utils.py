"""
utils.py - Hilfsfunktionen für das Projekt.

Enthält: Timer, Logging, Seed-Setting, GPU-Info, Plotting-Hilfsfunktionen.
"""

import io
import json
import os
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

# Fix Windows cp1252 encoding issues with emoji/unicode output
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


# ============================================================
# Reproduzierbarkeit
# ============================================================

def set_seed(seed: int = 42):
    """Setzt den Random Seed für Reproduzierbarkeit."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic mode (kann Training verlangsamen)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"🎲 Random Seed gesetzt: {seed}")


# ============================================================
# Timer
# ============================================================

@contextmanager
def timer(description: str = "Operation"):
    """Context Manager zum Zeitmessen."""
    start = time.time()
    yield
    elapsed = time.time() - start
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)
    print(f"{description}: {hours:02d}:{minutes:02d}:{seconds:02d} ({elapsed:.1f}s)")


class TrainingTimer:
    """Tracker für Trainingszeiten über mehrere Experimente."""

    def __init__(self):
        self.records = []

    def start(self, experiment: str):
        self._current = {
            "experiment": experiment,
            "start_time": time.time(),
        }

    def stop(self) -> float:
        elapsed = time.time() - self._current["start_time"]
        self._current["elapsed_seconds"] = elapsed
        self._current["elapsed_readable"] = format_time(elapsed)
        self.records.append(self._current)
        return elapsed

    def summary(self) -> str:
        total = sum(r["elapsed_seconds"] for r in self.records)
        lines = ["\nTraining-Zeit Zusammenfassung:", "-" * 50]
        for r in self.records:
            lines.append(f"  {r['experiment']:>30s}: {r['elapsed_readable']}")
        lines.append("-" * 50)
        lines.append(f"  {'TOTAL':>30s}: {format_time(total)}")
        return "\n".join(lines)

    def save(self, filepath: Path):
        with open(filepath, "w") as f:
            json.dump(self.records, f, indent=2, default=str)


def format_time(seconds: float) -> str:
    """Formatiert Sekunden als HH:MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


# ============================================================
# GPU-Informationen
# ============================================================

def get_gpu_info() -> Dict:
    """Gibt GPU-Informationen zurück."""
    if not torch.cuda.is_available():
        return {"available": False}

    return {
        "available": True,
        "device_name": torch.cuda.get_device_name(0),
        "total_memory_gb": round(torch.cuda.get_device_properties(0).total_mem / 1e9, 2),
        "allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
        "reserved_gb": round(torch.cuda.memory_reserved() / 1e9, 2),
        "cuda_version": torch.version.cuda,
        "pytorch_version": torch.__version__,
    }


def print_gpu_status():
    """Druckt den aktuellen GPU-Status."""
    info = get_gpu_info()
    if not info["available"]:
        print("⚠ Keine GPU verfügbar.")
        return

    print(f"\n🖥 GPU Status:")
    print(f"  Gerät:     {info['device_name']}")
    print(f"  VRAM:      {info['total_memory_gb']} GB total")
    print(f"  Belegt:    {info['allocated_gb']} GB")
    print(f"  Reserviert: {info['reserved_gb']} GB")
    print(f"  CUDA:      {info['cuda_version']}")
    print(f"  PyTorch:   {info['pytorch_version']}")


# ============================================================
# Logging
# ============================================================

def setup_logging(log_file: Optional[Path] = None):
    """Konfiguriert Logging für Konsole und optional Datei."""
    import logging
    import sys

    logger = logging.getLogger("hate_speech")
    logger.setLevel(logging.INFO)

    # Console Handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    console.setFormatter(fmt)
    logger.addHandler(console)

    # File Handler
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger


# ============================================================
# Ergebnis-Aggregation
# ============================================================

def aggregate_fold_metrics(fold_metrics: List[Dict]) -> Dict:
    """
    Aggregiert Metriken über CV-Folds (Mean ± Std).

    Args:
        fold_metrics: Liste von Metriken-Dicts.

    Returns:
        Aggregiertes Dict mit _mean und _std Suffixen.
    """
    if not fold_metrics:
        return {}

    numeric_keys = [
        k for k in fold_metrics[0]
        if isinstance(fold_metrics[0][k], (int, float))
    ]

    result = {}
    for key in numeric_keys:
        values = [m[key] for m in fold_metrics if key in m]
        result[f"{key}_mean"] = float(np.mean(values))
        result[f"{key}_std"] = float(np.std(values))
        result[f"{key}_values"] = values

    return result


# ============================================================
# Dateiverwaltung
# ============================================================

def ensure_dir(path: Path) -> Path:
    """Stellt sicher, dass ein Verzeichnis existiert."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: dict, filepath: Path):
    """Speichert ein Dict als JSON."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str, ensure_ascii=False)


def load_json(filepath: Path) -> dict:
    """Lädt ein JSON-File."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)
