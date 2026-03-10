"""
models.py - Modell-Definitionen und Laden von BERT-Modellen.

Unterstützt: mBERT, GBERT, HateBERT
"""

from typing import Optional, Tuple

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from src.config import MODELS, NUM_LABELS, SEED


def load_model_and_tokenizer(
    model_key: str,
    num_labels: int = NUM_LABELS,
    device: Optional[str] = None,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Lädt ein BERT-Modell und den zugehörigen Tokenizer.

    Args:
        model_key: Schlüssel aus MODELS dict (z.B. "GBERT", "mBERT", "HateBERT").
        num_labels: Anzahl der Klassen.
        device: Zielgerät ("cuda", "cpu", oder None für Auto).

    Returns:
        (model, tokenizer)
    """
    if model_key not in MODELS:
        raise ValueError(
            f"Unbekanntes Modell: {model_key}. "
            f"Verfügbar: {list(MODELS.keys())}"
        )

    model_path = MODELS[model_key]
    print(f"🔄 Lade {model_key} ({model_path}) ...")

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=num_labels,
    )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    print(f"  ✅ Modell geladen auf {device}")
    print(f"  📊 Parameter: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    return model, tokenizer


def get_model_info(model_key: str) -> dict:
    """Gibt Informationen über ein Modell zurück."""
    model_path = MODELS[model_key]

    # Lade nur Config, nicht das ganze Modell
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_path)

    return {
        "name": model_key,
        "hub_path": model_path,
        "hidden_size": config.hidden_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "vocab_size": config.vocab_size,
        "max_position_embeddings": config.max_position_embeddings,
    }


def free_model_memory(model: PreTrainedModel):
    """Gibt GPU-Speicher frei nach dem Training eines Modells."""
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("🧹 GPU-Speicher freigegeben.")


def check_gpu():
    """Prüft GPU-Verfügbarkeit und gibt Informationen aus."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU verfuegbar: {gpu_name} ({gpu_mem:.1f} GB)")
        return True
    else:
        print("⚠ Keine GPU verfügbar! Training wird auf CPU ausgeführt (sehr langsam).")
        return False
