

import re
import argparse
from pathlib import Path
from typing import Callable, Dict, List

import pandas as pd

from src.config import (
    LABEL_MAP,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    TRAIN_FILE,
    TEST_FILE,
)


# Einzelne Preprocessing-Schritte


def remove_urls(text: str) -> str:
    """Ersetzt URLs durch ein Platzhalter-Token."""
    return re.sub(r"http\S+|www\S+|https\S+", "[URL]", text)


def normalize_usernames(text: str) -> str:
    """Ersetzt @-Mentions durch ein generisches Token."""
    return re.sub(r"@\w+", "@USER", text)


def remove_emojis(text: str) -> str:
    """
    Entfernt Emojis (alles, was nicht ASCII ist).
    
    Nutzt einen regulären Ausdruck, der die gängigen Unicode-Bereiche 
    für Emojis und Piktogramme abdeckt.
    """
    # Breites Emoji-Pattern (Unicode Emoji-Blöcke)
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # Emoticons
        "\U0001F300-\U0001F5FF"  # Misc Symbols
        "\U0001F680-\U0001F6FF"  # Transport
        "\U0001F1E0-\U0001F1FF"  # Flags
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001f926-\U0001f937"
        "\U00010000-\U0010ffff"
        "\u200d\u2640-\u2642\u2600-\u2B55\u23cf\u23e9\u231a\ufe0f\u3030"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub("", text)


def lowercase(text: str) -> str:
    """Konvertiert Text in Kleinbuchstaben."""
    return text.lower()


def normalize_whitespace(text: str) -> str:
    """Normalisiert mehrfache Leerzeichen."""
    return re.sub(r"\s+", " ", text).strip()


def remove_hashtag_symbol(text: str) -> str:
    """Entfernt nur das #-Zeichen, behält das Wort."""
    return re.sub(r"#(\w+)", r"\1", text)



# Zusammengesetzte Varianten für Ablation Study


def full_preprocessing(text: str) -> str:
    """Kombiniert alle Preprocessing-Schritte."""
    text = remove_urls(text)
    text = normalize_usernames(text)
    text = remove_emojis(text)
    text = remove_hashtag_symbol(text)
    text = normalize_whitespace(text)
    return text


def full_preprocessing_with_lowercase(text: str) -> str:
    """Wie full_preprocessing, zusätzlich Lowercase."""
    text = full_preprocessing(text)
    text = lowercase(text)
    return text


PREPROCESSING_VARIANTS: Dict[str, Callable[[str], str]] = {
    "original": lambda text: text,
    "remove_urls": remove_urls,
    "normalize_usernames": normalize_usernames,
    "remove_emojis": remove_emojis,
    "lowercase": lowercase,
    "full_preprocessing": full_preprocessing,
    "full_preprocessing_lowercase": full_preprocessing_with_lowercase,
}

# Daten laden (GermEval 2018 TSV-Format)


def load_germeval_file(filepath: Path) -> pd.DataFrame:
    """
    Lädt eine GermEval 2018 TSV-Datei.
    Format: Text \\t Coarse_Label \\t Fine_Label
    """
    rows = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                text = parts[0].strip()
                coarse_label = parts[1].strip()
                fine_label = parts[2].strip() if len(parts) > 2 else ""
                rows.append({
                    "text": text,
                    "coarse_label": coarse_label,
                    "fine_label": fine_label,
                })
    df = pd.DataFrame(rows)
    return df


def prepare_binary_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Konvertiert Coarse-Labels in binäre numerische Labels.
    OTHER → 0, OFFENSE → 1
    """
    df = df.copy()
    df["label"] = df["coarse_label"].map(LABEL_MAP)
    # Entferne Zeilen mit unbekannten Labels
    before = len(df)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    after = len(df)
    if before != after:
        print(f"  ⚠ {before - after} Zeilen mit unbekannten Labels entfernt.")
    return df


def apply_preprocessing(
    df: pd.DataFrame,
    variant: str = "original"
) -> pd.DataFrame:
    """Wendet eine Preprocessing-Variante auf die Text-Spalte an."""
    preprocess_fn = PREPROCESSING_VARIANTS[variant]
    df = df.copy()
    df["text"] = df["text"].apply(preprocess_fn)
    return df



# CLI: Rohdaten vorverarbeiten und speichern


def preprocess_and_save(input_dir: Path, output_dir: Path, variant: str = "full_preprocessing"):
    """Lädt Rohdaten, verarbeitet sie vor und speichert als CSV."""

    print(f"📦 Preprocessing-Variante: {variant}")
    print(f"📂 Input:  {input_dir}")
    print(f"📂 Output: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, filepath in [("train", TRAIN_FILE), ("test", TEST_FILE)]:
        if not filepath.exists():
            print(f"  ⚠ Datei nicht gefunden: {filepath}")
            continue

        print(f"\n  Verarbeite {split_name} ({filepath.name}) ...")
        df = load_germeval_file(filepath)
        print(f"    Rohdaten:  {len(df)} Einträge")

        df = prepare_binary_labels(df)
        df = apply_preprocessing(df, variant=variant)
        print(f"    Nach Preprocessing: {len(df)} Einträge")
        print(f"    Label-Verteilung:")
        for label_name, count in df["coarse_label"].value_counts().items():
            pct = count / len(df) * 100
            print(f"      {label_name}: {count} ({pct:.1f}%)")

        out_path = output_dir / f"{split_name}.csv"
        df.to_csv(out_path, index=False)
        print(f"    ✅ Gespeichert: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Preprocessing für GermEval 2018 Daten")
    parser.add_argument(
        "--input", type=str,
        default=str(RAW_DATA_DIR),
        help="Pfad zum GermEval-Rohdaten-Ordner",
    )
    parser.add_argument(
        "--output", type=str,
        default=str(PROCESSED_DATA_DIR),
        help="Ausgabe-Ordner für vorverarbeitete Daten",
    )
    parser.add_argument(
        "--variant", type=str,
        default="full_preprocessing",
        choices=list(PREPROCESSING_VARIANTS.keys()),
        help="Preprocessing-Variante",
    )
    args = parser.parse_args()
    preprocess_and_save(Path(args.input), Path(args.output), args.variant)


if __name__ == "__main__":
    main()
