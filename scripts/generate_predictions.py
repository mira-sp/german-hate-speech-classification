"""
Generate predictions for all saved models and save detailed results for error analysis.

Creates:
- predictions/MODEL_predictions.csv: All predictions with probabilities
- predictions/MODEL_errors.csv: Only misclassified examples
- predictions/MODEL_false_positives.csv: FP examples sorted by confidence
- predictions/MODEL_false_negatives.csv: FN examples sorted by confidence

Usage:
    python scripts/generate_predictions.py
    python scripts/generate_predictions.py --model GBERT --fold 0
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import MODELS_DIR, LABEL_NAMES, MAX_LENGTH, RESULTS_DIR
from src.data_loader import load_data


def generate_predictions(
    model_path: Path,
    test_df: pd.DataFrame,
    batch_size: int = 64,
    device: str = "cuda",
) -> pd.DataFrame:
    """
    Generiert Vorhersagen für alle Beispiele.
    
    Returns:
        DataFrame mit Spalten: text, label, prediction, prob_OTHER, prob_OFFENSE, correct
    """
    print(f"  📦 Lade Modell von {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
    model.eval()
    model = model.to(device)
    
    predictions = []
    probabilities = []
    
    print(f"  🔮 Generiere Vorhersagen für {len(test_df)} Beispiele...")
    with torch.no_grad():
        for i in tqdm(range(0, len(test_df), batch_size), desc="  Batches"):
            batch_texts = test_df["text"].iloc[i:i+batch_size].tolist()
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            ).to(device)
            
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            
            predictions.extend(preds.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())
    
    # Ergebnisse zusammenführen
    results_df = test_df.copy()
    results_df["prediction"] = predictions
    results_df["prob_OTHER"] = [p[0] for p in probabilities]
    results_df["prob_OFFENSE"] = [p[1] for p in probabilities]
    results_df["correct"] = results_df["label"] == results_df["prediction"]
    
    # Prediction confidence (probability of predicted class)
    results_df["confidence"] = results_df.apply(
        lambda row: row["prob_OFFENSE"] if row["prediction"] == 1 else row["prob_OTHER"],
        axis=1
    )
    
    accuracy = results_df["correct"].mean()
    n_errors = (~results_df["correct"]).sum()
    print(f"  ✅ Accuracy: {accuracy:.4f} ({n_errors}/{len(results_df)} Fehler)")
    
    return results_df


def save_error_analysis(
    predictions_df: pd.DataFrame,
    output_dir: Path,
    model_name: str,
):
    """Speichert verschiedene Fehler-Ansichten."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Alle Vorhersagen
    all_path = output_dir / f"{model_name}_predictions.csv"
    predictions_df.to_csv(all_path, index=False)
    print(f"  💾 Alle Vorhersagen: {all_path}")
    
    # 2. Nur Fehler
    errors = predictions_df[~predictions_df["correct"]].copy()
    errors = errors.sort_values("confidence", ascending=False)
    errors_path = output_dir / f"{model_name}_errors.csv"
    errors.to_csv(errors_path, index=False)
    print(f"  💾 Fehler ({len(errors)}): {errors_path}")
    
    # 3. False Positives (Modell sagt OFFENSE, aber OTHER)
    fp = predictions_df[(predictions_df["label"] == 0) & (predictions_df["prediction"] == 1)].copy()
    fp = fp.sort_values("prob_OFFENSE", ascending=False)
    fp_path = output_dir / f"{model_name}_false_positives.csv"
    fp.to_csv(fp_path, index=False)
    print(f"  💾 False Positives ({len(fp)}): {fp_path}")
    
    # 4. False Negatives (Modell sagt OTHER, aber OFFENSE)
    fn = predictions_df[(predictions_df["label"] == 1) & (predictions_df["prediction"] == 0)].copy()
    fn = fn.sort_values("prob_OTHER", ascending=False)
    fn_path = output_dir / f"{model_name}_false_negatives.csv"
    fn.to_csv(fn_path, index=False)
    print(f"  💾 False Negatives ({len(fn)}): {fn_path}")
    
    # 5. Zusammenfassung
    summary = {
        "model": model_name,
        "total": len(predictions_df),
        "correct": int(predictions_df["correct"].sum()),
        "errors": int((~predictions_df["correct"]).sum()),
        "accuracy": float(predictions_df["correct"].mean()),
        "false_positives": len(fp),
        "false_negatives": len(fn),
        "fp_avg_confidence": float(fp["prob_OFFENSE"].mean()) if len(fp) > 0 else 0.0,
        "fn_avg_confidence": float(fn["prob_OTHER"].mean()) if len(fn) > 0 else 0.0,
    }
    
    summary_path = output_dir / f"{model_name}_error_summary.json"
    import json
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  💾 Zusammenfassung: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Generiere Vorhersagen für Fehleranalyse")
    parser.add_argument("--model", type=str, default=None, help="Modell-Name (z.B. GBERT, mBERT)")
    parser.add_argument("--fold", type=int, default=0, help="Fold-Nummer (0-4)")
    parser.add_argument("--preprocessing", type=str, default="full_preprocessing", help="Preprocessing-Variante")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch-Größe")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Device: {device}")
    
    # Testdaten laden
    print(f"\n📂 Lade Testdaten (Preprocessing: {args.preprocessing})...")
    _, test_df = load_data(preprocessing_variant=args.preprocessing)
    print(f"  ✅ {len(test_df)} Testbeispiele geladen")
    
    output_dir = RESULTS_DIR / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Modelle bestimmen
    if args.model:
        models_to_process = [(args.model.upper(), args.fold)]
    else:
        # Alle Modelle, bester Fold
        models_to_process = [
            ("GBERT", 0),
            ("mBERT", 0),
            ("HateBERT", 0),
        ]
    
    for model_name, fold in models_to_process:
        print(f"\n{'='*60}")
        print(f"  {model_name} (Fold {fold})")
        print(f"{'='*60}")
        
        model_path = MODELS_DIR / f"{model_name.lower()}_best" / f"fold_{fold}"
        
        if not model_path.exists():
            print(f"  ⚠️  Modell nicht gefunden: {model_path}")
            continue
        
        predictions_df = generate_predictions(
            model_path=model_path,
            test_df=test_df,
            batch_size=args.batch_size,
            device=device,
        )
        
        save_error_analysis(
            predictions_df=predictions_df,
            output_dir=output_dir,
            model_name=f"{model_name}_fold{fold}",
        )
    
    print(f"\n✅ Fertig! Alle Vorhersagen gespeichert in: {output_dir}")


if __name__ == "__main__":
    main()
