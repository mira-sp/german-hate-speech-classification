"""
Rebuild data_size_variation_results.json from CSV file.
Fixes the issue where 10%, 25%, 50% data was lost.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

# Paths
csv_path = Path("results/metrics/data_size_variation_results.csv")
json_path = Path("results/metrics/data_size_variation_results.json")

# Load CSV
df = pd.read_csv(csv_path)

# Group by data_size
results_by_size = {}

for size in df['data_size'].unique():
    size_data = df[df['data_size'] == size]
    
    # Get fold results
    fold_results = []
    for _, row in size_data.iterrows():
        fold_results.append({
            "f1_macro": row['f1_macro'],
            "accuracy": row['accuracy'],
            "train_time_sec": row['train_time_sec'],
            "fold": int(row['fold'])
        })
    
    # Compute aggregated stats
    f1_values = size_data['f1_macro'].tolist()
    acc_values = size_data['accuracy'].tolist()
    train_size = int(size_data['train_n'].iloc[0])
    
    # Note: We're missing some detailed per-class metrics
    # But we can at least restore the main metrics
    results_by_size[size] = {
        "model": "GBERT",
        "accuracy_mean": float(np.mean(acc_values)),
        "accuracy_std": float(np.std(acc_values)),
        "accuracy_values": acc_values,
        "f1_macro_mean": float(np.mean(f1_values)),
        "f1_macro_std": float(np.std(f1_values)),
        "f1_macro_values": f1_values,
        "f1_micro_mean": float(np.mean(acc_values)),  # micro == accuracy for this task
        "f1_micro_std": float(np.std(acc_values)),
        "f1_micro_values": acc_values,
        "fold_results": fold_results,
        "train_size": train_size,
        "data_fraction": float(size.strip('%')) / 100
    }

# Create final JSON structure
final_json = {
    "experiment": "data_size_variation",
    "model": "GBERT",
    "results": results_by_size
}

# Save
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(final_json, f, indent=2)

print(f"✅ Rebuilt JSON with {len(results_by_size)} data sizes:")
for size in sorted(results_by_size.keys(), key=lambda x: float(x.strip('%'))):
    r = results_by_size[size]
    print(f"  {size}: F1={r['f1_macro_mean']:.4f}±{r['f1_macro_std']:.4f}, "
          f"Acc={r['accuracy_mean']:.4f}±{r['accuracy_std']:.4f}, n={r['train_size']}")
print(f"\n💾 Saved to: {json_path}")
