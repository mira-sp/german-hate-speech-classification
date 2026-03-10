"""
Regenerate the learning curve plot with all data sizes.
"""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Paths
json_path = Path("results/metrics/data_size_variation_results.json")
plot_path = Path("results/plots/learning_curve.png")

# Load data
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

results = data['results']
model_name = data['model']

# Extract data
sizes_pct = []
train_sizes = []
f1_means = []
f1_stds = []
acc_means = []
acc_stds = []

for size_key in sorted(results.keys(), key=lambda x: float(x.strip("%")) / 100):
    r = results[size_key]
    pct = float(size_key.strip("%"))
    sizes_pct.append(pct)
    train_sizes.append(r.get("train_size", 0))
    f1_means.append(r["f1_macro_mean"])
    f1_stds.append(r["f1_macro_std"])
    acc_means.append(r["accuracy_mean"])
    acc_stds.append(r["accuracy_std"])

f1_means = np.array(f1_means)
f1_stds = np.array(f1_stds)
acc_means = np.array(acc_means)
acc_stds = np.array(acc_stds)

# Create plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# F1 Score
ax1.plot(sizes_pct, f1_means, "o-", linewidth=2, markersize=8, color="#2196F3", label="F1 (Macro)")
ax1.fill_between(sizes_pct, f1_means - f1_stds, f1_means + f1_stds, alpha=0.2, color="#2196F3")
ax1.set_xlabel("Training Data Size (%)", fontsize=12)
ax1.set_ylabel("F1-Score (Macro)", fontsize=12)
ax1.set_title(f"Learning Curve: {model_name} – F1 Score", fontsize=14)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=11)
ax1.set_xticks(sizes_pct)

# Accuracy
ax2.plot(sizes_pct, acc_means, "s-", linewidth=2, markersize=8, color="#4CAF50", label="Accuracy")
ax2.fill_between(sizes_pct, acc_means - acc_stds, acc_means + acc_stds, alpha=0.2, color="#4CAF50")
ax2.set_xlabel("Training Data Size (%)", fontsize=12)
ax2.set_ylabel("Accuracy", fontsize=12)
ax2.set_title(f"Learning Curve: {model_name} – Accuracy", fontsize=14)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=11)
ax2.set_xticks(sizes_pct)

plt.tight_layout()
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"✅ Learning curve regenerated with {len(sizes_pct)} data sizes:")
for i, pct in enumerate(sizes_pct):
    print(f"  {pct:.0f}%: F1={f1_means[i]:.4f}±{f1_stds[i]:.4f}, Acc={acc_means[i]:.4f}±{acc_stds[i]:.4f}")
print(f"\n📊 Plot saved to: {plot_path}")
