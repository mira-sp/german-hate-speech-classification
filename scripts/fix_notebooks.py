"""
Quick fix for notebook issues. 
"""
import json
from pathlib import Path

# Fix notebook 02 - change the baseline loading code
nb02_path = Path('notebooks/02_baseline_results.ipynb')
with open(nb02_path, 'r', encoding='utf-8') as f:
    nb02 = json.load(f)

# Find and fix the problematic cell
for cell in nb02['cells']:
    if cell['cell_type'] == 'code':
        source_text = ''.join(cell['source'])
        if 'baseline_df = pd.DataFrame.from_dict(baseline_results' in source_text:
            # Replace with corrected code
            cell['source'] = [
                "# Baseline Results\n",
                "baseline_path = METRICS_DIR / 'baseline_results.json'\n",
                "with open(baseline_path, 'r') as f:\n",
                "    baseline_data = json.load(f)\n",
                "\n",
                "# Convert list to DataFrame\n",
                "baseline_df = pd.DataFrame(baseline_data['baselines'])\n",
                "baseline_df = baseline_df.sort_values('f1_macro', ascending=False)\n",
                "baseline_df = baseline_df.set_index('model')\n",
                "\n",
                "print('Baseline-Modelle:')\n",
                "print(baseline_df[['f1_macro', 'precision_macro', 'recall_macro', 'accuracy']])"
            ]
            print("✅ Fixed cell in 02_baseline_results.ipynb")

with open(nb02_path, 'w', encoding='utf-8') as f:
    json.dump(nb02, f, indent=1)
print(f"Saved {nb02_path}")

# Fix notebook 03 - change the import
nb03_path = Path('notebooks/03_bert_model_comparison.ipynb')
with open(nb03_path, 'r', encoding='utf-8') as f:
    nb03 = json.load(f)

# Find and fix the problematic cell  
for cell in nb03['cells']:
    if cell['cell_type'] == 'code':
        source_text = ''.join(cell['source'])
        if 'plot_per_class_metrics' in source_text:
            # Replace with corrected import
            new_source = source_text.replace('plot_per_class_metrics', 'plot_metrics_per_class')
            cell['source'] = new_source.split('\n')
            # Ensure each line ends with \n except the last
            cell['source'] = [line + '\n' if i < len(cell['source'])-1 else line
                             for i, line in enumerate(cell['source'])]
            print("✅ Fixed import in 03_bert_model_comparison.ipynb")

with open(nb03_path, 'w', encoding='utf-8') as f:
    json.dump(nb03, f, indent=1)
print(f"Saved {nb03_path}")

print("\n✅ All fixes applied!")
