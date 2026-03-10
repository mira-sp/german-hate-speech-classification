"""
Fix notebook 03 - data size results loading.
"""
import json
from pathlib import Path

nb03_path = Path('notebooks/03_bert_model_comparison.ipynb')
with open(nb03_path, 'r', encoding='utf-8') as f:
    nb03 = json.load(f)

# Find and fix the data size loading cell
for cell in nb03['cells']:
    if cell['cell_type'] == 'code':
        source_text = ''.join(cell['source'])
        if 'for size_key in sorted(data_size_results.keys()' in source_text:
            # Replace the problematic loop
            new_source = source_text.replace(
                'for size_key in sorted(data_size_results.keys(),',
                'for size_key in sorted(data_size_results.get("results", data_size_results).keys(),'
            )
            new_source = new_source.replace(
                'data = data_size_results[size_key]',
                'data = data_size_results.get("results", data_size_results)[size_key]'
            )
            cell['source'] = [line + '\n' if line != new_source.split('\n')[-1] else line
                             for line in new_source.split('\n')]
            print("✅ Fixed data size loading in 03_bert_model_comparison.ipynb")

with open(nb03_path, 'w', encoding='utf-8') as f:
    json.dump(nb03, f, indent=1)
print(f"Saved {nb03_path}")
