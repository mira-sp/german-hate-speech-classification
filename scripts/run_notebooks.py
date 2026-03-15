"""
Execute all analysis notebooks programmatically.
"""
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from pathlib import Path
import sys

def execute_notebook(notebook_path):
    """Execute a Jupyter notebook."""
    print(f"\n{'='*70}")
    print(f"Executing: {notebook_path.name}")
    print('='*70)
    
    # Read notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    # Configure executor
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    
    try:
        # Execute
        ep.preprocess(nb, {'metadata': {'path': str(notebook_path.parent)}})
        
        # Write back
        with open(notebook_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        
        print(f"{notebook_path.name} completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error executing {notebook_path.name}:")
        print(f"   {str(e)}")
        return False

def main():
    notebooks_dir = Path(__file__).parent.parent / 'notebooks'
    
    # Notebooks to execute
    notebooks = [
        '01_data_exploration.ipynb',
        '02_baseline_results.ipynb',
        '03_bert_model_comparison.ipynb',
        '04_error_analysis.ipynb',
        '05_statistical_significance.ipynb',
    ]
    
    results = {}
    for nb_name in notebooks:
        nb_path = notebooks_dir / nb_name
        if nb_path.exists():
            results[nb_name] = execute_notebook(nb_path)
        else:
            print(f"{nb_name} not found!")
            results[nb_name] = False
    
    # Summary
    print(f"\n{'='*70}")
    print("Summary:")
    print('='*70)
    for nb_name, success in results.items():
        status = "OK" if success else "FAIL"
        print(f"  [{status}] {nb_name}")
    
    print(f"\n{'='*70}")
    if all(results.values()):
        print("All notebooks executed successfully!")
    else:
        print("Some notebooks failed - check output above")
        sys.exit(1)

if __name__ == '__main__':
    main()
