import json
import numpy as np
import math
import os
import sys

# Add src to path if needed, but this script seems standalone
# base_path is relative to notebook location in notebook, so adjusted here
base_path = os.path.join(os.path.dirname(__file__), "../results/metrics")

# Funktion zur Berechnung des gepaarten t-Tests
def paired_t_test(model_a_scores, model_b_scores, name_a="Model A", name_b="Model B"):
    """
    Berechnet den gepaarten t-Test für zwei Arrays von Scores.
    Gibt t-Wert und p-Wert zurück.
    """
    a = np.array(model_a_scores)
    b = np.array(model_b_scores)
    
    n = len(a)
    diffs = a - b
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs, ddof=1) # Sample Standard Deviation
    se_diff = std_diff / np.sqrt(n)
    
    if se_diff == 0:
        t_stat = 0
    else:
        t_stat = mean_diff / se_diff
    
    # p-Wert Berechnung manuell (ohne scipy, falls nicht installiert)
    # Für df=4 (n=5-1)
    df = n - 1
    
    # PDF der t-Verteilung
    def t_pdf(x, df):
        return (1 / (np.sqrt(df * np.pi) * math.gamma(df / 2)) * 
                math.gamma((df + 1) / 2) * 
                (1 + x**2 / df) ** (-(df + 1) / 2))

    # Numerical Integration für p-Value (wenn scipy fehlt)
    try:
        from scipy import stats
        p_value = stats.ttest_rel(a, b).pvalue
        method = "scipy.stats.ttest_rel"
    except ImportError:
        # Fallback: Numerische Integration
        method = "manual integration"
        if t_stat == 0:
            p_value = 1.0
        else:
            num_steps = 10000
            x_start = abs(t_stat)
            x_end = 100 # Approx infinity
            dx = (x_end - x_start) / num_steps
            
            area = 0
            for i in range(num_steps):
                x = x_start + i * dx
                area += t_pdf(x, df) * dx
            p_two_sided = area * 2
            p_value = p_two_sided

    print(f"\n--- {name_a} vs {name_b} ---")
    print(f"Mean Difference: {mean_diff:.4f}")
    print(f"Std Dev of Diff: {std_diff:.4f}")
    print(f"t-statistic:     {t_stat:.4f}")
    print(f"p-value (2-side): {p_value:.5f}")
    print(f"p-value (1-side): {p_value/2:.5f}")
    print(f"Method: {method}")
    
    return t_stat, p_value

def load_f1_scores(filename):
    filepath = os.path.join(base_path, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data['f1_macro_values']

try:
    gbert_f1 = load_f1_scores("GBERT_cv_results.json")
    mbert_f1 = load_f1_scores("mBERT_cv_results.json")
    hatebert_f1 = load_f1_scores("HateBERT_cv_results.json")
    
    print("Daten geladen:")
    print(f"GBERT Scores:    {gbert_f1}")
    print(f"mBERT Scores:    {mbert_f1}")
    print(f"HateBERT Scores: {hatebert_f1}")
    
    # Test 1: GBERT vs. mBERT
    t_gm, p_gm = paired_t_test(gbert_f1, mbert_f1, "GBERT", "mBERT")

    # Test 2: GBERT vs. HateBERT
    t_gh, p_gh = paired_t_test(gbert_f1, hatebert_f1, "GBERT", "HateBERT")

except Exception as e:
    print(f"Fehler: {e}")
    # Fallback Values
    gbert_f1 = [0.7959, 0.7961, 0.8136, 0.7936, 0.8408]
    mbert_f1 = [0.7947312018145308, 0.7716141168983255, 0.7615186615186615, 0.7670234458812064, 0.7472964335776289]
    hatebert_f1 = [0.6793593724464781, 0.673568233075757, 0.6663708255106104, 0.6775190026642299, 0.6649365072808961]
    print("\nVerwende Fallback-Werte (Log-basiert):")
    paired_t_test(gbert_f1, mbert_f1, "GBERT", "mBERT")
    paired_t_test(gbert_f1, hatebert_f1, "GBERT", "HateBERT")
