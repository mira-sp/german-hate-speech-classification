# Writing Plan: Monolingual Hate Speech Classification for German Text

**Target Length:** 4 pages (approx. 2,000-2,400 words)  
**Dataset:** GermEval 2018 (~8,500 German tweets)  
**Best Model:** GBERT (F1=0.8097±0.0106)

---

## 1. Title & Abstract (0.2 pages, ~100-150 words)

### Title

- **Suggested:** "German Hate Speech Detection: Comparing Language-Specific BERT Models with Traditional Baselines"
- **Alternative:** "Monolingual vs. Multilingual BERT for German Hate Speech Classification on GermEval 2018"

### Abstract

**TODO - Write concise summary covering:**

- [ ] Problem statement (hate speech detection in German social media)
- [ ] Approach (comparison of baselines vs. BERT models)
- [ ] Dataset (GermEval 2018, binary classification)
- [ ] Key finding (GBERT achieves F1=0.8097, outperforms multilingual models by 2.3%)
- [ ] Main conclusion (language-specific models > multilingual for hate speech)

---

## 2. Introduction (0.8 pages, ~400-450 words)

### Subsection 2.1: Motivation & Background

**TODO - Write about:**

- [ ] Rising importance of automatic hate speech detection on social media
- [ ] Challenges specific to German language (compound words, grammatical complexity)
- [ ] Need for language-specific vs. multilingual models
- [ ] Gap: Limited comparison of BERT variants on German hate speech

### Subsection 2.2: Research Questions

**TODO - Clearly state:**

- [ ] **RQ1:** How do traditional ML baselines compare to transformer-based models?
- [ ] **RQ2:** Does language-specific BERT (GBERT) outperform multilingual BERT (mBERT)?
- [ ] **RQ3:** What is the impact of training data size on model performance?
- [ ] **RQ4:** How does text preprocessing affect classification accuracy?

### Subsection 2.3: Hypothesis

**TODO - State clear hypotheses:**

- [ ] **H1:** BERT-based models will significantly outperform traditional baselines
- [ ] **H2:** Language-specific GBERT will achieve higher F1 than multilingual mBERT
- [ ] **H3:** HateBERT (trained on English hate speech) will underperform on German text
- [ ] **H4:** Aggressive preprocessing will improve model performance

**Expected outcome:** GBERT > mBERT > Baselines; HateBERT underperforms due to language mismatch

---

## 3. Related Work (Optional, 0.3 pages)

**TODO - Brief literature review (if space allows):**

- [ ] Cite GermEval 2018 shared task papers
- [ ] Mention mBERT and language-specific BERT models
- [ ] Reference hate speech detection benchmarks
- [ ] Position your work in context

**NOTE:** Can be condensed or merged into Introduction if space is tight.

---

## 4. Dataset & Methodology (1.0 pages, ~500-550 words)

### Subsection 4.1: GermEval 2018 Dataset

**TODO - Describe dataset:**

- [ ] Source: GermEval 2018 Shared Task on German hate speech
- [ ] Size: ~8,500 German tweets
- [ ] Task: Binary classification (OFFENSE vs. OTHER)
- [ ] Class distribution: ~66% OTHER, ~34% OFFENSE (imbalanced)
- [ ] Language characteristics: informal German, slang, emojis, URLs, @mentions

**REFERENCE DATA:**

- From `notebooks/01_data_exploration.ipynb` (if you have distribution stats)
- Mention training set size: ~6,000 tweets for training

### Subsection 4.2: Experimental Setup

#### 4.2.1: Baseline Models

**TODO - Describe each baseline:**

- [ ] **Majority Classifier:** Always predicts most frequent class
- [ ] **Lexicon-Based:** Uses German offensive word dictionary
- [ ] **Random Forest:** TF-IDF features (n-grams 1-3) + 100 trees

**REFERENCE RESULTS:**

```
Majority:      F1 = 0.3966
Lexicon:       F1 = 0.5366
Random Forest: F1 = 0.6707
```

#### 4.2.2: BERT Models

**TODO - Describe each BERT variant:**

- [ ] **GBERT:** deepset/gbert-base (German-specific, trained on German Wikipedia/news)
- [ ] **mBERT:** bert-base-multilingual-cased (104 languages)
- [ ] **HateBERT:** GroNLP/hateBERT (English hate speech corpus)

**TODO - Training details:**

- [ ] Fine-tuning setup: Learning rate 2e-5, batch size 16, 3 epochs
- [ ] Sequence length: 128 tokens
- [ ] Optimizer: AdamW with linear warmup
- [ ] 5-fold stratified cross-validation for robust evaluation
- [ ] Hardware: NVIDIA RTX 3060, ~8 minutes per fold

**REFERENCE CONFIG:**

- From `src/config.py` lines 40-77

#### 4.2.3: Evaluation Metrics

**TODO - Explain metrics:**

- [ ] Primary metric: **F1 (Macro)** - balances precision/recall, handles class imbalance
- [ ] Secondary: Accuracy, per-class F1, precision, recall
- [ ] Why F1 Macro: Equal importance to both OFFENSE and OTHER classes
- [ ] Cross-validation: Report mean ± std across 5 folds

---

## 5. Experiments (0.5 pages, ~250-300 words)

### Subsection 5.1: Four Key Experiments

**TODO - Briefly describe each experiment:**

#### Experiment 1: Baseline Comparison

- [ ] Purpose: Establish non-transformer baselines
- [ ] Models: Majority, Lexicon, Random Forest
- [ ] Finding: Random Forest achieves F1=0.6707 (best baseline)

**FIGURE REFERENCE:** `results/plots/model_comparison.png`

#### Experiment 2: BERT Full Data Comparison

- [ ] Purpose: Compare BERT variants on full training set
- [ ] Models: GBERT, mBERT, HateBERT
- [ ] Finding: GBERT wins (F1=0.8097)

**TABLE REFERENCE:** Generate from `results/metrics/bert_full_data_results.json`

#### Experiment 3: Data Size Variation (Learning Curve)

- [ ] Purpose: Assess data efficiency
- [ ] Tested: 10%, 25%, 50%, 75%, 100% of training data
- [ ] Finding: Performance plateaus after 50% (diminishing returns)

**FIGURE REFERENCE:** `results/plots/learning_curve.png`

#### Experiment 4: Preprocessing Ablation

- [ ] Purpose: Determine optimal preprocessing strategy
- [ ] Tested 7 variants: original, remove_urls, normalize_usernames, remove_emojis, lowercase, full_preprocessing, full_preprocessing_lowercase
- [ ] Finding: Full preprocessing (F1=0.8097) slightly better than original (F1=0.8036), lowercasing hurts performance

**FIGURE REFERENCE:** `results/plots/preprocessing_ablation.png`

---

## 6. Results (0.8 pages, ~400-450 words)

### Subsection 6.1: Overall Model Comparison

**TODO - Present main findings:**

- [ ] GBERT achieves best F1=0.8097±0.0106 (accuracy 83.34%)
- [ ] mBERT second: F1=0.7874±0.0082 (−2.3% vs GBERT)
- [ ] HateBERT worst BERT: F1=0.6724±0.0058 (−13.7% vs GBERT)
- [ ] Random Forest best baseline: F1=0.6707
- [ ] BERT improvement over baseline: +20.7% relative improvement

**TABLE 1: Overall Model Performance**

```
Model          | F1 (Macro)     | Accuracy       | Training Time
---------------|----------------|----------------|---------------
GBERT          | 0.8097 ± 0.011 | 0.8334 ± 0.008 | ~7.9 min
mBERT          | 0.7874 ± 0.008 | 0.8110 ± 0.006 | ~7.8 min
HateBERT       | 0.6724 ± 0.006 | 0.7123 ± 0.010 | ~7.7 min
Random Forest  | 0.6707         | 0.6939         | <1 min
Lexicon        | 0.5366         | 0.6939         | <1 sec
Majority       | 0.3966         | 0.6573         | <1 sec
```

**SOURCE:** `results/metrics/bert_full_data_results.json` + `baseline_results.json`  
**FIGURE:** `results/plots/model_comparison.png`

### Subsection 6.2: Per-Class Analysis

**TODO - Analyze class-specific performance:**

- [ ] GBERT excels at both classes (OTHER: F1=0.88, OFFENSE: F1=0.74)
- [ ] All models struggle more with OFFENSE (minority class)
- [ ] HateBERT particularly weak on OFFENSE detection (wrong language domain)

**FIGURE REFERENCE:** `results/plots/per_class_metrics.png`

### Subsection 6.3: Learning Curve Analysis

**TODO - Discuss data efficiency:**

- [ ] 10% data: F1=0.5214 (baseline level)
- [ ] 50% data: F1=0.7880 (97% of full performance)
- [ ] 75% data: F1=0.7914 (minimal improvement)
- [ ] 100% data: F1=0.8097
- [ ] Conclusion: Diminishing returns after 50%, but full data needed for peak performance

**FIGURE REFERENCE:** `results/plots/learning_curve.png`

### Subsection 6.4: Preprocessing Impact

**TODO - Analyze preprocessing variants:**

- [ ] Full preprocessing best: F1=0.8097 (remove URLs, normalize @mentions, remove emojis)
- [ ] Remove URLs alone: F1=0.8053 (−0.4%)
- [ ] Normalize usernames: F1=0.8050 (−0.5%)
- [ ] Original (minimal preprocessing): F1=0.8036 (−0.6%)
- [ ] **Lowercase HURTS performance:** F1=0.7891 (−2.1%) — case information is important!

**FIGURE REFERENCE:** `results/plots/preprocessing_ablation.png`  
**TABLE REFERENCE:** `results/metrics/preprocessing_ablation_results.csv`

### Subsection 6.5: Confusion Matrix Analysis

**TODO - Error analysis:**

- [ ] GBERT: 715 errors on fold 0 (3,414 test samples = 79.1% accuracy)
- [ ] False Negatives (590): Model predicts OTHER but true label is OFFENSE
  - Users express offense implicitly/sarcastically
  - Mild offensive language missed
- [ ] False Positives (125): Model predicts OFFENSE but true label is OTHER
  - Aggressive but non-offensive political statements
  - Quoted offensive content

**FIGURE REFERENCE:** `results/plots/cm_gbert.png`  
**DATA REFERENCE:** `results/predictions/GBERT_fold0_false_negatives.csv` (inspect top examples)

**TODO - Select 2-3 specific error examples for qualitative discussion:**

- [ ] Example 1 (FN): [Pick from false_negatives.csv - high confidence error]
- [ ] Example 2 (FP): [Pick from false_positives.csv - high confidence error]

---

## 7. Discussion (0.4 pages, ~200-250 words)

### Why These Results Make Sense

**TODO - Provide theoretical justification:**

#### GBERT Outperforms mBERT

- [ ] Language-specific pre-training captures German linguistic nuances
- [ ] German compound words, case system, word order
- [ ] mBERT dilutes language-specific knowledge across 104 languages
- [ ] Confirms prior findings: monolingual > multilingual for single-language tasks

#### HateBERT Underperforms Dramatically

- [ ] Pre-trained on **English** hate speech, applied to **German**
- [ ] Language mismatch more important than domain match
- [ ] Vocabulary mismatch: German words treated as subwords/unknowns
- [ ] Lesson: Language alignment > domain alignment

#### Lowercase Hurts Performance

- [ ] German has productive case distinctions (nouns capitalized)
- [ ] Case carries semantic information (e.g., "Türken" vs. "türken")
- [ ] Offensive terms may have distinctive capitalization patterns
- [ ] BERT tokenizers are case-sensitive for good reason

#### Preprocessing Helps Modestly

- [ ] URLs, @mentions are noise (don't carry hate speech signal)
- [ ] Removing them focuses model on actual content
- [ ] But improvement is small (+0.6%) — BERT already robust to noise
- [ ] Emojis removal has minimal impact (model learns to use/ignore them)

#### Learning Curve Plateau

- [ ] 50% data captures most linguistic patterns
- [ ] Remaining data adds edge cases, rare patterns
- [ ] Diminishing returns typical for neural models
- [ ] Full data still recommended for competition/production

---

## 8. Limitations & Future Work (Optional, 0.2 pages)

**TODO - Discuss limitations (if space allows):**

- [ ] Dataset: Twitter-specific, may not generalize to other platforms
- [ ] Binary classification: Doesn't distinguish severity of offense
- [ ] Class imbalance: Could explore oversampling/undersampling
- [ ] Single language: Findings may not transfer to other languages

**TODO - Suggest future directions:**

- [ ] Multi-task learning (severity + binary)
- [ ] Ensemble methods (combine GBERT + preprocessing variants)
- [ ] Larger models (GBERT-large, GPT-based)
- [ ] Cross-platform evaluation (Reddit, Facebook comments)

---

## 9. Conclusion (0.4 pages, ~200-250 words)

**TODO - Summarize key takeaways:**

### Main Findings

- [ ] **RQ1 Answer:** BERT models substantially outperform traditional baselines (+20% relative)
- [ ] **RQ2 Answer:** GBERT (language-specific) beats mBERT by 2.3% F1
- [ ] **RQ3 Answer:** 50% data achieves 97% of full performance (data-efficient)
- [ ] **RQ4 Answer:** Full preprocessing improves F1 slightly (+0.6%), lowercase hurts (−2.1%)

### Hypothesis Validation

- [ ] ✅ H1 confirmed: BERT >> baselines
- [ ] ✅ H2 confirmed: GBERT > mBERT
- [ ] ✅ H3 confirmed: HateBERT underperforms (language mismatch)
- [ ] ❌ H4 partially rejected: Preprocessing helps minimally; lowercase harms

### Practical Implications

- [ ] For German hate speech detection: Use GBERT
- [ ] Language-specific models worth the training cost
- [ ] 50% data may suffice for prototyping; use 100% for production
- [ ] Preserve case information in preprocessing

### Closing Statement

- [ ] Final sentence: Emphasize importance of language-specific models for low-resource languages
- [ ] Call to action: Encourage development of more monolingual BERT variants

---

## 10. References

**TODO - Cite key papers:**

- [ ] GermEval 2018 shared task paper
- [ ] BERT original paper (Devlin et al., 2019)
- [ ] GBERT paper (Chan et al., 2020 / deepset documentation)
- [ ] HateBERT paper (Caselli et al., 2021)
- [ ] Relevant hate speech detection surveys

---

## Appendix: Figure & Table Placement Guide

### Figures to Include (Select 3-4 for 4 pages):

**Priority Figures:**

1. **Figure 1: Model Comparison Bar Chart** → Section 6.1  
   `results/plots/model_comparison.png`
2. **Figure 2: Learning Curve** → Section 6.3  
   `results/plots/learning_curve.png`
3. **Figure 3: Preprocessing Ablation** → Section 6.4  
   `results/plots/preprocessing_ablation.png`

**Optional Figures (if space allows):** 4. **Figure 4: GBERT Confusion Matrix** → Section 6.5  
 `results/plots/cm_gbert.png`

5. **Figure 5: Per-Class Metrics** → Section 6.2  
   `results/plots/per_class_metrics.png`

### Tables to Include (2-3 tables max):

**Priority Tables:**

1. **Table 1: Overall Model Performance** → Section 6.1  
   Source: Combine `bert_full_data_results.json` + `baseline_results.json`
2. **Table 2: Learning Curve Results** → Section 6.3  
   Source: `data_size_variation_results.json`

**Optional Tables:** 3. **Table 3: Preprocessing Ablation** → Section 6.4  
 Source: `preprocessing_ablation_results.csv`

---

## Writing Tips & Word Count Allocation

| Section        | Target Words    | % of Paper |
| -------------- | --------------- | ---------- |
| Abstract       | 100-150         | 5%         |
| Introduction   | 400-450         | 20%        |
| Dataset/Method | 500-550         | 25%        |
| Experiments    | 250-300         | 12%        |
| Results        | 400-450         | 20%        |
| Discussion     | 200-250         | 10%        |
| Conclusion     | 200-250         | 10%        |
| **TOTAL**      | **2,050-2,400** | **100%**   |

---

## File References (For Writing)

### Data Files

- `results/metrics/baseline_results.json` → Baseline numbers
- `results/metrics/bert_full_data_results.json` → Main BERT results
- `results/metrics/GBERT_cv_results.json` → GBERT details
- `results/metrics/data_size_variation_results.json` → Learning curve
- `results/metrics/preprocessing_ablation_results.json` → Preprocessing experiments
- `results/predictions/GBERT_fold0_false_negatives.csv` → Error examples

### Visualization Files

- `results/plots/model_comparison.png` → Main comparison
- `results/plots/learning_curve.png` → Data efficiency
- `results/plots/preprocessing_ablation.png` → Preprocessing impact
- `results/plots/cm_gbert.png` → Confusion matrix
- `results/plots/per_class_metrics.png` → Class-specific performance

### Notebook Files (For Reference)

- `notebooks/02_baseline_results.ipynb` → Baseline analysis + LaTeX tables
- `notebooks/03_bert_model_comparison.ipynb` → Complete BERT analysis
- `notebooks/04_error_analysis.ipynb` → Error patterns, example selection

---

## Next Steps

1. **Review all result files** to extract exact numbers
2. **Select 2-3 error examples** from false positive/negative CSVs
3. **Start writing sections 1-3** (Title, Introduction, Dataset)
4. **Generate LaTeX tables** from notebooks for copy-paste
5. **Write results section** with figures embedded
6. **Draft Discussion** with theoretical justification
7. **Polish Conclusion** and Abstract last
8. **Format figures** (ensure high quality, readable labels)
9. **Proofread** for consistency in terminology and numbers
10. **Check page count** and adjust if needed

**Estimated writing time:** 6-8 hours for first draft

---

**Good luck with your paper! 🎓**
