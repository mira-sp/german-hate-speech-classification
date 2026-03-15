"""Generate plots for data exploration"""
import sys
import os
# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from src.config import LABEL_NAMES, LABEL_MAP
from src.preprocessing import PREPROCESSING_VARIANTS
from src.data_loader import load_data, get_data_stats

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# Load data
print("Loading data...")
train_df, test_df = load_data(preprocessing_variant='original')
print(f'Training Set: {len(train_df)} Samples')
print(f'Test Set: {len(test_df)} Samples')

# Define output path
output_dir = os.path.join(project_root, 'results', 'plots')

# 1. Class distribution plot
print("\nGenerating class distribution plot...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, df, title in zip(axes, [train_df, test_df], ['Training Set', 'Test Set']):
    counts = df['coarse_label'].value_counts()
    colors = ['#4CAF50', '#F44336']
    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor='gray')
    ax.set_title(f'{title} - Klassenverteilung', fontsize=14)
    ax.set_ylabel('Anzahl Tweets')
    for bar, count in zip(bars, counts.values):
        pct = count / len(df) * 100
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 30,
                f'{count}\n({pct:.1f}%)', ha='center', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'class_distribution.png'), dpi=300, bbox_inches='tight')
print("✓ Saved class_distribution.png")
plt.close()

# 2. Text length distribution
print("Generating text length distribution plot...")
train_df['word_count'] = train_df['text'].str.split().str.len()
train_df['char_count'] = train_df['text'].str.len()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for label, color, name in [(0, '#4CAF50', 'OTHER'), (1, '#F44336', 'OFFENSE')]:
    subset = train_df[train_df['label'] == label]
    axes[0].hist(subset['word_count'], bins=30, alpha=0.6, color=color, label=name, edgecolor='gray')
axes[0].set_xlabel('Wortanzahl')
axes[0].set_ylabel('Häufigkeit')
axes[0].set_title('Verteilung der Wortanzahl pro Klasse')
axes[0].legend()

for label, color, name in [(0, '#4CAF50', 'OTHER'), (1, '#F44336', 'OFFENSE')]:
    subset = train_df[train_df['label'] == label]
    axes[1].hist(subset['char_count'], bins=30, alpha=0.6, color=color, label=name, edgecolor='gray')
axes[1].set_xlabel('Zeichenanzahl')
axes[1].set_ylabel('Häufigkeit')
axes[1].set_title('Verteilung der Zeichenanzahl pro Klasse')
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'text_length_distribution.png'), dpi=300, bbox_inches='tight')
print("✓ Saved text_length_distribution.png")
plt.close()

# 3. Top words per class
print("Generating top words plot...")
STOPWORDS_DE = {'der', 'die', 'das', 'und', 'ist', 'in', 'den', 'von', 'zu', 'für',
                'mit', 'auf', 'des', 'im', 'ein', 'eine', 'es', 'an', 'dem', 'nicht',
                'als', 'auch', 'aus', 'dass', 'sich', 'wie', 'ich', 'er', 'sie', 'hat',
                'wir', 'was', 'so', 'oder', 'sind', 'aber', 'bei', 'nur', 'noch', 'man',
                'da', 'nach', 'schon', 'wenn', 'kann', 'werden', 'dort', 'über', 'haben',
                'wird', 'einem', 'doch', 'war', 'diese', 'du', 'mir', 'hier', 'rt'}

def get_top_words(texts, n=20):
    words = []
    for text in texts:
        for word in text.lower().split():
            word = word.strip('.,!?:;"\'-#@()')
            if word and word not in STOPWORDS_DE and len(word) > 2:
                words.append(word)
    return Counter(words).most_common(n)

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
for ax, label, title, color in [
    (axes[0], 0, 'OTHER (Top 20 Wörter)', '#4CAF50'),
    (axes[1], 1, 'OFFENSE (Top 20 Wörter)', '#F44336'),
]:
    texts = train_df[train_df['label'] == label]['text'].tolist()
    top = get_top_words(texts, 20)
    words, counts = zip(*top)
    ax.barh(range(len(words)), counts, color=color, edgecolor='gray')
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel('Häufigkeit')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'top_words_per_class.png'), dpi=300, bbox_inches='tight')
print("✓ Saved top_words_per_class.png")
plt.close()

# 4. Fine label distribution
print("Generating fine label distribution plot...")
fig, ax = plt.subplots(figsize=(10, 5))
fine_counts = train_df['fine_label'].value_counts()
fine_counts.plot(kind='bar', ax=ax, color=sns.color_palette('Set2', len(fine_counts)), edgecolor='gray')
ax.set_title('Fine-grained Label-Verteilung (Training)')
ax.set_ylabel('Anzahl')
ax.set_xlabel('Fine Label')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'fine_label_distribution.png'), dpi=300, bbox_inches='tight')
print("✓ Saved fine_label_distribution.png")
plt.close()

print("\nAll data exploration plots generated successfully!")
