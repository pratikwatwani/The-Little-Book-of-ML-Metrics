"""Generate figures for NLP chapter."""
import sys
sys.path.insert(0, '.')

import matplotlib
matplotlib.use('Agg')
from style import *

np.random.seed(42)

# ============================================================
# 1. BLEU — score vs n-gram order, showing how BLEU drops with higher n
# ============================================================
fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
n_grams = [1, 2, 3, 4]
# Simulated: good translation vs mediocre
good_scores = [0.85, 0.72, 0.61, 0.52]
mediocre_scores = [0.70, 0.45, 0.28, 0.15]
bad_scores = [0.55, 0.22, 0.08, 0.02]

ax.plot(n_grams, good_scores, color=start_color, marker='o', markersize=10, **LINE_KW, label='Good translation')
ax.plot(n_grams, mediocre_scores, color=middle_color, marker='s', markersize=10, **LINE_KW, label='Mediocre translation')
ax.plot(n_grams, bad_scores, color=end_color, marker='^', markersize=10, **LINE_KW, label='Poor translation')
ax.set_xlabel('N-gram Order')
ax.set_ylabel('Precision')
ax.set_xticks(n_grams)
ax.set_xticklabels(['Unigram', 'Bigram', 'Trigram', '4-gram'])
ax.set_ylim(0, 1)
ax.legend(fontsize=14)
ax.grid(True, linestyle='--', alpha=0.3)
fig.tight_layout()
save_figure(fig, 'BLEU_ngram_precision')
plt.close()
print("1. BLEU: done")

# ============================================================
# 2. METEOR — comparison with BLEU showing METEOR rewards synonyms
# ============================================================
fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
scenarios = ['Exact\nmatch', 'Synonym\nused', 'Word\nreordered', 'Paraphrase', 'Wrong\nmeaning']
bleu_scores = [0.95, 0.30, 0.40, 0.15, 0.10]
meteor_scores = [0.95, 0.75, 0.70, 0.55, 0.12]

x = np.arange(len(scenarios))
width = 0.35
bars1 = ax.bar(x - width/2, bleu_scores, width, label='BLEU', color=end_color, edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x + width/2, meteor_scores, width, label='METEOR', color=start_color, edgecolor='black', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(scenarios, fontsize=12)
ax.set_ylabel('Score')
ax.set_ylim(0, 1.1)
ax.legend(fontsize=14)
fig.tight_layout()
save_figure(fig, 'METEOR_vs_BLEU')
plt.close()
print("2. METEOR: done")

# ============================================================
# 3. ROUGE — ROUGE-1 vs ROUGE-2 vs ROUGE-L for different summary lengths
# ============================================================
fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
compression = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
rouge1 = [0.25, 0.40, 0.52, 0.62, 0.70, 0.78, 0.85, 0.90, 0.95]
rouge2 = [0.10, 0.22, 0.35, 0.45, 0.55, 0.65, 0.73, 0.82, 0.90]
rougeL = [0.20, 0.35, 0.47, 0.56, 0.64, 0.72, 0.80, 0.87, 0.93]

ax.plot(compression, rouge1, color=start_color, marker='o', markersize=8, linewidth=4, solid_capstyle='round', label='ROUGE-1')
ax.plot(compression, rouge2, color=middle_color, marker='s', markersize=8, linewidth=4, solid_capstyle='round', label='ROUGE-2')
ax.plot(compression, rougeL, color=end_color, marker='^', markersize=8, linewidth=4, solid_capstyle='round', label='ROUGE-L')
ax.set_xlabel('Summary Length / Reference Length')
ax.set_ylabel('ROUGE Score (Recall)')
ax.set_ylim(0, 1)
ax.legend(fontsize=14)
ax.grid(True, linestyle='--', alpha=0.3)
fig.tight_layout()
save_figure(fig, 'ROUGE_variants')
plt.close()
print("3. ROUGE: done")

# ============================================================
# 4. TER — edit operations breakdown for good vs bad translation
# ============================================================
fig, axs = plt.subplots(1, 2, figsize=FIGSIZE_COMPARISON)

ops = ['Insertions', 'Deletions', 'Substitutions', 'Shifts']

good_ops = [1, 0, 1, 1]  # TER = 3/15 = 0.20
bad_ops = [4, 3, 5, 2]   # TER = 14/15 = 0.93

for op_counts, ax, title, color, ter in [
    (good_ops, axs[0], 'Good Translation', start_color, 0.20),
    (bad_ops, axs[1], 'Poor Translation', end_color, 0.93)
]:
    bars = ax.barh(ops, op_counts, color=color, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Number of Edits')
    ax.set_title(f'{title} (TER = {ter:.2f})', fontsize=16, color=color)
    ax.set_xlim(0, 7)
    for bar, val in zip(bars, op_counts):
        if val > 0:
            ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2.,
                    str(val), va='center', fontsize=14)

fig.tight_layout()
save_figure(fig, 'TER_edit_breakdown')
plt.close()
print("4. TER: done")

# ============================================================
# 5. EM — strict vs partial matching comparison
# ============================================================
fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
categories = ['Exact\nmatch', 'Partial\noverlap', 'Off by\npunctuation', 'Synonym\nused', 'Completely\nwrong']
em_scores = [1.0, 0.0, 0.0, 0.0, 0.0]
f1_scores = [1.0, 0.75, 0.90, 0.60, 0.0]

x = np.arange(len(categories))
width = 0.35
bars1 = ax.bar(x - width/2, em_scores, width, label='Exact Match', color=start_color, edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x + width/2, f1_scores, width, label='Token F1', color=middle_color, edgecolor='black', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=11)
ax.set_ylabel('Score')
ax.set_ylim(0, 1.15)
ax.legend(fontsize=14)
fig.tight_layout()
save_figure(fig, 'EM_vs_F1')
plt.close()
print("5. EM: done")

# ============================================================
# 6. WER — WER vs number of errors for different reference lengths
# ============================================================
fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
errors = np.arange(0, 21)
for ref_len, color in [(10, start_color), (20, middle_color), (50, end_color)]:
    wer = errors / ref_len
    ax.plot(errors, wer, color=color, marker='o' if ref_len == 10 else ('s' if ref_len == 20 else '^'),
            markersize=6, linewidth=3, solid_capstyle='round', label=f'Ref length = {ref_len}')

ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.3, linewidth=1)
ax.annotate('WER = 100%', xy=(15, 1.02), fontsize=12, color='gray')
ax.set_xlabel('Number of Errors (S + D + I)')
ax.set_ylabel('Word Error Rate')
ax.set_ylim(0, 2.2)
ax.legend(fontsize=14)
ax.grid(True, linestyle='--', alpha=0.3)
fig.tight_layout()
save_figure(fig, 'WER_vs_errors')
plt.close()
print("6. WER: done")

print("\nAll NLP figures generated!")
