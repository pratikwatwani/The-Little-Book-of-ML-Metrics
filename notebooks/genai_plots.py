"""Generate figures for GenAI chapter."""
import sys
sys.path.insert(0, '.')

import matplotlib
matplotlib.use('Agg')
from style import *

np.random.seed(42)

# ============================================================
# 1. Perplexity — PPL vs training steps (learning curve)
# ============================================================
fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
steps = np.arange(0, 1001, 50)
ppl = 500 * np.exp(-0.004 * steps) + 15 + np.random.randn(len(steps)) * 3
ax.plot(steps, ppl, color=start_color, **LINE_KW)
ax.set_xlabel('Training Steps')
ax.set_ylabel('Perplexity')
ax.set_ylim(0, 550)
ax.grid(True, linestyle='--', alpha=0.3)
ax.annotate('converged', xy=(800, 18), fontsize=14, color='gray')
fig.tight_layout()
save_figure(fig, 'Perplexity_training')
plt.close()
print("1. Perplexity: done")

# ============================================================
# 2. BERTScore — comparison bar chart: BERTScore vs BLEU
# ============================================================
fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
scenarios = ['Exact\nmatch', 'Synonym', 'Paraphrase', 'Negation', 'Random']
bertscore = [0.98, 0.88, 0.82, 0.45, 0.20]
bleu = [1.0, 0.15, 0.10, 0.35, 0.05]

x = np.arange(len(scenarios))
width = 0.35
ax.bar(x - width/2, bleu, width, label='BLEU', color=end_color, edgecolor='black', linewidth=0.5)
ax.bar(x + width/2, bertscore, width, label='BERTScore', color=start_color, edgecolor='black', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(scenarios, fontsize=12)
ax.set_ylabel('Score')
ax.set_ylim(0, 1.1)
ax.legend(fontsize=14)
fig.tight_layout()
save_figure(fig, 'BERTScore_vs_BLEU')
plt.close()
print("2. BERTScore: done")

# ============================================================
# 3. MAUVE — divergence curve (MAUVE frontier)
# ============================================================
fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
# Simulated MAUVE frontier curve
lambda_vals = np.linspace(0, 1, 100)
# Good model: high MAUVE area
kl_p = 0.5 * (1 - lambda_vals)**2
kl_q = 0.5 * lambda_vals**2
good_curve_x = np.exp(-kl_q)
good_curve_y = np.exp(-kl_p)
# Bad model: lower MAUVE area
kl_p_bad = 2.0 * (1 - lambda_vals)**2
kl_q_bad = 2.0 * lambda_vals**2
bad_curve_x = np.exp(-kl_q_bad)
bad_curve_y = np.exp(-kl_p_bad)

ax.plot(good_curve_x, good_curve_y, color=start_color, linewidth=4, label='Good model (MAUVE ≈ 0.92)')
ax.fill_between(good_curve_x, good_curve_y, alpha=0.1, color=start_color)
ax.plot(bad_curve_x, bad_curve_y, color=end_color, linewidth=4, label='Poor model (MAUVE ≈ 0.65)')
ax.fill_between(bad_curve_x, bad_curve_y, alpha=0.1, color=end_color)
ax.set_xlabel('$\\exp(-KL(Q \\| R_\\lambda))$', fontsize=14)
ax.set_ylabel('$\\exp(-KL(P \\| R_\\lambda))$', fontsize=14)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.legend(fontsize=13)
ax.grid(True, linestyle='--', alpha=0.3)
fig.tight_layout()
save_figure(fig, 'MAUVE_frontier')
plt.close()
print("3. MAUVE: done")

# ============================================================
# 4. IS — bar chart showing IS for different quality levels
# ============================================================
fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
models = ['Random\nNoise', 'Early\nTraining', 'Mid\nTraining', 'Well\nTrained', 'State of\nthe Art']
is_scores = [1.2, 3.5, 12.0, 35.0, 85.0]
colors = [end_color, end_color, middle_color, start_color, start_color]
bars = ax.bar(models, is_scores, color=colors, edgecolor='black', linewidth=0.5, width=0.6)
ax.set_ylabel('Inception Score')
for bar, val in zip(bars, is_scores):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
            f'{val:.1f}', ha='center', fontsize=13, fontweight='bold')
fig.tight_layout()
save_figure(fig, 'IS_quality_levels')
plt.close()
print("4. IS: done")

# ============================================================
# 5. FID — lower is better, comparison across models
# ============================================================
fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
models = ['State of\nthe Art', 'Well\nTrained', 'Mid\nTraining', 'Early\nTraining', 'Random\nNoise']
fid_scores = [5.2, 18.5, 45.0, 120.0, 280.0]
colors = [start_color, start_color, middle_color, end_color, end_color]
bars = ax.barh(models, fid_scores, color=colors, edgecolor='black', linewidth=0.5)
ax.set_xlabel('FID (lower is better)')
for bar, val in zip(bars, fid_scores):
    ax.text(bar.get_width() + 3, bar.get_y() + bar.get_height()/2.,
            f'{val:.1f}', va='center', fontsize=13, fontweight='bold')
ax.set_xlim(0, 320)
fig.tight_layout()
save_figure(fig, 'FID_quality_levels')
plt.close()
print("5. FID: done")

# ============================================================
# 6. LPIPS — perceptual distance vs pixel distance
# ============================================================
fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
# Simulated: LPIPS captures perceptual differences better than L2
distortion_levels = ['None', 'Blur', 'Noise', 'Color\nshift', 'Texture\nchange', 'Structure\nchange']
lpips = [0.0, 0.08, 0.12, 0.15, 0.35, 0.65]
l2 = [0.0, 0.10, 0.25, 0.30, 0.15, 0.20]

x = np.arange(len(distortion_levels))
width = 0.35
ax.bar(x - width/2, l2, width, label='L2 (pixel)', color=end_color, edgecolor='black', linewidth=0.5)
ax.bar(x + width/2, lpips, width, label='LPIPS (perceptual)', color=start_color, edgecolor='black', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(distortion_levels, fontsize=11)
ax.set_ylabel('Distance')
ax.legend(fontsize=14)
fig.tight_layout()
save_figure(fig, 'LPIPS_vs_L2')
plt.close()
print("6. LPIPS: done")

# ============================================================
# 7. CLIP Score — text-image alignment scatter
# ============================================================
fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
n = 40
# Good alignment
good_clip = 0.28 + np.random.rand(n//2) * 0.07
good_fid = 5 + np.random.rand(n//2) * 15
# Poor alignment
bad_clip = 0.15 + np.random.rand(n//2) * 0.08
bad_fid = 30 + np.random.rand(n//2) * 50

ax.scatter(good_clip, good_fid, color=start_color, s=80, label='Good alignment', edgecolors='black', linewidth=0.5)
ax.scatter(bad_clip, bad_fid, color=end_color, s=80, label='Poor alignment', edgecolors='black', linewidth=0.5)
ax.set_xlabel('CLIP Score (higher = better alignment)')
ax.set_ylabel('FID (lower = better quality)')
ax.legend(fontsize=13)
ax.grid(True, linestyle='--', alpha=0.3)
fig.tight_layout()
save_figure(fig, 'CLIP_Score_scatter')
plt.close()
print("7. CLIP Score: done")

# ============================================================
# 8. MOS — distribution of listener ratings
# ============================================================
fig, axs = plt.subplots(1, 2, figsize=FIGSIZE_COMPARISON)

# High quality audio
high_ratings = np.random.choice([4, 5], size=50, p=[0.3, 0.7])
low_ratings = np.random.choice([1, 2, 3], size=50, p=[0.3, 0.5, 0.2])

for ratings, ax, title, color, mos in [
    (high_ratings, axs[0], 'High Quality Audio', start_color, np.mean(high_ratings)),
    (low_ratings, axs[1], 'Low Quality Audio', end_color, np.mean(low_ratings))
]:
    bins = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    ax.hist(ratings, bins=bins, color=color, edgecolor='black', linewidth=0.5, rwidth=0.8)
    ax.axvline(x=np.mean(ratings), color='black', linestyle='--', linewidth=2)
    ax.set_xlabel('Listener Rating')
    ax.set_ylabel('Count')
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_title(f'{title}\n(MOS = {mos:.1f})', fontsize=15, color=color)
    ax.set_ylim(0, 40)

fig.tight_layout()
save_figure(fig, 'MOS_distributions')
plt.close()
print("8. MOS: done")

# ============================================================
# 9. VQAScore — alignment examples bar chart
# ============================================================
fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
prompts = ['Exact\nmatch', 'Partial\nmatch', 'Wrong\nattribute', 'Wrong\nobject', 'Unrelated']
vqa_scores = [0.95, 0.72, 0.35, 0.12, 0.03]
colors_list = [start_color, start_color, middle_color, end_color, end_color]
bars = ax.bar(prompts, vqa_scores, color=colors_list, edgecolor='black', linewidth=0.5, width=0.6)
ax.set_ylabel('VQAScore')
ax.set_ylim(0, 1.1)
for bar, val in zip(bars, vqa_scores):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
            f'{val:.2f}', ha='center', fontsize=13)
fig.tight_layout()
save_figure(fig, 'VQAScore_alignment')
plt.close()
print("9. VQAScore: done")

# ============================================================
# 10. PESQ — score range visualization
# ============================================================
fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
conditions = ['Clean\nspeech', 'Light\nnoise', 'Moderate\nnoise', 'Heavy\nnoise', 'Very\ndegraded']
pesq_scores = [4.5, 3.8, 2.9, 2.0, 1.2]
colors_list = [start_color, start_color, middle_color, end_color, end_color]
bars = ax.bar(conditions, pesq_scores, color=colors_list, edgecolor='black', linewidth=0.5, width=0.6)
ax.set_ylabel('PESQ Score')
ax.set_ylim(0, 5)
ax.axhline(y=4.5, color='gray', linestyle='--', alpha=0.3)
ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)
ax.annotate('Excellent', xy=(4.2, 4.6), fontsize=11, color='gray')
ax.annotate('Poor', xy=(4.2, 1.1), fontsize=11, color='gray')
for bar, val in zip(bars, pesq_scores):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.08,
            f'{val:.1f}', ha='center', fontsize=13, fontweight='bold')
fig.tight_layout()
save_figure(fig, 'PESQ_conditions')
plt.close()
print("10. PESQ: done")

# ============================================================
# 11. STOI — intelligibility vs noise level
# ============================================================
fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
snr = np.array([-10, -5, 0, 5, 10, 15, 20, 25, 30])
stoi = 1 / (1 + np.exp(-0.4 * (snr - 5)))  # Sigmoid curve
ax.plot(snr, stoi, color=start_color, marker='o', markersize=8, **LINE_KW)
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
ax.annotate('50% intelligibility', xy=(-8, 0.52), fontsize=12, color='gray')
ax.set_xlabel('Signal-to-Noise Ratio (dB)')
ax.set_ylabel('STOI')
ax.set_ylim(0, 1.05)
ax.grid(True, linestyle='--', alpha=0.3)
fig.tight_layout()
save_figure(fig, 'STOI_vs_SNR')
plt.close()
print("11. STOI: done")

print("\nAll GenAI figures generated!")
