"""Generate remaining figures for Computer Vision chapter."""
import sys
sys.path.insert(0, '.')

import matplotlib
matplotlib.use('Agg')
from style import *
from matplotlib.patches import Rectangle

np.random.seed(42)

# ============================================================
# 1. Dice Coefficient — overlap visualization (similar to Jaccard)
# ============================================================
fig, axs = plt.subplots(1, 3, figsize=(16, 5))

for ax, overlap_pct, dice_val, title, color in [
    (axs[0], 0.9, 0.95, 'High Overlap', start_color),
    (axs[1], 0.5, 0.67, 'Moderate Overlap', middle_color),
    (axs[2], 0.1, 0.18, 'Low Overlap', end_color)
]:
    # Create simple segmentation masks
    mask_size = 20
    gt = np.zeros((mask_size, mask_size))
    pred = np.zeros((mask_size, mask_size))
    gt[4:16, 4:16] = 1  # Ground truth square
    offset = int((1 - overlap_pct) * 12)
    pred[4+offset:16+offset, 4:16] = 1  # Predicted, shifted

    # Color: green=both, blue=pred only, red=gt only
    display = np.zeros((mask_size, mask_size, 3))
    both = (gt == 1) & (pred == 1)
    gt_only = (gt == 1) & (pred == 0)
    pred_only = (pred == 1) & (gt == 0)
    # Use NannyML colors
    display[both] = [0.04, 0.65, 0.83]  # cyan - intersection
    display[gt_only] = [0.87, 0.25, 0.25]  # red - missed
    display[pred_only] = [0.23, 0.01, 0.50]  # purple - false positive
    display[(gt == 0) & (pred == 0)] = [0.95, 0.95, 0.95]  # light gray

    ax.imshow(display, interpolation='nearest')
    ax.set_title(f'{title}\nDice = {dice_val:.2f}', fontsize=16, color=color)
    ax.set_xticks([])
    ax.set_yticks([])

fig.tight_layout()
save_figure(fig, 'Dice_overlap_examples')
plt.close()
print("1. Dice: done")

# ============================================================
# 2. PQ — decomposition bar chart (SQ x RQ = PQ)
# ============================================================
fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
models = ['Model A\n(Good)', 'Model B\n(Good seg,\nbad detect)', 'Model C\n(Bad seg,\ngood detect)', 'Model D\n(Poor)']
sq = [0.85, 0.90, 0.55, 0.50]
rq = [0.90, 0.50, 0.85, 0.40]
pq = [s*r for s, r in zip(sq, rq)]

x = np.arange(len(models))
width = 0.25
ax.bar(x - width, sq, width, label='SQ (Segmentation)', color=start_color, edgecolor='black', linewidth=0.5)
ax.bar(x, rq, width, label='RQ (Recognition)', color=middle_color, edgecolor='black', linewidth=0.5)
ax.bar(x + width, pq, width, label='PQ = SQ × RQ', color=end_color, edgecolor='black', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11)
ax.set_ylabel('Score')
ax.set_ylim(0, 1.05)
ax.legend(fontsize=12)
fig.tight_layout()
save_figure(fig, 'PQ_decomposition')
plt.close()
print("2. PQ: done")

# ============================================================
# 3. SSIM — score vs distortion type (similar to LPIPS approach)
# ============================================================
fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
distortions = ['Original', 'Slight\nblur', 'Moderate\nblur', 'JPEG\ncompress', 'Salt &\npepper', 'Heavy\nnoise']
ssim_scores = [1.0, 0.92, 0.75, 0.68, 0.55, 0.30]
psnr_scores_norm = [1.0, 0.85, 0.60, 0.55, 0.40, 0.25]  # normalized to 0-1 range for comparison

x = np.arange(len(distortions))
width = 0.35
ax.bar(x - width/2, ssim_scores, width, label='SSIM', color=start_color, edgecolor='black', linewidth=0.5)
ax.bar(x + width/2, psnr_scores_norm, width, label='PSNR (normalized)', color=middle_color, edgecolor='black', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(distortions, fontsize=11)
ax.set_ylabel('Score')
ax.set_ylim(0, 1.1)
ax.legend(fontsize=13)
fig.tight_layout()
save_figure(fig, 'SSIM_vs_PSNR')
plt.close()
print("3. SSIM: done")

# ============================================================
# 4. OKS — similarity vs distance (Gaussian falloff)
# ============================================================
fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
distances = np.linspace(0, 3, 200)
for k, color, label in [
    (0.05, start_color, 'k=0.05 (strict, e.g. eyes)'),
    (0.10, middle_color, 'k=0.10 (moderate, e.g. shoulders)'),
    (0.20, end_color, 'k=0.20 (lenient, e.g. hips)')
]:
    oks = np.exp(-distances**2 / (2 * k**2))
    ax.plot(distances, oks, color=color, linewidth=4, solid_capstyle='round', label=label)

ax.set_xlabel('Normalized Distance (d / scale)')
ax.set_ylabel('OKS Similarity')
ax.set_ylim(0, 1.05)
ax.legend(fontsize=12)
ax.grid(True, linestyle='--', alpha=0.3)
fig.tight_layout()
save_figure(fig, 'OKS_gaussian_falloff')
plt.close()
print("4. OKS: done")

# ============================================================
# 5. Pixel Accuracy — class imbalance problem (pie + bar)
# ============================================================
fig, axs = plt.subplots(1, 2, figsize=FIGSIZE_COMPARISON)

# Left: class distribution (imbalanced)
sizes = [85, 10, 5]
labels_pie = ['Background (85%)', 'Class 1 (10%)', 'Class 2 (5%)']
colors_pie = ['#dddddd', start_color, middle_color]
axs[0].pie(sizes, labels=labels_pie, colors=colors_pie, autopct='%1.0f%%', startangle=90, textprops={'fontsize': 12})
axs[0].set_title('Class Distribution', fontsize=16)

# Right: PA vs mPA comparison
metrics_labels = ['Pixel\nAccuracy', 'Mean Pixel\nAccuracy']
pa = [0.90, 0.55]  # PA high because background dominates
colors_bar = [end_color, start_color]
bars = axs[1].bar(metrics_labels, pa, color=colors_bar, edgecolor='black', linewidth=0.5, width=0.5)
axs[1].set_ylim(0, 1)
axs[1].set_ylabel('Score')
axs[1].set_title('Accuracy Paradox', fontsize=16)
for bar, val in zip(bars, pa):
    axs[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{val:.0%}', ha='center', fontsize=15, fontweight='bold')

fig.tight_layout()
save_figure(fig, 'Pixel_Accuracy_imbalance')
plt.close()
print("5. Pixel Accuracy: done")

print("\nAll CV figures generated!")
