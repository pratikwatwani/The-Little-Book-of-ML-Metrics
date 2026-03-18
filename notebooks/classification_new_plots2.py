"""Generate remaining classification metric figures."""
import sys
sys.path.insert(0, '.')

import matplotlib
matplotlib.use('Agg')
from style import *
from sklearn.metrics import confusion_matrix

np.random.seed(42)

# ============================================================
# 1. TNR (Specificity) — 3D surface + 2D line (match FPR/FNR pattern)
# ============================================================
tn = np.linspace(0, 100, 101)
fp = np.linspace(0, 100, 101)
TN, FP = np.meshgrid(tn, fp)
TNR = TN / (TN + FP)
TNR = np.nan_to_num(TNR, 0)

fig = plt.figure(figsize=FIGSIZE_LARGE)
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(TN, FP, TNR, cmap=nml_cmap.reversed(), linewidth=0, antialiased=False)
ax.set_xlabel('True Negatives (TN)')
ax.set_ylabel('False Positives (FP)')
ax.set_zlabel('TNR')
ax.set_ylim(100, 0)
ax.set_xlim(0, 100)
ax.set_xticks([0, 25, 50, 75])
ax.set_yticks([0, 25, 50, 75, 100])
ax.set_zticks([0.2, 0.4, 0.6, 0.8, 1.0])
cbar = fig.colorbar(surf)
cbar.set_label('TNR')
ax.view_init(elev=20, azim=-65)
fig.tight_layout()
save_figure(fig, 'TNR_3d_surface')
plt.close()
print("1. TNR: done")

# ============================================================
# 2. Balanced Accuracy — imbalanced scenario comparison
# ============================================================
fig, axs = plt.subplots(1, 2, figsize=FIGSIZE_COMPARISON)

metrics = ['Accuracy', 'Balanced\nAccuracy']
# Scenario: 95% negative class, model always predicts negative
always_neg = [0.95, 0.50]
decent_model = [0.85, 0.82]

x = np.arange(len(metrics))
width = 0.35
axs[0].bar(x - width/2, always_neg, width, label='Always predict negative', color=end_color, edgecolor='black', linewidth=0.5)
axs[0].bar(x + width/2, decent_model, width, label='Decent model', color=start_color, edgecolor='black', linewidth=0.5)
axs[0].set_xticks(x)
axs[0].set_xticklabels(metrics, fontsize=13)
axs[0].set_ylim(0, 1.1)
axs[0].set_ylabel('Score')
axs[0].set_title('Imbalanced Dataset (95% negative)', fontsize=14)
axs[0].legend(fontsize=11)

# Balanced scenario
always_neg_bal = [0.50, 0.50]
decent_bal = [0.85, 0.85]
axs[1].bar(x - width/2, always_neg_bal, width, label='Always predict negative', color=end_color, edgecolor='black', linewidth=0.5)
axs[1].bar(x + width/2, decent_bal, width, label='Decent model', color=start_color, edgecolor='black', linewidth=0.5)
axs[1].set_xticks(x)
axs[1].set_xticklabels(metrics, fontsize=13)
axs[1].set_ylim(0, 1.1)
axs[1].set_title('Balanced Dataset (50/50)', fontsize=14)
axs[1].legend(fontsize=11)

fig.tight_layout()
save_figure(fig, 'Balanced_Accuracy_comparison')
plt.close()
print("2. Balanced Accuracy: done")

# ============================================================
# 3. Precision — 3D surface
# Already generated earlier, but check if it exists
# ============================================================
import os
if not os.path.exists('../book/figures/Precision_3d_surface.png'):
    tp = np.linspace(0, 100, 101)
    fp = np.linspace(0, 100, 101)
    TP, FP = np.meshgrid(tp, fp)
    Precision = TP / (TP + FP)
    Precision = np.nan_to_num(Precision, 0)
    fig = plt.figure(figsize=FIGSIZE_LARGE)
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(TP, FP, Precision, cmap=nml_cmap.reversed(), linewidth=0, antialiased=False)
    ax.set_xlabel('True Positives (TP)')
    ax.set_ylabel('False Positives (FP)')
    ax.set_zlabel('Precision')
    ax.set_ylim(100, 0)
    ax.set_xlim(0, 100)
    cbar = fig.colorbar(surf)
    cbar.set_label('Precision')
    ax.view_init(elev=20, azim=-65)
    fig.tight_layout()
    save_figure(fig, 'Precision_3d_surface')
    plt.close()
print("3. Precision: already exists or regenerated")

# ============================================================
# 4. F-beta — F-score vs recall for different beta values
# ============================================================
fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
recall = np.linspace(0.01, 1.0, 200)
precision = 0.8  # fixed
for beta, color, label in [
    (0.5, start_color, r'$\beta=0.5$ (favor precision)'),
    (1.0, middle_color, r'$\beta=1.0$ (F1)'),
    (2.0, end_color, r'$\beta=2.0$ (favor recall)')
]:
    fbeta = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
    ax.plot(recall, fbeta, color=color, linewidth=4, solid_capstyle='round', label=label)

ax.set_xlabel('Recall')
ax.set_ylabel(r'$F_\beta$ Score')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.legend(fontsize=13)
ax.grid(True, linestyle='--', alpha=0.3)
fig.tight_layout()
save_figure(fig, 'F_beta_curves')
plt.close()
print("4. F-beta: done")

# ============================================================
# 5. ROC AUC — example ROC curves
# ============================================================
fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
fpr_good = np.array([0, 0.02, 0.05, 0.1, 0.2, 0.4, 1.0])
tpr_good = np.array([0, 0.5, 0.75, 0.88, 0.95, 0.98, 1.0])
fpr_bad = np.array([0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
tpr_bad = np.array([0, 0.15, 0.35, 0.55, 0.72, 0.88, 1.0])

ax.plot(fpr_good, tpr_good, color=start_color, linewidth=4, solid_capstyle='round', label='Good model (AUC ≈ 0.95)')
ax.fill_between(fpr_good, tpr_good, alpha=0.1, color=start_color)
ax.plot(fpr_bad, tpr_bad, color=end_color, linewidth=4, solid_capstyle='round', label='Weak model (AUC ≈ 0.62)')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random (AUC = 0.50)')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.legend(fontsize=13, loc='lower right')
ax.grid(True, linestyle='--', alpha=0.3)
fig.tight_layout()
save_figure(fig, 'ROC_AUC_curves')
plt.close()
print("5. ROC AUC: done")

# ============================================================
# 6. PR AUC — example PR curves
# ============================================================
fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
recall_good = np.array([0, 0.3, 0.5, 0.7, 0.85, 0.95, 1.0])
prec_good = np.array([1.0, 0.95, 0.92, 0.88, 0.80, 0.65, 0.40])
recall_bad = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0])
prec_bad = np.array([0.6, 0.45, 0.35, 0.25, 0.18, 0.10])

ax.plot(recall_good, prec_good, color=start_color, linewidth=4, solid_capstyle='round', label='Good model (PR AUC ≈ 0.85)')
ax.fill_between(recall_good, prec_good, alpha=0.1, color=start_color)
ax.plot(recall_bad, prec_bad, color=end_color, linewidth=4, solid_capstyle='round', label='Weak model (PR AUC ≈ 0.30)')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.legend(fontsize=13, loc='upper right')
ax.grid(True, linestyle='--', alpha=0.3)
fig.tight_layout()
save_figure(fig, 'PR_AUC_curves')
plt.close()
print("6. PR AUC: done")

# ============================================================
# 7. Jaccard — IoU visualization with overlapping sets
# ============================================================
fig, axs = plt.subplots(1, 2, figsize=FIGSIZE_COMPARISON)

from matplotlib.patches import Circle

for ax, overlap, jaccard, title, color in [
    (axs[0], 0.7, 0.82, 'High Overlap', start_color),
    (axs[1], 1.8, 0.15, 'Low Overlap', end_color)
]:
    c1 = Circle((-overlap/2, 0), 1.0, alpha=0.3, color=start_color, label='Predicted (A)')
    c2 = Circle((overlap/2, 0), 1.0, alpha=0.3, color=middle_color, label='Ground Truth (B)')
    ax.add_patch(c1)
    ax.add_patch(c2)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-1.8, 1.8)
    ax.set_aspect('equal')
    ax.set_title(f'{title}\nJaccard ≈ {jaccard:.2f}', fontsize=16, color=color)
    ax.legend(fontsize=11, loc='lower center')
    ax.set_xticks([])
    ax.set_yticks([])

fig.tight_layout()
save_figure(fig, 'Jaccard_overlap')
plt.close()
print("7. Jaccard: done")

# ============================================================
# 8. Cohen's Kappa — agreement levels bar chart
# ============================================================
fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
levels = ['Poor\n(<0.0)', 'Slight\n(0.0-0.2)', 'Fair\n(0.2-0.4)', 'Moderate\n(0.4-0.6)', 'Substantial\n(0.6-0.8)', 'Almost Perfect\n(0.8-1.0)']
kappa_mid = [-0.1, 0.1, 0.3, 0.5, 0.7, 0.9]
colors_k = [end_color, end_color, middle_color, middle_color, start_color, start_color]
bars = ax.bar(range(len(levels)), kappa_mid, color=colors_k, edgecolor='black', linewidth=0.5)
ax.set_xticks(range(len(levels)))
ax.set_xticklabels(levels, fontsize=10)
ax.set_ylabel("Cohen's Kappa")
ax.set_ylim(-0.2, 1.05)
ax.axhline(y=0, color='black', linewidth=0.5)
fig.tight_layout()
save_figure(fig, 'Cohens_Kappa_levels')
plt.close()
print("8. Cohen's Kappa: done")

# ============================================================
# 9. D-squared Log Loss — D2 vs Log Loss relationship
# ============================================================
fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
model_ll = np.array([0.1, 0.2, 0.35, 0.5, 0.69, 0.9, 1.2])
null_ll = 0.69  # log loss of always predicting 50%
d2 = 1 - model_ll / null_ll
ax.plot(model_ll, d2, color=start_color, marker='o', markersize=10, **LINE_KW)
ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax.axvline(x=null_ll, color='gray', linestyle='--', alpha=0.5)
ax.annotate('null model\n(D²=0)', xy=(null_ll, 0.02), fontsize=12, color='gray', ha='center')
ax.set_xlabel('Model Log Loss')
ax.set_ylabel('D² Log Loss Score')
ax.grid(True, linestyle='--', alpha=0.3)
fig.tight_layout()
save_figure(fig, 'D2_Log_Loss_curve')
plt.close()
print("9. D-squared Log Loss: done")

# ============================================================
# 10. P4 — comparison with F1 on imbalanced data
# ============================================================
fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
scenarios = ['Balanced\n(good model)', 'Imbalanced\n(good model)', 'Imbalanced\n(ignores TN)', 'All\nnegative']
f1_scores = [0.90, 0.85, 0.80, 0.0]
p4_scores = [0.88, 0.82, 0.45, 0.0]

x = np.arange(len(scenarios))
width = 0.35
ax.bar(x - width/2, f1_scores, width, label='F1-score', color=end_color, edgecolor='black', linewidth=0.5)
ax.bar(x + width/2, p4_scores, width, label='P4-metric', color=start_color, edgecolor='black', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(scenarios, fontsize=11)
ax.set_ylabel('Score')
ax.set_ylim(0, 1.05)
ax.legend(fontsize=14)
fig.tight_layout()
save_figure(fig, 'P4_vs_F1')
plt.close()
print("10. P4: done")

# ============================================================
# 11. EC (Expected Cost) — cost matrix heatmap
# ============================================================
fig, ax = plt.subplots(figsize=(6, 5))
cost_matrix = np.array([[0, 40], [80, 0]])
im = ax.imshow(cost_matrix, cmap=nml_cmap, aspect='auto')
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Approve ($D_1$)', 'Reject ($D_2$)'], fontsize=13)
ax.set_yticklabels(['Creditworthy ($H_1$)', 'Not Creditworthy ($H_2$)'], fontsize=13)
ax.set_xlabel('Decision', fontsize=14)
ax.set_ylabel('True State', fontsize=14)
for i in range(2):
    for j in range(2):
        color = 'white' if cost_matrix[i, j] > 20 else 'black'
        ax.text(j, i, f'Cost = {cost_matrix[i,j]}', ha='center', va='center', fontsize=16, color=color, fontweight='bold')
fig.tight_layout()
save_figure(fig, 'EC_cost_matrix')
plt.close()
print("11. EC: done")

print("\nAll remaining classification figures generated!")
