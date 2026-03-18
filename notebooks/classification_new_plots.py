"""Generate missing classification metric figures for the book."""
import sys
sys.path.insert(0, '.')

import matplotlib
matplotlib.use('Agg')
from style import *

# ============================================================
# Precision: 3D surface over (TP, FP) grid
# ============================================================
tp = np.linspace(0, 100, 101)
fp = np.linspace(0, 100, 101)
TP, FP = np.meshgrid(tp, fp)
Precision = TP / (TP + FP)
Precision = np.nan_to_num(Precision, 0)

fig = plt.figure(figsize=FIGSIZE_LARGE)
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(TP, FP, Precision, cmap=nml_cmap.reversed(),
                       linewidth=0, antialiased=False)
ax.set_xlabel('True Positives (TP)')
ax.set_ylabel('False Positives (FP)')
ax.set_zlabel('Precision')
ax.set_ylim(100, 0)
ax.set_xlim(0, 100)
ax.set_xticks([0, 25, 50, 75])
ax.set_yticks([0, 25, 50, 75, 100])
ax.set_zticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.tick_params(axis='z', which='major', pad=5)
cbar = fig.colorbar(surf)
cbar.set_label('Precision')
ax.view_init(elev=20, azim=-65)
fig.tight_layout()
save_figure(fig, 'Precision_3d_surface')
plt.close()

# Precision: 2D line plot
fig, ax = plt.subplots(figsize=FIGSIZE_LARGE)
x = np.linspace(1, 200, 200)
colors = [start_color, middle_color, end_color, end_end_color]
for tp_val, color in zip([10, 50, 100, 150], colors):
    precision = tp_val / x
    ax.plot(x, precision, label=f'TP = {tp_val}', color=color, **LINE_KW)
ax.set_xlabel('TP + FP')
ax.set_ylabel('Precision')
ax.set_xlim(0, 200)
ax.set_ylim(0, 1)
ax.legend(fontsize=16)
ax.grid(True, linestyle='--', alpha=0.7)
fig.tight_layout()
save_figure(fig, 'Precision_2d_line_plot')
plt.close()
print("Precision: done")

# ============================================================
# Accuracy: 3D surface over (correct, total) or heatmap
# ============================================================
tp = np.linspace(0, 50, 101)
tn = np.linspace(0, 50, 101)
TP_grid, TN_grid = np.meshgrid(tp, tn)
# Fixed FP=25, FN=25 scenario, vary TP and TN
total = TP_grid + TN_grid + 50  # FP+FN = 50
Accuracy = (TP_grid + TN_grid) / total

fig = plt.figure(figsize=FIGSIZE_LARGE)
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(TP_grid, TN_grid, Accuracy, cmap=nml_cmap.reversed(),
                       linewidth=0, antialiased=False)
ax.set_xlabel('True Positives (TP)')
ax.set_ylabel('True Negatives (TN)')
ax.set_zlabel('Accuracy')
ax.set_xticks([0, 10, 20, 30, 40, 50])
ax.set_yticks([0, 10, 20, 30, 40, 50])
ax.set_zticks([0.0, 0.2, 0.4, 0.6, 0.8])
ax.tick_params(axis='z', which='major', pad=5)
cbar = fig.colorbar(surf)
cbar.set_label('Accuracy')
ax.view_init(elev=20, azim=-65)
fig.tight_layout()
save_figure(fig, 'Accuracy_3d_surface')
plt.close()
print("Accuracy: done")

# ============================================================
# F1-score: surface over (Precision, Recall)
# ============================================================
p = np.linspace(0.01, 1, 100)
r = np.linspace(0.01, 1, 100)
P, R = np.meshgrid(p, r)
F1 = 2 * P * R / (P + R)

fig = plt.figure(figsize=FIGSIZE_LARGE)
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(P, R, F1, cmap=nml_cmap.reversed(),
                       linewidth=0, antialiased=False)
ax.set_xlabel('Precision')
ax.set_ylabel('Recall')
ax.set_zlabel('F1-score')
ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
ax.set_zticks([0.0, 0.25, 0.5, 0.75, 1.0])
ax.tick_params(axis='z', which='major', pad=5)
cbar = fig.colorbar(surf)
cbar.set_label('F1-score')
ax.view_init(elev=20, azim=-65)
fig.tight_layout()
save_figure(fig, 'F1_3d_surface')
plt.close()

# F1: 2D cross-section — F1 vs Recall for fixed Precision values
fig, ax = plt.subplots(figsize=FIGSIZE_LARGE)
recall = np.linspace(0.01, 1, 200)
for prec_val, color in zip([0.25, 0.5, 0.75, 1.0], [end_end_color, end_color, middle_color, start_color]):
    f1 = 2 * prec_val * recall / (prec_val + recall)
    ax.plot(recall, f1, label=f'Precision = {prec_val}', color=color, **LINE_KW)
ax.set_xlabel('Recall')
ax.set_ylabel('F1-score')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.legend(fontsize=16)
ax.grid(True, linestyle='--', alpha=0.7)
fig.tight_layout()
save_figure(fig, 'F1_2d_line_plot')
plt.close()
print("F1: done")

# ============================================================
# Brier Score: loss curve vs predicted probability
# ============================================================
p = np.linspace(0, 1, 200)

fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
brier_y1 = (1 - p) ** 2  # actual = 1
brier_y0 = p ** 2          # actual = 0
ax.plot(p, brier_y1, label='Actual = 1', color=start_color, **LINE_KW)
ax.plot(p, brier_y0, label='Actual = 0', color=end_color, **LINE_KW)
ax.set_xlabel('Predicted Probability')
ax.set_ylabel('Brier Score')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.legend(fontsize=16)
ax.grid(True, linestyle='--', alpha=0.7)
fig.tight_layout()
save_figure(fig, 'Brier_Score_curves')
plt.close()
print("Brier Score: done")

# ============================================================
# Log Loss: loss curve vs predicted probability
# ============================================================
p = np.linspace(0.001, 0.999, 200)

fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
ll_y1 = -np.log(p)      # actual = 1
ll_y0 = -np.log(1 - p)  # actual = 0
ax.plot(p, ll_y1, label='Actual = 1', color=start_color, **LINE_KW)
ax.plot(p, ll_y0, label='Actual = 0', color=end_color, **LINE_KW)
ax.set_xlabel('Predicted Probability')
ax.set_ylabel('Log Loss')
ax.set_xlim(0, 1)
ax.set_ylim(0, 5)
ax.legend(fontsize=16)
ax.grid(True, linestyle='--', alpha=0.7)
fig.tight_layout()
save_figure(fig, 'Log_Loss_curves')
plt.close()
print("Log Loss: done")

# ============================================================
# MCC: 3D surface over (TP, FP) with fixed FN=20, TN=80
# ============================================================
tp = np.linspace(1, 100, 100)
fp = np.linspace(1, 100, 100)
TP_g, FP_g = np.meshgrid(tp, fp)
FN_val = 20
TN_val = 80
numerator = TP_g * TN_val - FP_g * FN_val
denominator = np.sqrt((TP_g + FP_g) * (TP_g + FN_val) * (TN_val + FP_g) * (TN_val + FN_val))
MCC = numerator / denominator
MCC = np.nan_to_num(MCC, 0)

fig = plt.figure(figsize=FIGSIZE_LARGE)
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(TP_g, FP_g, MCC, cmap=nml_cmap.reversed(),
                       linewidth=0, antialiased=False)
ax.set_xlabel('True Positives (TP)')
ax.set_ylabel('False Positives (FP)')
ax.set_zlabel('MCC')
ax.set_xticks([0, 25, 50, 75, 100])
ax.set_yticks([0, 25, 50, 75, 100])
ax.tick_params(axis='z', which='major', pad=5)
cbar = fig.colorbar(surf)
cbar.set_label('MCC')
ax.view_init(elev=20, azim=-65)
fig.tight_layout()
save_figure(fig, 'MCC_3d_surface')
plt.close()
print("MCC: done")

print("\nAll classification figures generated!")
