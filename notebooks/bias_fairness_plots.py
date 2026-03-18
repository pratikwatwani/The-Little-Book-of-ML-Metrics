"""Generate figures for Bias & Fairness chapter."""
import sys
sys.path.insert(0, '.')

import matplotlib
matplotlib.use('Agg')
from style import *

# ============================================================
# 1. Demographic Parity — compare selection rates across groups
# Shows: fair (equal rates) vs unfair (unequal rates)
# ============================================================
fig, axs = plt.subplots(1, 2, figsize=FIGSIZE_COMPARISON)

groups = ['Group A', 'Group B', 'Group C']
x = np.arange(len(groups))
width = 0.5

# Fair scenario
fair_rates = [0.42, 0.40, 0.41]
bars1 = axs[0].bar(x, fair_rates, width, color=[start_color]*3, edgecolor='black', linewidth=0.5)
axs[0].set_ylim(0, 1)
axs[0].set_ylabel('P(Ŷ = 1)', fontsize=14)
axs[0].set_xticks(x)
axs[0].set_xticklabels(groups, fontsize=12)
axs[0].set_title('Fair (Parity Satisfied)', fontsize=18, color=start_color)
axs[0].axhline(y=0.41, color='black', linestyle='--', alpha=0.5, linewidth=1)
for bar, val in zip(bars1, fair_rates):
    axs[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{val:.0%}', ha='center', fontsize=13)

# Unfair scenario
unfair_rates = [0.65, 0.30, 0.20]
colors = [start_color, end_color, end_color]
bars2 = axs[1].bar(x, unfair_rates, width, color=colors, edgecolor='black', linewidth=0.5)
axs[1].set_ylim(0, 1)
axs[1].set_xticks(x)
axs[1].set_xticklabels(groups, fontsize=12)
axs[1].set_title('Unfair (Parity Violated)', fontsize=18, color=end_color)
for bar, val in zip(bars2, unfair_rates):
    axs[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{val:.0%}', ha='center', fontsize=13)

fig.tight_layout()
save_figure(fig, 'Demographic_Parity')
plt.close()
print("1. Demographic Parity: done")

# ============================================================
# 2. Equality of Opportunity — compare TPR across groups
# Shows: equal TPR (fair) vs unequal TPR (unfair)
# ============================================================
fig, axs = plt.subplots(1, 2, figsize=FIGSIZE_COMPARISON)

# Fair
fair_tpr = [0.82, 0.80, 0.81]
bars1 = axs[0].bar(x, fair_tpr, width, color=[start_color]*3, edgecolor='black', linewidth=0.5)
axs[0].set_ylim(0, 1)
axs[0].set_ylabel('True Positive Rate', fontsize=14)
axs[0].set_xticks(x)
axs[0].set_xticklabels(groups, fontsize=12)
axs[0].set_title('Fair (Equal Opportunity)', fontsize=18, color=start_color)
axs[0].axhline(y=0.81, color='black', linestyle='--', alpha=0.5, linewidth=1)
for bar, val in zip(bars1, fair_tpr):
    axs[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{val:.0%}', ha='center', fontsize=13)

# Unfair
unfair_tpr = [0.90, 0.55, 0.45]
colors = [start_color, end_color, end_color]
bars2 = axs[1].bar(x, unfair_tpr, width, color=colors, edgecolor='black', linewidth=0.5)
axs[1].set_ylim(0, 1)
axs[1].set_ylabel('True Positive Rate', fontsize=14)
axs[1].set_xticks(x)
axs[1].set_xticklabels(groups, fontsize=12)
axs[1].set_title('Unfair (Unequal Opportunity)', fontsize=18, color=end_color)
for bar, val in zip(bars2, unfair_tpr):
    axs[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{val:.0%}', ha='center', fontsize=13)

fig.tight_layout()
save_figure(fig, 'Equality_of_Opportunity')
plt.close()
print("2. Equality of Opportunity: done")

# ============================================================
# 3. Equality of Odds — compare both TPR and FPR across groups
# Shows: grouped bars for TPR and FPR per group
# ============================================================
fig, axs = plt.subplots(1, 2, figsize=FIGSIZE_COMPARISON)
width = 0.35

# Fair scenario
fair_tpr = [0.80, 0.78, 0.81]
fair_fpr = [0.10, 0.11, 0.09]
bars_tpr = axs[0].bar(x - width/2, fair_tpr, width, label='TPR', color=start_color, edgecolor='black', linewidth=0.5)
bars_fpr = axs[0].bar(x + width/2, fair_fpr, width, label='FPR', color=middle_color, edgecolor='black', linewidth=0.5)
axs[0].set_ylim(0, 1)
axs[0].set_ylabel('Rate', fontsize=14)
axs[0].set_xticks(x)
axs[0].set_xticklabels(groups, fontsize=12)
axs[0].set_title('Fair (Equalized Odds)', fontsize=18, color=start_color)
axs[0].legend(fontsize=12)

# Unfair scenario
unfair_tpr = [0.90, 0.55, 0.50]
unfair_fpr = [0.05, 0.25, 0.30]
bars_tpr = axs[1].bar(x - width/2, unfair_tpr, width, label='TPR', color=start_color, edgecolor='black', linewidth=0.5)
bars_fpr = axs[1].bar(x + width/2, unfair_fpr, width, label='FPR', color=end_color, edgecolor='black', linewidth=0.5)
axs[1].set_ylim(0, 1)
axs[1].set_xticks(x)
axs[1].set_xticklabels(groups, fontsize=12)
axs[1].set_title('Unfair (Odds Violated)', fontsize=18, color=end_color)
axs[1].legend(fontsize=12)

fig.tight_layout()
save_figure(fig, 'Equality_of_Odds')
plt.close()
print("3. Equality of Odds: done")

# ============================================================
# 4. Predictive Parity — compare PPV/Precision across groups
# Shows: equal precision (fair) vs unequal (unfair)
# ============================================================
fig, axs = plt.subplots(1, 2, figsize=FIGSIZE_COMPARISON)
width = 0.5

# Fair
fair_ppv = [0.75, 0.73, 0.74]
bars1 = axs[0].bar(x, fair_ppv, width, color=[start_color]*3, edgecolor='black', linewidth=0.5)
axs[0].set_ylim(0, 1)
axs[0].set_ylabel('Positive Predictive Value (PPV)', fontsize=12)
axs[0].set_xticks(x)
axs[0].set_xticklabels(groups, fontsize=12)
axs[0].set_title('Fair (Predictive Parity)', fontsize=18, color=start_color)
axs[0].axhline(y=0.74, color='black', linestyle='--', alpha=0.5, linewidth=1)
for bar, val in zip(bars1, fair_ppv):
    axs[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{val:.0%}', ha='center', fontsize=13)

# Unfair
unfair_ppv = [0.85, 0.50, 0.40]
colors = [start_color, end_color, end_color]
bars2 = axs[1].bar(x, unfair_ppv, width, color=colors, edgecolor='black', linewidth=0.5)
axs[1].set_ylim(0, 1)
axs[1].set_xticks(x)
axs[1].set_xticklabels(groups, fontsize=12)
axs[1].set_title('Unfair (Parity Violated)', fontsize=18, color=end_color)
for bar, val in zip(bars2, unfair_ppv):
    axs[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{val:.0%}', ha='center', fontsize=13)

fig.tight_layout()
save_figure(fig, 'Predictive_Parity')
plt.close()
print("4. Predictive Parity: done")

# ============================================================
# 5. Calibration within Groups — show calibration curves per group
# Shows: predicted probability vs actual frequency, per group
# ============================================================
fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

# Simulated calibration curves
prob_bins = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

# Well-calibrated (both groups close to diagonal)
group_a_freq = prob_bins + np.array([0.02, -0.01, 0.01, -0.02, 0.01, 0.02, -0.01, 0.01, -0.02])
group_b_freq = prob_bins + np.array([-0.01, 0.02, -0.02, 0.01, -0.01, 0.01, 0.02, -0.02, 0.01])

# Poorly calibrated (Group B scores mean different things)
group_b_bad = prob_bins + np.array([0.15, 0.12, 0.10, 0.08, 0.05, 0.0, -0.05, -0.10, -0.12])

ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect calibration', alpha=0.5)
ax.plot(prob_bins, group_a_freq, 'o-', color=start_color, linewidth=3, markersize=8,
        label='Group A (well calibrated)', solid_capstyle='round')
ax.plot(prob_bins, group_b_bad, 's-', color=end_color, linewidth=3, markersize=8,
        label='Group B (poorly calibrated)', solid_capstyle='round')

ax.set_xlabel('Predicted Probability', fontsize=16)
ax.set_ylabel('Observed Frequency', fontsize=16)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.legend(fontsize=14, loc='upper left')
ax.grid(True, linestyle='--', alpha=0.3)

fig.tight_layout()
save_figure(fig, 'Calibration_within_Groups')
plt.close()
print("5. Calibration within Groups: done")

print("\nAll bias & fairness figures generated!")
