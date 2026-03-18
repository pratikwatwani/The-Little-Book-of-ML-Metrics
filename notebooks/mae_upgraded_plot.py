"""Upgraded MAE figure: comparative plot showing MAE vs MSE behavior with outliers."""
import sys
sys.path.insert(0, '.')

import matplotlib
matplotlib.use('Agg')
from style import *

np.random.seed(42)

# Create a realistic scenario: predictions with one outlier
y_true = np.array([10, 12, 15, 14, 13, 11, 16, 14, 12, 50])  # last one is outlier
y_pred = np.array([11, 13, 14, 15, 12, 12, 15, 13, 13, 15])
errors = y_true - y_pred

# Compute metrics
mae = np.mean(np.abs(errors))
mse = np.mean(errors**2)
rmse = np.sqrt(mse)

# Per-sample contribution to MAE vs MSE
mae_contrib = np.abs(errors)
mse_contrib = errors**2

fig, axs = plt.subplots(1, 2, figsize=FIGSIZE_COMPARISON)

# Left: per-sample error contribution
x = np.arange(len(errors))
width = 0.35
axs[0].bar(x - width/2, mae_contrib, width, label=f'|error| (MAE={mae:.1f})',
           color=start_color, edgecolor='black', linewidth=0.5)
axs[0].bar(x + width/2, mse_contrib, width, label=f'error² (MSE={mse:.1f})',
           color=end_color, edgecolor='black', linewidth=0.5, alpha=0.8)
axs[0].set_xlabel('Sample')
axs[0].set_ylabel('Error Contribution')
axs[0].set_xticks(x)
axs[0].set_xticklabels([f'{i+1}' for i in x])
axs[0].legend(fontsize=12, loc='upper left')
# Highlight the outlier
axs[0].annotate('outlier', xy=(9, mse_contrib[9]), xytext=(7, mse_contrib[9]*0.8),
                fontsize=13, color=end_color, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=end_color))

# Right: cumulative effect — MAE vs RMSE with and without outlier
labels = ['With Outlier', 'Without Outlier']
mae_vals = [mae, np.mean(np.abs(errors[:-1]))]
rmse_vals = [rmse, np.sqrt(np.mean(errors[:-1]**2))]

x2 = np.arange(len(labels))
axs[1].bar(x2 - width/2, mae_vals, width, label='MAE', color=start_color, edgecolor='black', linewidth=0.5)
axs[1].bar(x2 + width/2, rmse_vals, width, label='RMSE', color=end_color, edgecolor='black', linewidth=0.5)
axs[1].set_xticks(x2)
axs[1].set_xticklabels(labels, fontsize=13)
axs[1].set_ylabel('Error')
axs[1].legend(fontsize=13)
# Add value labels
for i, (m, r) in enumerate(zip(mae_vals, rmse_vals)):
    axs[1].text(i - width/2, m + 0.3, f'{m:.1f}', ha='center', fontsize=12, fontweight='bold', color=start_color)
    axs[1].text(i + width/2, r + 0.3, f'{r:.1f}', ha='center', fontsize=12, fontweight='bold', color=end_color)

fig.tight_layout()
save_figure(fig, 'MAE_vs_MSE_outlier')
plt.close()
print("MAE upgraded figure: done")
