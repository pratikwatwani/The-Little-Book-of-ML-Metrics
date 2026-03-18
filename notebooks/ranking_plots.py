"""Generate figures for Ranking chapter."""
import sys
sys.path.insert(0, '.')

import matplotlib
matplotlib.use('Agg')
from style import *

np.random.seed(42)

# ============================================================
# 1. MRR — show how rank of first relevant item affects score
# ============================================================
fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
ranks = np.arange(1, 21)
mrr_values = 1.0 / ranks
ax.plot(ranks, mrr_values, color=start_color, marker='o', markersize=8, **LINE_KW)
ax.set_xlabel('Rank of First Relevant Item')
ax.set_ylabel('Reciprocal Rank')
ax.set_xticks([1, 5, 10, 15, 20])
ax.set_ylim(0, 1.05)
ax.grid(True, linestyle='--', alpha=0.3)
fig.tight_layout()
save_figure(fig, 'MRR_reciprocal_rank')
plt.close()
print("1. MRR: done")

# ============================================================
# 2. MAP — AP for good vs bad ranking
# ============================================================
fig, axs = plt.subplots(1, 2, figsize=FIGSIZE_COMPARISON)

# Good ranking: relevant items at top
good_ranking = [1, 1, 1, 0, 0, 1, 0, 0, 0, 0]  # 4 relevant, mostly at top
bad_ranking =  [0, 0, 0, 1, 0, 0, 1, 0, 1, 1]  # 4 relevant, at bottom

for ranking, ax, title, color in [
    (good_ranking, axs[0], 'Good Ranking', start_color),
    (bad_ranking, axs[1], 'Poor Ranking', end_color)
]:
    positions = np.arange(1, len(ranking) + 1)
    colors = [color if r == 1 else '#dddddd' for r in ranking]
    ax.barh(positions, [1]*len(ranking), color=colors, edgecolor='black', linewidth=0.5)
    # Compute precision at each relevant position
    precisions = []
    tp = 0
    for i, r in enumerate(ranking):
        if r == 1:
            tp += 1
            precisions.append((i+1, tp/(i+1)))
    ap = np.mean([p for _, p in precisions])
    ax.set_yticks(positions)
    ax.set_yticklabels([f'Rank {i}' for i in positions], fontsize=11)
    ax.set_xlim(0, 1.1)
    ax.set_xticks([])
    ax.set_title(f'{title} (AP = {ap:.2f})', fontsize=16, color=color)
    ax.invert_yaxis()
    # Mark relevant items
    for i, r in enumerate(ranking):
        if r == 1:
            ax.text(0.5, i+1, 'Relevant', ha='center', va='center', fontsize=12, fontweight='bold')
        else:
            ax.text(0.5, i+1, 'Not relevant', ha='center', va='center', fontsize=12, color='gray')

fig.tight_layout()
save_figure(fig, 'MAP_good_vs_bad')
plt.close()
print("2. MAP: done")

# ============================================================
# 3. CG vs DCG vs nDCG — show position discounting effect
# ============================================================
fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
relevance = [3, 2, 3, 0, 1, 2, 0, 0, 1, 0]
positions = np.arange(1, len(relevance) + 1)

cg_cumulative = np.cumsum(relevance)
dcg_cumulative = np.cumsum([r / np.log2(i + 1) for i, r in zip(positions, relevance)])

ax.plot(positions, cg_cumulative, color=end_color, marker='s', markersize=8, **LINE_KW, label='CG (no discount)')
ax.plot(positions, dcg_cumulative, color=start_color, marker='o', markersize=8, **LINE_KW, label='DCG (discounted)')
ax.set_xlabel('Position (k)')
ax.set_ylabel('Cumulative Score')
ax.set_xticks(positions)
ax.legend(fontsize=14)
ax.grid(True, linestyle='--', alpha=0.3)
fig.tight_layout()
save_figure(fig, 'CG_vs_DCG')
plt.close()
print("3. CG vs DCG: done")

# ============================================================
# 4. nDCG — perfect vs degraded ranking
# ============================================================
fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

relevance_perfect = sorted(relevance, reverse=True)  # ideal
# Progressively shuffle
ndcg_scores = []
n_shuffles = 20
for i in range(n_shuffles):
    if i == 0:
        r = relevance_perfect.copy()
    else:
        r = relevance_perfect.copy()
        # Randomly swap i pairs
        for _ in range(i):
            a, b = np.random.randint(0, len(r), 2)
            r[a], r[b] = r[b], r[a]
    dcg = sum(rel / np.log2(pos + 1) for pos, rel in zip(range(1, len(r)+1), r))
    idcg = sum(rel / np.log2(pos + 1) for pos, rel in zip(range(1, len(relevance_perfect)+1), relevance_perfect))
    ndcg_scores.append(dcg / idcg if idcg > 0 else 0)

ax.plot(range(n_shuffles), ndcg_scores, color=start_color, marker='o', markersize=6, linewidth=3)
ax.set_xlabel('Number of Random Swaps')
ax.set_ylabel('nDCG')
ax.set_ylim(0.4, 1.05)
ax.grid(True, linestyle='--', alpha=0.3)
fig.tight_layout()
save_figure(fig, 'nDCG_degradation')
plt.close()
print("4. nDCG: done")

# ============================================================
# 5. Hit Rate — simple visual: hit vs miss at different K
# ============================================================
fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
K_values = [1, 3, 5, 10, 20, 50]
# Simulated hit rates for a recommender
hit_rates = [0.15, 0.35, 0.52, 0.72, 0.88, 0.96]
ax.plot(K_values, hit_rates, color=start_color, marker='o', markersize=10, **LINE_KW)
ax.set_xlabel('K (recommendation list size)')
ax.set_ylabel('Hit Rate')
ax.set_ylim(0, 1.05)
ax.set_xscale('log')
ax.set_xticks(K_values)
ax.set_xticklabels([str(k) for k in K_values])
ax.grid(True, linestyle='--', alpha=0.3)
fig.tight_layout()
save_figure(fig, 'Hit_Rate_vs_K')
plt.close()
print("5. Hit Rate: done")

# ============================================================
# 6. FCP — concordant vs discordant pairs visualization
# ============================================================
fig, axs = plt.subplots(1, 2, figsize=FIGSIZE_COMPARISON)

# Good ranking (mostly concordant)
true_prefs = [5, 4, 3, 2, 1]
good_pred =  [5, 4, 2, 3, 1]
bad_pred =   [1, 3, 5, 2, 4]

for pred, ax, title, color in [
    (good_pred, axs[0], 'Good Ranking', start_color),
    (bad_pred, axs[1], 'Poor Ranking', end_color)
]:
    # Count concordant pairs
    concordant = 0
    total = 0
    for i in range(len(true_prefs)):
        for j in range(i+1, len(true_prefs)):
            total += 1
            if (true_prefs[i] - true_prefs[j]) * (pred[i] - pred[j]) > 0:
                concordant += 1
    fcp = concordant / total

    ax.scatter(true_prefs, pred, color=color, s=150, zorder=5, edgecolors='black')
    ax.plot([0.5, 5.5], [0.5, 5.5], 'k--', alpha=0.3)
    ax.set_xlabel('True Preference', fontsize=14)
    ax.set_ylabel('Predicted Rank', fontsize=14)
    ax.set_title(f'{title} (FCP = {fcp:.2f})', fontsize=16, color=color)
    ax.set_xlim(0.5, 5.5)
    ax.set_ylim(0.5, 5.5)
    ax.set_xticks([1,2,3,4,5])
    ax.set_yticks([1,2,3,4,5])

fig.tight_layout()
save_figure(fig, 'FCP_comparison')
plt.close()
print("6. FCP: done")

# ============================================================
# 7. Diversity — high vs low diversity recommendation lists
# ============================================================
fig, axs = plt.subplots(1, 2, figsize=FIGSIZE_COMPARISON)

categories = ['Action', 'Comedy', 'Drama', 'Sci-Fi', 'Horror', 'Romance']

# Low diversity (all same genre)
low_div = [5, 0, 0, 0, 0, 0]
# High diversity
high_div = [1, 1, 1, 1, 1, 0]

x = np.arange(len(categories))
axs[0].bar(x, high_div, color=[start_color, middle_color, end_color, NML_DARK_RED, start_color, middle_color],
           edgecolor='black', linewidth=0.5)
axs[0].set_xticks(x)
axs[0].set_xticklabels(categories, fontsize=10, rotation=30)
axs[0].set_ylabel('Items Recommended')
axs[0].set_title('High Diversity', fontsize=18, color=start_color)
axs[0].set_ylim(0, 6)

axs[1].bar(x, low_div, color=[end_color]*6, edgecolor='black', linewidth=0.5)
axs[1].set_xticks(x)
axs[1].set_xticklabels(categories, fontsize=10, rotation=30)
axs[1].set_title('Low Diversity', fontsize=18, color=end_color)
axs[1].set_ylim(0, 6)

fig.tight_layout()
save_figure(fig, 'Diversity_comparison')
plt.close()
print("7. Diversity: done")

# ============================================================
# 8. Novelty — popularity vs novelty trade-off
# ============================================================
fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
popularity = np.linspace(0.01, 1.0, 100)
novelty = -np.log2(popularity)
ax.plot(popularity, novelty, color=start_color, **LINE_KW)
ax.set_xlabel('Item Popularity P(i)')
ax.set_ylabel('Novelty $(-\\log_2 P(i))$')
ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
ax.annotate('popular items\n(low novelty)', xy=(0.7, 0.5), fontsize=13, color='gray', ha='center')
ax.annotate('rare items\n(high novelty)', xy=(0.1, 5), fontsize=13, color=start_color, ha='center')
ax.grid(True, linestyle='--', alpha=0.3)
fig.tight_layout()
save_figure(fig, 'Novelty_curve')
plt.close()
print("8. Novelty: done")

# ============================================================
# 9. Coverage — catalog utilization bar chart
# ============================================================
fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
systems = ['System A\n(Popular only)', 'System B\n(Moderate)', 'System C\n(Full catalog)']
coverage = [0.12, 0.45, 0.89]
colors = [end_color, middle_color, start_color]
bars = ax.bar(systems, coverage, color=colors, edgecolor='black', linewidth=0.5, width=0.5)
ax.set_ylabel('Catalog Coverage')
ax.set_ylim(0, 1)
for bar, val in zip(bars, coverage):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
            f'{val:.0%}', ha='center', fontsize=15, fontweight='bold')
fig.tight_layout()
save_figure(fig, 'Coverage_comparison')
plt.close()
print("9. Coverage: done")

# ============================================================
# 10. Serendipity — relevance x unexpectedness scatter
# ============================================================
fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
n_items = 50
relevance = np.random.rand(n_items)
unexpectedness = np.random.rand(n_items)
serendipity = relevance * unexpectedness

scatter = ax.scatter(relevance, unexpectedness, c=serendipity, cmap=nml_cmap.reversed(),
                     s=100, edgecolors='black', linewidth=0.5)
cbar = fig.colorbar(scatter)
cbar.set_label('Serendipity Score', fontsize=14)
ax.set_xlabel('Relevance')
ax.set_ylabel('Unexpectedness')
# Annotate quadrants
ax.annotate('High Serendipity\n(relevant + surprising)', xy=(0.8, 0.8), fontsize=12,
            ha='center', color=middle_color, fontweight='bold')
ax.annotate('Low Serendipity\n(irrelevant or expected)', xy=(0.2, 0.2), fontsize=12,
            ha='center', color='gray')
ax.grid(True, linestyle='--', alpha=0.3)
fig.tight_layout()
save_figure(fig, 'Serendipity_scatter')
plt.close()
print("10. Serendipity: done")

print("\nAll ranking figures generated!")
