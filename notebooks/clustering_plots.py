"""Generate figures for Clustering chapter."""
import sys
sys.path.insert(0, '.')

import matplotlib
matplotlib.use('Agg')
from style import *
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score, silhouette_samples,
    davies_bouldin_score, calinski_harabasz_score,
    mutual_info_score, adjusted_rand_score,
    fowlkes_mallows_score, completeness_score,
    homogeneity_score, v_measure_score
)

np.random.seed(42)

# ============================================================
# Generate synthetic data: good vs bad clustering
# ============================================================
X_good, y_good = make_blobs(n_samples=300, centers=3, cluster_std=0.8, random_state=42)
X_bad, y_bad = make_blobs(n_samples=300, centers=3, cluster_std=2.5, random_state=42)

# Get KMeans labels
km_good = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X_good)
km_bad = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X_bad)

# ============================================================
# 1. Silhouette Score — silhouette plot for good vs bad clustering
# ============================================================
fig, axs = plt.subplots(1, 2, figsize=FIGSIZE_COMPARISON)

for idx, (X, labels, ax, title_color, title) in enumerate([
    (X_good, km_good.labels_, axs[0], start_color, f'Well Separated (S = {silhouette_score(X_good, km_good.labels_):.2f})'),
    (X_bad, km_bad.labels_, axs[1], end_color, f'Overlapping (S = {silhouette_score(X_bad, km_bad.labels_):.2f})')
]):
    sil_values = silhouette_samples(X, labels)
    n_clusters = len(set(labels))
    y_lower = 10
    colors_list = [start_color, middle_color, end_color]
    for i in range(n_clusters):
        cluster_sil = np.sort(sil_values[labels == i])
        ax.fill_betweenx(np.arange(y_lower, y_lower + len(cluster_sil)),
                         0, cluster_sil, alpha=0.7, color=colors_list[i % 3])
        y_lower += len(cluster_sil) + 10
    avg = silhouette_score(X, labels)
    ax.axvline(x=avg, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Silhouette Coefficient')
    ax.set_yticks([])
    ax.set_title(title, fontsize=16, color=title_color)
    ax.set_xlim(-0.3, 1)

fig.tight_layout()
save_figure(fig, 'Silhouette_Score_comparison')
plt.close()
print("1. Silhouette Score: done")

# ============================================================
# 2. Davies-Bouldin + Calinski-Harabasz — elbow-style plot
# Shows: metric value vs number of clusters (k=2..8)
# ============================================================
X_elbow, _ = make_blobs(n_samples=400, centers=4, cluster_std=1.0, random_state=42)
k_range = range(2, 9)
db_scores = []
ch_scores = []
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_elbow)
    db_scores.append(davies_bouldin_score(X_elbow, km.labels_))
    ch_scores.append(calinski_harabasz_score(X_elbow, km.labels_))

# Davies-Bouldin (lower is better)
fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
ax.plot(list(k_range), db_scores, color=start_color, marker='o', markersize=10, **LINE_KW)
ax.axvline(x=4, color='black', linestyle='--', alpha=0.5, linewidth=1)
ax.annotate('optimal k=4', xy=(4, db_scores[2]), xytext=(5.5, db_scores[2] + 0.15),
            fontsize=14, arrowprops=dict(arrowstyle='->', color='black'))
ax.set_xlabel('Number of Clusters (k)')
ax.set_ylabel('Davies-Bouldin Index')
ax.set_xticks(list(k_range))
fig.tight_layout()
save_figure(fig, 'Davies_Bouldin_elbow')
plt.close()
print("2. Davies-Bouldin: done")

# Calinski-Harabasz (higher is better)
fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
ax.plot(list(k_range), ch_scores, color=start_color, marker='o', markersize=10, **LINE_KW)
ax.axvline(x=4, color='black', linestyle='--', alpha=0.5, linewidth=1)
ax.annotate('optimal k=4', xy=(4, ch_scores[2]), xytext=(5.5, ch_scores[2] - 100),
            fontsize=14, arrowprops=dict(arrowstyle='->', color='black'))
ax.set_xlabel('Number of Clusters (k)')
ax.set_ylabel('Calinski-Harabasz Index')
ax.set_xticks(list(k_range))
fig.tight_layout()
save_figure(fig, 'Calinski_Harabasz_elbow')
plt.close()
print("3. Calinski-Harabasz: done")

# ============================================================
# 4. External metrics comparison — bar chart
# MI, ARI, FMI, Completeness, Homogeneity, V-Measure for good vs bad clustering
# ============================================================
metrics_good = {
    'MI': mutual_info_score(y_good, km_good.labels_),
    'ARI': adjusted_rand_score(y_good, km_good.labels_),
    'FMI': fowlkes_mallows_score(y_good, km_good.labels_),
    'Completeness': completeness_score(y_good, km_good.labels_),
    'Homogeneity': homogeneity_score(y_good, km_good.labels_),
    'V-Measure': v_measure_score(y_good, km_good.labels_),
}
metrics_bad = {
    'MI': mutual_info_score(y_bad, km_bad.labels_),
    'ARI': adjusted_rand_score(y_bad, km_bad.labels_),
    'FMI': fowlkes_mallows_score(y_bad, km_bad.labels_),
    'Completeness': completeness_score(y_bad, km_bad.labels_),
    'Homogeneity': homogeneity_score(y_bad, km_bad.labels_),
    'V-Measure': v_measure_score(y_bad, km_bad.labels_),
}

fig, ax = plt.subplots(figsize=(12, 7))
x = np.arange(len(metrics_good))
width = 0.35
bars1 = ax.bar(x - width/2, list(metrics_good.values()), width, label='Well Separated',
               color=start_color, edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x + width/2, list(metrics_bad.values()), width, label='Overlapping',
               color=end_color, edgecolor='black', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(list(metrics_good.keys()), fontsize=13)
ax.set_ylabel('Score')
ax.legend(fontsize=14)
ax.set_ylim(0, 1.15)
fig.tight_layout()
save_figure(fig, 'Clustering_external_metrics_comparison')
plt.close()
print("4. External metrics comparison: done")

# ============================================================
# 5. Cluster visualization — good vs bad, colored by labels
# Used as shared figure for multiple metrics
# ============================================================
fig, axs = plt.subplots(1, 2, figsize=FIGSIZE_COMPARISON)
colors_map = {0: start_color, 1: middle_color, 2: end_color}

for X, labels, ax, title in [
    (X_good, km_good.labels_, axs[0], 'Well Separated Clusters'),
    (X_bad, km_bad.labels_, axs[1], 'Overlapping Clusters')
]:
    for c in range(3):
        mask = labels == c
        ax.scatter(X[mask, 0], X[mask, 1], c=colors_map[c], s=30, alpha=0.7)
    ax.set_title(title, fontsize=16)
    ax.set_xticks([])
    ax.set_yticks([])

fig.tight_layout()
save_figure(fig, 'Clustering_good_vs_bad')
plt.close()
print("5. Cluster visualization: done")

# ============================================================
# 6. Contingency Matrix — heatmap example
# ============================================================
from sklearn.metrics import confusion_matrix
# Use bad clustering to show interesting misalignment
cont_matrix = confusion_matrix(y_bad, km_bad.labels_)
# Reorder to best match
from scipy.optimize import linear_sum_assignment
row_ind, col_ind = linear_sum_assignment(-cont_matrix)
cont_matrix = cont_matrix[:, col_ind]

fig, ax = plt.subplots(figsize=(7, 5))
im = ax.imshow(cont_matrix, cmap=nml_cmap.reversed(), aspect='auto')
ax.set_xlabel('Cluster Label', fontsize=14)
ax.set_ylabel('True Label', fontsize=14)
ax.set_xticks(range(3))
ax.set_yticks(range(3))
ax.set_xticklabels([f'Cluster {i}' for i in range(3)])
ax.set_yticklabels([f'Class {i}' for i in range(3)])
for i in range(3):
    for j in range(3):
        ax.text(j, i, str(cont_matrix[i, j]), ha='center', va='center',
                fontsize=18, color='white' if cont_matrix[i, j] > cont_matrix.max()/2 else 'black')
cbar = fig.colorbar(im)
cbar.set_label('Count', fontsize=14)
fig.tight_layout()
save_figure(fig, 'Contingency_Matrix_heatmap')
plt.close()
print("6. Contingency Matrix: done")

# ============================================================
# 7. Pair Confusion Matrix — 2x2 heatmap
# ============================================================
from sklearn.metrics.cluster import pair_confusion_matrix
pcm = pair_confusion_matrix(y_bad, km_bad.labels_)

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(pcm, cmap=nml_cmap.reversed(), aspect='auto')
labels_pcm = [['$C_{00}$\n(Both differ)', '$C_{01}$\n(Same pred,\ndiff true)'],
              ['$C_{10}$\n(Same true,\ndiff pred)', '$C_{11}$\n(Both same)']]
for i in range(2):
    for j in range(2):
        ax.text(j, i, f'{labels_pcm[i][j]}\n{int(pcm[i,j])}', ha='center', va='center',
                fontsize=12, color='white' if pcm[i, j] > pcm.max()/2 else 'black')
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Different Pred', 'Same Pred'], fontsize=11)
ax.set_yticklabels(['Different True', 'Same True'], fontsize=11)
fig.tight_layout()
save_figure(fig, 'Pair_Confusion_Matrix')
plt.close()
print("7. Pair Confusion Matrix: done")

print("\nAll clustering figures generated!")
