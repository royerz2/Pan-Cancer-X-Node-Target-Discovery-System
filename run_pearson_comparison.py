#!/usr/bin/env python3
"""
Pearson vs Jaccard Co-essentiality Clustering Comparison
=========================================================

Systematically compares Jaccard-binarized and Pearson-correlation co-essentiality
distance matrices for Ward hierarchical clustering across multiple cancer types
and cluster counts (k=3–15).

Outputs:
  - pearson_comparison_results/nmi_comparison.csv: per-cancer, per-k NMI values
  - pearson_comparison_results/summary.json: aggregate statistics
  - figures/figS_pearson_nmi_comparison.pdf: NMI comparison figure
"""

import sys
import json
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore', category=FutureWarning)


def compute_jaccard_distance_matrix(crispr_sub, selective_genes, dep_threshold=-0.5):
    """Compute Jaccard-based distance matrix from binarized essentiality."""
    gene_list = list(selective_genes)
    n = len(gene_list)
    
    # Binary essentiality matrix: (cell_lines x genes)
    binary = np.zeros((crispr_sub.shape[0], n))
    for j, g in enumerate(gene_list):
        if g in crispr_sub.columns:
            binary[:, j] = (crispr_sub[g].values < dep_threshold).astype(float)
    
    # Jaccard via dot products
    intersection = binary.T @ binary
    row_sums = binary.sum(axis=0)
    union = row_sums[:, None] + row_sums[None, :] - intersection
    union = np.maximum(union, 1)
    sim = intersection / union
    np.fill_diagonal(sim, 1.0)
    
    dist = 1 - sim
    np.fill_diagonal(dist, 0)
    return dist


def compute_pearson_distance_matrix(crispr_sub, selective_genes):
    """Compute Pearson correlation-based distance matrix from continuous Chronos scores."""
    gene_list = list(selective_genes)
    n = len(gene_list)
    
    # Extract continuous scores
    scores = np.zeros((crispr_sub.shape[0], n))
    for j, g in enumerate(gene_list):
        if g in crispr_sub.columns:
            scores[:, j] = crispr_sub[g].values
    
    # Filter out zero-variance genes (replace with small noise to avoid NaN)
    stds_raw = scores.std(axis=0)
    zero_var = stds_raw < 1e-10
    if zero_var.any():
        # Add tiny noise to zero-variance columns
        scores[:, zero_var] += np.random.normal(0, 1e-8, 
                                                 size=(scores.shape[0], zero_var.sum()))
    
    # Pearson correlation matrix using numpy corrcoef (handles centering/scaling)
    corr = np.corrcoef(scores, rowvar=False)
    
    # Handle any remaining NaN (shouldn't happen but safety)
    corr = np.nan_to_num(corr, nan=0.0, posinf=1.0, neginf=-1.0)
    corr = np.clip(corr, -1, 1)
    np.fill_diagonal(corr, 1.0)
    
    # Distance = 1 - |correlation| (absolute correlation captures both positive
    # and negative associations as biologically informative)
    dist = 1 - np.abs(corr)
    np.fill_diagonal(dist, 0)
    # Ensure no negative values from floating point
    dist = np.maximum(dist, 0)
    return dist


def compute_nmi(labels_a, labels_b):
    """Compute Normalized Mutual Information between two clusterings."""
    try:
        from sklearn.metrics import normalized_mutual_info_score
        return normalized_mutual_info_score(labels_a, labels_b)
    except ImportError:
        # Manual NMI implementation
        from collections import Counter
        import math
        
        n = len(labels_a)
        if n == 0:
            return 0.0
        
        ca = Counter(labels_a)
        cb = Counter(labels_b)
        
        # Joint counts
        joint = Counter(zip(labels_a, labels_b))
        
        # Mutual information
        mi = 0.0
        for (a, b), nij in joint.items():
            if nij > 0:
                mi += (nij / n) * math.log((nij * n) / (ca[a] * cb[b]))
        
        # Entropies
        ha = -sum((c / n) * math.log(c / n) for c in ca.values() if c > 0)
        hb = -sum((c / n) * math.log(c / n) for c in cb.values() if c > 0)
        
        if ha + hb == 0:
            return 0.0
        return 2 * mi / (ha + hb)


def compute_silhouette(dist_matrix, labels):
    """Compute mean silhouette score from a distance matrix."""
    try:
        from sklearn.metrics import silhouette_score
        return silhouette_score(dist_matrix, labels, metric='precomputed')
    except ImportError:
        return np.nan


def main():
    from pan_cancer_xnode import DepMapLoader
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    
    out_dir = Path("pearson_comparison_results")
    out_dir.mkdir(exist_ok=True)
    
    print("Loading DepMap data...")
    depmap = DepMapLoader("./depmap_data")
    crispr = depmap.load_crispr_dependencies()
    pan_essential = depmap.get_pan_essential_genes(threshold=0.9)
    
    # Use top cancer types by cell line count for robust comparison
    available_ct = depmap.get_available_cancer_types()
    # Filter for cancer types with >= 20 cell lines for statistical power
    test_cancers = [(name, count) for name, count in available_ct if count >= 20][:10]
    
    print(f"Testing {len(test_cancers)} cancer types: "
          f"{', '.join(n for n,c in test_cancers)}")
    
    k_values = list(range(3, 16))  # k = 3 to 15
    results = []
    
    for cancer_name, n_lines in test_cancers:
        print(f"\n  {cancer_name} ({n_lines} cell lines)...")
        
        cell_lines = depmap.get_cell_lines_for_cancer(cancer_name)
        available = [cl for cl in cell_lines if cl in crispr.index]
        if len(available) < 10:
            print(f"    Skipping: too few available lines ({len(available)})")
            continue
        
        crispr_sub = crispr.loc[available]
        n_avail = len(available)
        min_ess = max(1, int(n_avail * 0.3))
        
        # Get selective genes
        gene_names = crispr_sub.columns.tolist()
        non_pan = np.array([g not in pan_essential for g in gene_names])
        ess_mask = (crispr_sub.values < -0.5)
        gene_ess_counts = ess_mask.sum(axis=0)
        selective_mask = (gene_ess_counts >= min_ess) & non_pan
        selective = [gene_names[i] for i in range(len(gene_names)) 
                     if selective_mask[i]]
        
        if len(selective) < 10:
            print(f"    Skipping: too few selective genes ({len(selective)})")
            continue
        
        print(f"    {len(selective)} selective genes")
        
        # Compute both distance matrices
        jaccard_dist = compute_jaccard_distance_matrix(crispr_sub, selective)
        pearson_dist = compute_pearson_distance_matrix(crispr_sub, selective)
        
        # Ensure valid distance matrices
        jaccard_dist = (jaccard_dist + jaccard_dist.T) / 2
        pearson_dist = (pearson_dist + pearson_dist.T) / 2
        np.fill_diagonal(jaccard_dist, 0)
        np.fill_diagonal(pearson_dist, 0)
        
        # Ward linkage for both
        try:
            jac_condensed = squareform(jaccard_dist, checks=False)
            pear_condensed = squareform(pearson_dist, checks=False)
            Z_jac = linkage(jac_condensed, method='ward')
            Z_pear = linkage(pear_condensed, method='ward')
        except Exception as e:
            print(f"    Linkage failed: {e}")
            continue
        
        for k in k_values:
            if k >= len(selective):
                continue
            
            labels_jac = fcluster(Z_jac, k, criterion='maxclust')
            labels_pear = fcluster(Z_pear, k, criterion='maxclust')
            
            # NMI between the two clusterings
            nmi = compute_nmi(labels_jac, labels_pear)
            
            # Silhouette scores for each
            sil_jac = compute_silhouette(jaccard_dist, labels_jac)
            sil_pear = compute_silhouette(pearson_dist, labels_pear)
            
            results.append({
                'cancer': cancer_name,
                'n_cell_lines': n_avail,
                'n_selective': len(selective),
                'k': k,
                'nmi': round(nmi, 4),
                'silhouette_jaccard': round(sil_jac, 4) if not np.isnan(sil_jac) else None,
                'silhouette_pearson': round(sil_pear, 4) if not np.isnan(sil_pear) else None,
            })
        
        print(f"    Done: {len(k_values)} k values tested")
    
    if not results:
        print("ERROR: No results generated!")
        sys.exit(1)
    
    # Save raw results
    df = pd.DataFrame(results)
    df.to_csv(out_dir / "nmi_comparison.csv", index=False)
    print(f"\nSaved {len(df)} rows to {out_dir / 'nmi_comparison.csv'}")
    
    # Aggregate statistics
    mean_by_k = df.groupby('k').agg({
        'nmi': ['mean', 'std', 'min', 'max'],
        'silhouette_jaccard': 'mean',
        'silhouette_pearson': 'mean',
    }).reset_index()
    
    # Flatten column names
    mean_by_k.columns = ['k', 'nmi_mean', 'nmi_std', 'nmi_min', 'nmi_max',
                         'sil_jaccard_mean', 'sil_pearson_mean']
    
    overall_nmi_mean = float(df['nmi'].mean())
    overall_nmi_std = float(df['nmi'].std())
    best_k = int(mean_by_k.loc[mean_by_k['nmi_mean'].idxmax(), 'k'])
    
    # Per-cancer mean NMI
    per_cancer = df.groupby('cancer')['nmi'].mean().to_dict()
    
    summary = {
        'overall_nmi_mean': round(overall_nmi_mean, 4),
        'overall_nmi_std': round(overall_nmi_std, 4),
        'best_agreement_k': best_k,
        'best_agreement_nmi': round(float(mean_by_k.loc[mean_by_k['k'] == best_k, 'nmi_mean'].values[0]), 4),
        'n_cancers_tested': len(df['cancer'].unique()),
        'n_total_comparisons': len(df),
        'per_cancer_mean_nmi': {k: round(v, 4) for k, v in per_cancer.items()},
        'per_k_stats': [
            {
                'k': int(row['k']),
                'nmi_mean': round(row['nmi_mean'], 4),
                'nmi_std': round(row['nmi_std'], 4),
                'sil_jaccard': round(row['sil_jaccard_mean'], 4) if not pd.isna(row['sil_jaccard_mean']) else None,
                'sil_pearson': round(row['sil_pearson_mean'], 4) if not pd.isna(row['sil_pearson_mean']) else None,
            }
            for _, row in mean_by_k.iterrows()
        ],
    }
    
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 65)
    print("Jaccard vs Pearson Co-essentiality Clustering Comparison")
    print("=" * 65)
    print(f"Cancer types tested: {summary['n_cancers_tested']}")
    print(f"Overall NMI (mean ± std): {overall_nmi_mean:.4f} ± {overall_nmi_std:.4f}")
    print(f"\n{'k':>4}  {'NMI mean':>10}  {'NMI std':>10}  {'Sil(Jac)':>10}  {'Sil(Pear)':>10}")
    print("-" * 50)
    for _, row in mean_by_k.iterrows():
        marker = " *" if int(row['k']) == best_k else ""
        print(f"{int(row['k']):>4}  {row['nmi_mean']:>10.4f}  {row['nmi_std']:>10.4f}  "
              f"{row['sil_jaccard_mean']:>10.4f}  {row['sil_pearson_mean']:>10.4f}{marker}")
    
    print(f"\n* Best agreement at k={best_k}")
    
    print("\nPer-cancer mean NMI:")
    for cancer, nmi_val in sorted(per_cancer.items(), key=lambda x: -x[1]):
        print(f"  {cancer:<35} {nmi_val:.4f}")
    
    # Generate figure
    generate_figure(df, mean_by_k, summary)
    
    return summary


def generate_figure(df, mean_by_k, summary):
    """Generate NMI comparison figure."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    
    # Panel A: NMI vs k (mean ± std across cancers)
    ax = axes[0]
    k_vals = mean_by_k['k'].values
    nmi_means = mean_by_k['nmi_mean'].values
    nmi_stds = mean_by_k['nmi_std'].values
    
    ax.fill_between(k_vals, nmi_means - nmi_stds, nmi_means + nmi_stds,
                    alpha=0.25, color='#2196F3')
    ax.plot(k_vals, nmi_means, 'o-', color='#1565C0', linewidth=2,
            markersize=5, label='Mean NMI')
    ax.set_xlabel('Number of clusters (k)', fontsize=11)
    ax.set_ylabel('NMI (Jaccard vs Pearson)', fontsize=11)
    ax.set_title('A', fontsize=13, fontweight='bold', loc='left')
    ax.set_xticks(k_vals)
    ax.set_ylim(0, 1)
    ax.axhline(y=summary['overall_nmi_mean'], color='gray', linestyle='--',
               alpha=0.5, label=f"Overall mean = {summary['overall_nmi_mean']:.2f}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Panel B: Per-cancer NMI distribution (box plot)
    ax = axes[1]
    cancers = sorted(df['cancer'].unique())
    cancer_nmis = [df[df['cancer'] == c]['nmi'].values for c in cancers]
    # Abbreviate cancer names
    short_names = []
    for c in cancers:
        if len(c) > 20:
            parts = c.split()
            short = ' '.join(parts[:2])
            if len(short) > 18:
                short = short[:16] + '..'
            short_names.append(short)
        else:
            short_names.append(c)
    
    bp = ax.boxplot(cancer_nmis, patch_artist=True, vert=True)
    colors = plt.cm.Set3(np.linspace(0, 1, len(cancers)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('NMI', fontsize=11)
    ax.set_title('B', fontsize=13, fontweight='bold', loc='left')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Panel C: Silhouette comparison (Jaccard vs Pearson)
    ax = axes[2]
    sil_jac = mean_by_k['sil_jaccard_mean'].values
    sil_pear = mean_by_k['sil_pearson_mean'].values
    
    width = 0.35
    x = np.arange(len(k_vals))
    ax.bar(x - width/2, sil_jac, width, label='Jaccard', color='#FF9800', alpha=0.8)
    ax.bar(x + width/2, sil_pear, width, label='Pearson', color='#4CAF50', alpha=0.8)
    ax.set_xlabel('Number of clusters (k)', fontsize=11)
    ax.set_ylabel('Mean silhouette score', fontsize=11)
    ax.set_title('C', fontsize=13, fontweight='bold', loc='left')
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(k)) for k in k_vals])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    fig_dir = Path("figures")
    fig_dir.mkdir(exist_ok=True)
    for ext in ['pdf', 'png']:
        fig.savefig(fig_dir / f"figS_pearson_nmi_comparison.{ext}",
                    dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\nFigure saved to figures/figS_pearson_nmi_comparison.pdf")


if __name__ == "__main__":
    main()
