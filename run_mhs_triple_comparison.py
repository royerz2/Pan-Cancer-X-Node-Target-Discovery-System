#!/usr/bin/env python3
"""
MHS-to-Triple Comparison Analysis
==================================
Analyse the relationship between MHS (minimal hitting set) predictions
and ranked triple combinations across gold-standard cancer types.

Produces:
  1. Supplementary Table S4: Cancer | MHS targets | Top triple | Added | Removed | Hub penalty
  2. Overlap statistics: superset / subset / partial / disjoint
  3. STAT3 tracking: in MHS, in triple, or both
  4. UpSet-style overlap plot + hub penalty redistribution figure
  5. MHS cost vs. ranked triple score correlation
  6. Worked-example text for 3 cancer types

All outputs saved to mhs_triple_results/.
"""

import sys
import time
import json
import logging
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

logging.basicConfig(level=logging.WARNING)

from gold_standard import CANCER_ALIASES

# Derive gold standard cancer types
GOLD_STANDARD_CANCERS = sorted({
    pipeline_name
    for aliases in CANCER_ALIASES.values()
    for pipeline_name in aliases
    if pipeline_name
})


def run_analysis():
    """Run the full pipeline for each cancer type and capture MHS + triple data."""
    from pan_cancer_xnode import PanCancerXNodeAnalyzer

    print(f"Running ALIN pipeline for {len(GOLD_STANDARD_CANCERS)} cancer types...\n")

    analyzer = PanCancerXNodeAnalyzer()

    records = []
    for i, ct in enumerate(GOLD_STANDARD_CANCERS, 1):
        print(f"  [{i}/{len(GOLD_STANDARD_CANCERS)}] {ct}...", end="", flush=True)
        try:
            analysis = analyzer.analyze_cancer_type(ct)
        except Exception as e:
            print(f" ERROR: {e}", flush=True)
            continue

        # MHS targets (best = lowest cost, smallest size)
        mhs_targets = set()
        mhs_cost = 0.0
        mhs_coverage = 0.0
        mhs_size = 0
        if analysis.minimal_hitting_sets:
            best_mhs = analysis.minimal_hitting_sets[0]
            mhs_targets = set(best_mhs.targets)
            mhs_cost = best_mhs.total_cost
            mhs_coverage = best_mhs.coverage
            mhs_size = len(best_mhs.targets)

        # Triple targets
        triple_targets = set()
        triple_score = 0.0
        triple_synergy = 0.0
        triple_resistance = 0.0
        triple_coverage = 0.0
        if analysis.best_triple:
            triple_targets = set(analysis.best_triple.targets)
            triple_score = analysis.best_triple.combined_score
            triple_synergy = analysis.best_triple.synergy_score
            triple_resistance = analysis.best_triple.resistance_score
            triple_coverage = analysis.best_triple.coverage

        # Compute hub penalty for each gene in the triple
        # We need the gene_path_freqs: fraction of paths containing each gene
        n_paths = len(analysis.viability_paths)
        gene_path_freq = {}
        all_genes_in_paths = set()
        for vp in analysis.viability_paths:
            all_genes_in_paths.update(vp.nodes)
        for g in all_genes_in_paths:
            gene_path_freq[g] = sum(
                1 for vp in analysis.viability_paths if g in vp.nodes
            ) / max(n_paths, 1)

        freq_values = sorted(gene_path_freq.values())
        median_freq = freq_values[len(freq_values) // 2] if freq_values else 0.3

        hub_penalty_info = {}
        for g in mhs_targets | triple_targets:
            excess = gene_path_freq.get(g, 0) - median_freq
            hub_penalty_info[g] = {
                'path_freq': gene_path_freq.get(g, 0),
                'excess': max(excess, 0),
                'penalty': max(excess, 0) * 1.5,
            }

        # Determine overlap category
        shared = mhs_targets & triple_targets
        added = triple_targets - mhs_targets
        removed = mhs_targets - triple_targets

        if triple_targets <= mhs_targets and triple_targets:
            overlap_cat = 'triple_subset_of_mhs'
        elif mhs_targets <= triple_targets and mhs_targets:
            overlap_cat = 'triple_superset_of_mhs'
        elif shared:
            overlap_cat = 'partial_overlap'
        elif mhs_targets and triple_targets:
            overlap_cat = 'disjoint'
        else:
            overlap_cat = 'missing_data'

        record = {
            'cancer_type': ct,
            'n_paths': n_paths,
            'mhs_targets': '+'.join(sorted(mhs_targets)),
            'mhs_size': mhs_size,
            'mhs_cost': round(mhs_cost, 3),
            'mhs_coverage': round(mhs_coverage, 3),
            'triple_targets': '+'.join(sorted(triple_targets)),
            'triple_score': round(triple_score, 4),
            'triple_synergy': round(triple_synergy, 3),
            'triple_resistance': round(triple_resistance, 3),
            'triple_coverage': round(triple_coverage, 3),
            'shared': '+'.join(sorted(shared)) if shared else '',
            'added_in_triple': '+'.join(sorted(added)) if added else '',
            'removed_from_mhs': '+'.join(sorted(removed)) if removed else '',
            'n_shared': len(shared),
            'n_added': len(added),
            'n_removed': len(removed),
            'overlap_category': overlap_cat,
            'stat3_in_mhs': 'STAT3' in mhs_targets,
            'stat3_in_triple': 'STAT3' in triple_targets,
            'stat3_status': (
                'both' if 'STAT3' in mhs_targets and 'STAT3' in triple_targets else
                'mhs_only' if 'STAT3' in mhs_targets else
                'triple_only' if 'STAT3' in triple_targets else
                'neither'
            ),
            'hub_penalties': json.dumps({
                g: round(v['penalty'], 3) for g, v in hub_penalty_info.items()
            }),
            'hub_path_freqs': json.dumps({
                g: round(v['path_freq'], 3) for g, v in hub_penalty_info.items()
            }),
            'median_path_freq': round(median_freq, 4),
        }

        records.append(record)
        print(f"  MHS={'+'.join(sorted(mhs_targets))} → Triple={'+'.join(sorted(triple_targets))}"
              f"  [{overlap_cat}]")

    return pd.DataFrame(records)


def compute_statistics(df: pd.DataFrame) -> dict:
    """Compute summary statistics from the comparison data."""
    stats = {}

    # Overlap categories
    cat_counts = df['overlap_category'].value_counts().to_dict()
    stats['overlap_categories'] = cat_counts
    n = len(df)
    for cat in ['triple_subset_of_mhs', 'triple_superset_of_mhs', 'partial_overlap', 'disjoint']:
        stats[f'{cat}_frac'] = cat_counts.get(cat, 0) / n

    # STAT3 tracking
    stat3_counts = df['stat3_status'].value_counts().to_dict()
    stats['stat3_tracking'] = stat3_counts
    stats['stat3_in_mhs_frac'] = df['stat3_in_mhs'].mean()
    stats['stat3_in_triple_frac'] = df['stat3_in_triple'].mean()
    stats['stat3_removed_by_ranking'] = stat3_counts.get('mhs_only', 0)
    stats['stat3_kept'] = stat3_counts.get('both', 0)

    # Mean targets shared / added / removed
    stats['mean_shared'] = df['n_shared'].mean()
    stats['mean_added'] = df['n_added'].mean()
    stats['mean_removed'] = df['n_removed'].mean()

    # MHS cost vs triple score correlation
    valid = df.dropna(subset=['mhs_cost', 'triple_score'])
    valid = valid[(valid['mhs_cost'] > 0) & (valid['triple_score'] > 0)]
    if len(valid) >= 5:
        r, p = sp_stats.pearsonr(valid['mhs_cost'], valid['triple_score'])
        stats['cost_score_pearson_r'] = round(r, 3)
        stats['cost_score_pearson_p'] = round(p, 4)
        rho, rho_p = sp_stats.spearmanr(valid['mhs_cost'], valid['triple_score'])
        stats['cost_score_spearman_rho'] = round(rho, 3)
        stats['cost_score_spearman_p'] = round(rho_p, 4)
    else:
        stats['cost_score_note'] = 'too few valid pairs'

    return stats


def generate_worked_examples(df: pd.DataFrame) -> str:
    """Generate 2-3 worked example paragraphs for the main text."""
    lines = []

    # Pick 3 cancer types with different overlap patterns:
    # 1. PDAC (well-known) — partial overlap expected
    # 2. Melanoma — probably STAT3 removed
    # 3. One where triple is subset of MHS

    examples = []
    for target_ct in ['Pancreatic Adenocarcinoma', 'Melanoma', 'Invasive Breast Carcinoma',
                       'Non-Small Cell Lung Cancer', 'Colorectal Adenocarcinoma']:
        row = df[df['cancer_type'] == target_ct]
        if not row.empty:
            examples.append(row.iloc[0])
        if len(examples) >= 3:
            break

    for row in examples:
        ct = row['cancer_type']
        mhs = row['mhs_targets']
        triple = row['triple_targets']
        added = row['added_in_triple'] if row['added_in_triple'] else 'none'
        removed = row['removed_from_mhs'] if row['removed_from_mhs'] else 'none'
        stat3 = row['stat3_status']
        cat = row['overlap_category']

        hub_data = json.loads(row['hub_penalties'])
        high_penalty_genes = [f"{g} ({v:.2f})" for g, v in sorted(hub_data.items(), key=lambda x: -x[1])
                              if v > 0.1]

        lines.append(f"\\textbf{{{ct}:}} MHS selects \\{{{mhs.replace('+', ', ')}\\}} "
                     f"(cost-optimal mechanism coverage). "
                     f"The ranked triple is \\{{{triple.replace('+', ', ')}\\}}. ")
        if removed != 'none':
            lines.append(f"Targets removed: {removed.replace('+', ', ')} "
                         f"(hub penalties: {'; '.join(high_penalty_genes) if high_penalty_genes else 'minimal'}). ")
        if added != 'none':
            lines.append(f"Targets added: {added.replace('+', ', ')} "
                         f"(selected for synergy, druggability, or cancer-specific scoring). ")
        lines.append(f"STAT3 status: {stat3}. Overlap category: {cat.replace('_', ' ')}.\n")

    return '\n'.join(lines)


def generate_figures(df: pd.DataFrame, output_dir: Path):
    """Generate figures for MHS-to-triple comparison."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
        import matplotlib.patches as mpatches
    except ImportError:
        print("  matplotlib not available; skipping figures.")
        return

    fig = plt.figure(figsize=(16, 12))

    # ── Panel A: Overlap category bar chart ──
    ax1 = fig.add_subplot(2, 3, 1)
    cats = df['overlap_category'].value_counts()
    cat_labels = {
        'triple_subset_of_mhs': 'Triple ⊂ MHS',
        'triple_superset_of_mhs': 'Triple ⊃ MHS',
        'partial_overlap': 'Partial overlap',
        'disjoint': 'Disjoint',
    }
    labels = [cat_labels.get(c, c) for c in cats.index]
    colors = ['#0072B2', '#009E73', '#D55E00', '#CC79A7']
    ax1.bar(range(len(cats)), cats.values, color=colors[:len(cats)], edgecolor='white')
    ax1.set_xticks(range(len(cats)))
    ax1.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
    ax1.set_ylabel('Cancer types')
    ax1.set_title('A. MHS–Triple overlap', fontsize=10, fontweight='bold')
    ax1.spines[['top', 'right']].set_visible(False)

    # ── Panel B: STAT3 tracking ──
    ax2 = fig.add_subplot(2, 3, 2)
    stat3_cats = df['stat3_status'].value_counts()
    stat3_labels = {'both': 'Both', 'mhs_only': 'MHS only', 'triple_only': 'Triple only', 'neither': 'Neither'}
    s_labels = [stat3_labels.get(c, c) for c in stat3_cats.index]
    s_colors = ['#E69F00', '#56B4E9', '#F0E442', '#999999']
    ax2.bar(range(len(stat3_cats)), stat3_cats.values, color=s_colors[:len(stat3_cats)], edgecolor='white')
    ax2.set_xticks(range(len(stat3_cats)))
    ax2.set_xticklabels(s_labels, fontsize=9)
    ax2.set_ylabel('Cancer types')
    ax2.set_title('B. STAT3 in MHS vs. Triple', fontsize=10, fontweight='bold')
    ax2.spines[['top', 'right']].set_visible(False)

    # ── Panel C: Hub penalty redistribution ──
    ax3 = fig.add_subplot(2, 3, 3)
    # For each cancer type, show hub penalty for genes in MHS that were removed vs kept
    removed_penalties = []
    kept_penalties = []
    for _, row in df.iterrows():
        hub_data = json.loads(row['hub_penalties'])
        removed_genes = set(row['removed_from_mhs'].split('+')) if row['removed_from_mhs'] else set()
        triple_genes = set(row['triple_targets'].split('+')) if row['triple_targets'] else set()
        mhs_genes = set(row['mhs_targets'].split('+')) if row['mhs_targets'] else set()
        for g in mhs_genes:
            penalty = hub_data.get(g, 0)
            if g in removed_genes:
                removed_penalties.append(penalty)
            elif g in triple_genes:
                kept_penalties.append(penalty)

    bp_data = []
    bp_labels = []
    if kept_penalties:
        bp_data.append(kept_penalties)
        bp_labels.append(f'Kept in triple\n(n={len(kept_penalties)})')
    if removed_penalties:
        bp_data.append(removed_penalties)
        bp_labels.append(f'Removed\n(n={len(removed_penalties)})')
    if bp_data:
        parts = ax3.violinplot(bp_data, showmeans=True, showmedians=True)
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(['#0072B2', '#D55E00'][i] if i < 2 else '#999')
            pc.set_alpha(0.7)
        ax3.set_xticks(range(1, len(bp_labels) + 1))
        ax3.set_xticklabels(bp_labels, fontsize=8)
    ax3.set_ylabel('Hub penalty')
    ax3.set_title('C. Hub penalty: kept vs. removed', fontsize=10, fontweight='bold')
    ax3.spines[['top', 'right']].set_visible(False)

    # ── Panel D: MHS cost vs Triple score scatter ──
    ax4 = fig.add_subplot(2, 3, 4)
    valid = df[(df['mhs_cost'] > 0) & (df['triple_score'] > 0)]
    if not valid.empty:
        ax4.scatter(valid['mhs_cost'], valid['triple_score'], c='#0072B2',
                    s=40, edgecolor='white', alpha=0.8, zorder=3)
        for _, row in valid.iterrows():
            ax4.annotate(row['cancer_type'][:8], (row['mhs_cost'], row['triple_score']),
                         fontsize=6, alpha=0.7)
        # Correlation line
        if len(valid) >= 3:
            z = np.polyfit(valid['mhs_cost'], valid['triple_score'], 1)
            x_line = np.linspace(valid['mhs_cost'].min(), valid['mhs_cost'].max(), 50)
            ax4.plot(x_line, np.polyval(z, x_line), '--', color='#D55E00', alpha=0.6)
            r, p = sp_stats.pearsonr(valid['mhs_cost'], valid['triple_score'])
            ax4.text(0.05, 0.95, f'r = {r:.2f}, p = {p:.3f}',
                     transform=ax4.transAxes, fontsize=8, va='top')
    ax4.set_xlabel('MHS cost')
    ax4.set_ylabel('Triple combined score')
    ax4.set_title('D. MHS cost vs. Triple score', fontsize=10, fontweight='bold')
    ax4.spines[['top', 'right']].set_visible(False)

    # ── Panel E: N targets shared/added/removed per cancer ──
    ax5 = fig.add_subplot(2, 3, 5)
    y = np.arange(len(df))
    bar_h = 0.25
    df_sorted = df.sort_values('cancer_type')
    ax5.barh(y - bar_h, df_sorted['n_shared'], bar_h, label='Shared', color='#009E73')
    ax5.barh(y, df_sorted['n_added'], bar_h, label='Added in triple', color='#0072B2')
    ax5.barh(y + bar_h, df_sorted['n_removed'], bar_h, label='Removed from MHS', color='#D55E00')
    ax5.set_yticks(y)
    ax5.set_yticklabels(df_sorted['cancer_type'], fontsize=6)
    ax5.set_xlabel('N targets')
    ax5.set_title('E. Target changes per cancer', fontsize=10, fontweight='bold')
    ax5.legend(fontsize=7, loc='lower right')
    ax5.spines[['top', 'right']].set_visible(False)

    # ── Panel F: Gene frequency heatmap (most common across MHS and triples) ──
    ax6 = fig.add_subplot(2, 3, 6)
    mhs_gene_counts = Counter()
    triple_gene_counts = Counter()
    for _, row in df.iterrows():
        if row['mhs_targets']:
            for g in row['mhs_targets'].split('+'):
                mhs_gene_counts[g] += 1
        if row['triple_targets']:
            for g in row['triple_targets'].split('+'):
                triple_gene_counts[g] += 1
    # Top 12 genes across both
    all_gene_counts = Counter()
    for g in set(list(mhs_gene_counts.keys()) + list(triple_gene_counts.keys())):
        all_gene_counts[g] = mhs_gene_counts.get(g, 0) + triple_gene_counts.get(g, 0)
    top_genes = [g for g, _ in all_gene_counts.most_common(12)]

    x_pos = np.arange(len(top_genes))
    bar_w = 0.35
    mhs_vals = [mhs_gene_counts.get(g, 0) for g in top_genes]
    triple_vals = [triple_gene_counts.get(g, 0) for g in top_genes]
    ax6.bar(x_pos - bar_w/2, mhs_vals, bar_w, label='MHS', color='#E69F00', edgecolor='white')
    ax6.bar(x_pos + bar_w/2, triple_vals, bar_w, label='Ranked triple', color='#0072B2', edgecolor='white')
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(top_genes, rotation=45, ha='right', fontsize=7)
    ax6.set_ylabel(f'Cancer types (/{len(df)})')
    ax6.set_title('F. Gene frequency: MHS vs. Triple', fontsize=10, fontweight='bold')
    ax6.legend(fontsize=7)
    ax6.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    fig_path = output_dir / 'mhs_triple_comparison.png'
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Copy to figures/
    figures_dir = Path('figures')
    figures_dir.mkdir(exist_ok=True)
    import shutil
    shutil.copy(fig_path, figures_dir / 'fig_s11_mhs_triple_comparison.png')
    print(f"  Figure saved: {fig_path}")
    print(f"  Copied to figures/fig_s11_mhs_triple_comparison.png")


def main():
    output_dir = Path('mhs_triple_results')
    output_dir.mkdir(exist_ok=True)

    t0 = time.time()

    # 1. Run pipeline and collect MHS + triple data
    df = run_analysis()
    csv_path = output_dir / 'mhs_triple_comparison.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved comparison table: {csv_path}")

    # 2. Compute statistics
    stats = compute_statistics(df)

    # 3. Generate worked examples
    examples_text = generate_worked_examples(df)

    # 4. Generate figures
    generate_figures(df, output_dir)

    elapsed = time.time() - t0

    # 5. Print summary
    print(f"\n{'='*70}")
    print("  MHS-TO-TRIPLE COMPARISON SUMMARY")
    print(f"{'='*70}")

    print(f"\n  Overlap categories (n={len(df)}):")
    for cat, count in stats['overlap_categories'].items():
        frac = count / len(df)
        print(f"    {cat}: {count} ({frac:.1%})")

    print(f"\n  STAT3 tracking:")
    for status, count in stats['stat3_tracking'].items():
        print(f"    {status}: {count}")
    print(f"    STAT3 in MHS: {stats['stat3_in_mhs_frac']:.1%}")
    print(f"    STAT3 in triple: {stats['stat3_in_triple_frac']:.1%}")
    print(f"    Removed by ranking: {stats['stat3_removed_by_ranking']}")

    print(f"\n  Target changes (mean):")
    print(f"    Shared: {stats['mean_shared']:.1f}")
    print(f"    Added:  {stats['mean_added']:.1f}")
    print(f"    Removed: {stats['mean_removed']:.1f}")

    if 'cost_score_pearson_r' in stats:
        print(f"\n  MHS cost vs Triple score:")
        print(f"    Pearson r = {stats['cost_score_pearson_r']}, p = {stats['cost_score_pearson_p']}")
        print(f"    Spearman ρ = {stats['cost_score_spearman_rho']}, p = {stats['cost_score_spearman_p']}")

    print(f"\n  Elapsed: {elapsed:.0f}s")

    # 6. Save everything
    json_path = output_dir / 'mhs_triple_summary.json'
    with open(json_path, 'w') as f:
        json.dump(stats, f, indent=2)

    examples_path = output_dir / 'worked_examples.txt'
    with open(examples_path, 'w') as f:
        f.write(examples_text)

    # Save supplementary table in LaTeX-ready format
    supp_table_path = output_dir / 'table_s4_mhs_triple.csv'
    supp_df = df[['cancer_type', 'mhs_targets', 'mhs_cost', 'triple_targets',
                   'triple_score', 'added_in_triple', 'removed_from_mhs',
                   'overlap_category', 'stat3_status']].copy()
    supp_df.columns = ['Cancer Type', 'MHS Targets', 'MHS Cost', 'Top Ranked Triple',
                        'Triple Score', 'Targets Added', 'Targets Removed',
                        'Overlap Category', 'STAT3 Status']
    supp_df.to_csv(supp_table_path, index=False)

    print(f"\n  All outputs in: {output_dir}/")


if __name__ == '__main__':
    main()
