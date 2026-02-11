#!/usr/bin/env python3
"""
Lineage-Aware vs Welch t-test Cancer-Specific Dependency Comparison.

Runs the ALIN pipeline twice:
  1. Default (Welch t-test) — cancer vs all others, no lineage control
  2. Lineage-aware (OLS regression) — Chronos ~ lineage + is_target_cancer

Compares predictions, genes identified, and benchmark concordance.
Generates summary statistics for paper and supplementary figure.
"""

import sys
import time
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.WARNING)

from gold_standard import CANCER_ALIASES, run_benchmark

# Derive gold standard cancer types
GOLD_STANDARD_CANCERS = sorted({
    pipeline_name
    for aliases in CANCER_ALIASES.values()
    for pipeline_name in aliases
    if pipeline_name
})

CONDITIONS = {
    'welch_default': {'use_lineage_aware_statistical': False},
    'lineage_aware': {'use_lineage_aware_statistical': True},
}


def run_condition(condition_name: str, kwargs: dict, output_dir: Path):
    """Run full pipeline for one condition and benchmark."""
    from pan_cancer_xnode import PanCancerXNodeAnalyzer, generate_triple_summary_table

    print(f"\n{'='*70}")
    print(f"  CONDITION: {condition_name}")
    print(f"  kwargs: {kwargs}")
    print(f"{'='*70}\n")

    t0 = time.time()
    analyzer = PanCancerXNodeAnalyzer(**kwargs)

    results = {}
    per_cancer_genes = {}  # cancer_type → set of genes in MHS input paths

    for i, ct in enumerate(GOLD_STANDARD_CANCERS, 1):
        print(f"  [{i}/{len(GOLD_STANDARD_CANCERS)}] {ct}...", end="", flush=True)
        try:
            analysis = analyzer.analyze_cancer_type(ct)
            results[ct] = analysis
            # Collect cancer-specific genes from paths
            cs_genes = set()
            for vp in analysis.viability_paths:
                if vp.path_type == 'cancer_specific':
                    cs_genes.update(vp.nodes)
            per_cancer_genes[ct] = cs_genes

            if analysis.best_triple:
                targets = sorted(analysis.best_triple.targets)
                print(f" {'+'.join(targets)}", flush=True)
            else:
                print(" (no triple)", flush=True)
        except Exception as e:
            print(f" ERROR: {e}", flush=True)

    if not results:
        return None, {}

    df = generate_triple_summary_table(results)
    csv_path = output_dir / f'triple_combinations_{condition_name}.csv'
    df.to_csv(csv_path, index=False)

    # Run benchmark
    bench = run_benchmark(str(csv_path), verbose=False)
    elapsed = time.time() - t0
    recall = bench.get('recall', bench)

    print(f"  Done in {elapsed:.1f}s — any_overlap={recall.get('any_overlap', 0):.1%}")

    return {
        'condition': condition_name,
        'any_overlap': recall.get('any_overlap', 0),
        'pair_overlap': recall.get('pair_overlap', 0),
        'exact': recall.get('exact', 0),
        'superset': recall.get('superset', 0),
        'precision': recall.get('precision', 0),
        'any_overlap_testable': recall.get('testable_any_overlap', 0),
        'pair_overlap_testable': recall.get('testable_pair_overlap', 0),
        'n_predictions': len(df),
        'elapsed_s': round(elapsed, 1),
        'csv_path': str(csv_path),
    }, per_cancer_genes


def compare_predictions(genes_welch: dict, genes_lineage: dict):
    """Compare cancer-specific genes identified by each method."""
    comparison_rows = []
    all_cancers = sorted(set(genes_welch.keys()) | set(genes_lineage.keys()))

    for ct in all_cancers:
        g_w = genes_welch.get(ct, set())
        g_l = genes_lineage.get(ct, set())
        shared = g_w & g_l
        welch_only = g_w - g_l
        lineage_only = g_l - g_w

        comparison_rows.append({
            'cancer_type': ct,
            'welch_n': len(g_w),
            'lineage_n': len(g_l),
            'shared_n': len(shared),
            'welch_only_n': len(welch_only),
            'lineage_only_n': len(lineage_only),
            'jaccard': len(shared) / len(g_w | g_l) if (g_w | g_l) else float('nan'),
            'welch_genes': ', '.join(sorted(g_w)),
            'lineage_genes': ', '.join(sorted(g_l)),
            'shared_genes': ', '.join(sorted(shared)),
            'welch_only_genes': ', '.join(sorted(welch_only)),
            'lineage_only_genes': ', '.join(sorted(lineage_only)),
        })

    return pd.DataFrame(comparison_rows)


def compare_triples(csv_welch: str, csv_lineage: str):
    """Compare which target triples change between methods."""
    df_w = pd.read_csv(csv_welch)
    df_l = pd.read_csv(csv_lineage)

    merged = df_w[['Cancer Type', 'Target 1', 'Target 2', 'Target 3']].merge(
        df_l[['Cancer Type', 'Target 1', 'Target 2', 'Target 3']],
        on='Cancer Type', how='outer', suffixes=('_welch', '_lineage')
    )

    n_total = len(merged)
    n_changed = 0
    change_rows = []
    for _, row in merged.iterrows():
        w_set = {row.get('Target 1_welch', ''), row.get('Target 2_welch', ''), row.get('Target 3_welch', '')} - {'', np.nan}
        l_set = {row.get('Target 1_lineage', ''), row.get('Target 2_lineage', ''), row.get('Target 3_lineage', '')} - {'', np.nan}
        # Remove NaN
        w_set = {x for x in w_set if pd.notna(x)}
        l_set = {x for x in l_set if pd.notna(x)}
        changed = w_set != l_set
        if changed:
            n_changed += 1
        change_rows.append({
            'cancer_type': row['Cancer Type'],
            'welch_triple': '+'.join(sorted(w_set)),
            'lineage_triple': '+'.join(sorted(l_set)),
            'changed': changed,
            'targets_shared': len(w_set & l_set),
            'targets_total': len(w_set | l_set),
        })

    return pd.DataFrame(change_rows), n_changed, n_total


def generate_figure(output_dir: Path, gene_comparison: pd.DataFrame,
                    triple_comparison: pd.DataFrame, results: list):
    """Generate supplementary figure: gene Jaccard per cancer + benchmark bar."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available; skipping figure.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Panel A: Gene Jaccard similarity per cancer type
    ax = axes[0]
    gc = gene_comparison.dropna(subset=['jaccard']).sort_values('jaccard')
    if not gc.empty:
        ax.barh(gc['cancer_type'], gc['jaccard'], color='steelblue', edgecolor='white')
        ax.set_xlabel('Jaccard similarity')
        ax.set_title('A. Cancer-specific gene overlap')
        ax.set_xlim(0, 1)
    ax.spines[['top', 'right']].set_visible(False)

    # Panel B: N genes per method
    ax = axes[1]
    gc2 = gene_comparison.sort_values('cancer_type')
    if not gc2.empty:
        y = np.arange(len(gc2))
        bar_h = 0.35
        ax.barh(y - bar_h/2, gc2['welch_n'], bar_h, label='Welch', color='#D55E00')
        ax.barh(y + bar_h/2, gc2['lineage_n'], bar_h, label='Lineage-aware', color='#0072B2')
        ax.set_yticks(y)
        ax.set_yticklabels(gc2['cancer_type'], fontsize=7)
        ax.set_xlabel('N cancer-specific genes')
        ax.set_title('B. Genes per method')
        ax.legend(fontsize=8)
    ax.spines[['top', 'right']].set_visible(False)

    # Panel C: Benchmark comparison bars
    ax = axes[2]
    metrics = ['any_overlap', 'pair_overlap', 'exact']
    labels = ['Any overlap', 'Pair overlap', 'Exact']
    x = np.arange(len(metrics))
    bar_w = 0.3
    for i, r in enumerate(results):
        vals = [r.get(m, 0) for m in metrics]
        offset = -bar_w/2 + i * bar_w
        color = '#D55E00' if 'welch' in r['condition'] else '#0072B2'
        ax.bar(x + offset, vals, bar_w, label=r['condition'].replace('_', ' ').title(), color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Concordance')
    ax.set_title('C. Benchmark')
    ax.legend(fontsize=8)
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    fig_path = output_dir / 'lineage_aware_comparison.png'
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Also save to figures/ for paper
    figures_dir = Path('figures')
    figures_dir.mkdir(exist_ok=True)
    import shutil
    shutil.copy(fig_path, figures_dir / 'fig_s10_lineage_aware_comparison.png')

    print(f"  Figure saved: {fig_path}")
    print(f"  Copied to figures/fig_s10_lineage_aware_comparison.png")


def main():
    output_dir = Path('lineage_control_results')
    output_dir.mkdir(exist_ok=True)

    all_results = []
    all_genes = {}

    for cond_name, kwargs in CONDITIONS.items():
        result, genes = run_condition(cond_name, kwargs, output_dir)
        if result:
            all_results.append(result)
            all_genes[cond_name] = genes

    if len(all_results) < 2:
        print("ERROR: Need both conditions to compare.")
        sys.exit(1)

    # ── Gene-level comparison ──
    gene_comparison = compare_predictions(
        all_genes['welch_default'], all_genes['lineage_aware']
    )
    gene_path = output_dir / 'gene_comparison.csv'
    gene_comparison.to_csv(gene_path, index=False)

    median_jaccard = gene_comparison['jaccard'].median()
    mean_jaccard = gene_comparison['jaccard'].mean()

    # ── Triple-level comparison ──
    triple_changes, n_changed, n_total = compare_triples(
        all_results[0]['csv_path'], all_results[1]['csv_path']
    )
    triple_path = output_dir / 'triple_comparison.csv'
    triple_changes.to_csv(triple_path, index=False)

    # ── Generate figure ──
    generate_figure(output_dir, gene_comparison, triple_changes, all_results)

    # ── Print summary ──
    print(f"\n{'='*70}")
    print("  LINEAGE-AWARE vs WELCH COMPARISON SUMMARY")
    print(f"{'='*70}")

    print(f"\n  {'Metric':<30} {'Welch':>10} {'Lineage':>10}")
    print(f"  {'-'*50}")
    for metric in ['any_overlap', 'pair_overlap', 'exact', 'precision',
                    'any_overlap_testable', 'n_predictions']:
        w_val = all_results[0].get(metric, 0)
        l_val = all_results[1].get(metric, 0)
        if isinstance(w_val, float):
            print(f"  {metric:<30} {w_val:>9.1%} {l_val:>9.1%}")
        else:
            print(f"  {metric:<30} {w_val:>10} {l_val:>10}")

    print(f"\n  Gene-level Jaccard: median={median_jaccard:.3f}, mean={mean_jaccard:.3f}")
    print(f"  Triple predictions changed: {n_changed}/{n_total} ({n_changed/n_total:.1%})")

    # ── Save summary JSON ──
    summary = {
        'conditions': all_results,
        'gene_comparison': {
            'median_jaccard': round(median_jaccard, 3),
            'mean_jaccard': round(mean_jaccard, 3),
        },
        'triple_comparison': {
            'n_changed': n_changed,
            'n_total': n_total,
            'fraction_changed': round(n_changed / n_total, 3) if n_total else 0,
        },
    }
    json_path = output_dir / 'lineage_comparison_summary.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Results saved to: {output_dir}/")
    print(f"    gene_comparison.csv")
    print(f"    triple_comparison.csv")
    print(f"    lineage_comparison_summary.json")
    print(f"    lineage_aware_comparison.png")


if __name__ == '__main__':
    main()
