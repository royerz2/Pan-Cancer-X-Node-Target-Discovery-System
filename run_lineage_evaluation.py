#!/usr/bin/env python3
"""
Lineage-Stratified Benchmark Evaluation
========================================

Groups gold-standard entries by lineage (Carcinoma, Hematologic, CNS, Sarcoma, Other),
recalculates benchmark concordance per lineage, runs Fisher's exact tests comparing
within-lineage vs across-lineage performance, and generates a supplementary figure.

Outputs:
  - lineage_evaluation_results/lineage_benchmark.csv
  - lineage_evaluation_results/summary.json
  - figures/figS_lineage_benchmark.pdf
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact

# Import benchmark components
from gold_standard import (
    GOLD_STANDARD, CANCER_ALIASES, GENE_EQUIVALENTS,
    check_match, is_testable, _resolve_pipeline_cancers,
    NON_CRISPR_TARGETS,
)

# ============================================================================
# LINEAGE ASSIGNMENTS
# ============================================================================
# Each gold-standard cancer type → lineage category.
# Categories: Carcinoma (epithelial solid tumors), Hematologic (blood/marrow),
#             CNS (brain tumors), Sarcoma (mesenchymal), Other (doesn't fit).

CANCER_LINEAGE = {
    # Carcinomas (epithelial origin)
    'Melanoma': 'Carcinoma',
    'Non-Small Cell Lung Cancer': 'Carcinoma',
    'Anaplastic Thyroid Cancer': 'Carcinoma',
    'Colorectal Adenocarcinoma': 'Carcinoma',
    'Invasive Breast Carcinoma': 'Carcinoma',
    'Renal Cell Carcinoma': 'Carcinoma',
    'Head and Neck Squamous Cell Carcinoma': 'Carcinoma',
    'Ovarian Epithelial Tumor': 'Carcinoma',
    'Low-Grade Serous Ovarian Cancer': 'Carcinoma',
    'Prostate Adenocarcinoma': 'Carcinoma',
    'Pancreatic Adenocarcinoma': 'Carcinoma',
    'Bladder Urothelial Carcinoma': 'Carcinoma',
    'Endometrial Carcinoma': 'Carcinoma',
    'Hepatocellular Carcinoma': 'Carcinoma',
    'Stomach Adenocarcinoma': 'Carcinoma',
    'Biliary Tract Cancer': 'Carcinoma',
    'Cholangiocarcinoma': 'Carcinoma',
    'Medullary Thyroid Carcinoma': 'Carcinoma',

    # Hematologic malignancies
    'Acute Myeloid Leukemia': 'Hematologic',
    'Chronic Lymphocytic Leukemia': 'Hematologic',
    'Plasma Cell Myeloma': 'Hematologic',

    # CNS tumors
    'Low-Grade Glioma': 'CNS',
    'High-Grade Glioma': 'CNS',
    'Diffuse Glioma': 'CNS',  # pipeline name for glioma

    # Sarcomas (mesenchymal)
    'Liposarcoma': 'Sarcoma',
    'Gastrointestinal Stromal Tumor': 'Sarcoma',
}


def assign_lineage(cancer_type: str) -> str:
    """Assign lineage to a cancer type, defaulting to 'Other'."""
    return CANCER_LINEAGE.get(cancer_type, 'Other')


def run_lineage_benchmark(predictions_csv: str = 'results/triple_combinations.csv'):
    """Run benchmark stratified by lineage."""
    df = pd.read_csv(predictions_csv)
    df.columns = [c.replace(' ', '_') for c in df.columns]

    # Group gold standard entries by lineage
    lineage_entries = defaultdict(list)
    for entry in GOLD_STANDARD:
        lineage = assign_lineage(entry['cancer'])
        lineage_entries[lineage].append(entry)

    print("=" * 70)
    print("LINEAGE-STRATIFIED BENCHMARK EVALUATION")
    print("=" * 70)
    print(f"\nGold standard: {len(GOLD_STANDARD)} entries across {len(lineage_entries)} lineages")
    for lin, entries in sorted(lineage_entries.items()):
        n_test = sum(1 for e in entries if is_testable(e))
        print(f"  {lin}: {len(entries)} entries ({n_test} testable)")

    # Evaluate per lineage
    lineage_results = {}
    all_rows = []

    for lineage, entries in sorted(lineage_entries.items()):
        results = _evaluate_entries(df, entries)
        n = len(results)
        testable_results = [r for r, e in zip(results, entries) if is_testable(e)]
        nt = len(testable_results)

        # All entries
        n_exact = sum(1 for r in results if r['best_match'] == 'exact')
        n_pair = sum(1 for r in results if r['best_match'] in ('exact', 'superset', 'pair_overlap'))
        n_any = sum(1 for r in results if r['best_match'] in ('exact', 'superset', 'pair_overlap', 'any_overlap'))

        # Testable entries
        nt_exact = sum(1 for r in testable_results if r['best_match'] == 'exact')
        nt_pair = sum(1 for r in testable_results if r['best_match'] in ('exact', 'superset', 'pair_overlap'))
        nt_any = sum(1 for r in testable_results if r['best_match'] in ('exact', 'superset', 'pair_overlap', 'any_overlap'))

        lineage_results[lineage] = {
            'n_total': n,
            'n_testable': nt,
            'exact': n_exact, 'pair_overlap': n_pair, 'any_overlap': n_any,
            'exact_pct': n_exact / n * 100 if n > 0 else 0,
            'pair_pct': n_pair / n * 100 if n > 0 else 0,
            'any_pct': n_any / n * 100 if n > 0 else 0,
            'testable_exact': nt_exact, 'testable_pair': nt_pair, 'testable_any': nt_any,
            'testable_exact_pct': nt_exact / nt * 100 if nt > 0 else 0,
            'testable_pair_pct': nt_pair / nt * 100 if nt > 0 else 0,
            'testable_any_pct': nt_any / nt * 100 if nt > 0 else 0,
            'results': results,
        }

        all_rows.append({
            'Lineage': lineage,
            'N_entries': n,
            'N_testable': nt,
            'Any_overlap': f"{n_any}/{n}",
            'Any_overlap_pct': round(n_any / n * 100, 1) if n > 0 else 0,
            'Pair_overlap': f"{n_pair}/{n}",
            'Pair_overlap_pct': round(n_pair / n * 100, 1) if n > 0 else 0,
            'Exact': f"{n_exact}/{n}",
            'Exact_pct': round(n_exact / n * 100, 1) if n > 0 else 0,
            'Testable_any_pct': round(nt_any / nt * 100, 1) if nt > 0 else 0,
            'Testable_pair_pct': round(nt_pair / nt * 100, 1) if nt > 0 else 0,
            'Testable_exact_pct': round(nt_exact / nt * 100, 1) if nt > 0 else 0,
        })

    # Print results
    print(f"\n{'Lineage':<14} {'N':>3} {'Any%':>8} {'Pair%':>8} {'Exact%':>8}   {'T-Any%':>8} {'T-Pair%':>8}")
    print("-" * 70)
    for row in all_rows:
        print(f"{row['Lineage']:<14} {row['N_entries']:>3} "
              f"{row['Any_overlap_pct']:>7.1f}% {row['Pair_overlap_pct']:>7.1f}% "
              f"{row['Exact_pct']:>7.1f}%   "
              f"{row['Testable_any_pct']:>7.1f}% {row['Testable_pair_pct']:>7.1f}%")

    # Statistical tests: within-lineage vs across-lineage (Fisher's exact)
    print("\n" + "=" * 70)
    print("STATISTICAL TESTS: Within-lineage vs across-lineage")
    print("=" * 70)
    stat_results = _run_statistical_tests(lineage_results)

    # Save results
    outdir = Path('lineage_evaluation_results')
    outdir.mkdir(exist_ok=True)

    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(outdir / 'lineage_benchmark.csv', index=False)
    print(f"\nSaved to {outdir / 'lineage_benchmark.csv'}")

    # Summary JSON
    summary = {
        'lineage_metrics': {k: {kk: vv for kk, vv in v.items() if kk != 'results'}
                            for k, v in lineage_results.items()},
        'statistical_tests': stat_results,
        'pooled_any_overlap': sum(r['any_overlap'] for r in lineage_results.values()) / len(GOLD_STANDARD) * 100,
        'pooled_pair_overlap': sum(r['pair_overlap'] for r in lineage_results.values()) / len(GOLD_STANDARD) * 100,
    }
    with open(outdir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Saved to {outdir / 'summary.json'}")

    # Generate figure
    _generate_figure(lineage_results, stat_results)

    return lineage_results, stat_results


def _evaluate_entries(df, entries):
    """Evaluate a list of gold-standard entries against predictions."""
    results = []
    for entry in entries:
        gold_cancer = entry['cancer']
        gold_targets = entry['targets']
        pipeline_cancers = _resolve_pipeline_cancers(gold_cancer)

        matched_predictions = []
        for _, row in df.iterrows():
            if row['Cancer_Type'] in pipeline_cancers:
                pred_targets = frozenset({row['Target_1'], row['Target_2'], row['Target_3']})
                match_type = check_match(pred_targets, gold_targets)
                matched_predictions.append({
                    'cancer_type': row['Cancer_Type'],
                    'predicted': pred_targets,
                    'match_type': match_type,
                })
                # Also check best-combination columns
                bc1 = row.get('Best_Combo_1', '')
                bc2 = row.get('Best_Combo_2', '')
                bc3 = row.get('Best_Combo_3', '')
                if bc1 and str(bc1) != 'nan' and bc2 and str(bc2) != 'nan':
                    combo_set = {str(bc1), str(bc2)}
                    if bc3 and str(bc3) != 'nan' and str(bc3) != '':
                        combo_set.add(str(bc3))
                    combo_targets = frozenset(combo_set)
                    if combo_targets != pred_targets:
                        combo_match = check_match(combo_targets, gold_targets)
                        matched_predictions.append({
                            'cancer_type': row['Cancer_Type'],
                            'predicted': combo_targets,
                            'match_type': combo_match,
                        })

        match_priority = {'exact': 4, 'superset': 3, 'pair_overlap': 2, 'any_overlap': 1, 'none': 0}
        if matched_predictions:
            best = max(matched_predictions, key=lambda x: match_priority[x['match_type']])
        else:
            best = {'cancer_type': 'NO_PREDICTION', 'predicted': frozenset(), 'match_type': 'none'}

        results.append({
            'gold_cancer': gold_cancer,
            'gold_targets': gold_targets,
            'best_match': best['match_type'],
            'predicted': best['predicted'],
            'pipeline_cancer': best['cancer_type'],
        })

    return results


def _run_statistical_tests(lineage_results):
    """Run Fisher's exact tests comparing each lineage vs rest."""
    stat_results = {}

    # For each metric, test lineage vs rest
    for metric in ['any_overlap', 'pair_overlap']:
        metric_tests = {}
        total_hits = sum(r[metric] for r in lineage_results.values())
        total_n = sum(r['n_total'] for r in lineage_results.values())

        for lineage, res in lineage_results.items():
            if res['n_total'] < 2:
                continue
            # 2x2 table: [hits_in, misses_in; hits_out, misses_out]
            hits_in = res[metric]
            misses_in = res['n_total'] - hits_in
            hits_out = total_hits - hits_in
            misses_out = (total_n - res['n_total']) - hits_out

            table = [[hits_in, misses_in], [hits_out, misses_out]]
            odds_ratio, p_value = fisher_exact(table, alternative='two-sided')

            pct_in = hits_in / res['n_total'] * 100 if res['n_total'] > 0 else 0
            pct_out = hits_out / (total_n - res['n_total']) * 100 if (total_n - res['n_total']) > 0 else 0

            metric_tests[lineage] = {
                'hits': hits_in,
                'total': res['n_total'],
                'pct': round(pct_in, 1),
                'rest_pct': round(pct_out, 1),
                'odds_ratio': round(odds_ratio, 3) if not np.isinf(odds_ratio) else 'inf',
                'p_value': round(p_value, 4),
                'significant': p_value < 0.05,
            }

            print(f"  {metric:>14} | {lineage:<14}: {pct_in:.1f}% vs rest {pct_out:.1f}% "
                  f"(OR={odds_ratio:.2f}, p={p_value:.4f}{'*' if p_value < 0.05 else ''})")

        stat_results[metric] = metric_tests

    # Overall heterogeneity: chi-square test across lineages
    from scipy.stats import chi2_contingency
    for metric in ['any_overlap', 'pair_overlap']:
        contingency = []
        labels = []
        for lineage, res in sorted(lineage_results.items()):
            if res['n_total'] >= 2:
                hits = res[metric]
                misses = res['n_total'] - hits
                contingency.append([hits, misses])
                labels.append(lineage)
        if len(contingency) >= 2:
            try:
                chi2, p, dof, expected = chi2_contingency(contingency)
                print(f"\n  {metric} heterogeneity across lineages: "
                      f"chi2={chi2:.2f}, df={dof}, p={p:.4f}")
                stat_results[f'{metric}_heterogeneity'] = {
                    'chi2': round(chi2, 3),
                    'df': dof,
                    'p_value': round(p, 4),
                    'lineages': labels,
                }
            except ValueError:
                pass

    return stat_results


def _generate_figure(lineage_results, stat_results):
    """Generate supplementary figure: lineage-stratified benchmark."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    lineages = sorted(lineage_results.keys(),
                      key=lambda x: lineage_results[x]['any_pct'], reverse=True)
    colors = {
        'Carcinoma': '#2196F3',
        'Hematologic': '#E91E63',
        'CNS': '#4CAF50',
        'Sarcoma': '#FF9800',
        'Other': '#9E9E9E',
    }

    # Panel A: Grouped bar chart — all entries
    x = np.arange(len(lineages))
    width = 0.25
    any_vals = [lineage_results[l]['any_pct'] for l in lineages]
    pair_vals = [lineage_results[l]['pair_pct'] for l in lineages]
    exact_vals = [lineage_results[l]['exact_pct'] for l in lineages]

    bars1 = axes[0].bar(x - width, any_vals, width, label='Any overlap',
                        color='#42A5F5', edgecolor='white', linewidth=0.5)
    bars2 = axes[0].bar(x, pair_vals, width, label='Pair overlap',
                        color='#FF7043', edgecolor='white', linewidth=0.5)
    bars3 = axes[0].bar(x + width, exact_vals, width, label='Exact',
                        color='#66BB6A', edgecolor='white', linewidth=0.5)

    axes[0].set_xlabel('Lineage')
    axes[0].set_ylabel('Recall (%)')
    axes[0].set_title('A. All entries', fontweight='bold', loc='left')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(lineages, rotation=30, ha='right', fontsize=9)
    axes[0].legend(fontsize=8, loc='upper right')
    axes[0].set_ylim(0, max(any_vals + [10]) * 1.15)
    # Add counts
    for i, lineage in enumerate(lineages):
        n = lineage_results[lineage]['n_total']
        axes[0].text(i, max(any_vals[i], pair_vals[i], exact_vals[i]) + 1.5,
                     f'n={n}', ha='center', fontsize=8, style='italic')

    # Pooled reference line
    pooled_any = sum(r['any_overlap'] for r in lineage_results.values()) / len(GOLD_STANDARD) * 100
    axes[0].axhline(y=pooled_any, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    axes[0].text(len(lineages) - 0.5, pooled_any + 1, f'Pooled: {pooled_any:.1f}%',
                 fontsize=7, color='gray', ha='right')

    # Panel B: Testable entries only
    tany = [lineage_results[l]['testable_any_pct'] for l in lineages]
    tpair = [lineage_results[l]['testable_pair_pct'] for l in lineages]
    texact = [lineage_results[l]['testable_exact_pct'] for l in lineages]

    axes[1].bar(x - width, tany, width, label='Any overlap',
                color='#42A5F5', edgecolor='white', linewidth=0.5)
    axes[1].bar(x, tpair, width, label='Pair overlap',
                color='#FF7043', edgecolor='white', linewidth=0.5)
    axes[1].bar(x + width, texact, width, label='Exact',
                color='#66BB6A', edgecolor='white', linewidth=0.5)

    axes[1].set_xlabel('Lineage')
    axes[1].set_ylabel('Recall (%)')
    axes[1].set_title('B. Testable entries only', fontweight='bold', loc='left')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(lineages, rotation=30, ha='right', fontsize=9)
    axes[1].legend(fontsize=8, loc='upper right')
    axes[1].set_ylim(0, max(tany + [10]) * 1.15)
    for i, lineage in enumerate(lineages):
        nt = lineage_results[lineage]['n_testable']
        axes[1].text(i, max(tany[i], tpair[i], texact[i]) + 1.5,
                     f'n={nt}', ha='center', fontsize=8, style='italic')

    # Panel C: Per-cancer breakdown within each lineage (dot plot)
    y_pos = 0
    y_labels = []
    y_positions = []
    lineage_boundaries = []

    for lineage in lineages:
        res = lineage_results[lineage]
        entries_with_results = [(r['gold_cancer'], r['best_match']) for r in res['results']]
        # Deduplicate by cancer name (show best match per cancer)
        cancer_best = {}
        match_priority = {'exact': 4, 'superset': 3, 'pair_overlap': 2, 'any_overlap': 1, 'none': 0}
        for cancer, match in entries_with_results:
            if cancer not in cancer_best or match_priority[match] > match_priority[cancer_best[cancer]]:
                cancer_best[cancer] = match

        lineage_boundaries.append(y_pos)
        for cancer, match in sorted(cancer_best.items()):
            match_color = {
                'exact': '#2E7D32',
                'superset': '#66BB6A',
                'pair_overlap': '#FF9800',
                'any_overlap': '#42A5F5',
                'none': '#E57373',
            }
            score = match_priority[match]
            axes[2].barh(y_pos, score, color=match_color[match], height=0.7,
                         edgecolor='white', linewidth=0.3)
            # Truncate long cancer names
            short = cancer[:25] + '…' if len(cancer) > 25 else cancer
            y_labels.append(short)
            y_positions.append(y_pos)
            y_pos += 1
        y_pos += 0.5  # gap between lineages

    axes[2].set_yticks(y_positions)
    axes[2].set_yticklabels(y_labels, fontsize=6)
    axes[2].set_xlabel('Match level')
    axes[2].set_xticks([0, 1, 2, 3, 4])
    axes[2].set_xticklabels(['None', 'Any', 'Pair', 'Super', 'Exact'], fontsize=8)
    axes[2].set_title('C. Per-cancer match level', fontweight='bold', loc='left')
    axes[2].invert_yaxis()

    # Add lineage labels
    for i, lineage in enumerate(lineages):
        y_start = lineage_boundaries[i]
        n_cancers = len(set(r['gold_cancer'] for r in lineage_results[lineage]['results']))
        y_mid = y_start + n_cancers / 2 - 0.5
        axes[2].text(-0.5, y_mid, lineage, fontsize=7, fontweight='bold',
                     ha='right', va='center', color=colors.get(lineage, 'gray'))

    plt.tight_layout()

    figpath = Path('figures') / 'figS_lineage_benchmark.pdf'
    figpath.parent.mkdir(exist_ok=True)
    plt.savefig(figpath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nFigure saved to {figpath}")


if __name__ == '__main__':
    predictions = sys.argv[1] if len(sys.argv) > 1 else 'results/triple_combinations.csv'
    run_lineage_benchmark(predictions)
