#!/usr/bin/env python3
"""
Compute expanded ablation table with triple-level change tracking.

Reads existing ablation CSVs + lineage-aware results, computes:
  1. Benchmark concordance per condition
  2. Top-1 triple changes vs full pipeline
  3. Top-5 triple overlap counts
  4. Which cancers changed and how

Outputs: ablation_results/expanded_ablation_table.json
"""
import json
import pandas as pd
from pathlib import Path
from collections import OrderedDict

ABLATION_DIR = Path('ablation_results')
LINEAGE_DIR = Path('lineage_control_results')

# Conditions to include: label → (csv_path, source_for_metrics)
CONDITIONS = OrderedDict([
    ('Full pipeline', {
        'csv': ABLATION_DIR / 'triple_combinations_full_pipeline.csv',
        'metrics_key': 'full_pipeline',
    }),
    ('No OmniPath paths', {
        'csv': ABLATION_DIR / 'triple_combinations_no_omnipath.csv',
        'metrics_key': 'no_omnipath',
    }),
    ('No perturbation priors', {
        'csv': ABLATION_DIR / 'triple_combinations_no_perturbation.csv',
        'metrics_key': 'no_perturbation',
    }),
    ('No hub penalty', {
        'csv': ABLATION_DIR / 'triple_combinations_no_hub_penalty.csv',
        'metrics_key': 'no_hub_penalty',
    }),
    ('Lineage-aware statistical', {
        'csv': LINEAGE_DIR / 'triple_combinations_lineage_aware.csv',
        'metrics_key': 'lineage_aware',
    }),
])


def load_triples(csv_path: Path) -> dict:
    """Load triple predictions from CSV, return {cancer_type: (top1_set, top5_list)}."""
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().replace(' ', '_') for c in df.columns]

    result = {}
    for _, row in df.iterrows():
        ct = row['Cancer_Type']
        t1 = frozenset(filter(pd.notna, [row.get('Target_1'), row.get('Target_2'), row.get('Target_3')]))
        result[ct] = {'top1': t1}
    return result


def compare_triples(baseline: dict, ablation: dict, condition_name: str):
    """Compare triples between baseline and ablation condition."""
    all_cancers = sorted(set(baseline.keys()) | set(ablation.keys()))

    changes = []
    n_same = 0
    n_changed = 0
    n_missing = 0

    for ct in all_cancers:
        if ct not in baseline:
            continue
        if ct not in ablation:
            n_missing += 1
            changes.append({
                'cancer': ct,
                'baseline': sorted(baseline[ct]['top1']),
                'ablation': 'NO PREDICTION',
                'type': 'lost'
            })
            continue

        b_set = baseline[ct]['top1']
        a_set = ablation[ct]['top1']

        if b_set == a_set:
            n_same += 1
        else:
            n_changed += 1
            shared = b_set & a_set
            changes.append({
                'cancer': ct,
                'baseline': sorted(b_set),
                'ablation': sorted(a_set),
                'shared': sorted(shared),
                'n_shared': len(shared),
                'type': 'changed'
            })

    return {
        'n_same': n_same,
        'n_changed': n_changed,
        'n_missing': n_missing,
        'n_total': len(baseline),
        'changes': changes,
    }


def main():
    # Load pre-computed metrics
    with open(ABLATION_DIR / 'ablation_summary.json') as f:
        ablation_metrics = {r['condition']: r for r in json.load(f)}

    with open(LINEAGE_DIR / 'lineage_comparison_summary.json') as f:
        lineage_data = json.load(f)
        lineage_conds = {c['condition']: c for c in lineage_data['conditions']}

    # Load baseline triples
    baseline = load_triples(CONDITIONS['Full pipeline']['csv'])
    print(f"Baseline (full pipeline): {len(baseline)} cancer types\n")

    # Build table
    table_rows = []

    for label, info in CONDITIONS.items():
        # Get metrics
        key = info['metrics_key']
        if key in ablation_metrics:
            m = ablation_metrics[key]
        elif key in lineage_conds:
            m = lineage_conds[key]
        else:
            print(f"WARNING: No metrics for {key}")
            continue

        any_ovlp = m.get('any_overlap', 0)
        pair_ovlp = m.get('pair_overlap', 0)
        exact = m.get('exact', 0)
        n_preds = m.get('n_predictions', 0)

        # Load triples and compare
        triples = load_triples(info['csv'])
        comparison = compare_triples(baseline, triples, label)

        row = {
            'condition': label,
            'any_overlap': any_ovlp,
            'pair_overlap': pair_ovlp,
            'exact': exact,
            'n_predictions': n_preds,
            'top1_changed': comparison['n_changed'],
            'top1_same': comparison['n_same'],
            'top1_lost': comparison['n_missing'],
            'changes': comparison['changes'],
        }
        table_rows.append(row)

        # Print
        delta_any = (any_ovlp - ablation_metrics['full_pipeline']['any_overlap']) * 100
        delta_pair = (pair_ovlp - ablation_metrics['full_pipeline']['pair_overlap']) * 100
        delta_str = f"  Δany={delta_any:+.1f}pp  Δpair={delta_pair:+.1f}pp" if label != 'Full pipeline' else ""
        print(f"{label:<28} AnyOvlp={any_ovlp:.1%}  PairOvlp={pair_ovlp:.1%}  "
              f"Exact={exact:.1%}  Preds={n_preds}  "
              f"Top1-changed={comparison['n_changed']}/{len(baseline)}{delta_str}")

        if comparison['changes']:
            for ch in comparison['changes']:
                if ch['type'] == 'changed':
                    print(f"    {ch['cancer']}: {'+'.join(ch['baseline'])} → {'+'.join(ch['ablation'])} "
                          f"(shared={ch['n_shared']})")
                elif ch['type'] == 'lost':
                    print(f"    {ch['cancer']}: {'+'.join(ch['baseline'])} → LOST")

    # Save
    out_path = ABLATION_DIR / 'expanded_ablation_table.json'
    with open(out_path, 'w') as f:
        # Convert frozensets for JSON serialization
        def default_ser(o):
            if isinstance(o, frozenset):
                return sorted(o)
            raise TypeError(f"Cannot serialize {type(o)}")
        json.dump(table_rows, f, indent=2, default=default_ser)
    print(f"\nSaved to {out_path}")

    # Print LaTeX-ready table
    print(f"\n{'='*100}")
    print("LaTeX-ready table:")
    print(f"{'='*100}")
    full_any = ablation_metrics['full_pipeline']['any_overlap']
    full_pair = ablation_metrics['full_pipeline']['pair_overlap']
    full_exact = ablation_metrics['full_pipeline']['exact']
    print(f"{'Condition':<28} {'AnyOvlp':>8} {'PairOvlp':>9} {'Exact':>6} {'Preds':>5} {'Top-1 Δ':>8}")
    print("-" * 100)
    for row in table_rows:
        delta_any = (row['any_overlap'] - full_any) * 100
        delta_pair = (row['pair_overlap'] - full_pair) * 100
        delta_str = f"({delta_any:+.1f})" if row['condition'] != 'Full pipeline' else ""
        top1_str = f"{row['top1_changed']}/{row['n_predictions']}" if row['condition'] != 'Full pipeline' else "---"
        print(f"{row['condition']:<28} {row['any_overlap']:>7.1%} {row['pair_overlap']:>8.1%} "
              f"{row['exact']:>5.1%} {row['n_predictions']:>5} {top1_str:>8}")


if __name__ == '__main__':
    main()
