#!/usr/bin/env python3
"""
ALIN Ablation Suite — remove each pipeline component and re-run benchmark.

Produces ablation_results/ablation_summary.csv with concordance metrics per condition.
"""
import sys
import time
import logging
import json
from pathlib import Path

logging.basicConfig(level=logging.WARNING)

from gold_standard import CANCER_ALIASES, run_benchmark

# Derive gold standard cancer types
GOLD_STANDARD_CANCERS = sorted({
    pipeline_name
    for aliases in CANCER_ALIASES.values()
    for pipeline_name in aliases
    if pipeline_name
})

# Ablation conditions: name → kwargs for PanCancerXNodeAnalyzer
ABLATION_CONDITIONS = {
    'full_pipeline': {},
    'no_omnipath': {'disable_omnipath': True},
    'no_perturbation': {'disable_perturbation': True},
    'no_coessentiality': {'disable_coessentiality': True},
    'no_statistical': {'disable_statistical': True},
    'no_hub_penalty': {'disable_hub_penalty': True},
}


def run_condition(condition_name: str, kwargs: dict, output_dir: Path):
    """Run pipeline for one ablation condition and benchmark."""
    from pan_cancer_xnode import PanCancerXNodeAnalyzer, generate_triple_summary_table

    print(f"\n{'='*70}")
    print(f"  ABLATION: {condition_name}")
    print(f"  kwargs: {kwargs}")
    print(f"{'='*70}\n")

    t0 = time.time()
    analyzer = PanCancerXNodeAnalyzer(**kwargs)

    results = {}
    for i, ct in enumerate(GOLD_STANDARD_CANCERS, 1):
        print(f"  [{i}/{len(GOLD_STANDARD_CANCERS)}] {ct}...", end="", flush=True)
        try:
            analysis = analyzer.analyze_cancer_type(ct)
            results[ct] = analysis
            if analysis.best_triple:
                targets = sorted(analysis.best_triple.targets)
                print(f" {'+'.join(targets)}", flush=True)
            else:
                print(" (no triple)", flush=True)
        except Exception as e:
            print(f" ERROR: {e}", flush=True)

    if not results:
        return None

    df = generate_triple_summary_table(results)
    csv_path = output_dir / f'triple_combinations_{condition_name}.csv'
    df.to_csv(csv_path, index=False)

    # Run benchmark
    bench = run_benchmark(str(csv_path), verbose=False)
    elapsed = time.time() - t0
    recall = bench.get('recall', bench)  # handle both nested and flat
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
    }


def main():
    output_dir = Path('ablation_results')
    output_dir.mkdir(exist_ok=True)

    all_results = []
    for cond_name, kwargs in ABLATION_CONDITIONS.items():
        result = run_condition(cond_name, kwargs, output_dir)
        if result:
            all_results.append(result)

    # Write summary
    import pandas as pd
    summary_df = pd.DataFrame(all_results)
    summary_path = output_dir / 'ablation_summary.csv'
    summary_df.to_csv(summary_path, index=False)

    # Pretty-print table
    print(f"\n{'='*90}")
    print("  ABLATION SUMMARY")
    print(f"{'='*90}")
    print(f"{'Condition':<22} {'AnyOvlp':>8} {'PairOvlp':>9} {'Exact':>6} {'Prec':>6} {'AnyOvlp_T':>10} {'Time':>6}")
    print("-" * 90)
    for r in all_results:
        print(f"{r['condition']:<22} {r['any_overlap']:>7.1%} {r['pair_overlap']:>8.1%} "
              f"{r['exact']:>5.1%} {r['precision']:>5.1%} {r['any_overlap_testable']:>9.1%} "
              f"{r['elapsed_s']:>5.0f}s")
    print(f"\nSaved to {summary_path}")

    # Also save JSON for paper integration
    json_path = output_dir / 'ablation_summary.json'
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)


if __name__ == '__main__':
    main()
