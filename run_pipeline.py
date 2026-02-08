#!/usr/bin/env python3
"""
Run ALIN pipeline for gold standard cancer types.
Produces predictions for benchmarking against curated combination therapies.

Saves predictions to results/triple_combinations.csv

Flags:
  --no-known-synergies  Disable KNOWN_SYNERGIES lookup in synergy scoring.
                        Saves to results/triple_combinations_no_ks.csv
"""
import sys
import time
import logging
import argparse
from pathlib import Path

logging.basicConfig(level=logging.WARNING)

from gold_standard import CANCER_ALIASES

# Derive gold standard cancers: pipeline cancer types that map to gold standard entries
GOLD_STANDARD_CANCERS = sorted({
    pipeline_name
    for aliases in CANCER_ALIASES.values()
    for pipeline_name in aliases
    if pipeline_name  # skip empty strings
})

def main():
    parser = argparse.ArgumentParser(description='Run ALIN pipeline for gold standard cancers')
    parser.add_argument('--no-known-synergies', action='store_true',
                        help='Disable KNOWN_SYNERGIES in synergy scoring')
    args = parser.parse_args()

    from pan_cancer_xnode import PanCancerXNodeAnalyzer, generate_triple_summary_table

    use_ks = not args.no_known_synergies
    mode_label = 'WITHOUT KNOWN_SYNERGIES' if not use_ks else 'with KNOWN_SYNERGIES'
    print(f"Initializing ALIN pipeline ({mode_label})...", flush=True)
    t0 = time.time()
    analyzer = PanCancerXNodeAnalyzer(use_known_synergies=use_ks)
    t_init = time.time() - t0
    print(f"Initialization: {t_init:.1f}s", flush=True)

    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)

    results = {}
    for i, ct in enumerate(GOLD_STANDARD_CANCERS, 1):
        print(f"\n[{i}/{len(GOLD_STANDARD_CANCERS)}] {ct}...", flush=True)
        t1 = time.time()
        try:
            analysis = analyzer.analyze_cancer_type(ct)
            results[ct] = analysis
            if analysis.best_triple:
                targets = sorted(analysis.best_triple.targets)
                print(f"  -> Best triple: {' + '.join(targets)} (score={analysis.best_triple.combined_score:.3f})",
                      flush=True)
            else:
                print("  -> No triple found", flush=True)
            if analysis.best_combination and analysis.best_combination != analysis.best_triple:
                bc = analysis.best_combination
                bc_targets = sorted(bc.targets)
                print(f"  -> Best combo ({len(bc_targets)}): {' + '.join(bc_targets)} (score={bc.combined_score:.3f})",
                      flush=True)
        except Exception as e:
            print(f"  -> ERROR: {e}", flush=True)
        t2 = time.time()
        print(f"  [{t2-t1:.1f}s]", flush=True)

    # Generate output CSV
    if results:
        df = generate_triple_summary_table(results)
        csv_name = 'triple_combinations_no_ks.csv' if not use_ks else 'triple_combinations.csv'
        csv_path = output_dir / csv_name
        df.to_csv(csv_path, index=False)
        print(f"\n{'='*70}")
        print(f"Saved {len(df)} predictions to {csv_path}")
        print(f"{'='*70}")

    print(f"\nTotal time: {time.time()-t0:.1f}s")

if __name__ == '__main__':
    main()
