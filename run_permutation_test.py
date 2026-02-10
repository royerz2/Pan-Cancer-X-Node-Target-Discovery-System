#!/usr/bin/env python3
"""
Permutation test for ALIN benchmark significance.

Protocol: For each of 1,000 iterations, within each cancer type's candidate
pool (selective-essential genes), randomly sample 3 genes to form a null
triple. Compute gold-standard concordance under this null. Report the
empirical p-value: fraction of null iterations achieving >= observed
concordance.

This replaces the binomial test (which assumes independence across cancer
types — violated because cancers share gene pools and DepMap cell lines).
"""
import sys
import time
import random
import logging
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

logging.basicConfig(level=logging.WARNING)

N_PERMUTATIONS = 1000


def load_candidate_pools():
    """
    Load per-cancer candidate gene pools (selective-essential genes).
    These are the genes from which the pipeline selects its triples.
    """
    from pan_cancer_xnode import DepMapLoader, ViabilityPathInference, OmniPathLoader
    
    depmap = DepMapLoader("./depmap_data")
    omnipath = OmniPathLoader("./depmap_data")
    path_inf = ViabilityPathInference(depmap, omnipath)
    crispr = depmap.load_crispr_dependencies()
    
    from gold_standard import CANCER_ALIASES
    pipeline_cancers = sorted({
        pipeline_name
        for aliases in CANCER_ALIASES.values()
        for pipeline_name in aliases
        if pipeline_name
    })
    
    pools = {}
    for ct in pipeline_cancers:
        cell_lines = depmap.get_cell_lines_for_cancer(ct)
        avail = [cl for cl in cell_lines if cl in crispr.index]
        if len(avail) == 0:
            continue
        subset = crispr.loc[avail]
        mean_dep = subset.mean(axis=0)
        # Selective-essential: mean Chronos < -0.5 in this cancer
        essential = set(mean_dep[mean_dep < -0.5].index)
        if len(essential) >= 3:
            pools[ct] = list(essential)
    
    return pools


def compute_null_concordance(predictions_df, gold_entries, candidate_pools, rng):
    """
    Generate one null iteration: for each cancer type, randomly sample
    3 genes from its candidate pool.
    """
    from gold_standard import check_match, _resolve_pipeline_cancers, is_testable
    
    # Build null predictions: replace each cancer's targets with random draws
    null_rows = []
    for _, row in predictions_df.iterrows():
        ct = row['Cancer_Type']
        if ct in candidate_pools and len(candidate_pools[ct]) >= 3:
            sampled = rng.sample(candidate_pools[ct], 3)
        else:
            sampled = [row['Target_1'], row['Target_2'], row['Target_3']]
        null_rows.append({
            'Cancer_Type': ct,
            'Target_1': sampled[0],
            'Target_2': sampled[1],
            'Target_3': sampled[2],
        })
    
    null_df = pd.DataFrame(null_rows)
    
    # Evaluate against gold standard
    n_any = 0
    n_pair = 0
    n_exact = 0
    
    for entry in gold_entries:
        gold_targets = entry['targets']
        pipeline_cancers = _resolve_pipeline_cancers(entry['cancer'])
        
        best_match = 'none'
        match_priority = {'exact': 4, 'superset': 3, 'pair_overlap': 2, 'any_overlap': 1, 'none': 0}
        
        for _, row in null_df.iterrows():
            if row['Cancer_Type'] in pipeline_cancers:
                pred = frozenset({row['Target_1'], row['Target_2'], row['Target_3']})
                mt = check_match(pred, gold_targets)
                if match_priority[mt] > match_priority[best_match]:
                    best_match = mt
        
        if best_match in ('exact', 'superset', 'pair_overlap', 'any_overlap'):
            n_any += 1
        if best_match in ('exact', 'superset', 'pair_overlap'):
            n_pair += 1
        if best_match == 'exact':
            n_exact += 1
    
    n = len(gold_entries)
    return {
        'any_overlap': n_any / n if n > 0 else 0,
        'pair_overlap': n_pair / n if n > 0 else 0,
        'exact': n_exact / n if n > 0 else 0,
    }


def main():
    output_dir = Path('ablation_results')
    output_dir.mkdir(exist_ok=True)
    
    # Load actual predictions
    pred_path = 'results/triple_combinations.csv'
    if not Path(pred_path).exists():
        print(f"ERROR: {pred_path} not found. Run run_pipeline.py first.")
        sys.exit(1)
    
    pred_df = pd.read_csv(pred_path)
    pred_df.columns = [c.replace(' ', '_') for c in pred_df.columns]
    
    from gold_standard import GOLD_STANDARD, run_benchmark
    
    # Run observed benchmark
    print("Running observed benchmark...")
    observed = run_benchmark(pred_path, verbose=False)
    obs_any = observed['recall']['any_overlap']
    obs_pair = observed['recall']['pair_overlap']
    obs_exact = observed['recall']['exact']
    
    print(f"  Observed: any_overlap={obs_any:.1%}, pair_overlap={obs_pair:.1%}, exact={obs_exact:.1%}")
    
    # Load candidate pools
    print("Loading per-cancer candidate pools...")
    pools = load_candidate_pools()
    print(f"  {len(pools)} cancer types with candidate pools")
    pool_sizes = [len(v) for v in pools.values()]
    print(f"  Pool sizes: median={np.median(pool_sizes):.0f}, "
          f"range=[{min(pool_sizes)}, {max(pool_sizes)}]")
    
    # Run permutation test
    print(f"\nRunning {N_PERMUTATIONS} permutations...")
    rng = random.Random(42)
    
    null_any = []
    null_pair = []
    null_exact = []
    
    t0 = time.time()
    for i in range(N_PERMUTATIONS):
        if (i + 1) % 100 == 0 or i == 0:
            print(f"  Permutation {i + 1}/{N_PERMUTATIONS}...", flush=True)
        
        result = compute_null_concordance(pred_df, GOLD_STANDARD, pools, rng)
        null_any.append(result['any_overlap'])
        null_pair.append(result['pair_overlap'])
        null_exact.append(result['exact'])
    
    elapsed = time.time() - t0
    
    null_any = np.array(null_any)
    null_pair = np.array(null_pair)
    null_exact = np.array(null_exact)
    
    # Compute p-values (one-sided: P(null >= observed))
    p_any = np.mean(null_any >= obs_any)
    p_pair = np.mean(null_pair >= obs_pair)
    p_exact = np.mean(null_exact >= obs_exact)
    
    print(f"\n{'='*70}")
    print(f"  PERMUTATION TEST RESULTS ({N_PERMUTATIONS} iterations, {elapsed:.1f}s)")
    print(f"{'='*70}")
    print(f"  {'Metric':<20} {'Observed':>10} {'Null mean':>10} {'Null std':>10} {'p-value':>10}")
    print(f"  {'-'*60}")
    print(f"  {'Any-overlap':<20} {obs_any:>9.1%} {null_any.mean():>9.1%} {null_any.std():>9.1%} {p_any:>9.4f}")
    print(f"  {'Pair-overlap':<20} {obs_pair:>9.1%} {null_pair.mean():>9.1%} {null_pair.std():>9.1%} {p_pair:>9.4f}")
    print(f"  {'Exact':<20} {obs_exact:>9.1%} {null_exact.mean():>9.1%} {null_exact.std():>9.1%} {p_exact:>9.4f}")
    
    result = {
        'n_permutations': N_PERMUTATIONS,
        'n_cancers_with_pools': len(pools),
        'median_pool_size': float(np.median(pool_sizes)),
        'observed': {'any_overlap': obs_any, 'pair_overlap': obs_pair, 'exact': obs_exact},
        'null_mean': {'any_overlap': float(null_any.mean()), 'pair_overlap': float(null_pair.mean()),
                      'exact': float(null_exact.mean())},
        'null_std': {'any_overlap': float(null_any.std()), 'pair_overlap': float(null_pair.std()),
                     'exact': float(null_exact.std())},
        'p_values': {'any_overlap': float(p_any), 'pair_overlap': float(p_pair),
                     'exact': float(p_exact)},
        'null_distributions': {
            'any_overlap': null_any.tolist(),
            'pair_overlap': null_pair.tolist(),
            'exact': null_exact.tolist(),
        },
    }
    
    json_path = output_dir / 'permutation_test_results.json'
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {json_path}")


if __name__ == '__main__':
    main()
