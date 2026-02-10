#!/usr/bin/env python3
"""
Optimized degree-preserving network null test for STAT3 enrichment.

Key optimization: pre-computes co-essentiality, statistical, and perturbation
modules ONCE (they don't use OmniPath), then only re-runs OmniPath signaling
path inference for each shuffle. ~3-4x faster than the naive approach.
"""
import sys
import time
import random
import logging
import json
import numpy as np
from pathlib import Path
from collections import Counter

logging.basicConfig(level=logging.WARNING)

N_PERMUTATIONS = 20
TEST_CANCERS = [
    'Non-Small Cell Lung Cancer',
    'Melanoma',
    'Colorectal Adenocarcinoma',
    'Pancreatic Adenocarcinoma',
    'Invasive Breast Carcinoma',
]


def degree_preserving_shuffle(edges, rng, n_swaps=None):
    """Degree-preserving edge randomization on a directed graph."""
    edge_list = list(edges)
    if n_swaps is None:
        n_swaps = len(edge_list) * 10
    edge_set = set(edge_list)
    for _ in range(n_swaps):
        if len(edge_list) < 2:
            break
        i, j = rng.sample(range(len(edge_list)), 2)
        u, v = edge_list[i]
        x, y = edge_list[j]
        if u == y or x == v:
            continue
        if (u, y) in edge_set or (x, v) in edge_set:
            continue
        edge_set.discard((u, v))
        edge_set.discard((x, y))
        edge_set.add((u, y))
        edge_set.add((x, v))
        edge_list[i] = (u, y)
        edge_list[j] = (x, v)
    return edge_list


def run_null_test():
    import pandas as pd
    from pan_cancer_xnode import (
        DepMapLoader, OmniPathLoader, ViabilityPathInference,
        MinimalHittingSetSolver, CostFunction, DrugTargetDB
    )

    output_dir = Path('ablation_results')
    output_dir.mkdir(exist_ok=True)

    t0 = time.time()
    print("Loading data...")
    depmap = DepMapLoader("./depmap_data")
    omnipath = OmniPathLoader("./depmap_data")
    drug_db = DrugTargetDB()

    # Get original network edges
    network_df = omnipath.load_signaling_network()
    original_edges = list(zip(network_df['source'], network_df['target']))
    original_edge_types = dict(zip(
        zip(network_df['source'], network_df['target']),
        zip(network_df['interaction_type'], network_df['database'])
    ))
    print(f"Network: {len(original_edges)} edges, "
          f"{len(set(s for s, _ in original_edges) | set(t for _, t in original_edges))} nodes")

    # ---- PHASE 1: Pre-compute non-OmniPath modules for all test cancers ----
    print("\n--- Pre-computing non-OmniPath modules (once) ---")
    # Create a path inferrer with OmniPath DISABLED
    precomp_inf = ViabilityPathInference(depmap, omnipath,
                                          disable_omnipath=True)
    cached_non_omnipath = {}
    for ct in TEST_CANCERS:
        print(f"  Pre-computing modules for {ct}...")
        paths = precomp_inf.infer_all_paths(ct)
        cached_non_omnipath[ct] = paths
        print(f"    -> {len(paths)} non-OmniPath mechanisms")

    # ---- PHASE 2: Observed (real network) ----
    print("\n--- Observed (real network) ---")
    cost_fn = CostFunction(depmap, drug_db)
    solver = MinimalHittingSetSolver(cost_fn)
    # Create inferrer with ONLY OmniPath (to get signaling paths)
    omni_only_inf = ViabilityPathInference(depmap, omnipath,
                                            disable_coessentiality=True,
                                            disable_statistical=True,
                                            disable_perturbation=True)
    observed_targets = Counter()
    for ct in TEST_CANCERS:
        # Combine pre-computed modules + real OmniPath paths
        omni_paths = omni_only_inf.infer_all_paths(ct)
        all_paths = cached_non_omnipath[ct] + omni_paths
        if all_paths:
            hits = solver.solve(all_paths, ct, max_size=4)
            if hits:
                for g in hits[0].targets:
                    observed_targets[g] += 1
                print(f"  {ct}: {hits[0].targets}")

    observed_stat3_freq = observed_targets.get('STAT3', 0)
    print(f"\nObserved STAT3 frequency: {observed_stat3_freq}/{len(TEST_CANCERS)}")

    # ---- PHASE 3: Null distribution (shuffle OmniPath only) ----
    print(f"\n--- Null distribution ({N_PERMUTATIONS} permutations) ---")
    null_stat3_freqs = []
    null_all_targets = Counter()
    rng = random.Random(42)

    for perm_i in range(N_PERMUTATIONS):
        perm_start = time.time()
        if (perm_i + 1) % 5 == 0 or perm_i == 0:
            elapsed = time.time() - t0
            print(f"  Permutation {perm_i + 1}/{N_PERMUTATIONS} "
                  f"(elapsed: {elapsed/60:.1f} min)...", flush=True)

        # Shuffle edges preserving degree
        shuffled_edges = degree_preserving_shuffle(original_edges, rng)

        # Build shuffled OmniPath loader
        shuffled_data = []
        for (s, t) in shuffled_edges:
            orig_key = (s, t)
            if orig_key in original_edge_types:
                itype, db = original_edge_types[orig_key]
            else:
                itype, db = 'activation', 'shuffled'
            shuffled_data.append({'source': s, 'target': t,
                                  'interaction_type': itype, 'database': db})

        null_omnipath = OmniPathLoader("./depmap_data")
        null_omnipath._network_df = pd.DataFrame(shuffled_data)

        # Only compute OmniPath signaling paths with shuffled network
        null_omni_inf = ViabilityPathInference(
            depmap, null_omnipath,
            disable_coessentiality=True,
            disable_statistical=True,
            disable_perturbation=True
        )

        perm_targets = Counter()
        for ct in TEST_CANCERS:
            try:
                omni_paths = null_omni_inf.infer_all_paths(ct)
                # Combine with PRE-COMPUTED non-OmniPath modules
                all_paths = cached_non_omnipath[ct] + omni_paths
                if all_paths:
                    null_cost_fn = CostFunction(depmap, drug_db)
                    null_solver = MinimalHittingSetSolver(null_cost_fn)
                    hits = null_solver.solve(all_paths, ct, max_size=4)
                    if hits:
                        for g in hits[0].targets:
                            perm_targets[g] += 1
            except Exception:
                pass

        null_stat3_freqs.append(perm_targets.get('STAT3', 0))
        for g, c in perm_targets.items():
            null_all_targets[g] += c

    # Compute p-value
    null_stat3_freqs = np.array(null_stat3_freqs)
    p_value = float(np.mean(null_stat3_freqs >= observed_stat3_freq))
    mean_null = float(np.mean(null_stat3_freqs))
    std_null = float(np.std(null_stat3_freqs))

    total_time = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"  STAT3 Degree-Preserving Null Results")
    print(f"{'=' * 60}")
    print(f"  Observed STAT3 freq:  {observed_stat3_freq}/{len(TEST_CANCERS)}")
    print(f"  Null mean +/- std:    {mean_null:.2f} +/- {std_null:.2f}")
    print(f"  Null range:           [{null_stat3_freqs.min()}, {null_stat3_freqs.max()}]")
    print(f"  p-value (one-sided):  {p_value:.4f}")
    print(f"  Top null genes:       {null_all_targets.most_common(10)}")
    print(f"  Total time:           {total_time/60:.1f} min")

    # Save results
    result = {
        'n_permutations': N_PERMUTATIONS,
        'test_cancers': TEST_CANCERS,
        'observed_stat3_freq': int(observed_stat3_freq),
        'null_mean': float(mean_null),
        'null_std': float(std_null),
        'null_min': int(null_stat3_freqs.min()),
        'null_max': int(null_stat3_freqs.max()),
        'p_value': float(p_value),
        'observed_all_targets': dict(observed_targets),
        'null_stat3_distribution': null_stat3_freqs.tolist(),
        'top_null_genes': dict(null_all_targets.most_common(20)),
        'elapsed_minutes': round(total_time / 60, 1),
    }

    json_path = output_dir / 'degree_null_results.json'
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {json_path}")


if __name__ == '__main__':
    run_null_test()
