#!/usr/bin/env python3
"""
Sensitivity analysis for exploratory-mode biology reward weights.

Sweeps W_ESSENTIALITY, W_MUTATION, W_CENTRALITY on a small number of
cancer types and measures:
  - target diversity (unique targets across cancers)
  - combo diversity (unique #1 combos across cancers)
  - biology score (mean essentiality + mutation burden + pathway coverage)
  - drug-novelty (fraction of #1 combos containing at least one undrugged target)

The sweep helps determine the optimal biology reward weights for the
exploratory config.

Usage:
    python scripts/sensitivity_exploratory.py [--cancers 10] [--output results/sensitivity/]
"""
import sys
import json
import time
import argparse
import itertools
from pathlib import Path
from collections import Counter
from dataclasses import asdict
from datetime import datetime

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from alin.run_modes import ModeConfig, RunMode, exploratory_config


def build_sweep_configs():
    """Generate parameter combinations to sweep."""
    # Sweep ranges for biology reward weights
    w_ess_values = [0.0, 0.06, 0.12, 0.18, 0.24]
    w_mut_values = [0.0, 0.05, 0.10, 0.15, 0.20]
    w_cent_values = [0.0, 0.04, 0.08, 0.12]

    configs = []
    for w_ess, w_mut, w_cent in itertools.product(w_ess_values, w_mut_values, w_cent_values):
        # Skip the "all zero" case (that's just actionable without drug bonus)
        if w_ess == 0.0 and w_mut == 0.0 and w_cent == 0.0:
            continue
        cfg = exploratory_config()
        cfg.W_ESSENTIALITY = w_ess
        cfg.W_MUTATION = w_mut
        cfg.W_CENTRALITY = w_cent
        configs.append(cfg)
    return configs


def evaluate_config(cfg: ModeConfig, cancer_types: list, analyzer_kwargs: dict):
    """Run the pipeline with a given config on a subset of cancer types."""
    from pan_cancer_xnode import PanCancerXNodeAnalyzer

    analyzer = PanCancerXNodeAnalyzer(
        mode_config=cfg,
        **analyzer_kwargs,
    )

    all_targets = set()
    all_combos = set()
    essentiality_scores = []
    mutation_scores = []
    coverage_scores = []
    undrugged_combos = 0
    total_combos = 0

    for ct in cancer_types:
        try:
            analysis = analyzer.analyze_cancer_type(ct)
            bt = analysis.best_triple
            if bt and bt.targets:
                targets = tuple(sorted(bt.targets))
                all_combos.add(targets)
                for t in targets:
                    all_targets.add(t)
                essentiality_scores.append(bt.synergy_score)
                coverage_scores.append(bt.coverage)
                total_combos += 1
                # Check if any target is undrugged
                has_undrugged = any(
                    analyzer.drug_db.get_druggability_score(t, cancer_type=ct) < 0.3
                    for t in targets
                )
                if has_undrugged:
                    undrugged_combos += 1
        except Exception as e:
            print(f"  [SKIP] {ct}: {e}", flush=True)
            continue

    return {
        "W_ESSENTIALITY": cfg.W_ESSENTIALITY,
        "W_MUTATION": cfg.W_MUTATION,
        "W_CENTRALITY": cfg.W_CENTRALITY,
        "unique_targets": len(all_targets),
        "unique_combos": len(all_combos),
        "mean_coverage": float(np.mean(coverage_scores)) if coverage_scores else 0.0,
        "drug_novelty": undrugged_combos / max(total_combos, 1),
        "cancers_analyzed": total_combos,
        "targets": sorted(all_targets),
    }


def main():
    parser = argparse.ArgumentParser(description="Sensitivity analysis for exploratory mode")
    parser.add_argument("--cancers", type=int, default=10,
                        help="Number of cancer types to test (default: 10)")
    parser.add_argument("--output", type=str, default="results/sensitivity",
                        help="Output directory")
    parser.add_argument("--data-dir", type=str, default="./depmap_data")
    parser.add_argument("--max-configs", type=int, default=20,
                        help="Max configs to sweep (randomly sampled if more)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    analyzer_kwargs = {
        "data_dir": args.data_dir,
        "enable_api": False,
    }

    # Pick representative cancer types (mix of common & rare)
    from pan_cancer_xnode import DepMapLoader
    depmap = DepMapLoader(args.data_dir)
    all_types = depmap.get_available_cancer_types()
    # Sort by cell line count, pick evenly from top
    selected = [ct for ct, _ in all_types[:args.cancers]]
    print(f"Sweeping on {len(selected)} cancer types: {selected}", flush=True)

    # Build sweep configs
    all_configs = build_sweep_configs()
    if len(all_configs) > args.max_configs:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(all_configs), size=args.max_configs, replace=False)
        all_configs = [all_configs[i] for i in sorted(indices)]
    print(f"Testing {len(all_configs)} parameter configurations", flush=True)

    # Also run with the default actionable config as baseline
    from alin.run_modes import actionable_config
    act_cfg = actionable_config()

    results = []

    # Baseline: actionable
    print("\n[Baseline] Running actionable mode...", flush=True)
    t0 = time.time()
    baseline = evaluate_config(act_cfg, selected, analyzer_kwargs)
    baseline["mode"] = "actionable"
    baseline["time_sec"] = time.time() - t0
    results.append(baseline)
    print(f"  Actionable: {baseline['unique_combos']} unique combos, "
          f"{baseline['unique_targets']} targets, "
          f"novelty={baseline['drug_novelty']:.2f}", flush=True)

    # Sweep exploratory configs
    for i, cfg in enumerate(all_configs, 1):
        label = (f"W_ESS={cfg.W_ESSENTIALITY:.2f}, "
                 f"W_MUT={cfg.W_MUTATION:.2f}, "
                 f"W_CENT={cfg.W_CENTRALITY:.2f}")
        print(f"\n[{i}/{len(all_configs)}] {label}", flush=True)
        t0 = time.time()
        result = evaluate_config(cfg, selected, analyzer_kwargs)
        result["mode"] = "exploratory"
        result["time_sec"] = time.time() - t0
        results.append(result)
        print(f"  -> {result['unique_combos']} combos, "
              f"{result['unique_targets']} targets, "
              f"novelty={result['drug_novelty']:.2f}, "
              f"coverage={result['mean_coverage']:.2f}", flush=True)

    # Find best config by composite metric
    # Maximise: unique_combos + unique_targets + drug_novelty + coverage
    for r in results:
        r["composite"] = (
            r["unique_combos"] * 2 +
            r["unique_targets"] +
            r["drug_novelty"] * 10 +
            r["mean_coverage"] * 5
        )

    results.sort(key=lambda x: -x["composite"])

    print("\n" + "=" * 70)
    print("SENSITIVITY ANALYSIS RESULTS (sorted by composite score)")
    print("=" * 70)
    print(f"{'Rank':>4} {'Mode':>11} {'W_ESS':>5} {'W_MUT':>5} {'W_CENT':>6} "
          f"{'Combos':>6} {'Targets':>7} {'Novelty':>7} {'Coverage':>8} {'Score':>6}")
    for i, r in enumerate(results[:25], 1):
        print(f"{i:4d} {r['mode']:>11} "
              f"{r.get('W_ESSENTIALITY', '-'):>5} "
              f"{r.get('W_MUTATION', '-'):>5} "
              f"{r.get('W_CENTRALITY', '-'):>6} "
              f"{r['unique_combos']:6d} {r['unique_targets']:7d} "
              f"{r['drug_novelty']:7.2f} {r['mean_coverage']:8.2f} "
              f"{r['composite']:6.1f}")

    # Save results
    out_file = output_dir / "sensitivity_sweep.json"
    with open(out_file, "w") as f:
        json.dump({
            "date": datetime.now().isoformat(),
            "cancer_types": selected,
            "n_configs": len(results),
            "results": results,
        }, f, indent=2, default=str)
    print(f"\nResults saved to {out_file}")

    # Save best exploratory config
    best_exploratory = next(r for r in results if r["mode"] == "exploratory")
    best_file = output_dir / "best_exploratory_config.json"
    with open(best_file, "w") as f:
        json.dump({
            "W_ESSENTIALITY": best_exploratory["W_ESSENTIALITY"],
            "W_MUTATION": best_exploratory["W_MUTATION"],
            "W_CENTRALITY": best_exploratory["W_CENTRALITY"],
            "composite_score": best_exploratory["composite"],
            "unique_targets": best_exploratory["unique_targets"],
            "unique_combos": best_exploratory["unique_combos"],
            "drug_novelty": best_exploratory["drug_novelty"],
        }, f, indent=2)
    print(f"Best exploratory config: {best_file}")


if __name__ == "__main__":
    main()
