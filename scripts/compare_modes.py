#!/usr/bin/env python3
"""
Compare Actionable vs Exploratory pipeline results.

Reads two result directories and produces a side-by-side comparison
suitable for a paper figure / supplementary table.

Usage:
    python scripts/compare_modes.py [--actionable results/] [--exploratory results_exploratory/]
"""
import json
import argparse
from pathlib import Path
from collections import Counter


def load_results(path: Path):
    """Load all_findings.json from a results directory."""
    findings = path / "all_findings.json"
    if not findings.exists():
        raise FileNotFoundError(f"No all_findings.json in {path}")
    with open(findings) as f:
        data = json.load(f)
    return data.get("results", data)


def summarise(results: dict, label: str):
    """Compute summary stats from a results dict."""
    combos = set()
    targets = set()
    target_freq = Counter()
    combo_usage = Counter()
    drugged = 0
    undrugged = 0
    synergy_vals = []
    coverage_vals = []

    for cancer, info in results.items():
        best = info.get("best_triple_combination") or {}
        tgt = best.get("targets", [])
        if not tgt:
            continue
        key = tuple(sorted(tgt))
        combos.add(key)
        combo_usage[key] += 1
        for t in tgt:
            targets.add(t)
            target_freq[t] += 1
        d_count = best.get("druggable_count", 0)
        if d_count == len(tgt):
            drugged += 1
        elif d_count < len(tgt):
            undrugged += 1
        synergy_vals.append(best.get("synergy_score", 0))
        coverage_vals.append(best.get("coverage", 0) or best.get("path_coverage", 0))

    max_reuse = max(combo_usage.values()) if combo_usage else 0
    mean_syn = sum(synergy_vals) / len(synergy_vals) if synergy_vals else 0
    mean_cov = sum(coverage_vals) / len(coverage_vals) if coverage_vals else 0

    return {
        "label": label,
        "cancers": len(results),
        "unique_combos": len(combos),
        "unique_targets": len(targets),
        "max_combo_reuse": max_reuse,
        "fully_drugged_combos": drugged,
        "has_undrugged_target": undrugged,
        "mean_synergy": mean_syn,
        "mean_coverage": mean_cov,
        "target_freq": target_freq,
        "combo_usage": combo_usage,
    }


def print_comparison(act, exp):
    w = 40
    print("=" * 70)
    print("ACTIONABLE vs EXPLORATORY — Side-by-Side Comparison")
    print("=" * 70)
    print(f"{'Metric':<{w}} {'Actionable':>12} {'Exploratory':>12}")
    print("-" * 70)
    metrics = [
        ("Cancer types", "cancers"),
        ("Unique #1 combos", "unique_combos"),
        ("Unique targets", "unique_targets"),
        ("Max combo reuse", "max_combo_reuse"),
        ("All-drugged combos", "fully_drugged_combos"),
        ("Has undrugged target", "has_undrugged_target"),
        ("Mean synergy", "mean_synergy"),
        ("Mean path coverage", "mean_coverage"),
    ]
    for label, key in metrics:
        a_val = act[key]
        e_val = exp[key]
        if isinstance(a_val, float):
            print(f"{label:<{w}} {a_val:>12.3f} {e_val:>12.3f}")
        else:
            print(f"{label:<{w}} {a_val:>12} {e_val:>12}")

    # Targets only in one mode
    act_targets = set(act["target_freq"].keys())
    exp_targets = set(exp["target_freq"].keys())
    only_act = act_targets - exp_targets
    only_exp = exp_targets - act_targets
    shared = act_targets & exp_targets

    print()
    print(f"Shared targets:            {len(shared)}")
    print(f"Only in Actionable:        {len(only_act)}: {sorted(only_act)}")
    print(f"Only in Exploratory:       {len(only_exp)}: {sorted(only_exp)}")

    # Top targets per mode
    print()
    print("Top 15 targets — Actionable:")
    for t, f in act["target_freq"].most_common(15):
        marker = "  *NEW*" if t in only_act else ""
        print(f"  {t:15s}: {f:3d}/{act['cancers']}{marker}")

    print()
    print("Top 15 targets — Exploratory:")
    for t, f in exp["target_freq"].most_common(15):
        marker = "  *NEW*" if t in only_exp else ""
        print(f"  {t:15s}: {f:3d}/{exp['cancers']}{marker}")

    # Cancer types where the two modes disagree
    # (Need the raw results for this)
    return only_act, only_exp, shared


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--actionable", type=str, default="results")
    parser.add_argument("--exploratory", type=str, default="results_exploratory")
    args = parser.parse_args()

    act_path = Path(args.actionable)
    exp_path = Path(args.exploratory)

    act_results = load_results(act_path)
    exp_results = load_results(exp_path)

    act_summary = summarise(act_results, "Actionable")
    exp_summary = summarise(exp_results, "Exploratory")

    print_comparison(act_summary, exp_summary)

    # Per-cancer comparison
    shared_cancers = set(act_results.keys()) & set(exp_results.keys())
    changed = 0
    print()
    print("=" * 70)
    print("Per-cancer #1 combo differences (first 30)")
    print("=" * 70)
    diffs = []
    for cancer in sorted(shared_cancers):
        act_best = (act_results[cancer].get("best_triple_combination") or {})
        exp_best = (exp_results[cancer].get("best_triple_combination") or {})
        act_t = sorted(act_best.get("targets", []))
        exp_t = sorted(exp_best.get("targets", []))
        if act_t != exp_t:
            changed += 1
            diffs.append((cancer, act_t, exp_t))
    
    for cancer, at, et in diffs[:30]:
        print(f"  {cancer:42s}  {'+'.join(at) or '(none)':30s} -> {'+'.join(et) or '(none)'}")
    
    print(f"\nTotal changed: {changed}/{len(shared_cancers)} ({100*changed/max(len(shared_cancers),1):.0f}%)")

    # Save comparison JSON
    out = Path(args.actionable) / "mode_comparison.json"
    with open(out, "w") as f:
        json.dump({
            "actionable": {k: v for k, v in act_summary.items() if k not in ("target_freq", "combo_usage")},
            "exploratory": {k: v for k, v in exp_summary.items() if k not in ("target_freq", "combo_usage")},
            "only_actionable_targets": sorted(set(act_summary["target_freq"]) - set(exp_summary["target_freq"])),
            "only_exploratory_targets": sorted(set(exp_summary["target_freq"]) - set(act_summary["target_freq"])),
            "cancers_changed": changed,
            "cancers_shared": len(shared_cancers),
        }, f, indent=2)
    print(f"\nComparison saved to {out}")


if __name__ == "__main__":
    main()
