#!/usr/bin/env python3
"""
Post-Pipeline Validation Suite
===============================
Run immediately after ``pan_cancer_xnode.py --all-cancers`` finishes.

Checks the three known red flags from v1:
  1. CDK6 dominance (appearing in too many cancer types)
  2. Binary/near-binary synergy scores
  3. Quantized resistance scores (handful of discrete values)

Also produces:
  - Full score-component distribution statistics
  - Target diversity metrics
  - LINCS perturbation-bonus impact analysis
    - Optional legacy result comparison (if a legacy result directory is provided)
  - Per-cancer quality flags

Usage:
    python scripts/post_pipeline_validation.py [--results-dir results/]
                                                [--v1-dir path/to/legacy_results]
                                                [--output outputs/reports/validation_reports/validation_report.json]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ───────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────

def _load_triples(path: str) -> pd.DataFrame:
    """Load triple_combinations.csv with safe column normalization."""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def _load_findings(path: str) -> Dict[str, Any]:
    """Load all_findings.json."""
    with open(path) as f:
        return json.load(f)


def _pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%"


def _stat_summary(values: List[float], label: str) -> Dict[str, Any]:
    """Compute min/max/mean/median/std/n_unique for a list of floats."""
    if not values:
        return {"label": label, "n": 0}
    arr = np.array(values)
    unique_vals = np.unique(np.round(arr, 6))
    return {
        "label": label,
        "n": len(values),
        "min": round(float(np.min(arr)), 6),
        "max": round(float(np.max(arr)), 6),
        "mean": round(float(np.mean(arr)), 6),
        "median": round(float(np.median(arr)), 6),
        "std": round(float(np.std(arr, ddof=1)), 6) if len(values) > 1 else 0.0,
        "n_unique": int(len(unique_vals)),
        "unique_ratio": round(len(unique_vals) / len(values), 4),
    }


# ───────────────────────────────────────────────────────────────────────
# 1. CDK6 Dominance Check
# ───────────────────────────────────────────────────────────────────────

def check_target_dominance(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Flag genes that appear in an unusually large fraction of cancer types.

    Healthy range: no single gene in > 50% of cancer types.
    """
    n_cancers = df["Cancer_Type"].nunique()
    target_cols = [c for c in df.columns if c.startswith("Target_")]
    if not target_cols:
        # Might be alt format
        target_cols = ["Target_1", "Target_2", "Target_3"]
        target_cols = [c for c in target_cols if c in df.columns]

    gene_counts: Counter = Counter()
    for _, row in df.iterrows():
        for col in target_cols:
            gene = row.get(col)
            if pd.notna(gene):
                gene_counts[gene] += 1

    # Per-gene cancer-type count (deduplicated: count each cancer once)
    gene_cancer_sets: Dict[str, set] = {}
    for _, row in df.iterrows():
        ct = row.get("Cancer_Type", "")
        for col in target_cols:
            gene = row.get(col)
            if pd.notna(gene):
                gene_cancer_sets.setdefault(gene, set()).add(ct)

    top_genes = sorted(gene_cancer_sets.items(), key=lambda x: -len(x[1]))[:10]
    dominant = []
    for gene, cancers in top_genes:
        frac = len(cancers) / n_cancers
        dominant.append({
            "gene": gene,
            "n_cancer_types": len(cancers),
            "fraction": round(frac, 4),
        })

    worst = dominant[0] if dominant else {}
    return {
        "n_cancer_types": n_cancers,
        "n_unique_combos": df["Triple_Targets"].nunique() if "Triple_Targets" in df.columns else -1,
        "top_genes": dominant,
        "red_flag": worst.get("fraction", 0) > 0.50,
        "verdict": (
            f"RED FLAG: {worst['gene']} in {_pct(worst['fraction'])} of cancers"
            if worst.get("fraction", 0) > 0.50
            else f"OK: top gene ({worst.get('gene', '?')}) in {_pct(worst.get('fraction', 0))} of cancers"
        ),
    }


# ───────────────────────────────────────────────────────────────────────
# 2. Synergy Distribution Check
# ───────────────────────────────────────────────────────────────────────

def check_synergy_distribution(df: pd.DataFrame, findings: Dict) -> Dict[str, Any]:
    """
    Check whether synergy scores are continuous or clustered.

    Red flag: > 80% of scores are the same value (binary-like).
    """
    # Get synergy from CSV
    syn_csv = df["Synergy_Score"].dropna().tolist() if "Synergy_Score" in df.columns else []

    # Get synergy from all_findings.json (more precise, includes alternatives)
    syn_json = []
    for cancer, data in findings.get("results", {}).items():
        bt = data.get("best_triple_combination")
        if bt is None:
            continue
        s = bt.get("synergy_score")
        if s is not None:
            syn_json.append(float(s))

    synergy_values = syn_json if syn_json else syn_csv
    stats = _stat_summary(synergy_values, "synergy_score")

    # Check for mode dominance
    if synergy_values:
        rounded = [round(v, 2) for v in synergy_values]
        counter = Counter(rounded)
        mode_val, mode_count = counter.most_common(1)[0]
        mode_frac = mode_count / len(synergy_values)
    else:
        mode_val, mode_frac = None, 0

    return {
        **stats,
        "mode_value": mode_val,
        "mode_fraction": round(mode_frac, 4),
        "red_flag": mode_frac > 0.80,
        "verdict": (
            f"RED FLAG: {_pct(mode_frac)} of synergy scores = {mode_val} (binary-like)"
            if mode_frac > 0.80
            else f"OK: mode covers {_pct(mode_frac)} (n_unique={stats.get('n_unique', 0)})"
        ),
    }


# ───────────────────────────────────────────────────────────────────────
# 3. Resistance Distribution Check
# ───────────────────────────────────────────────────────────────────────

def check_resistance_distribution(df: pd.DataFrame, findings: Dict) -> Dict[str, Any]:
    """
    Check whether resistance scores are continuous or quantized.

    Red flag: fewer than 10 unique values across 50+ cancer types.
    """
    res_csv = df["Resistance_Score"].dropna().tolist() if "Resistance_Score" in df.columns else []

    res_json = []
    for cancer, data in findings.get("results", {}).items():
        bt = data.get("best_triple_combination")
        if bt is None:
            continue
        r = bt.get("resistance_score")
        if r is not None:
            res_json.append(float(r))

    resistance_values = res_json if res_json else res_csv
    stats = _stat_summary(resistance_values, "resistance_score")

    n = len(resistance_values)
    n_unique = stats.get("n_unique", 0)

    # Quantization: if < 10 unique values among 50+ cancers, it's quantized
    quantized = n >= 20 and n_unique < 10

    # Also check how many are exactly 0.0
    n_zero = sum(1 for v in resistance_values if abs(v) < 1e-6)
    zero_frac = n_zero / n if n else 0

    return {
        **stats,
        "n_zero": n_zero,
        "zero_fraction": round(zero_frac, 4),
        "red_flag": quantized,
        "verdict": (
            f"RED FLAG: only {n_unique} unique resistance values across {n} cancers (quantized)"
            if quantized
            else f"OK: {n_unique} unique values across {n} cancers"
        ),
    }


# ───────────────────────────────────────────────────────────────────────
# 4. Full Score-Component Distributions
# ───────────────────────────────────────────────────────────────────────

def score_component_distributions(findings: Dict) -> Dict[str, Any]:
    """
    Extract distribution stats for every numeric score component.
    """
    components = {
        "combined_score": [],
        "synergy_score": [],
        "resistance_score": [],
        "path_coverage": [],
        "druggable_count": [],
    }

    for cancer, data in findings.get("results", {}).items():
        bt = data.get("best_triple_combination")
        if bt is None:
            continue
        for key in components:
            v = bt.get(key)
            if v is not None:
                components[key].append(float(v))

    return {k: _stat_summary(v, k) for k, v in components.items()}


# ───────────────────────────────────────────────────────────────────────
# 5. LINCS Perturbation Impact
# ───────────────────────────────────────────────────────────────────────

def check_lincs_impact(findings: Dict) -> Dict[str, Any]:
    """
    Assess whether LINCS data is actually contributing to scores.

    Reads per-cancer reports for perturbation bonus mentions, and checks
    whether there's any variance in LINCS-driven metrics.
    """
    # Check if perturbation_bonus or cancer-relevance weighting is tracked
    # in the all_findings.json
    synergy_vals = []
    resistance_vals = []
    coverage_vals = []

    for cancer, data in findings.get("results", {}).items():
        bt = data.get("best_triple_combination")
        if bt is None:
            continue
        s = bt.get("synergy_score")
        r = bt.get("resistance_score")
        c = bt.get("path_coverage")
        if s is not None:
            synergy_vals.append(float(s))
        if r is not None:
            resistance_vals.append(float(r))
        if c is not None:
            coverage_vals.append(float(c))

    # The LINCS integration should make synergy more varied (not all 0.94)
    # and resistance scores non-zero
    syn_variance = float(np.var(synergy_vals)) if synergy_vals else 0
    res_variance = float(np.var(resistance_vals)) if resistance_vals else 0

    # Count distinct synergy values
    syn_unique = len(set(round(v, 4) for v in synergy_vals))
    res_unique = len(set(round(v, 4) for v in resistance_vals))

    # Check for LINCS coverage by examining combo targets
    lincs_covered = 0
    total_targets = 0
    for cancer, data in findings.get("results", {}).items():
        bt = data.get("best_triple_combination")
        if bt is None:
            continue
        targets = bt.get("targets", [])
        total_targets += len(targets)
        # We can't directly tell from the JSON, but if synergy != 0.94
        # (the curated-only default), LINCS is probably influencing it
        s = bt.get("synergy_score", 0.94)
        if abs(s - 0.94) > 0.01:  # different from default
            lincs_covered += len(targets)

    return {
        "synergy_variance": round(syn_variance, 6),
        "resistance_variance": round(res_variance, 6),
        "synergy_unique_values": syn_unique,
        "resistance_unique_values": res_unique,
        "targets_with_non_default_synergy": lincs_covered,
        "total_targets": total_targets,
        "lincs_influence_estimate": (
            f"{_pct(lincs_covered / total_targets)} of target-slots show non-default synergy"
            if total_targets > 0 else "N/A"
        ),
    }


# ───────────────────────────────────────────────────────────────────────
# 6. Legacy Comparison
# ───────────────────────────────────────────────────────────────────────

def compare_with_v1(v2_df: pd.DataFrame, v1_path: str) -> Optional[Dict[str, Any]]:
    """
    Compare current results with a legacy result bundle when one is provided.
    """
    if not os.path.exists(v1_path):
        return None

    v1_df = pd.read_csv(v1_path)
    v1_df.columns = [c.strip() for c in v1_df.columns]

    # Normalize v1 column names to match v2
    col_map = {
        "Cancer Type": "Cancer_Type",
        "Target 1": "Target_1",
        "Target 2": "Target_2",
        "Target 3": "Target_3",
        "Synergy": "Synergy_Score",
        "Resistance": "Resistance_Score",
    }
    v1_df = v1_df.rename(columns=col_map)

    # Compare cancer types
    v1_cancers = set(v1_df["Cancer_Type"]) if "Cancer_Type" in v1_df.columns else set()
    v2_cancers = set(v2_df["Cancer_Type"]) if "Cancer_Type" in v2_df.columns else set()
    shared = v1_cancers & v2_cancers

    # Compare targets for shared cancer types
    changed_combos = []
    unchanged_combos = []
    for cancer in sorted(shared):
        v1_row = v1_df[v1_df["Cancer_Type"] == cancer].iloc[0] if len(v1_df[v1_df["Cancer_Type"] == cancer]) > 0 else None
        v2_row = v2_df[v2_df["Cancer_Type"] == cancer].iloc[0] if len(v2_df[v2_df["Cancer_Type"] == cancer]) > 0 else None
        if v1_row is None or v2_row is None:
            continue

        v1_targets = sorted([v1_row.get(f"Target_{i}", "") for i in range(1, 4)])
        v2_targets = sorted([v2_row.get(f"Target_{i}", "") for i in range(1, 4)])

        if v1_targets != v2_targets:
            changed_combos.append({
                "cancer": cancer,
                "v1": " + ".join(v1_targets),
                "v2": " + ".join(v2_targets),
            })
        else:
            unchanged_combos.append(cancer)

    # Synergy comparison
    v1_syn = v1_df["Synergy_Score"].dropna().tolist() if "Synergy_Score" in v1_df.columns else []
    v2_syn = v2_df["Synergy_Score"].dropna().tolist() if "Synergy_Score" in v2_df.columns else []

    return {
        "v1_cancer_types": len(v1_cancers),
        "v2_cancer_types": len(v2_cancers),
        "shared_cancer_types": len(shared),
        "combos_changed": len(changed_combos),
        "combos_unchanged": len(unchanged_combos),
        "change_rate": round(len(changed_combos) / max(len(shared), 1), 4),
        "changed_details": changed_combos[:20],  # first 20
        "v1_synergy_stats": _stat_summary(v1_syn, "v1_synergy"),
        "v2_synergy_stats": _stat_summary(v2_syn, "v2_synergy"),
    }


# ───────────────────────────────────────────────────────────────────────
# 7. Per-Cancer Quality Flags
# ───────────────────────────────────────────────────────────────────────

def per_cancer_quality_flags(df: pd.DataFrame, findings: Dict) -> List[Dict[str, Any]]:
    """
    Flag individual cancer types with suspicious results.

    Suspicious conditions:
    - resistance_score = 0.0 (no resistance data)
    - path_coverage < 50% (insufficient coverage despite passing min_coverage)
    - same combo as > 3 other cancers (may indicate undifferentiated scoring)
    """
    flags = []

    # Count combo frequency
    combo_counts: Counter = Counter()
    if "Triple_Targets" in df.columns:
        combo_counts = Counter(df["Triple_Targets"].dropna())

    for _, row in df.iterrows():
        cancer = row.get("Cancer_Type", "?")
        issues = []

        # Resistance = 0
        res = row.get("Resistance_Score")
        if pd.notna(res) and abs(float(res)) < 1e-6:
            issues.append("resistance=0.00 (no resistance data)")

        # Low coverage
        cov = row.get("Path_Coverage")
        if pd.notna(cov):
            cov_str = str(cov).replace("%", "")
            try:
                cov_val = float(cov_str)
                if cov_val < 50:
                    issues.append(f"low coverage ({cov_val:.1f}%)")
            except ValueError:
                pass

        # Non-unique combo
        combo = row.get("Triple_Targets")
        if pd.notna(combo) and combo_counts.get(combo, 0) > 5:
            issues.append(f"combo shared by {combo_counts[combo]} cancers")

        if issues:
            flags.append({"cancer": cancer, "issues": issues})

    return flags


# ───────────────────────────────────────────────────────────────────────
# 8. Target Diversity Index
# ───────────────────────────────────────────────────────────────────────

def target_diversity_index(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Measure how diverse the target selection is across cancer types.

    Shannon entropy of unique combos, normalised to [0,1].
    Higher = more diverse = better.
    """
    if "Triple_Targets" not in df.columns:
        return {"error": "Triple_Targets column not found"}

    combos = df["Triple_Targets"].dropna().tolist()
    n = len(combos)
    counter = Counter(combos)
    n_unique = len(counter)

    # Shannon entropy
    probs = [count / n for count in counter.values()]
    entropy = -sum(p * math.log2(p) for p in probs if p > 0)
    max_entropy = math.log2(n) if n > 1 else 1
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

    return {
        "n_cancer_types": n,
        "n_unique_combos": n_unique,
        "shannon_entropy": round(entropy, 4),
        "max_possible_entropy": round(max_entropy, 4),
        "normalised_entropy": round(normalized_entropy, 4),
        "most_common_combos": [
            {"combo": combo, "count": count}
            for combo, count in counter.most_common(10)
        ],
        "verdict": (
            f"LOW diversity: {n_unique} unique combos for {n} cancers (entropy={normalized_entropy:.2f})"
            if normalized_entropy < 0.5
            else f"OK: {n_unique} unique combos (normalised entropy={normalized_entropy:.2f})"
        ),
    }


# ───────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────

def run_validation(
    results_dir: str = "results",
    v1_dir: Optional[str] = None,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run all validation checks and return a structured report."""

    triples_path = os.path.join(results_dir, "triple_combinations.csv")
    findings_path = os.path.join(results_dir, "all_findings.json")
    v1_triples_path = os.path.join(v1_dir, "triple_combinations.csv") if v1_dir else ""

    if not os.path.exists(triples_path):
        print(f"ERROR: {triples_path} not found. Has the pipeline finished?")
        sys.exit(1)

    df = _load_triples(triples_path)
    findings = _load_findings(findings_path) if os.path.exists(findings_path) else {}

    print("=" * 70)
    print("POST-PIPELINE VALIDATION REPORT")
    print("=" * 70)
    print(f"Results dir: {results_dir}")
    print(f"Cancer types in CSV: {df['Cancer_Type'].nunique()}")
    print(f"Cancer types in JSON: {len(findings.get('results', {}))}")
    print()

    report: Dict[str, Any] = {"results_dir": results_dir}

    # ── 1. CDK6 dominance ──────────────────────────────────────────
    print("─" * 70)
    print("1. TARGET DOMINANCE CHECK")
    print("─" * 70)
    dom = check_target_dominance(df)
    report["target_dominance"] = dom
    print(f"  {dom['verdict']}")
    print(f"  Unique combos: {dom['n_unique_combos']}")
    for g in dom["top_genes"][:5]:
        marker = " ⚠" if g["fraction"] > 0.50 else ""
        print(f"    {g['gene']:10s}  {g['n_cancer_types']:3d} cancers  ({_pct(g['fraction'])}){marker}")
    print()

    # ── 2. Synergy distribution ────────────────────────────────────
    print("─" * 70)
    print("2. SYNERGY DISTRIBUTION CHECK")
    print("─" * 70)
    syn = check_synergy_distribution(df, findings)
    report["synergy_distribution"] = syn
    print(f"  {syn['verdict']}")
    print(f"  range: [{syn.get('min', '?')} .. {syn.get('max', '?')}]")
    print(f"  mean={syn.get('mean', '?')}, std={syn.get('std', '?')}")
    print(f"  n_unique={syn.get('n_unique', '?')}, mode={syn.get('mode_value')} ({_pct(syn.get('mode_fraction', 0))})")
    print()

    # ── 3. Resistance distribution ─────────────────────────────────
    print("─" * 70)
    print("3. RESISTANCE DISTRIBUTION CHECK")
    print("─" * 70)
    res = check_resistance_distribution(df, findings)
    report["resistance_distribution"] = res
    print(f"  {res['verdict']}")
    print(f"  range: [{res.get('min', '?')} .. {res.get('max', '?')}]")
    print(f"  mean={res.get('mean', '?')}, std={res.get('std', '?')}")
    print(f"  n_zero={res.get('n_zero', 0)} ({_pct(res.get('zero_fraction', 0))})")
    print()

    # ── 4. Score component distributions ───────────────────────────
    print("─" * 70)
    print("4. SCORE COMPONENT DISTRIBUTIONS")
    print("─" * 70)
    dists = score_component_distributions(findings)
    report["score_distributions"] = dists
    for key, stats in dists.items():
        print(f"  {key:20s}  "
              f"mean={stats.get('mean', '?'):>8}  "
              f"std={stats.get('std', '?'):>8}  "
              f"range=[{stats.get('min', '?')} .. {stats.get('max', '?')}]  "
              f"unique={stats.get('n_unique', '?')}")
    print()

    # ── 5. LINCS impact ────────────────────────────────────────────
    print("─" * 70)
    print("5. LINCS PERTURBATION IMPACT")
    print("─" * 70)
    lincs = check_lincs_impact(findings)
    report["lincs_impact"] = lincs
    print(f"  Synergy variance:    {lincs['synergy_variance']}")
    print(f"  Resistance variance: {lincs['resistance_variance']}")
    print(f"  Synergy unique vals: {lincs['synergy_unique_values']}")
    print(f"  Resist. unique vals: {lincs['resistance_unique_values']}")
    print(f"  LINCS influence:     {lincs['lincs_influence_estimate']}")
    print()

    # ── 6. legacy comparison ───────────────────────────────────────
    print("─" * 70)
    print("6. LEGACY COMPARISON (OPTIONAL)")
    print("─" * 70)
    comp = compare_with_v1(df, v1_triples_path)
    report["v1_comparison"] = comp
    if comp is None:
        if v1_triples_path:
            print(f"  SKIPPED: {v1_triples_path} not found")
        else:
            print("  SKIPPED: no legacy results directory provided")
    else:
        print(f"  Shared cancers: {comp['shared_cancer_types']}")
        print(f"  Combos changed: {comp['combos_changed']} / {comp['shared_cancer_types']}"
              f"  ({_pct(comp['change_rate'])})")
        if comp["changed_details"]:
            print("  Notable changes:")
            for ch in comp["changed_details"][:10]:
                print(f"    {ch['cancer']:40s}  {ch['v1']} → {ch['v2']}")
    print()

    # ── 7. Per-cancer quality flags ────────────────────────────────
    print("─" * 70)
    print("7. PER-CANCER QUALITY FLAGS")
    print("─" * 70)
    flags = per_cancer_quality_flags(df, findings)
    report["quality_flags"] = flags
    if not flags:
        print("  No issues detected.")
    else:
        print(f"  {len(flags)} cancer types flagged:")
        for fl in flags[:20]:
            print(f"    {fl['cancer']:40s}  {'; '.join(fl['issues'])}")
        if len(flags) > 20:
            print(f"    ... and {len(flags) - 20} more")
    print()

    # ── 8. Target diversity ────────────────────────────────────────
    print("─" * 70)
    print("8. TARGET DIVERSITY INDEX")
    print("─" * 70)
    div = target_diversity_index(df)
    report["target_diversity"] = div
    print(f"  {div.get('verdict', 'N/A')}")
    if "most_common_combos" in div:
        print("  Most common combos:")
        for c in div["most_common_combos"][:5]:
            print(f"    {c['combo']:40s}  × {c['count']}")
    print()

    # ── Summary ────────────────────────────────────────────────────
    red_flags = sum([
        dom.get("red_flag", False),
        syn.get("red_flag", False),
        res.get("red_flag", False),
    ])
    report["summary"] = {
        "red_flags": red_flags,
        "total_checks": 3,
        "status": "PASS" if red_flags == 0 else f"FAIL ({red_flags} red flag{'s' if red_flags != 1 else ''})",
    }

    print("=" * 70)
    if red_flags == 0:
        print("OVERALL: PASS — no red flags detected")
    else:
        print(f"OVERALL: FAIL — {red_flags} red flag(s) detected")
        for name, check in [("Target dominance", dom), ("Synergy", syn), ("Resistance", res)]:
            if check.get("red_flag"):
                print(f"  ✗ {name}: {check['verdict']}")
    print("=" * 70)

    # Save JSON report
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nFull report saved to {output_path}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-pipeline validation")
    parser.add_argument("--results-dir", default="results", help="Pipeline results directory")
    parser.add_argument("--v1-dir", default="", help="Optional legacy results directory for comparison")
    parser.add_argument(
        "--output",
        default="outputs/reports/validation_reports/validation_report.json",
        help="Output JSON report",
    )
    args = parser.parse_args()

    run_validation(
        results_dir=args.results_dir,
        v1_dir=args.v1_dir,
        output_path=args.output,
    )
