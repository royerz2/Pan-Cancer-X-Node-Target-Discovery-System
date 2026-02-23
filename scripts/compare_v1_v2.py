#!/usr/bin/env python3
"""Compare v1 (tier-based) vs v2 (sigmoid+adaptive) pan-cancer results.

Reads triple_combinations.csv from results_v1/ and results/ and produces
a detailed comparison report.
"""

import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
V1_DIR = ROOT / "results_v1"
V2_DIR = ROOT / "results"

def load_triples(path: Path) -> pd.DataFrame:
    f = path / "triple_combinations.csv"
    if not f.exists():
        print(f"ERROR: {f} not found")
        sys.exit(1)
    df = pd.read_csv(f)
    # Normalize column names (v1 uses spaces, v2 may use underscores)
    df.columns = [c.replace(' ', '_') for c in df.columns]
    return df

def load_protein_scores(path: Path) -> dict:
    f = path / "protein_druggability_scores.json"
    if not f.exists():
        return {}
    with open(f) as fh:
        return json.load(fh)

def main():
    print("=" * 80)
    print("  V1 vs V2 Pan-Cancer Comparison")
    print("=" * 80)

    # --- Load triples ---
    v1 = load_triples(V1_DIR)
    v2 = load_triples(V2_DIR)
    print(f"\nDataset sizes:  v1 = {len(v1)} triples,  v2 = {len(v2)} triples")

    # --- Cancer types ---
    v1_cancers = set(v1["Cancer_Type"].unique())
    v2_cancers = set(v2["Cancer_Type"].unique())
    print(f"Cancer types:   v1 = {len(v1_cancers)},  v2 = {len(v2_cancers)}")
    new_cancers = v2_cancers - v1_cancers
    lost_cancers = v1_cancers - v2_cancers
    if new_cancers:
        print(f"  New in v2: {sorted(new_cancers)}")
    if lost_cancers:
        print(f"  Lost in v2: {sorted(lost_cancers)}")

    # --- Best triple per cancer ---
    common = v1_cancers & v2_cancers
    
    # Get rank-1 triple per cancer for each version
    def best_per_cancer(df):
        out = {}
        for ct in df["Cancer_Type"].unique():
            sub = df[df["Cancer_Type"] == ct].sort_values("Combined_Score")
            if len(sub) > 0:
                row = sub.iloc[0]
                t1 = row.get("Target_1", row.get("Target 1", ""))
                t2 = row.get("Target_2", row.get("Target 2", ""))
                t3 = row.get("Target_3", row.get("Target 3", ""))
                targets = sorted([t1, t2, t3])
                out[ct] = {
                    "targets": targets,
                    "triple_str": " + ".join(targets),
                    "score": row["Combined_Score"],
                }
        return out

    v1_best = best_per_cancer(v1)
    v2_best = best_per_cancer(v2)

    # --- Triple stability ---
    same_count = 0
    changed = []
    for ct in sorted(common):
        if ct not in v1_best or ct not in v2_best:
            continue
        if v1_best[ct]["targets"] == v2_best[ct]["targets"]:
            same_count += 1
        else:
            changed.append(ct)

    total_compared = len([c for c in common if c in v1_best and c in v2_best])
    print(f"\n--- Triple Stability ---")
    print(f"Same best triple:    {same_count}/{total_compared} ({100*same_count/max(total_compared,1):.0f}%)")
    print(f"Changed best triple: {len(changed)}/{total_compared}")
    
    if changed:
        print(f"\nChanged triples (top 20):")
        for ct in changed[:20]:
            v1t = v1_best[ct]["triple_str"]
            v2t = v2_best[ct]["triple_str"]
            # Check overlap
            overlap = len(set(v1_best[ct]["targets"]) & set(v2_best[ct]["targets"]))
            print(f"  {ct}:")
            print(f"    v1: {v1t}  (score={v1_best[ct]['score']:.4f})")
            print(f"    v2: {v2t}  (score={v2_best[ct]['score']:.4f})")
            print(f"    overlap: {overlap}/3 targets")

    # --- Score distribution ---
    score_cols = [c for c in v2.columns if "Score" in c or "score" in c]
    print(f"\n--- Score Distribution (v2) ---")
    for col in score_cols[:8]:
        if col in v1.columns:
            print(f"  {col}: v1 mean={v1[col].mean():.4f}  v2 mean={v2[col].mean():.4f}  Δ={v2[col].mean()-v1[col].mean():+.4f}")
        else:
            print(f"  {col}: v2 mean={v2[col].mean():.4f}  (new in v2)")

    # --- Protein score comparison ---
    v1_prot = load_protein_scores(V1_DIR)
    v2_prot = load_protein_scores(V2_DIR)
    
    if v1_prot and v2_prot:
        print(f"\n--- Protein Druggability Scores ---")
        print(f"  v1: {len(v1_prot)} genes scored")
        print(f"  v2: {len(v2_prot)} genes scored")
        
        common_genes = set(v1_prot.keys()) & set(v2_prot.keys())
        if common_genes:
            v1_scores = []
            v2_scores = []
            diffs = []
            for g in common_genes:
                s1 = v1_prot[g].get("protein_score", v1_prot[g].get("blended_score", 0))
                s2 = v2_prot[g].get("protein_score", v2_prot[g].get("blended_score", 0))
                v1_scores.append(s1)
                v2_scores.append(s2)
                diffs.append((g, s2 - s1, s1, s2))
            
            v1_arr = np.array(v1_scores)
            v2_arr = np.array(v2_scores)
            print(f"  Mean protein score: v1={v1_arr.mean():.4f}  v2={v2_arr.mean():.4f}  Δ={v2_arr.mean()-v1_arr.mean():+.4f}")
            
            # Biggest movers
            diffs.sort(key=lambda x: abs(x[1]), reverse=True)
            print(f"\n  Top 10 biggest score changes:")
            for gene, delta, s1, s2 in diffs[:10]:
                print(f"    {gene:12s}: {s1:.3f} → {s2:.3f}  (Δ={delta:+.3f})")

    # --- Benchmark impact ---
    # Check if we can compare against gold standard
    gs_file = ROOT / "gold_standard.py"
    if gs_file.exists():
        try:
            sys.path.insert(0, str(ROOT))
            from gold_standard import GOLD_STANDARD
            
            def calc_overlap(best_dict):
                any_hit = 0
                pair_hit = 0
                exact_hit = 0
                total = 0
                for entry in GOLD_STANDARD:
                    ct = entry.get("cancer_type", "")
                    gs_targets = set(entry.get("targets", []))
                    if ct in best_dict:
                        pred = set(best_dict[ct]["targets"])
                        overlap = len(gs_targets & pred)
                        total += 1
                        if overlap >= 1:
                            any_hit += 1
                        if overlap >= 2:
                            pair_hit += 1
                        if gs_targets == pred:
                            exact_hit += 1
                return total, any_hit, pair_hit, exact_hit
            
            print(f"\n--- Benchmark Concordance ---")
            t1, a1, p1, e1 = calc_overlap(v1_best)
            t2, a2, p2, e2 = calc_overlap(v2_best)
            print(f"  v1: {a1}/{t1} any-overlap ({100*a1/max(t1,1):.1f}%),  {p1}/{t1} pair ({100*p1/max(t1,1):.1f}%),  {e1}/{t1} exact")
            print(f"  v2: {a2}/{t2} any-overlap ({100*a2/max(t2,1):.1f}%),  {p2}/{t2} pair ({100*p2/max(t2,1):.1f}%),  {e2}/{t2} exact")
            if a2 != a1:
                print(f"  Δ any-overlap: {a2-a1:+d}  ({100*(a2-a1)/max(t1,1):+.1f}pp)")
            if p2 != p1:
                print(f"  Δ pair-overlap: {p2-p1:+d}  ({100*(p2-p1)/max(t1,1):+.1f}pp)")
        except Exception as e:
            print(f"  (Could not compute benchmark: {e})")

    # --- Unresolved gene gaps (wet-lab priorities) ---
    unresolved_file = V2_DIR / "unresolved_genes_wetlab_gaps.txt"
    if unresolved_file.exists():
        with open(unresolved_file) as f:
            genes = [l.strip() for l in f if l.strip() and not l.startswith('#')]
        print(f"\n--- Unresolved Gene Gaps (Wet-Lab Priorities) ---")
        print(f"  Total genes without Swiss-Prot ID: {len(genes)}")
        if genes:
            print(f"  First 15: {', '.join(genes[:15])}")
            print(f"  These genes used fallback protein scores (flat 0.3).")
            print(f"  Structural, abundance, and degradability data would improve accuracy.")

    # --- Summary ---
    print(f"\n{'='*80}")
    print(f"  SUMMARY")
    print(f"{'='*80}")
    print(f"  Triple stability: {same_count}/{total_compared} same ({100*same_count/max(total_compared,1):.0f}%)")
    print(f"  Changed: {len(changed)} cancer types")
    if v1_prot and v2_prot:
        print(f"  Protein score Δ: {v2_arr.mean()-v1_arr.mean():+.4f} (v2 {'higher' if v2_arr.mean() > v1_arr.mean() else 'lower'})")
    print()

if __name__ == "__main__":
    main()
