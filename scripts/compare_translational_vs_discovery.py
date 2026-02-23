#!/usr/bin/env python3
"""
Compare Translational (druggability-biased) vs Discovery (biology-first) pipeline results.

Generates:
  1. Side-by-side comparison table of triples per cancer type
  2. Targets unique to discovery mode (undruggable biology)
  3. Targets shared between modes (robust biology + druggable)
  4. Drug-development priority list: targets that appear in discovery but not translational
  5. Summary statistics for the manuscript

Usage:
    python scripts/compare_translational_vs_discovery.py
"""
import sys, json
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TRANS_DIR = ROOT / "results"           # v3 translational results
DISC_DIR  = ROOT / "results_discovery" # discovery-mode results
OUT_DIR   = ROOT / "results_discovery"

def load_triples(results_dir: Path) -> pd.DataFrame:
    f = results_dir / "triple_combinations.csv"
    if not f.exists():
        sys.exit(f"Missing: {f}")
    return pd.read_csv(f)

def get_targets(row):
    """Extract target set from a row."""
    targets = set()
    for col in ['Target_1', 'Target_2', 'Target_3']:
        if col in row and pd.notna(row[col]):
            targets.add(row[col])
    return frozenset(targets)

def main():
    trans = load_triples(TRANS_DIR)
    disc  = load_triples(DISC_DIR)

    print("=" * 90)
    print("TRANSLATIONAL vs DISCOVERY MODE — COMPARATIVE ANALYSIS")
    print("=" * 90)

    # ── 1. Overview ──────────────────────────────────────────────────────
    print(f"\n{'Metric':<40} {'Translational':>15} {'Discovery':>15}")
    print("-" * 70)
    print(f"{'Cancer types with triples':<40} {len(trans):>15} {len(disc):>15}")

    trans_targets = set()
    disc_targets  = set()
    for _, r in trans.iterrows():
        trans_targets |= get_targets(r)
    for _, r in disc.iterrows():
        disc_targets |= get_targets(r)

    print(f"{'Unique targets':<40} {len(trans_targets):>15} {len(disc_targets):>15}")

    trans_combos = set(tuple(sorted(get_targets(r))) for _, r in trans.iterrows())
    disc_combos  = set(tuple(sorted(get_targets(r))) for _, r in disc.iterrows())
    print(f"{'Unique triple combinations':<40} {len(trans_combos):>15} {len(disc_combos):>15}")

    # ── 2. Targets only in discovery mode (the interesting ones) ─────────
    discovery_only_targets = disc_targets - trans_targets
    shared_targets = disc_targets & trans_targets
    trans_only_targets = trans_targets - disc_targets

    print(f"\n{'Targets shared (both modes)':<40} {len(shared_targets):>15}")
    print(f"{'Targets in discovery only':<40} {len(discovery_only_targets):>15}")
    print(f"{'Targets in translational only':<40} {len(trans_only_targets):>15}")

    if discovery_only_targets:
        print(f"\n--- DISCOVERY-ONLY TARGETS (drug-development priorities) ---")
        # Count how many cancers each discovery-only target appears in
        disc_target_counts = Counter()
        for _, r in disc.iterrows():
            for t in get_targets(r):
                disc_target_counts[t] += 1
        for t in sorted(discovery_only_targets, key=lambda x: disc_target_counts[x], reverse=True):
            print(f"  {t:<20} ({disc_target_counts[t]} cancers)")

    if trans_only_targets:
        print(f"\n--- TRANSLATIONAL-ONLY TARGETS (lost when removing druggability bias) ---")
        trans_target_counts = Counter()
        for _, r in trans.iterrows():
            for t in get_targets(r):
                trans_target_counts[t] += 1
        for t in sorted(trans_only_targets, key=lambda x: trans_target_counts[x], reverse=True):
            print(f"  {t:<20} ({trans_target_counts[t]} cancers)")

    # ── 3. Per-cancer comparison ─────────────────────────────────────────
    print(f"\n{'=' * 90}")
    print("PER-CANCER COMPARISON")
    print(f"{'=' * 90}")

    # Build lookup by cancer type
    trans_by_cancer = {}
    for _, r in trans.iterrows():
        trans_by_cancer[r['Cancer_Type']] = get_targets(r)
    disc_by_cancer = {}
    for _, r in disc.iterrows():
        disc_by_cancer[r['Cancer_Type']] = get_targets(r)

    all_cancers = sorted(set(trans_by_cancer) | set(disc_by_cancer))

    comparison_rows = []
    identical = partial = different = trans_only_ct = disc_only_ct = 0

    print(f"\n{'Cancer Type':<50} {'Status':<15} {'Translational':<30} {'Discovery':<30}")
    print("-" * 125)

    for ct in all_cancers:
        t_triple = trans_by_cancer.get(ct)
        d_triple = disc_by_cancer.get(ct)

        if t_triple and d_triple:
            overlap = len(t_triple & d_triple)
            if overlap == 3:
                status = "IDENTICAL"
                identical += 1
            elif overlap >= 2:
                status = f"{overlap}/3 overlap"
                partial += 1
            elif overlap == 1:
                status = "1/3 overlap"
                partial += 1
            else:
                status = "DIFFERENT"
                different += 1
            t_str = " + ".join(sorted(t_triple))
            d_str = " + ".join(sorted(d_triple))
        elif t_triple:
            status = "trans only"
            trans_only_ct += 1
            t_str = " + ".join(sorted(t_triple))
            d_str = "—"
        else:
            status = "disc only"
            disc_only_ct += 1
            t_str = "—"
            d_str = " + ".join(sorted(d_triple))

        print(f"  {ct:<48} {status:<15} {t_str:<30} {d_str:<30}")
        comparison_rows.append({
            'Cancer_Type': ct,
            'Translational_Triple': " + ".join(sorted(t_triple)) if t_triple else "",
            'Discovery_Triple': " + ".join(sorted(d_triple)) if d_triple else "",
            'Overlap': len(t_triple & d_triple) if (t_triple and d_triple) else 0,
            'Status': status,
        })

    # ── 4. Summary statistics ────────────────────────────────────────────
    shared_cancers = set(trans_by_cancer) & set(disc_by_cancer)
    n_shared = len(shared_cancers)

    print(f"\n{'=' * 90}")
    print("AGREEMENT SUMMARY (cancers with triples in BOTH modes)")
    print(f"{'=' * 90}")
    print(f"  Cancers in both modes:    {n_shared}")
    print(f"  Identical triples:        {identical} ({100*identical/max(n_shared,1):.1f}%)")
    print(f"  Partial overlap (≥1):     {partial} ({100*partial/max(n_shared,1):.1f}%)")
    print(f"  Completely different:      {different} ({100*different/max(n_shared,1):.1f}%)")
    if trans_only_ct: print(f"  Translational only:       {trans_only_ct}")
    if disc_only_ct:  print(f"  Discovery only:           {disc_only_ct}")

    # ── 5. Drug-development priority table ───────────────────────────────
    # Targets that appear in discovery but NOT in translational, or that
    # appear much more frequently in discovery mode.
    print(f"\n{'=' * 90}")
    print("DRUG-DEVELOPMENT PRIORITY TARGETS")
    print("(Targets that biology selects but druggability de-prioritises)")
    print(f"{'=' * 90}")

    trans_freq = Counter()
    for _, r in trans.iterrows():
        for t in get_targets(r):
            trans_freq[t] += 1
    disc_freq = Counter()
    for _, r in disc.iterrows():
        for t in get_targets(r):
            disc_freq[t] += 1

    all_targets = sorted(disc_targets | trans_targets)
    priority_rows = []
    for t in all_targets:
        df = disc_freq.get(t, 0)
        tf = trans_freq.get(t, 0)
        delta = df - tf
        priority_rows.append({
            'Target': t,
            'Discovery_Count': df,
            'Translational_Count': tf,
            'Delta': delta,
            'Category': 'Discovery-only' if tf == 0 and df > 0 else
                        'Translational-only' if df == 0 and tf > 0 else
                        'Upranked in discovery' if delta > 0 else
                        'Downranked in discovery' if delta < 0 else 'Same',
        })

    priority_df = pd.DataFrame(priority_rows).sort_values('Delta', ascending=False)
    print(f"\n{'Target':<15} {'Discovery':>10} {'Translational':>14} {'Delta':>7}  {'Category'}")
    print("-" * 65)
    for _, r in priority_df.iterrows():
        print(f"  {r['Target']:<13} {r['Discovery_Count']:>10} {r['Translational_Count']:>14} {r['Delta']:>+7}  {r['Category']}")

    # ── 6. Druggability assessment of discovery-only targets ─────────────
    if discovery_only_targets:
        print(f"\n{'=' * 90}")
        print("DRUGGABILITY ASSESSMENT OF DISCOVERY-ONLY TARGETS")
        print("(These are the targets for which new drugs or modalities are needed)")
        print(f"{'=' * 90}\n")

        try:
            sys.path.insert(0, str(ROOT))
            from pan_cancer_xnode import DrugTargetDB
            db = DrugTargetDB()
            for t in sorted(discovery_only_targets):
                d_score = db.get_druggability_score(t)
                info = db.get_drug_info(t)
                drugs = info.get('drugs', []) if info else []
                stage = info.get('stage', 'unknown') if info else 'unknown'
                drug_str = ", ".join(drugs[:3]) if drugs else "none"
                print(f"  {t:<15} druggability={d_score:.2f}  stage={stage:<12}  drugs=[{drug_str}]")
        except Exception as e:
            print(f"  (Could not load DrugTargetDB: {e})")

    # ── 7. Cancers where discovery mode reveals new biology ──────────────
    print(f"\n{'=' * 90}")
    print("KEY BIOLOGICAL INSIGHTS FROM DISCOVERY MODE")
    print("(Cancers where removing druggability bias changed ≥2 targets)")
    print(f"{'=' * 90}\n")

    for ct in sorted(shared_cancers):
        t = trans_by_cancer[ct]
        d = disc_by_cancer[ct]
        overlap = len(t & d)
        if overlap <= 1:
            gained = d - t
            lost = t - d
            print(f"  {ct}:")
            print(f"    Translational: {' + '.join(sorted(t))}")
            print(f"    Discovery:     {' + '.join(sorted(d))}")
            print(f"    Gained: {', '.join(sorted(gained))}  |  Lost: {', '.join(sorted(lost))}")
            print()

    # ── 8. Save outputs ─────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    comp_df = pd.DataFrame(comparison_rows)
    comp_df.to_csv(OUT_DIR / "translational_vs_discovery_comparison.csv", index=False)
    priority_df.to_csv(OUT_DIR / "drug_development_priorities.csv", index=False)

    print(f"\nSaved: {OUT_DIR / 'translational_vs_discovery_comparison.csv'}")
    print(f"Saved: {OUT_DIR / 'drug_development_priorities.csv'}")

    # ── 9. Manuscript-ready summary ──────────────────────────────────────
    print(f"\n{'=' * 90}")
    print("MANUSCRIPT SUMMARY")
    print(f"{'=' * 90}")
    print(f"""
Discovery mode identified {len(disc_targets)} unique targets across {len(disc)} cancer types,
compared to {len(trans_targets)} targets in {len(trans)} cancer types for the translational track.
{len(discovery_only_targets)} targets appeared exclusively in the discovery track, representing
novel drug-development priorities that are biologically important but currently
lack approved therapeutics.

Of {n_shared} cancer types analysed in both modes, {identical} ({100*identical/max(n_shared,1):.1f}%) produced
identical triples, confirming that the core biology is robust to the scoring
model. {partial + different} cancers ({100*(partial+different)/max(n_shared,1):.1f}%) showed partial or complete
divergence, revealing targets that are biologically prioritised but
de-emphasised by druggability scoring.
""")


if __name__ == "__main__":
    main()
