#!/usr/bin/env python3
"""Generate precise statistics for manuscript tables and figures."""
import csv
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

trans_targets = Counter()
disc_targets = Counter()
trans_triples = {}
disc_triples = {}

with open(ROOT / "results/triple_combinations.csv") as f:
    for row in csv.DictReader(f):
        ct = row["Cancer_Type"]
        targets = sorted([row["Target_1"].strip(), row["Target_2"].strip(), row["Target_3"].strip()])
        trans_triples[ct] = {
            "targets": targets,
            "drugs": [row.get("Drug_1",""), row.get("Drug_2",""), row.get("Drug_3","")],
            "druggable": row.get("Druggable_Count",""),
            "score": row.get("Combined_Score",""),
        }
        for t in targets:
            trans_targets[t] += 1

with open(ROOT / "results_discovery/triple_combinations.csv") as f:
    for row in csv.DictReader(f):
        ct = row["Cancer_Type"]
        targets = sorted([row["Target_1"].strip(), row["Target_2"].strip(), row["Target_3"].strip()])
        disc_triples[ct] = {
            "targets": targets,
            "drugs": [row.get("Drug_1",""), row.get("Drug_2",""), row.get("Drug_3","")],
            "druggable": row.get("Druggable_Count",""),
            "score": row.get("Combined_Score",""),
        }
        for t in targets:
            disc_targets[t] += 1

trans_set = set(trans_targets.keys())
disc_set = set(disc_targets.keys())

print("=" * 60)
print("TARGET LANDSCAPE COMPARISON")
print("=" * 60)
print(f"Translational unique targets: {len(trans_set)}")
print(f"Discovery unique targets: {len(disc_set)}")
print(f"Shared: {len(trans_set & disc_set)}")
print(f"Translational-only: {len(trans_set - disc_set)}")
print(f"Discovery-only: {len(disc_set - trans_set)}")

print("\nTop 15 translational:")
for g, c in trans_targets.most_common(15):
    print(f"  {g}: {c}")

print("\nTop 15 discovery:")
for g, c in disc_targets.most_common(15):
    print(f"  {g}: {c}")

print(f"\nShared targets ({len(trans_set & disc_set)}):", sorted(trans_set & disc_set))
print(f"\nTranslational-only ({len(trans_set - disc_set)}):", sorted(trans_set - disc_set))
disc_only = sorted(disc_set - trans_set, key=lambda x: -disc_targets[x])
print(f"\nDiscovery-only ({len(disc_only)}) top 20:", disc_only[:20])

# Per-cancer comparison
print("\n" + "=" * 60)
print("PER-CANCER COMPARISON")
print("=" * 60)
shared_cancers = set(trans_triples.keys()) & set(disc_triples.keys())
identical = 0
partial = 0
different = 0
for ct in sorted(shared_cancers):
    t_set = set(trans_triples[ct]["targets"])
    d_set = set(disc_triples[ct]["targets"])
    overlap = len(t_set & d_set)
    if overlap == 3:
        identical += 1
    elif overlap > 0:
        partial += 1
    else:
        different += 1

print(f"Shared cancer types: {len(shared_cancers)}")
print(f"Identical (3/3): {identical} ({100*identical/len(shared_cancers):.1f}%)")
print(f"Partial (1-2/3): {partial} ({100*partial/len(shared_cancers):.1f}%)")
print(f"Different (0/3): {different} ({100*different/len(shared_cancers):.1f}%)")

# Druggability comparison
print("\n" + "=" * 60)
print("DRUGGABILITY")
print("=" * 60)
trans_drug_counts = []
disc_drug_counts = []
for ct in trans_triples:
    try:
        trans_drug_counts.append(int(trans_triples[ct]["druggable"]))
    except (ValueError, KeyError):
        pass
for ct in disc_triples:
    try:
        disc_drug_counts.append(int(disc_triples[ct]["druggable"]))
    except (ValueError, KeyError):
        pass

if trans_drug_counts:
    fully_druggable_t = sum(1 for x in trans_drug_counts if x == 3)
    print(f"Translational: mean druggable = {sum(trans_drug_counts)/len(trans_drug_counts):.1f}, fully (3/3) = {fully_druggable_t}/{len(trans_drug_counts)} ({100*fully_druggable_t/len(trans_drug_counts):.1f}%)")
if disc_drug_counts:
    fully_druggable_d = sum(1 for x in disc_drug_counts if x == 3)
    zero_drug = sum(1 for x in disc_drug_counts if x == 0)
    print(f"Discovery: mean druggable = {sum(disc_drug_counts)/len(disc_drug_counts):.1f}, fully (3/3) = {fully_druggable_d}/{len(disc_drug_counts)} ({100*fully_druggable_d/len(disc_drug_counts):.1f}%), zero-drug = {zero_drug}/{len(disc_drug_counts)} ({100*zero_drug/len(disc_drug_counts):.1f}%)")

# 8 identical cancers
print("\n" + "=" * 60)
print("IDENTICAL TRIPLE CANCERS")
print("=" * 60)
for ct in sorted(shared_cancers):
    t_set = set(trans_triples[ct]["targets"])
    d_set = set(disc_triples[ct]["targets"])
    if t_set == d_set:
        print(f"  {ct}: {' + '.join(sorted(t_set))}")
