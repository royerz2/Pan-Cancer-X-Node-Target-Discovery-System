#!/usr/bin/env python3
"""
Complete Analysis of v3 Pipeline Run
=====================================
Compares v1 (pre-raw-CV) vs v3 (with raw replicate CV) results,
produces summary statistics, stability metrics, and a full report.

Outputs:
  - results/v3_analysis_report.txt   — Human-readable report
  - results/v1_vs_v3_comparison.csv  — Side-by-side cancer-level comparison
"""

import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

# ============================================================================
# 1. Load v3 results
# ============================================================================

v3_triples = pd.read_csv("results/triple_combinations.csv")
v3_summary = pd.read_csv("results/pan_cancer_summary.csv")
v3_target_freq = pd.read_csv("results/target_frequency_summary.csv")
v3_triple_freq = pd.read_csv("results/triple_target_frequency.csv")
v3_detailed = pd.read_csv("results/xnode_combinations_detailed.csv")
v3_drugs = pd.read_csv("results/drug_protocols.csv")

with open("results/all_findings.json") as f:
    v3_findings = json.load(f)

# ============================================================================
# 2. Load v1 results (if available)
# ============================================================================

v1_triples = None
v1_available = False
if os.path.isfile("results_v1/triple_combinations.csv"):
    v1_triples = pd.read_csv("results_v1/triple_combinations.csv")
    v1_available = True

# ============================================================================
# 3. v3 Run Summary
# ============================================================================

lines = []
lines.append("=" * 80)
lines.append("ALIN v3 PIPELINE RUN — COMPLETE ANALYSIS")
lines.append("=" * 80)
lines.append("")

# Basic stats
n_cancers = len(v3_summary)
n_with_triples = len(v3_triples)
n_without = n_cancers - n_with_triples

lines.append(f"Cancer types analysed:      {n_cancers}")
lines.append(f"Cancer types with triples:  {n_with_triples}")
lines.append(f"Cancer types without:       {n_without} (insufficient cell lines / paths)")
lines.append("")

# Cell line coverage
if "Cell Lines" in v3_summary.columns:
    cl = v3_summary["Cell Lines"]
    lines.append(f"Cell lines per cancer type:")
    lines.append(f"  Median: {cl.median():.0f}   Mean: {cl.mean():.1f}   "
                 f"Min: {cl.min()}   Max: {cl.max()}")
    lines.append(f"  Total unique cell lines used: {cl.sum():.0f}")
    lines.append("")

# Path coverage
if "Paths" in v3_summary.columns:
    paths = v3_summary["Paths"]
    lines.append(f"Viability paths per cancer type:")
    lines.append(f"  Median: {paths.median():.0f}   Mean: {paths.mean():.1f}   "
                 f"Min: {paths.min()}   Max: {paths.max()}")
    lines.append("")

# Coverage
if "Coverage" in v3_summary.columns:
    cov_raw = v3_summary["Coverage"].astype(str).str.replace("%", "").astype(float)
    lines.append(f"Path coverage by X-Node triples:")
    lines.append(f"  Median: {cov_raw.median():.1f}%   Mean: {cov_raw.mean():.1f}%   "
                 f"Min: {cov_raw.min():.1f}%   Max: {cov_raw.max():.1f}%")
    lines.append("")

# ============================================================================
# 4. Triple Target Analysis
# ============================================================================

lines.append("-" * 80)
lines.append("TRIPLE TARGET ANALYSIS")
lines.append("-" * 80)
lines.append("")

# Most frequent targets across all triples
all_targets = []
for _, row in v3_triples.iterrows():
    for col in ["Target_1", "Target_2", "Target_3"]:
        if col in row and pd.notna(row[col]):
            all_targets.append(str(row[col]))

target_counts = Counter(all_targets)
lines.append(f"Unique targets across all triples: {len(target_counts)}")
lines.append("")
lines.append("Top 15 most frequent targets:")
for gene, count in target_counts.most_common(15):
    pct = count / n_with_triples * 100
    lines.append(f"  {gene:12s}  {count:3d}/{n_with_triples} cancers ({pct:5.1f}%)")
lines.append("")

# Unique triple combinations
triple_combos = []
for _, row in v3_triples.iterrows():
    targets = sorted([str(row[c]) for c in ["Target_1", "Target_2", "Target_3"]
                       if c in row and pd.notna(row[c])])
    triple_combos.append(" + ".join(targets))

combo_counts = Counter(triple_combos)
lines.append(f"Unique triple combinations: {len(combo_counts)}")
lines.append("")
lines.append("Most common triple combinations:")
for combo, count in combo_counts.most_common(10):
    lines.append(f"  {combo:40s}  {count:3d} cancers")
lines.append("")

# ============================================================================
# 5. Synergy & Resistance Scores
# ============================================================================

lines.append("-" * 80)
lines.append("SYNERGY & RESISTANCE SCORES")
lines.append("-" * 80)
lines.append("")

if "Synergy_Score" in v3_triples.columns:
    syn = v3_triples["Synergy_Score"]
    lines.append(f"Synergy scores:")
    lines.append(f"  Median: {syn.median():.3f}   Mean: {syn.mean():.3f}   "
                 f"Min: {syn.min():.3f}   Max: {syn.max():.3f}")

if "Resistance_Score" in v3_triples.columns:
    res = v3_triples["Resistance_Score"]
    lines.append(f"Resistance scores:")
    lines.append(f"  Median: {res.median():.3f}   Mean: {res.mean():.3f}   "
                 f"Min: {res.min():.3f}   Max: {res.max():.3f}")

if "Combined_Score" in v3_triples.columns:
    comb = v3_triples["Combined_Score"]
    lines.append(f"Combined scores:")
    lines.append(f"  Median: {comb.median():.3f}   Mean: {comb.mean():.3f}   "
                 f"Min: {comb.min():.3f}   Max: {comb.max():.3f}")
lines.append("")

# Druggability
if "Druggable_Count" in v3_triples.columns:
    drug = v3_triples["Druggable_Count"]
    fully_druggable = (drug == 3).sum()
    two_druggable = (drug == 2).sum()
    one_or_less = (drug <= 1).sum()
    lines.append(f"Druggability of triples:")
    lines.append(f"  3/3 druggable: {fully_druggable} ({fully_druggable/n_with_triples*100:.1f}%)")
    lines.append(f"  2/3 druggable: {two_druggable} ({two_druggable/n_with_triples*100:.1f}%)")
    lines.append(f"  ≤1/3 druggable: {one_or_less}")
    lines.append("")

# ============================================================================
# 6. Drug Protocol Analysis
# ============================================================================

lines.append("-" * 80)
lines.append("DRUG PROTOCOL ANALYSIS")
lines.append("-" * 80)
lines.append("")

all_drugs = []
for _, row in v3_triples.iterrows():
    for col in ["Drug_1", "Drug_2", "Drug_3"]:
        if col in row and pd.notna(row[col]) and str(row[col]) != "None":
            all_drugs.append(str(row[col]))

drug_counts = Counter(all_drugs)
lines.append(f"Unique drugs used: {len(drug_counts)}")
lines.append("")
lines.append("Top 10 most prescribed drugs:")
for drug, count in drug_counts.most_common(10):
    lines.append(f"  {drug:25s}  {count:3d} cancers")
lines.append("")

# ============================================================================
# 7. Lineage Distribution
# ============================================================================

if "Lineage" in v3_triples.columns:
    lines.append("-" * 80)
    lines.append("LINEAGE DISTRIBUTION")
    lines.append("-" * 80)
    lines.append("")
    lineage_counts = v3_triples["Lineage"].value_counts()
    for lineage, count in lineage_counts.items():
        lines.append(f"  {lineage:30s}  {count:3d} cancers")
    lines.append("")

# ============================================================================
# 8. v1 vs v3 Comparison
# ============================================================================

if v1_available and v1_triples is not None:
    lines.append("=" * 80)
    lines.append("v1 vs v3 COMPARISON")
    lines.append("=" * 80)
    lines.append("")

    # Normalise cancer type columns
    v1_cancer_col = "Cancer Type" if "Cancer Type" in v1_triples.columns else v1_triples.columns[0]
    v3_cancer_col = "Cancer_Type" if "Cancer_Type" in v3_triples.columns else v3_triples.columns[0]

    v1_cancers = set(v1_triples[v1_cancer_col].str.strip())
    v3_cancers = set(v3_triples[v3_cancer_col].str.strip())

    lines.append(f"v1 cancer types with triples: {len(v1_cancers)}")
    lines.append(f"v3 cancer types with triples: {len(v3_cancers)}")
    lines.append(f"Shared cancer types:          {len(v1_cancers & v3_cancers)}")
    lines.append(f"New in v3:                    {len(v3_cancers - v1_cancers)}")
    lines.append(f"Dropped from v1:              {len(v1_cancers - v3_cancers)}")
    lines.append("")

    # Build lookup dicts for shared cancers
    v1_lookup = {}
    for _, row in v1_triples.iterrows():
        ct = str(row[v1_cancer_col]).strip()
        t1 = str(row.get("Target 1", row.get("Target_1", ""))).strip()
        t2 = str(row.get("Target 2", row.get("Target_2", ""))).strip()
        t3 = str(row.get("Target 3", row.get("Target_3", ""))).strip()
        v1_lookup[ct] = sorted([t1, t2, t3])

    v3_lookup = {}
    for _, row in v3_triples.iterrows():
        ct = str(row[v3_cancer_col]).strip()
        t1 = str(row.get("Target_1", "")).strip()
        t2 = str(row.get("Target_2", "")).strip()
        t3 = str(row.get("Target_3", "")).strip()
        v3_lookup[ct] = sorted([t1, t2, t3])

    shared = v1_cancers & v3_cancers
    same_triple = 0
    partial_overlap = 0
    completely_different = 0
    comparison_rows = []

    for ct in sorted(shared):
        v1_t = v1_lookup.get(ct, [])
        v3_t = v3_lookup.get(ct, [])
        v1_set = set(v1_t)
        v3_set = set(v3_t)
        overlap = len(v1_set & v3_set)

        if v1_t == v3_t:
            same_triple += 1
            status = "IDENTICAL"
        elif overlap >= 2:
            partial_overlap += 1
            status = f"2/3 overlap"
        elif overlap == 1:
            partial_overlap += 1
            status = f"1/3 overlap"
        else:
            completely_different += 1
            status = "DIFFERENT"

        comparison_rows.append({
            "Cancer_Type": ct,
            "v1_Triple": " + ".join(v1_t),
            "v3_Triple": " + ".join(v3_t),
            "Overlap": overlap,
            "Status": status,
        })

    lines.append(f"Triple stability across shared cancer types ({len(shared)}):")
    lines.append(f"  Identical triples:     {same_triple} ({same_triple/max(len(shared),1)*100:.1f}%)")
    lines.append(f"  Partial overlap (≥1):  {partial_overlap} ({partial_overlap/max(len(shared),1)*100:.1f}%)")
    lines.append(f"  Completely different:  {completely_different} ({completely_different/max(len(shared),1)*100:.1f}%)")
    lines.append("")

    # Target-level stability
    v1_all = []
    for t in v1_lookup.values():
        v1_all.extend(t)
    v1_tc = Counter(v1_all)

    v3_all = []
    for t in v3_lookup.values():
        v3_all.extend(t)
    v3_tc = Counter(v3_all)

    all_genes = sorted(set(v1_tc.keys()) | set(v3_tc.keys()))
    lines.append("Target frequency comparison (v1 → v3):")
    gene_diffs = []
    for g in all_genes:
        c1 = v1_tc.get(g, 0)
        c3 = v3_tc.get(g, 0)
        diff = c3 - c1
        gene_diffs.append((g, c1, c3, diff))
    gene_diffs.sort(key=lambda x: -abs(x[3]))

    for g, c1, c3, diff in gene_diffs[:15]:
        arrow = "↑" if diff > 0 else ("↓" if diff < 0 else "=")
        lines.append(f"  {g:12s}  v1={c1:3d}  v3={c3:3d}  {arrow}{abs(diff):+d}")
    lines.append("")

    # Detailed comparison per cancer
    lines.append("Per-cancer triple comparison (changes only):")
    lines.append(f"  {'Cancer Type':45s} {'v1 Triple':50s} {'v3 Triple':50s} Status")
    lines.append(f"  {'-'*45} {'-'*50} {'-'*50} ------")
    for r in comparison_rows:
        if r["Status"] != "IDENTICAL":
            lines.append(f"  {r['Cancer_Type']:45s} {r['v1_Triple']:50s} {r['v3_Triple']:50s} {r['Status']}")
    lines.append("")

    # Save comparison CSV
    comp_df = pd.DataFrame(comparison_rows)
    comp_df.to_csv("results/v1_vs_v3_comparison.csv", index=False)

# ============================================================================
# 9. Per-Cancer Best Triples Table
# ============================================================================

lines.append("-" * 80)
lines.append("COMPLETE TRIPLE RESULTS BY CANCER TYPE")
lines.append("-" * 80)
lines.append("")

lines.append(f"{'Cancer Type':50s} {'Triple':40s} {'Syn':>5s} {'Res':>5s} {'Cov':>6s} {'Drug':>4s}")
lines.append(f"{'-'*50} {'-'*40} {'-'*5} {'-'*5} {'-'*6} {'-'*4}")

for _, row in v3_triples.sort_values("Cancer_Type" if "Cancer_Type" in v3_triples.columns else v3_triples.columns[0]).iterrows():
    ct = str(row.get("Cancer_Type", row.iloc[0]))[:48]
    triple = f"{row.get('Target_1','')} + {row.get('Target_2','')} + {row.get('Target_3','')}"
    syn = f"{row.get('Synergy_Score', 0):.2f}"
    res = f"{row.get('Resistance_Score', 0):.2f}"
    cov = str(row.get("Path_Coverage", ""))
    drug = str(row.get("Druggable_Count", ""))
    lines.append(f"  {ct:48s} {triple:40s} {syn:>5s} {res:>5s} {cov:>6s} {drug:>4s}")

lines.append("")

# ============================================================================
# 10. Pathway Analysis
# ============================================================================

lines.append("-" * 80)
lines.append("PATHWAY COVERAGE ANALYSIS")
lines.append("-" * 80)
lines.append("")

if "Pathways_Covered" in v3_triples.columns:
    pw = v3_triples["Pathways_Covered"]
    lines.append(f"Pathways covered per triple:")
    lines.append(f"  Median: {pw.median():.0f}   Mean: {pw.mean():.1f}   "
                 f"Min: {pw.min()}   Max: {pw.max()}")
    lines.append("")

# ============================================================================
# 11. Quality Metrics
# ============================================================================

lines.append("-" * 80)
lines.append("QUALITY & CONFIDENCE METRICS")
lines.append("-" * 80)
lines.append("")

# Log-based metrics
try:
    with open("results/pipeline_run_v3.log") as f:
        log = f.read()
    
    # Count raw CV usage
    raw_cv_line = [l for l in log.split("\n") if "Loaded raw replicate CV" in l]
    if raw_cv_line:
        lines.append(f"Raw replicate CV: {raw_cv_line[0].split('INFO:')[-1].strip()}")
    
    # UniProt resolution
    uniprot_lines = [l for l in log.split("\n") if "UniProt resolution complete" in l]
    if uniprot_lines:
        # Parse the last one
        last = uniprot_lines[-1]
        lines.append(f"UniProt resolution (last): {last.split('INFO:')[-1].strip()}")
    
    # Count warnings
    n_warnings = log.count("WARNING")
    n_errors = log.count("ERROR")
    lines.append(f"Pipeline log: {n_warnings} warnings, {n_errors} errors")
    lines.append("")

except FileNotFoundError:
    lines.append("Pipeline log not found")
    lines.append("")

# Runtime
try:
    with open("results/pipeline_run_v3.log") as f:
        log_text = f.read()
    import re
    runtime_match = re.search(r'(\d+)/96 \[(\d+:\d+:\d+)', log_text)
    progress_match = re.search(r'96/96 \[(\d+:\d+:\d+)', log_text)
    if progress_match:
        lines.append(f"Total runtime: {progress_match.group(1)}")
    lines.append("")
except Exception:
    pass

# ============================================================================
# 12. Summary Statistics
# ============================================================================

lines.append("=" * 80)
lines.append("SUMMARY")
lines.append("=" * 80)
lines.append("")

lines.append(f"Pipeline version:           v3 (raw replicate CV, 6-layer protein scoring)")
lines.append(f"Cancer types:               {n_cancers}")
lines.append(f"Triples identified:         {n_with_triples}")
lines.append(f"Unique targets:             {len(target_counts)}")
lines.append(f"Unique triple combos:       {len(combo_counts)}")
lines.append(f"Most common target:         {target_counts.most_common(1)[0][0]} ({target_counts.most_common(1)[0][1]} cancers)")
lines.append(f"Most common triple:         {combo_counts.most_common(1)[0][0]} ({combo_counts.most_common(1)[0][1]} cancers)")
lines.append(f"Unique drugs:               {len(drug_counts)}")
if v1_available:
    lines.append(f"v1→v3 triple stability:     {same_triple}/{len(shared)} identical ({same_triple/max(len(shared),1)*100:.0f}%)")
lines.append("")

# Write report
report = "\n".join(lines)
print(report)

with open("results/v3_analysis_report.txt", "w") as f:
    f.write(report)

print(f"\nReport saved to: results/v3_analysis_report.txt")
if v1_available:
    print(f"Comparison CSV:  results/v1_vs_v3_comparison.csv")
