#!/usr/bin/env python3
"""
DrugComb v1.5 Synergy Validation for ALIN Framework
====================================================

Tests the hypothesis: ALIN-predicted target pairs exhibit significantly higher
drug combination synergy scores than random target pairs in tissue-matched
cancer cell lines.

Uses DrugComb v1.5 (739K combination rows) to independently validate ALIN's
triple combination predictions across 17 primary cancer types.
"""

import csv
import json
import os
import random
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np
from scipy import stats

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).parent
DRUGCOMB_PATH = BASE_DIR / "synergy_data" / "drugcomb_summary_v1.5.csv"
RESULTS_DIR = BASE_DIR / "synergy_validation_results"
RESULTS_DIR.mkdir(exist_ok=True)

# Synergy metric to use (options: synergy_zip, synergy_bliss, synergy_loewe, synergy_hsa)
PRIMARY_METRIC = "synergy_zip"
ALL_METRICS = ["synergy_zip", "synergy_bliss", "synergy_loewe", "synergy_hsa"]

N_PERMUTATIONS = 10000
RANDOM_SEED = 42

# ============================================================================
# DRUG → GENE TARGET MAPPING (from gold_standard.py, extended)
# ============================================================================

DRUG_TO_GENE: Dict[str, str] = {
    # EGFR family
    'erlotinib': 'EGFR', 'gefitinib': 'EGFR', 'osimertinib': 'EGFR',
    'afatinib': 'EGFR', 'cetuximab': 'EGFR', 'panitumumab': 'EGFR',
    'lapatinib': 'EGFR',
    'trastuzumab': 'ERBB2', 'pertuzumab': 'ERBB2', 'tucatinib': 'ERBB2',
    # BRAF / MEK
    'vemurafenib': 'BRAF', 'dabrafenib': 'BRAF', 'encorafenib': 'BRAF',
    'trametinib': 'MAP2K1', 'cobimetinib': 'MAP2K1', 'binimetinib': 'MAP2K1',
    'avutometinib': 'MAP2K1',
    'sorafenib': 'BRAF',  # multi-kinase: BRAF, VEGFR, PDGFR
    # KRAS
    'sotorasib': 'KRAS', 'adagrasib': 'KRAS',
    # CDK
    'palbociclib': 'CDK4', 'ribociclib': 'CDK4', 'abemaciclib': 'CDK4',
    'dinaciclib': 'CDK2',
    # mTOR / PI3K / AKT
    'everolimus': 'MTOR', 'temsirolimus': 'MTOR',
    'alpelisib': 'PIK3CA', 'idelalisib': 'PIK3CD', 'copanlisib': 'PIK3CA',
    'gedatolisib': 'PIK3CA',
    'capivasertib': 'AKT1', 'ipatasertib': 'AKT1',
    # BCL2 family
    'venetoclax': 'BCL2', 'navitoclax': 'BCL2',
    # Note: navitoclax also hits BCL2L1/BCL-xL — we handle in MULTI_TARGET
    # FLT3
    'midostaurin': 'FLT3', 'gilteritinib': 'FLT3',
    # MET
    'capmatinib': 'MET', 'tepotinib': 'MET', 'savolitinib': 'MET',
    'crizotinib': 'ALK',
    'cabozantinib': 'MET',
    # VEGFR
    'lenvatinib': 'KDR', 'sunitinib': 'KDR', 'axitinib': 'KDR',
    'cediranib': 'KDR', 'bevacizumab': 'VEGFA',
    # ALK / ROS1 / RET
    'alectinib': 'ALK', 'brigatinib': 'ALK', 'lorlatinib': 'ALK',
    'entrectinib': 'NTRK1',
    'selpercatinib': 'RET', 'pralsetinib': 'RET',
    # FGFR
    'erdafitinib': 'FGFR1', 'pemigatinib': 'FGFR2', 'futibatinib': 'FGFR2',
    'infigratinib': 'FGFR2',
    # PARP
    'olaparib': 'PARP1', 'rucaparib': 'PARP1', 'niraparib': 'PARP1',
    'talazoparib': 'PARP1',
    # JAK / STAT
    'ruxolitinib': 'JAK2', 'fedratinib': 'JAK2', 'baricitinib': 'JAK1',
    'napabucasin': 'STAT3',
    # SRC
    'dasatinib': 'SRC', 'bosutinib': 'SRC',
    # BTK
    'ibrutinib': 'BTK', 'acalabrutinib': 'BTK', 'zanubrutinib': 'BTK',
    # Hormone therapy
    'fulvestrant': 'ESR1', 'tamoxifen': 'ESR1',
    'letrozole': 'CYP19A1', 'anastrozole': 'CYP19A1', 'exemestane': 'CYP19A1',
    'abiraterone': 'CYP17A1', 'enzalutamide': 'AR',
    # FAK
    'defactinib': 'PTK2',
    # Menin
    'revumenib': 'MEN1', 'ziftomenib': 'MEN1',
    # SHP2
    'tno155': 'PTPN11', 'rmc-4630': 'PTPN11',
    # IDH
    'ivosidenib': 'IDH1', 'enasidenib': 'IDH2',
    # MDM2
    'milademetan': 'MDM2',
}

# Multi-target drugs: each maps to a set of gene targets
MULTI_TARGET_DRUGS: Dict[str, Set[str]] = {
    'lapatinib': {'EGFR', 'ERBB2'},
    'afatinib': {'EGFR', 'ERBB2', 'ERBB4'},
    'lenvatinib': {'KDR', 'FGFR1'},
    'cabozantinib': {'MET', 'KDR', 'RET', 'AXL'},
    'crizotinib': {'ALK', 'MET', 'ROS1'},
    'avutometinib': {'BRAF', 'MAP2K1'},
    'sorafenib': {'BRAF', 'KDR'},
    'dasatinib': {'SRC', 'ABL1', 'FYN', 'YES1'},  # SRC-family kinase inhibitor
    'navitoclax': {'BCL2', 'BCL2L1'},
    'palbociclib': {'CDK4', 'CDK6'},
    'ribociclib': {'CDK4', 'CDK6'},
    'abemaciclib': {'CDK4', 'CDK6'},
    'dinaciclib': {'CDK2', 'CDK1', 'CDK5', 'CDK9'},
    'bosutinib': {'SRC', 'ABL1'},
}


def get_drug_targets(drug_name: str) -> Set[str]:
    """Return all gene targets for a drug (primary + multi-target)."""
    drug_lower = drug_name.lower().strip()
    targets = set()
    if drug_lower in DRUG_TO_GENE:
        targets.add(DRUG_TO_GENE[drug_lower])
    if drug_lower in MULTI_TARGET_DRUGS:
        targets.update(MULTI_TARGET_DRUGS[drug_lower])
    return targets


# ============================================================================
# ALIN PREDICTIONS (17 primary cancer types)
# ============================================================================

ALIN_PREDICTIONS = {
    'Acute Myeloid Leukemia': ('CDK6', 'MCL1', 'MET'),
    'Esophagogastric Adenocarcinoma': ('CDK6', 'EGFR', 'PIK3CA'),
    'Colorectal Adenocarcinoma': ('EGFR', 'KRAS', 'MET'),
    'Pancreatic Adenocarcinoma': ('FYN', 'KRAS', 'STAT3'),
    'Melanoma': ('BRAF', 'CDK6', 'EGFR'),
    'Non-Small Cell Lung Cancer': ('CDK6', 'EGFR', 'MAP2K1'),
    'Anaplastic Thyroid Cancer': ('CDK6', 'EGFR', 'MAP2K1'),
    'Bladder Urothelial Carcinoma': ('CDK2', 'EGFR', 'MAP2K1'),
    'Diffuse Glioma': ('CDK2', 'EGFR', 'MAP2K1'),
    'Hepatocellular Carcinoma': ('BCL2L1', 'EGFR', 'MAP2K1'),
    'Invasive Breast Carcinoma': ('BCL2L1', 'EGFR', 'MAP2K1'),
    'Prostate Adenocarcinoma': ('CDK2', 'EGFR', 'MAP2K1'),
    'Renal Cell Carcinoma': ('CDK6', 'EGFR', 'MAP2K1'),
    'Head and Neck Squamous Cell Carcinoma': ('CDK4', 'CDK6', 'ERBB2'),
    'Endometrial Carcinoma': ('CDK2', 'EGFR', 'MET'),
    'Ovarian Epithelial Tumor': ('CDK6', 'EGFR', 'MET'),
    'Liposarcoma': ('EGFR', 'FGFR1', 'MET'),
}

# Map ALIN cancer type → DrugComb tissue_name(s)
CANCER_TO_TISSUE: Dict[str, List[str]] = {
    'Acute Myeloid Leukemia': ['haematopoietic_and_lymphoid'],
    'Esophagogastric Adenocarcinoma': ['stomach'],
    'Colorectal Adenocarcinoma': ['large_intestine'],
    'Pancreatic Adenocarcinoma': ['pancreas'],
    'Melanoma': ['skin'],
    'Non-Small Cell Lung Cancer': ['lung'],
    'Anaplastic Thyroid Cancer': [],  # No thyroid in DrugComb
    'Bladder Urothelial Carcinoma': ['urinary_tract'],
    'Diffuse Glioma': ['brain'],
    'Hepatocellular Carcinoma': ['liver'],
    'Invasive Breast Carcinoma': ['breast'],
    'Prostate Adenocarcinoma': ['prostate'],
    'Renal Cell Carcinoma': ['kidney'],
    'Head and Neck Squamous Cell Carcinoma': [],  # No HNSCC-specific tissue
    'Endometrial Carcinoma': ['endometrium'],
    'Ovarian Epithelial Tumor': ['ovary'],
    'Liposarcoma': ['soft_tissue', 'bone'],
}


def get_predicted_pairs(cancer_type: str) -> List[FrozenSet[str]]:
    """Get all 3 pairwise combinations from an ALIN triple prediction."""
    triple = ALIN_PREDICTIONS[cancer_type]
    return [frozenset(pair) for pair in combinations(triple, 2)]


# ============================================================================
# LOAD AND PROCESS DRUGCOMB DATA
# ============================================================================

def load_drugcomb() -> List[dict]:
    """Load DrugComb v1.5 combination rows with both drugs mapped to gene targets."""
    print(f"Loading DrugComb v1.5 from {DRUGCOMB_PATH}...")
    rows = []
    unmapped_drugs = set()
    mapped_drugs = set()
    n_total = 0
    n_combos = 0
    n_mapped = 0

    with open(DRUGCOMB_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            n_total += 1
            drug_row = row.get('drug_row', '').strip()
            drug_col = row.get('drug_col', '').strip()

            # Skip monotherapy rows
            if not drug_row or not drug_col:
                continue
            n_combos += 1

            # Map drugs to gene targets
            targets_row = get_drug_targets(drug_row)
            targets_col = get_drug_targets(drug_col)

            if not targets_row:
                unmapped_drugs.add(drug_row)
            else:
                mapped_drugs.add(drug_row)
            if not targets_col:
                unmapped_drugs.add(drug_col)
            else:
                mapped_drugs.add(drug_col)

            # Only keep rows where BOTH drugs map to known gene targets
            if not targets_row or not targets_col:
                continue

            # Parse synergy scores
            synergy = {}
            for metric in ALL_METRICS:
                val = row.get(metric, '')
                try:
                    synergy[metric] = float(val)
                except (ValueError, TypeError):
                    synergy[metric] = None

            # Skip if primary metric has no value
            if synergy.get(PRIMARY_METRIC) is None:
                continue

            n_mapped += 1
            rows.append({
                'drug_row': drug_row,
                'drug_col': drug_col,
                'targets_row': targets_row,
                'targets_col': targets_col,
                'tissue': row.get('tissue_name', '').strip(),
                'cell_line': row.get('cell_line_name', '').strip(),
                'study': row.get('study_name', '').strip(),
                'block_id': row.get('block_id', ''),
                **synergy,
            })

    print(f"  Total rows: {n_total:,}")
    print(f"  Combination rows: {n_combos:,}")
    print(f"  Both-drugs-mapped rows: {n_mapped:,}")
    print(f"  Mapped drugs: {len(mapped_drugs)}")
    print(f"  Unmapped drugs: {len(unmapped_drugs)}")
    return rows


def compute_gene_pair_key(targets_row: Set[str], targets_col: Set[str]) -> List[FrozenSet[str]]:
    """Generate all possible gene-pair keys from two sets of drug targets."""
    pairs = []
    for g1 in targets_row:
        for g2 in targets_col:
            if g1 != g2:
                pairs.append(frozenset({g1, g2}))
    return pairs


# ============================================================================
# CORE ANALYSIS
# ============================================================================

def run_synergy_validation(drugcomb_rows: List[dict]) -> dict:
    """
    For each ALIN cancer type:
    1. Get predicted gene target pairs from triple
    2. Find DrugComb rows in matched tissue where drug targets match predicted pairs
    3. Compare synergy scores of predicted pairs vs ALL other pairs in same tissue
    4. Run permutation test
    """
    results = {}

    # Group DrugComb rows by tissue
    tissue_rows = defaultdict(list)
    for row in drugcomb_rows:
        tissue_rows[row['tissue']].append(row)

    print(f"\n{'='*80}")
    print("SYNERGY VALIDATION: ALIN-predicted pairs vs. random pairs")
    print(f"{'='*80}")
    print(f"Primary metric: {PRIMARY_METRIC}")
    print(f"Permutations: {N_PERMUTATIONS:,}")
    print()

    all_predicted_synergies = []
    all_random_synergies = []

    for cancer_type in sorted(ALIN_PREDICTIONS.keys()):
        tissues = CANCER_TO_TISSUE.get(cancer_type, [])
        if not tissues:
            print(f"\n--- {cancer_type}: No matching tissue in DrugComb, skipping ---")
            results[cancer_type] = {'status': 'no_tissue'}
            continue

        predicted_pairs = get_predicted_pairs(cancer_type)
        triple = ALIN_PREDICTIONS[cancer_type]

        # Collect all rows in matched tissues
        matched_rows = []
        for tissue in tissues:
            matched_rows.extend(tissue_rows.get(tissue, []))

        if not matched_rows:
            print(f"\n--- {cancer_type}: No DrugComb data for tissue(s) {tissues}, skipping ---")
            results[cancer_type] = {'status': 'no_data'}
            continue

        # Classify rows as predicted vs non-predicted
        predicted_synergies = []
        predicted_details = []
        nonpredicted_synergies = []

        for row in matched_rows:
            gene_pairs = compute_gene_pair_key(row['targets_row'], row['targets_col'])
            synergy_val = row[PRIMARY_METRIC]

            is_predicted = any(gp in predicted_pairs for gp in gene_pairs)
            if is_predicted:
                predicted_synergies.append(synergy_val)
                predicted_details.append({
                    'drug_row': row['drug_row'],
                    'drug_col': row['drug_col'],
                    'cell_line': row['cell_line'],
                    'synergy': synergy_val,
                    'gene_pairs': [str(gp) for gp in gene_pairs if gp in predicted_pairs],
                })
            else:
                nonpredicted_synergies.append(synergy_val)

        if not predicted_synergies:
            print(f"\n--- {cancer_type} ({', '.join(tissues)}): "
                  f"No DrugComb entries for predicted pairs {[set(p) for p in predicted_pairs]}, "
                  f"skipping ---")
            results[cancer_type] = {
                'status': 'no_predicted_pairs',
                'total_tissue_rows': len(matched_rows),
                'predicted_pairs': [list(p) for p in predicted_pairs],
            }
            continue

        # Compute statistics
        pred_arr = np.array(predicted_synergies)
        nonpred_arr = np.array(nonpredicted_synergies)
        all_arr = np.concatenate([pred_arr, nonpred_arr])

        observed_diff = np.mean(pred_arr) - np.mean(nonpred_arr)

        # Permutation test
        rng = np.random.RandomState(RANDOM_SEED)
        count_ge = 0
        n_pred = len(pred_arr)
        for _ in range(N_PERMUTATIONS):
            perm = rng.permutation(len(all_arr))
            perm_pred = all_arr[perm[:n_pred]]
            perm_nonpred = all_arr[perm[n_pred:]]
            perm_diff = np.mean(perm_pred) - np.mean(perm_nonpred)
            if perm_diff >= observed_diff:
                count_ge += 1
        p_value = (count_ge + 1) / (N_PERMUTATIONS + 1)

        # Mann-Whitney U test (non-parametric)
        if len(pred_arr) >= 2 and len(nonpred_arr) >= 2:
            mw_stat, mw_p = stats.mannwhitneyu(pred_arr, nonpred_arr, alternative='greater')
        else:
            mw_stat, mw_p = None, None

        # Effect size (Cohen's d)
        pooled_std = np.sqrt(
            ((len(pred_arr) - 1) * np.std(pred_arr, ddof=1)**2 +
             (len(nonpred_arr) - 1) * np.std(nonpred_arr, ddof=1)**2) /
            (len(pred_arr) + len(nonpred_arr) - 2)
        ) if len(pred_arr) > 1 and len(nonpred_arr) > 1 else 1.0
        cohens_d = observed_diff / pooled_std if pooled_std > 0 else 0.0

        # Accumulate for global analysis
        all_predicted_synergies.extend(predicted_synergies)
        all_random_synergies.extend(nonpredicted_synergies)

        results[cancer_type] = {
            'status': 'analyzed',
            'triple': list(triple),
            'tissues': tissues,
            'n_predicted': len(pred_arr),
            'n_nonpredicted': len(nonpred_arr),
            'mean_predicted': float(np.mean(pred_arr)),
            'std_predicted': float(np.std(pred_arr, ddof=1)) if len(pred_arr) > 1 else 0.0,
            'median_predicted': float(np.median(pred_arr)),
            'mean_nonpredicted': float(np.mean(nonpred_arr)),
            'std_nonpredicted': float(np.std(nonpred_arr, ddof=1)) if len(nonpred_arr) > 1 else 0.0,
            'median_nonpredicted': float(np.median(nonpred_arr)),
            'observed_diff': float(observed_diff),
            'cohens_d': float(cohens_d),
            'permutation_p': float(p_value),
            'mannwhitney_U': float(mw_stat) if mw_stat is not None else None,
            'mannwhitney_p': float(mw_p) if mw_p is not None else None,
            'top_synergistic_combos': sorted(predicted_details, key=lambda x: x['synergy'], reverse=True)[:10],
        }

        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
        direction = "↑" if observed_diff > 0 else "↓"

        print(f"\n{'─'*70}")
        print(f"  {cancer_type}")
        print(f"  Triple: {' + '.join(triple)}")
        print(f"  Tissue: {', '.join(tissues)}")
        print(f"  Predicted pairs: n={len(pred_arr)}, mean {PRIMARY_METRIC}={np.mean(pred_arr):.3f} "
              f"(±{np.std(pred_arr, ddof=1):.3f})" if len(pred_arr) > 1 else
              f"  Predicted pairs: n={len(pred_arr)}, {PRIMARY_METRIC}={np.mean(pred_arr):.3f}")
        print(f"  Non-predicted:   n={len(nonpred_arr)}, mean {PRIMARY_METRIC}={np.mean(nonpred_arr):.3f} "
              f"(±{np.std(nonpred_arr, ddof=1):.3f})" if len(nonpred_arr) > 1 else
              f"  Non-predicted:   n={len(nonpred_arr)}")
        print(f"  Δ = {observed_diff:+.3f} {direction}  Cohen's d = {cohens_d:.3f}")
        print(f"  Permutation p = {p_value:.4f} {sig}")
        if mw_p is not None:
            print(f"  Mann-Whitney U p = {mw_p:.4f}")

    # ========================================================================
    # GLOBAL ANALYSIS (pooled across all cancer types)
    # ========================================================================
    print(f"\n{'='*80}")
    print("GLOBAL POOLED ANALYSIS")
    print(f"{'='*80}")

    if all_predicted_synergies and all_random_synergies:
        pred_global = np.array(all_predicted_synergies)
        rand_global = np.array(all_random_synergies)
        global_diff = np.mean(pred_global) - np.mean(rand_global)

        # Global permutation test
        rng = np.random.RandomState(RANDOM_SEED + 1)
        all_global = np.concatenate([pred_global, rand_global])
        n_pred_global = len(pred_global)
        count_ge_global = 0
        for _ in range(N_PERMUTATIONS):
            perm = rng.permutation(len(all_global))
            perm_diff = np.mean(all_global[perm[:n_pred_global]]) - np.mean(all_global[perm[n_pred_global:]])
            if perm_diff >= global_diff:
                count_ge_global += 1
        global_p = (count_ge_global + 1) / (N_PERMUTATIONS + 1)

        # Global Mann-Whitney
        mw_global_stat, mw_global_p = stats.mannwhitneyu(
            pred_global, rand_global, alternative='greater'
        )

        # Global effect size
        pooled_std_g = np.sqrt(
            ((len(pred_global) - 1) * np.std(pred_global, ddof=1)**2 +
             (len(rand_global) - 1) * np.std(rand_global, ddof=1)**2) /
            (len(pred_global) + len(rand_global) - 2)
        )
        cohens_d_g = global_diff / pooled_std_g if pooled_std_g > 0 else 0.0

        sig_g = "***" if global_p < 0.001 else "**" if global_p < 0.01 else "*" if global_p < 0.05 else "ns"

        print(f"\n  ALIN-predicted pairs: n={len(pred_global)}, "
              f"mean {PRIMARY_METRIC} = {np.mean(pred_global):.3f} "
              f"(±{np.std(pred_global, ddof=1):.3f})")
        print(f"  Non-predicted pairs:  n={len(rand_global)}, "
              f"mean {PRIMARY_METRIC} = {np.mean(rand_global):.3f} "
              f"(±{np.std(rand_global, ddof=1):.3f})")
        print(f"\n  Δ = {global_diff:+.3f}  Cohen's d = {cohens_d_g:.3f}")
        print(f"  Permutation p = {global_p:.6f} {sig_g}")
        print(f"  Mann-Whitney U p = {mw_global_p:.6e}")

        results['__global__'] = {
            'n_predicted': len(pred_global),
            'n_nonpredicted': len(rand_global),
            'mean_predicted': float(np.mean(pred_global)),
            'std_predicted': float(np.std(pred_global, ddof=1)),
            'mean_nonpredicted': float(np.mean(rand_global)),
            'std_nonpredicted': float(np.std(rand_global, ddof=1)),
            'observed_diff': float(global_diff),
            'cohens_d': float(cohens_d_g),
            'permutation_p': float(global_p),
            'mannwhitney_U': float(mw_global_stat),
            'mannwhitney_p': float(mw_global_p),
            'n_cancer_types_analyzed': sum(
                1 for v in results.values()
                if isinstance(v, dict) and v.get('status') == 'analyzed'
            ),
        }
    else:
        print("  Insufficient data for global analysis.")

    return results


# ============================================================================
# MULTI-METRIC ANALYSIS
# ============================================================================

def run_multimetric(drugcomb_rows: List[dict]) -> dict:
    """Run the analysis across all 4 synergy metrics for robustness."""
    global PRIMARY_METRIC
    metric_results = {}
    for metric in ALL_METRICS:
        PRIMARY_METRIC = metric
        print(f"\n{'#'*80}")
        print(f"# METRIC: {metric}")
        print(f"{'#'*80}")
        metric_results[metric] = run_synergy_validation(drugcomb_rows)

    # Summary table
    print(f"\n{'='*80}")
    print("MULTI-METRIC SUMMARY")
    print(f"{'='*80}")
    print(f"\n{'Metric':<20} {'Δ (pred-rand)':>15} {'Cohen d':>10} {'Perm p':>12} {'MW p':>15} {'n_pred':>8} {'n_types':>8}")
    print("─" * 90)
    for metric in ALL_METRICS:
        g = metric_results[metric].get('__global__', {})
        if g:
            print(f"{metric:<20} {g['observed_diff']:>+15.3f} {g['cohens_d']:>10.3f} "
                  f"{g['permutation_p']:>12.4f} {g['mannwhitney_p']:>15.2e} "
                  f"{g['n_predicted']:>8} {g['n_cancer_types_analyzed']:>8}")

    return metric_results


# ============================================================================
# PAIR-LEVEL ANALYSIS
# ============================================================================

def run_pair_level_analysis(drugcomb_rows: List[dict]) -> dict:
    """
    Additional analysis: for each specific ALIN-predicted gene pair (e.g., EGFR+MAP2K1),
    compute its mean synergy across ALL relevant tissues and compare to the overall
    background rate.
    """
    print(f"\n{'='*80}")
    print("PAIR-LEVEL ANALYSIS (across all tissues)")
    print(f"{'='*80}")

    # Collect all unique predicted pairs
    all_predicted_pairs = set()
    for cancer_type in ALIN_PREDICTIONS:
        for pair in get_predicted_pairs(cancer_type):
            all_predicted_pairs.add(pair)

    # Compute synergy per gene pair across all data
    pair_synergies = defaultdict(list)
    for row in drugcomb_rows:
        gene_pairs = compute_gene_pair_key(row['targets_row'], row['targets_col'])
        synergy_val = row.get(PRIMARY_METRIC)
        if synergy_val is None:
            continue
        for gp in gene_pairs:
            pair_synergies[gp].append(synergy_val)

    # Background rate (all gene pairs)
    all_synergies = [v for vals in pair_synergies.values() for v in vals]
    background_mean = np.mean(all_synergies) if all_synergies else 0.0

    print(f"\nBackground mean {PRIMARY_METRIC}: {background_mean:.3f} (n={len(all_synergies):,})")
    print(f"\n{'Gene Pair':<30} {'Mean Syn':>10} {'Median':>10} {'n':>8} {'vs bg':>10} {'Predicted':>10}")
    print("─" * 80)

    pair_results = []
    for pair in sorted(pair_synergies.keys(), key=lambda p: np.mean(pair_synergies[p]), reverse=True):
        vals = pair_synergies[pair]
        if len(vals) < 3:
            continue
        pair_name = ' + '.join(sorted(pair))
        is_predicted = pair in all_predicted_pairs
        mean_syn = np.mean(vals)
        median_syn = np.median(vals)
        pair_results.append({
            'pair': pair_name,
            'mean': float(mean_syn),
            'median': float(median_syn),
            'n': len(vals),
            'is_predicted': is_predicted,
        })
        tag = "★ ALIN" if is_predicted else ""
        print(f"{pair_name:<30} {mean_syn:>10.3f} {median_syn:>10.3f} {len(vals):>8} "
              f"{mean_syn - background_mean:>+10.3f} {tag:>10}")

    # Summary: predicted vs non-predicted pair means
    pred_means = [r['mean'] for r in pair_results if r['is_predicted']]
    nonpred_means = [r['mean'] for r in pair_results if not r['is_predicted']]
    if pred_means and nonpred_means:
        print(f"\nPredicted pairs mean-of-means: {np.mean(pred_means):.3f} (n={len(pred_means)} pairs)")
        print(f"Non-predicted pairs mean-of-means: {np.mean(nonpred_means):.3f} (n={len(nonpred_means)} pairs)")
        print(f"Δ = {np.mean(pred_means) - np.mean(nonpred_means):+.3f}")

    return {
        'background_mean': float(background_mean),
        'n_total': len(all_synergies),
        'pair_results': pair_results[:50],  # top 50 for JSON
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("DrugComb v1.5 Synergy Validation for ALIN Framework")
    print("=" * 60)

    if not DRUGCOMB_PATH.exists():
        print(f"ERROR: DrugComb file not found at {DRUGCOMB_PATH}")
        print("  Please download from: https://zenodo.org/records/15235991")
        sys.exit(1)

    # Load data
    drugcomb_rows = load_drugcomb()

    if not drugcomb_rows:
        print("ERROR: No usable rows loaded from DrugComb.")
        sys.exit(1)

    # Reset primary metric
    global PRIMARY_METRIC
    PRIMARY_METRIC = "synergy_zip"

    # Run primary validation
    results = run_synergy_validation(drugcomb_rows)

    # Run pair-level analysis
    pair_results = run_pair_level_analysis(drugcomb_rows)

    # Run multi-metric analysis
    metric_results = run_multimetric(drugcomb_rows)

    # Save results
    output = {
        'primary_validation': results,
        'pair_level': pair_results,
        'multimetric_global': {
            metric: metric_results[metric].get('__global__', {})
            for metric in ALL_METRICS
        },
        'config': {
            'primary_metric': 'synergy_zip',
            'n_permutations': N_PERMUTATIONS,
            'random_seed': RANDOM_SEED,
            'drugcomb_version': 'v1.5',
            'n_drugcomb_rows': len(drugcomb_rows),
        },
    }

    outfile = RESULTS_DIR / "synergy_validation_results.json"
    with open(outfile, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {outfile}")

    # Also save a summary CSV
    csv_file = RESULTS_DIR / "synergy_validation_summary.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'cancer_type', 'triple', 'tissue', 'n_predicted', 'n_nonpredicted',
            'mean_predicted', 'mean_nonpredicted', 'delta', 'cohens_d',
            'permutation_p', 'mannwhitney_p', 'significant',
        ])
        for cancer_type in sorted(ALIN_PREDICTIONS.keys()):
            r = results.get(cancer_type, {})
            if r.get('status') != 'analyzed':
                writer.writerow([cancer_type, '', '', '', '', '', '', '', '', '', '', r.get('status', 'N/A')])
                continue
            sig = r['permutation_p'] < 0.05
            writer.writerow([
                cancer_type,
                ' + '.join(r['triple']),
                ', '.join(r['tissues']),
                r['n_predicted'],
                r['n_nonpredicted'],
                f"{r['mean_predicted']:.4f}",
                f"{r['mean_nonpredicted']:.4f}",
                f"{r['observed_diff']:+.4f}",
                f"{r['cohens_d']:.4f}",
                f"{r['permutation_p']:.6f}",
                f"{r['mannwhitney_p']:.6e}" if r['mannwhitney_p'] is not None else 'N/A',
                'Yes' if sig else 'No',
            ])
    print(f"Summary CSV saved to {csv_file}")

    return results


if __name__ == '__main__':
    main()
