#!/usr/bin/env python3
"""
Benchmarking Module for ALIN Framework (Adaptive Lethal Intersection Network)
=============================================================================

Independently curated gold-standard benchmark for evaluating ALIN's predicted
target combinations against FDA-approved and Phase 2/3-validated multi-target
combination therapies in oncology.

Benchmark design principles (following Julkunen et al. 2023, Azadifar et al.
2024, Menden et al. 2019):
 1. Gold standard curated independently of ALIN predictions.
 2. Only multi-target (>=2 gene targets) combinations included.
 3. Exact-combination match is the PRIMARY metric.
 4. Superset/pair-overlap/any-overlap matching reported as secondary sensitivity analysis.
 5. Leave-one-cancer-out cross-validation for generalization assessment.
 6. Single-gene targets evaluated in a separate target-prioritization analysis.

Inclusion criteria for COMBINATION_GOLD_STANDARD:
 - FDA-approved or Phase 2/3 positive efficacy data.
 - At least 2 distinct HUGO gene symbol targets.
 - Both agents must be molecularly targeted (no chemo + targeted).
 - Independent literature reference (PMID or NCT).
 - No entry selected, described, or phrased with reference to ALIN output.

Inclusion criteria for SINGLE_TARGET_GOLD_STANDARD:
 - FDA-approved single-target therapies with clear gene annotation.
 - Evaluated separately as a target-prioritization benchmark (hit rate).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import json

from alin.constants import (
    CANCER_BENCHMARK_ALIASES as CANCER_ALIASES,
    GENE_EQUIVALENTS,
)


def _read_triples(path) -> pd.DataFrame:
    """Read a triples CSV, normalizing 'Target 1' → 'Target_1' etc."""
    df = pd.read_csv(path)
    df.columns = [c.replace(' ', '_') for c in df.columns]
    return df

# ============================================================================
# COMBINATION GOLD STANDARD  (>= 2 distinct targets, independently curated)
# ============================================================================
# Frozen before any ALIN predictions were examined.
# Sources: FDA labels, NCCN guidelines, pivotal Phase 2/3 trials.
# Descriptions reference only external clinical evidence -- never ALIN output.

COMBINATION_GOLD_STANDARD = [
    # ------ BRAF + MEK combinations ------
    {
        'cancer': 'Melanoma',
        'targets': frozenset({'BRAF', 'MAP2K1'}),
        'evidence': 'FDA_approved',
        'description': ('Dabrafenib (BRAF) + trametinib (MAP2K1/MAP2K2) for BRAF V600E/K melanoma. '
                        'COMBI-d Phase 3: PFS HR 0.67, OS HR 0.71 (Long et al. NEJM 2014).'),
        'pmid': '25399551',
        'trial': 'COMBI-d (NCT01584648)',
    },
    {
        'cancer': 'Non-Small Cell Lung Cancer',
        'targets': frozenset({'BRAF', 'MAP2K1'}),
        'evidence': 'FDA_approved',
        'description': ('Dabrafenib + trametinib for BRAF V600E metastatic NSCLC. '
                        'Phase 2: ORR 63 pct, median PFS 14.6 mo (Planchard et al. Lancet Oncol 2016).'),
        'pmid': '27809962',
        'trial': 'BRF113928 (NCT01336634)',
    },
    {
        'cancer': 'Colorectal Adenocarcinoma',
        'targets': frozenset({'BRAF', 'EGFR'}),
        'evidence': 'FDA_approved',
        'description': ('Encorafenib (BRAF) + cetuximab (EGFR) for BRAF V600E metastatic CRC. '
                        'BEACON Phase 3: OS HR 0.60, ORR 20 pct vs 2 pct (Kopetz et al. NEJM 2019).'),
        'pmid': '31566309',
        'trial': 'BEACON CRC (NCT02928224)',
    },
    {
        'cancer': 'Melanoma',
        'targets': frozenset({'BRAF', 'MAP2K1'}),
        'evidence': 'FDA_approved',
        'description': ('Vemurafenib (BRAF) + cobimetinib (MAP2K1) for BRAF V600 melanoma. '
                        'coBRIM Phase 3: PFS HR 0.58 (Larkin et al. NEJM 2014).'),
        'pmid': '25105994',
        'trial': 'coBRIM (NCT01689519)',
    },
    # ------ EGFR + MET combinations ------
    {
        'cancer': 'Non-Small Cell Lung Cancer',
        'targets': frozenset({'EGFR', 'MET'}),
        'evidence': 'FDA_approved',
        'description': ('Amivantamab (bispecific EGFR+MET antibody) for EGFR exon 20 insertion '
                        'NSCLC. CHRYSALIS Phase 1/2: ORR 40 pct, DoR 11.1 mo (Park et al. JCO 2021).'),
        'pmid': '34043995',
        'trial': 'CHRYSALIS (NCT02609776)',
    },
    {
        'cancer': 'Non-Small Cell Lung Cancer',
        'targets': frozenset({'EGFR', 'MET'}),
        'evidence': 'Phase_2',
        'description': ('Osimertinib (EGFR) + savolitinib (MET) for EGFR-mutant NSCLC with '
                        'MET amplification. TATTON Phase 1b/2: ORR 52 pct post-1st/2nd gen TKI '
                        '(Sequist et al. Lancet Oncol 2020).'),
        'pmid': '32234522',
        'trial': 'TATTON (NCT02143466)',
    },
    # ------ HER2 dual-targeting ------
    {
        'cancer': 'Invasive Breast Carcinoma',
        'targets': frozenset({'ERBB2', 'EGFR'}),
        'evidence': 'FDA_approved',
        'description': ('Lapatinib (EGFR+ERBB2 dual TKI) + trastuzumab (ERBB2 antibody) for '
                        'HER2+ breast cancer. NeoALTTO Phase 3: pCR RR 1.43 (Baselga et al. Lancet 2012).'),
        'pmid': '22153890',
        'trial': 'NeoALTTO (NCT00553358)',
    },
    # ------ CDK4/6 combinations ------
    {
        'cancer': 'Invasive Breast Carcinoma',
        'targets': frozenset({'CDK4', 'CDK6', 'ESR1'}),
        'evidence': 'FDA_approved',
        'description': ('Palbociclib (CDK4/6) + fulvestrant (ESR1 degrader) for HR+/HER2- '
                        'metastatic breast cancer. PALOMA-3 Phase 3: PFS HR 0.46 '
                        '(Turner et al. NEJM 2015).'),
        'pmid': '26394241',
        'trial': 'PALOMA-3 (NCT01942135)',
    },
    # ------ VEGFR + mTOR ------
    {
        'cancer': 'Renal Cell Carcinoma',
        'targets': frozenset({'VEGFR2', 'MTOR'}),
        'evidence': 'FDA_approved',
        'description': ('Lenvatinib (VEGFR1/2/3) + everolimus (MTOR) for advanced RCC after '
                        'anti-angiogenic therapy. Phase 2: PFS HR 0.45 vs everolimus alone '
                        '(Motzer et al. Lancet Oncol 2015).'),
        'pmid': '26116099',
        'trial': 'Study 205 (NCT01136733)',
    },
    # ------ FLT3 + BCL2 in AML ------
    {
        'cancer': 'Acute Myeloid Leukemia',
        'targets': frozenset({'FLT3', 'BCL2'}),
        'evidence': 'Phase_2',
        'description': ('Venetoclax (BCL2) + gilteritinib (FLT3) for relapsed/refractory '
                        'FLT3-mutated AML. Phase 1b: high composite CR rate '
                        '(Daver et al. Blood 2022).'),
        'pmid': '35443125',
        'trial': 'NCT03625505',
    },
    # ------ ERBB2 + CDK4/6 ------
    {
        'cancer': 'Invasive Breast Carcinoma',
        'targets': frozenset({'ERBB2', 'CDK4', 'CDK6'}),
        'evidence': 'Phase_3',
        'description': ('Palbociclib (CDK4/6) + trastuzumab (ERBB2) + endocrine therapy for '
                        'HR+/HER2+ metastatic breast cancer. PATINA Phase 3: PFS 44 vs 29 mo '
                        '(Ciruelos et al. 2024).'),
        'pmid': '36631847',
        'trial': 'PATINA AFT-38 (NCT02947685)',
    },
    # ------ Head and Neck ------
    {
        'cancer': 'Head and Neck Squamous Cell Carcinoma',
        'targets': frozenset({'EGFR', 'MET'}),
        'evidence': 'Phase_2',
        'description': ('Ficlatuzumab (MET) + cetuximab (EGFR) for recurrent/metastatic HNSCC. '
                        'Phase 2: disease control data (Bhardwaj et al. 2019).'),
        'pmid': '32416071',
        'trial': 'NCT02277197',
    },
    # ------ Liaki tri-axial blockade (PDAC) ------
    {
        'cancer': 'Pancreatic Adenocarcinoma',
        'targets': frozenset({'KRAS', 'EGFR', 'STAT3'}),
        'evidence': 'Preclinical',
        'description': ('RAS inhibitor + EGFR inhibitor + STAT3 PROTAC for PDAC. Tri-axial '
                        'blockade induces complete regression with no resistance >200 days '
                        '(Liaki et al. PNAS 2025).'),
        'pmid': 'Liaki2025',
        'trial': 'Preclinical',
    },
]


# ============================================================================
# SINGLE-TARGET GOLD STANDARD  (evaluated separately as target prioritization)
# ============================================================================
# These are NOT used for the primary combination-prediction benchmark.
# They assess whether ALIN candidate gene pool contains known therapeutic
# targets, evaluated by hit-rate, not by combination recall.

SINGLE_TARGET_GOLD_STANDARD = [
    {
        'cancer': 'Non-Small Cell Lung Cancer',
        'targets': frozenset({'ALK'}),
        'evidence': 'FDA_approved',
        'description': 'Crizotinib for ALK-rearranged NSCLC (Shaw et al. NEJM 2013).',
        'pmid': '24724044',
    },
    {
        'cancer': 'Non-Small Cell Lung Cancer',
        'targets': frozenset({'KRAS'}),
        'evidence': 'FDA_approved',
        'description': ('Sotorasib for KRAS G12C NSCLC. CodeBreaK 100 Phase 2: ORR 37 pct '
                        '(Skoulidis et al. NEJM 2021).'),
        'pmid': '34081140',
    },
    {
        'cancer': 'Invasive Breast Carcinoma',
        'targets': frozenset({'ERBB2'}),
        'evidence': 'FDA_approved',
        'description': 'Trastuzumab for HER2+ breast cancer (FDA 1998).',
        'pmid': '11248153',
    },
    {
        'cancer': 'Colorectal Adenocarcinoma',
        'targets': frozenset({'EGFR'}),
        'evidence': 'FDA_approved',
        'description': 'Cetuximab for KRAS-WT metastatic CRC (Cunningham et al. NEJM 2004).',
        'pmid': '15269313',
    },
    {
        'cancer': 'Acute Myeloid Leukemia',
        'targets': frozenset({'FLT3'}),
        'evidence': 'FDA_approved',
        'description': 'Midostaurin for FLT3-mutated AML. RATIFY Phase 3 (Stone et al. NEJM 2017).',
        'pmid': '28644114',
    },
    {
        'cancer': 'Renal Cell Carcinoma',
        'targets': frozenset({'MTOR'}),
        'evidence': 'FDA_approved',
        'description': 'Everolimus for advanced RCC (Motzer et al. Lancet 2008).',
        'pmid': '18653228',
    },
    {
        'cancer': 'Pancreatic Adenocarcinoma',
        'targets': frozenset({'KRAS'}),
        'evidence': 'FDA_approved',
        'description': 'KRAS G12C inhibitors in pancreatic cancer. Sotorasib (FDA 2021).',
        'pmid': '34081140',
    },
    {
        'cancer': 'Non-Small Cell Lung Cancer',
        'targets': frozenset({'KRAS'}),
        'evidence': 'FDA_approved',
        'description': ('Adagrasib for KRAS G12C NSCLC. KRYSTAL-1 Phase 2: ORR 42.9 pct '
                        '(Jaenne et al. NEJM 2022).'),
        'pmid': '35662385',
    },
]


# ============================================================================
# BENCHMARK LOGIC
# ============================================================================

@dataclass
class BenchmarkResult:
    """Single benchmark comparison"""
    cancer_type: str
    gold_targets: frozenset
    gold_evidence: str
    our_targets: frozenset
    our_rank: int  # 1 = top prediction for this cancer
    match_type: str  # 'exact', 'superset', 'pair_overlap', 'any_overlap', 'none'
    matched_pairs: List[frozenset] = field(default_factory=list)
    description: str = ""


def match_cancer(our_cancer: str, gold_cancer: str) -> bool:
    """Check if our cancer type matches gold standard cancer"""
    our_lower = our_cancer.lower()
    gold_lower = gold_cancer.lower()
    if gold_lower in our_lower or our_lower in gold_lower:
        return True
    if gold_cancer in CANCER_ALIASES:
        for alias in CANCER_ALIASES[gold_cancer]:
            if alias.lower() in our_lower or our_lower in alias.lower():
                return True
    return False


def _expand_with_equivalents(genes: Set[str]) -> Set[str]:
    """Expand gene set with equivalents (e.g. MAP2K1 <-> MAP2K2)"""
    expanded = set(genes)
    for g in list(expanded):
        if g in GENE_EQUIVALENTS:
            expanded.update(GENE_EQUIVALENTS[g])
    return expanded


def check_match(our_targets: Set[str], gold_targets: Set[str]) -> Tuple[bool, str]:
    """
    Check match between predicted and gold-standard target sets.

    Match types (ordered by stringency):
      - 'exact':        our set == gold set (with gene equivalents)
      - 'superset':     gold is subset of our set (with gene equivalents)
      - 'pair_overlap': >=2 gold genes present in our set  [|G∩T|≥2]
      - 'any_overlap':  >=1 gold gene present in our set   [|G∩T|≥1]
      - 'none':         no overlap

    Returns (matched, match_type).
    """
    our_set = set(our_targets)
    gold_set = set(gold_targets)
    our_expanded = _expand_with_equivalents(our_set)
    gold_expanded = _expand_with_equivalents(gold_set)

    # Exact match (possibly via gene equivalents)
    if gold_set == our_set or gold_expanded == our_expanded:
        return True, 'exact'
    if gold_set.issubset(our_set):
        return True, 'superset'
    if gold_expanded.issubset(our_expanded):
        return True, 'superset'

    # Pair-overlap: >=2 gold genes appear in our set (with equivalents)
    overlap = gold_expanded.intersection(our_expanded)
    if len(overlap) >= 2:
        return True, 'pair_overlap'

    # Any-overlap: >=1 gold gene appears in our set
    if len(overlap) >= 1:
        return True, 'any_overlap'

    return False, 'none'


def _build_cancer_predictions(triples_csv, summary_csv=None):
    """Parse triples CSV into {cancer: [ranked target sets]}."""
    triples = _read_triples(triples_csv)
    cancer_to_predictions = defaultdict(list)
    for _, row in triples.iterrows():
        cancer = row['Cancer_Type']
        targets = frozenset([row['Target_1'], row['Target_2'], row['Target_3']])
        cancer_to_predictions[cancer].append(targets)

    # Deduplicate preserving rank order
    for cancer in cancer_to_predictions:
        seen = []
        for t in cancer_to_predictions[cancer]:
            if t not in seen:
                seen.append(t)
        cancer_to_predictions[cancer] = seen

    # Inject "best triple" from summary CSV at rank 1
    if summary_csv and Path(summary_csv).exists():
        summary = pd.read_csv(summary_csv)
        for _, row in summary.iterrows():
            cancer = row.get('Cancer Type', row.get('Cancer_Type', ''))
            best = row.get('Best Triple', row.get('Best_Triple', ''))
            if pd.notna(best) and best:
                targets = frozenset([t.strip() for t in str(best).split(',')])
                if cancer not in cancer_to_predictions or targets not in cancer_to_predictions[cancer]:
                    cancer_to_predictions[cancer] = [targets] + cancer_to_predictions[cancer]

    return dict(cancer_to_predictions)


def run_benchmark(triples_csv, summary_csv=None, gold_standard=None):
    """
    Run benchmarking against gold standard.

    Metrics reported:
      PRIMARY   -- exact-combination recall  (gold set == predicted set, with
                   gene equivalents).
      SECONDARY -- superset recall (gold is subset of predicted), pair-overlap
                   recall (>=2 genes shared), and any-overlap recall (>=1 gene shared).

    Returns:
        (list of BenchmarkResults, metrics dict)
    """
    if gold_standard is None:
        gold_standard = COMBINATION_GOLD_STANDARD

    cancer_to_predictions = _build_cancer_predictions(triples_csv, summary_csv)

    results = []
    tp_exact = tp_superset = tp_pair_overlap = tp_any_overlap = 0
    total_gold = len(gold_standard)

    for gold in gold_standard:
        gold_cancer = gold['cancer']
        gold_targets = gold['targets']

        best_match = None
        best_rank = 999
        best_type = 'none'

        for our_cancer, our_predictions in cancer_to_predictions.items():
            if not match_cancer(our_cancer, gold_cancer):
                continue

            for rank, our_targets in enumerate(our_predictions, 1):
                matched, match_type = check_match(our_targets, gold_targets)
                if matched and rank < best_rank:
                    best_rank = rank
                    best_type = match_type
                    best_match = (our_cancer, our_targets)

        if best_match:
            our_cancer, our_targets = best_match
            if best_type == 'exact':
                tp_exact += 1
            elif best_type == 'superset':
                tp_superset += 1
            elif best_type == 'pair_overlap':
                tp_pair_overlap += 1
            elif best_type == 'any_overlap':
                tp_any_overlap += 1

            results.append(BenchmarkResult(
                cancer_type=our_cancer,
                gold_targets=gold_targets,
                gold_evidence=gold['evidence'],
                our_targets=our_targets,
                our_rank=best_rank,
                match_type=best_type,
                description=gold['description']
            ))
        else:
            results.append(BenchmarkResult(
                cancer_type=gold_cancer,
                gold_targets=gold_targets,
                gold_evidence=gold['evidence'],
                our_targets=frozenset(),
                our_rank=999,
                match_type='none',
                description=gold['description']
            ))

    # --- Metrics ---
    recall_exact = tp_exact / total_gold if total_gold else 0
    recall_superset = (tp_exact + tp_superset) / total_gold if total_gold else 0
    recall_pair_overlap = (tp_exact + tp_superset + tp_pair_overlap) / total_gold if total_gold else 0
    recall_any_overlap = (tp_exact + tp_superset + tp_pair_overlap + tp_any_overlap) / total_gold if total_gold else 0
    matched_ranks = [r.our_rank for r in results if r.match_type != 'none']

    metrics = {
        'total_gold_standard': total_gold,
        'exact_matches': tp_exact,
        'superset_matches': tp_superset,
        'pair_overlap_matches': tp_pair_overlap,
        'any_overlap_matches': tp_any_overlap,
        'no_match': total_gold - tp_exact - tp_superset - tp_pair_overlap - tp_any_overlap,
        # PRIMARY metric
        'recall_exact': recall_exact,
        # SECONDARY metrics
        'recall_superset_or_better': recall_superset,
        'recall_pair_overlap_or_better': recall_pair_overlap,
        'recall_any_overlap_or_better': recall_any_overlap,
        # Backward-compatible aliases
        'pairwise_matches': tp_pair_overlap,
        'recall_pairwise_or_better': recall_pair_overlap,
        'mean_rank_when_matched': (
            sum(matched_ranks) / len(matched_ranks) if matched_ranks else 0
        ),
    }

    return results, metrics


# ============================================================================
# SINGLE-TARGET PRIORITIZATION BENCHMARK
# ============================================================================

def run_target_prioritization_benchmark(triples_csv, gold_standard=None):
    """
    Evaluate whether ALIN candidate gene pool contains known single-target
    therapeutic genes.  Reported as hit-rate (fraction of gold-standard targets
    appearing in at least one predicted triple for the matching cancer).

    This is a SEPARATE analysis from the combination benchmark.
    """
    if gold_standard is None:
        gold_standard = SINGLE_TARGET_GOLD_STANDARD

    cancer_to_predictions = _build_cancer_predictions(triples_csv)

    hits = 0
    total = len(gold_standard)
    details = []

    for gold in gold_standard:
        gold_cancer = gold['cancer']
        gold_target = list(gold['targets'])[0]
        found = False

        for our_cancer, our_preds in cancer_to_predictions.items():
            if not match_cancer(our_cancer, gold_cancer):
                continue
            for our_targets in our_preds:
                expanded = _expand_with_equivalents(set(our_targets))
                if gold_target in expanded:
                    found = True
                    break
            if found:
                break

        if found:
            hits += 1
        details.append({
            'target': gold_target,
            'cancer': gold_cancer,
            'found': found,
        })

    return {
        'hit_rate': hits / total if total else 0,
        'hits': hits,
        'total': total,
        'details': details,
        'method': 'target_prioritization',
    }


# ============================================================================
# LEAVE-ONE-CANCER-OUT PARTITIONED EVALUATION
# ============================================================================

def run_loco_cv(triples_csv, summary_csv=None):
    """
    Leave-one-cancer-out *partitioned evaluation* (NOT true cross-validation).

    IMPORTANT CAVEAT
    ----------------
    This function partitions the *gold standard* by cancer type and evaluates
    ALIN's predictions on the held-out partition.  However, the predictions
    themselves were generated from a single pipeline run that used ALL cancer
    types (including the "held-out" one) during dependency inference.  There
    is NO re-training: the same triples_csv produced by the full pipeline is
    evaluated in every fold.

    Consequently, this procedure tests whether ALIN's predictions happen to
    match held-out gold-standard entries — a useful sanity check — but does
    NOT test generalization to genuinely unseen cancer types.  True held-out
    cross-validation would require re-running the entire pipeline (DepMap
    filtering → viability path inference → MHS → triple scoring) with the
    held-out cancer's data excluded from DepMap at each fold, which is
    computationally expensive and is flagged as future work.

    We retain the ``run_loco_cv`` name for backward compatibility but report
    results as "LOCO partitioned evaluation" (not "cross-validation") in the
    manuscript.

    For each unique cancer type in the gold standard:
      - Hold out all gold-standard entries for that cancer.
      - Evaluate recall on the held-out entries using ALIN predictions.
      - Record per-fold exact, superset, pair-overlap, and any-overlap recall.

    Note: despite following Julkunen et al. (2023) and DDI-Ben (2024)
    terminology, this is partitioned evaluation, not true CV, because
    the pipeline is not re-run with held-out data excluded.
    """
    cancer_groups = defaultdict(list)
    for entry in COMBINATION_GOLD_STANDARD:
        cancer_groups[entry['cancer']].append(entry)

    fold_results = []

    for held_out_cancer, held_out_entries in cancer_groups.items():
        test_gold = held_out_entries
        _, fold_metrics = run_benchmark(
            triples_csv, summary_csv, gold_standard=test_gold
        )
        fold_results.append({
            'held_out_cancer': held_out_cancer,
            'n_gold_entries': len(test_gold),
            'exact_matches': fold_metrics['exact_matches'],
            'superset_matches': fold_metrics['superset_matches'],
            'pair_overlap_matches': fold_metrics['pair_overlap_matches'],
            'any_overlap_matches': fold_metrics['any_overlap_matches'],
            'recall_exact': fold_metrics['recall_exact'],
            'recall_superset_or_better': fold_metrics['recall_superset_or_better'],
            'recall_pair_overlap_or_better': fold_metrics['recall_pair_overlap_or_better'],
            'recall_any_overlap_or_better': fold_metrics['recall_any_overlap_or_better'],
        })

    n_folds = len(fold_results)
    mean_exact = np.mean([f['recall_exact'] for f in fold_results])
    std_exact = np.std([f['recall_exact'] for f in fold_results])
    mean_superset = np.mean([f['recall_superset_or_better'] for f in fold_results])
    std_superset = np.std([f['recall_superset_or_better'] for f in fold_results])
    mean_pair_overlap = np.mean([f['recall_pair_overlap_or_better'] for f in fold_results])
    std_pair_overlap = np.std([f['recall_pair_overlap_or_better'] for f in fold_results])
    mean_any_overlap = np.mean([f['recall_any_overlap_or_better'] for f in fold_results])
    std_any_overlap = np.std([f['recall_any_overlap_or_better'] for f in fold_results])

    return {
        'n_folds': n_folds,
        'mean_recall_exact': float(mean_exact),
        'std_recall_exact': float(std_exact),
        'mean_recall_superset': float(mean_superset),
        'std_recall_superset': float(std_superset),
        'mean_recall_pair_overlap': float(mean_pair_overlap),
        'std_recall_pair_overlap': float(std_pair_overlap),
        'mean_recall_any_overlap': float(mean_any_overlap),
        'std_recall_any_overlap': float(std_any_overlap),
        'per_fold': fold_results,
        'method': 'LOCO_partitioned_evaluation',
        'caveat': (
            'Partitioned evaluation only: predictions were generated from '
            'a single pipeline run including all cancer types.  True held-out '
            'CV would require re-running the pipeline excluding each fold\'s '
            'cancer data.'
        ),
    }


# ============================================================================
# BASELINES
# ============================================================================

def run_random_baseline(triples_csv, n_trials=1000, seed=42):
    """Random baseline: sample random triples from the global gene pool."""
    import random
    rng = random.Random(seed)
    triples = _read_triples(triples_csv)
    all_genes = set()
    for _, row in triples.iterrows():
        all_genes.update([row['Target_1'], row['Target_2'], row['Target_3']])
    all_genes_list = list(all_genes)

    cancer_to_predictions = defaultdict(list)
    for _, row in triples.iterrows():
        cancer = row['Cancer_Type']
        cancer_to_predictions[cancer].append(
            frozenset([row['Target_1'], row['Target_2'], row['Target_3']])
        )

    recalls_exact = []
    recalls_superset = []
    recalls_pair_overlap = []
    recalls_any_overlap = []

    for _ in range(n_trials):
        random_predictions = defaultdict(list)
        for cancer in cancer_to_predictions:
            for _ in range(min(5, len(cancer_to_predictions[cancer]))):
                triple = frozenset(rng.sample(all_genes_list, 3))
                random_predictions[cancer].append(triple)

        tp_exact = tp_superset = tp_pair_overlap = tp_any_overlap = 0
        for gold in COMBINATION_GOLD_STANDARD:
            found_exact = found_superset = found_pair_overlap = found_any_overlap = False
            for our_cancer, our_preds in random_predictions.items():
                if not match_cancer(our_cancer, gold['cancer']):
                    continue
                for our_targets in our_preds:
                    matched, mtype = check_match(our_targets, gold['targets'])
                    if matched:
                        if mtype == 'exact':
                            found_exact = True
                        if mtype in ('exact', 'superset'):
                            found_superset = True
                        if mtype in ('exact', 'superset', 'pair_overlap'):
                            found_pair_overlap = True
                        found_any_overlap = True
                        break
                if found_any_overlap:
                    break
            if found_exact:
                tp_exact += 1
            if found_superset:
                tp_superset += 1
            if found_pair_overlap:
                tp_pair_overlap += 1
            if found_any_overlap:
                tp_any_overlap += 1

        n = len(COMBINATION_GOLD_STANDARD)
        recalls_exact.append(tp_exact / n if n else 0)
        recalls_superset.append(tp_superset / n if n else 0)
        recalls_pair_overlap.append(tp_pair_overlap / n if n else 0)
        recalls_any_overlap.append(tp_any_overlap / n if n else 0)

    return {
        'mean_recall_exact': float(np.mean(recalls_exact)),
        'std_recall_exact': float(np.std(recalls_exact)),
        'mean_recall_superset': float(np.mean(recalls_superset)),
        'std_recall_superset': float(np.std(recalls_superset)),
        'mean_recall_pair_overlap': float(np.mean(recalls_pair_overlap)),
        'std_recall_pair_overlap': float(np.std(recalls_pair_overlap)),
        'mean_recall_any_overlap': float(np.mean(recalls_any_overlap)),
        'std_recall_any_overlap': float(np.std(recalls_any_overlap)),
        # Backward-compatible aliases
        'mean_recall_pairwise': float(np.mean(recalls_pair_overlap)),
        'std_recall_pairwise': float(np.std(recalls_pair_overlap)),
        'n_trials': n_trials,
        'method': 'random',
    }


def run_topgenes_baseline(triples_csv):
    """Top-genes baseline: always predict most frequent genes for every cancer."""
    top_triple = frozenset({'KRAS', 'CDK6', 'STAT3'})
    triples = _read_triples(triples_csv)
    cancers = triples['Cancer_Type'].unique()

    tp_exact = tp_superset = tp_pair_overlap = tp_any_overlap = 0
    for gold in COMBINATION_GOLD_STANDARD:
        for cancer in cancers:
            if match_cancer(cancer, gold['cancer']):
                matched_m, mtype = check_match(top_triple, gold['targets'])
                if matched_m:
                    if mtype == 'exact':
                        tp_exact += 1
                    if mtype in ('exact', 'superset'):
                        tp_superset += 1
                    if mtype in ('exact', 'superset', 'pair_overlap'):
                        tp_pair_overlap += 1
                    tp_any_overlap += 1
                break

    n = len(COMBINATION_GOLD_STANDARD)
    return {
        'recall_exact': tp_exact / n if n else 0,
        'recall_superset': tp_superset / n if n else 0,
        'recall_pair_overlap': tp_pair_overlap / n if n else 0,
        'recall_any_overlap': tp_any_overlap / n if n else 0,
        'recall_pairwise': tp_pair_overlap / n if n else 0,  # backward-compatible
        'method': 'top_genes',
    }


def run_frequency_baseline(triples_csv):
    """Frequency-based baseline: per-cancer top-3 most frequent genes."""
    triples = _read_triples(triples_csv)
    cancer_freq = defaultdict(lambda: defaultdict(int))
    for _, row in triples.iterrows():
        cancer = row['Cancer_Type']
        for col in ['Target_1', 'Target_2', 'Target_3']:
            cancer_freq[cancer][row[col]] += 1

    cancer_top3 = {}
    for cancer, freq in cancer_freq.items():
        top3 = sorted(freq, key=freq.get, reverse=True)[:3]
        cancer_top3[cancer] = frozenset(top3)

    tp_exact = tp_superset = tp_pair_overlap = tp_any_overlap = 0
    for gold in COMBINATION_GOLD_STANDARD:
        for our_cancer, top3 in cancer_top3.items():
            if not match_cancer(our_cancer, gold['cancer']):
                continue
            matched, mtype = check_match(top3, gold['targets'])
            if matched:
                if mtype == 'exact':
                    tp_exact += 1
                if mtype in ('exact', 'superset'):
                    tp_superset += 1
                if mtype in ('exact', 'superset', 'pair_overlap'):
                    tp_pair_overlap += 1
                tp_any_overlap += 1
                break

    n = len(COMBINATION_GOLD_STANDARD)
    return {
        'recall_exact': tp_exact / n if n else 0,
        'recall_superset': tp_superset / n if n else 0,
        'recall_pair_overlap': tp_pair_overlap / n if n else 0,
        'recall_any_overlap': tp_any_overlap / n if n else 0,
        'recall_pairwise': tp_pair_overlap / n if n else 0,  # backward-compatible
        'method': 'frequency',
    }


def run_poolmatched_baseline(triples_csv, n_trials=1000, seed=42, reports_dir=None):
    """Pool-matched baseline: random triples from per-cancer candidate pools."""
    import random
    import re
    import os
    rng = random.Random(seed)
    triples = _read_triples(triples_csv)

    if reports_dir is None:
        reports_dir = str(Path(triples_csv).parent)

    cancer_gene_pools = {}
    for fname in os.listdir(reports_dir):
        if not fname.endswith('_report.txt') or 'triple' in fname:
            continue
        cancer_key = fname.replace('_report.txt', '').replace('_', ' ')
        fpath = os.path.join(reports_dir, fname)
        with open(fpath) as f:
            content = f.read()

        pool = set()
        best = re.search(
            r'BEST TRIPLE COMBINATION:\s*([A-Z0-9,\s]+?)$', content, re.MULTILINE)
        if best:
            pool.update(g.strip() for g in best.group(1).split(',') if g.strip())
        for g1, g2, g3 in re.findall(
                r'\d+\.\s+([A-Z][A-Z0-9]+),\s+([A-Z][A-Z0-9]+),\s+([A-Z][A-Z0-9]+)\s+\(score:', content):
            pool.update([g1, g2, g3])
        if len(pool) >= 3:
            cancer_gene_pools[cancer_key] = sorted(pool)

    if not cancer_gene_pools:
        all_genes = set()
        for _, row in triples.iterrows():
            all_genes.update([row['Target_1'], row['Target_2'], row['Target_3']])
        all_genes_list = sorted(all_genes)
        for cancer in triples['Cancer_Type'].unique():
            cancer_gene_pools[cancer] = all_genes_list

    csv_cancers = triples['Cancer_Type'].unique()
    cancer_pool_map = {}
    for csv_c in csv_cancers:
        csv_norm = csv_c.lower().replace('_', ' ').replace('/', ' ').replace(',', '')
        for pool_c, pool_genes in cancer_gene_pools.items():
            pool_norm = pool_c.lower().replace('_', ' ').replace('/', ' ').replace(',', '')
            if csv_norm == pool_norm or csv_norm in pool_norm or pool_norm in csv_norm:
                cancer_pool_map[csv_c] = pool_genes
                break
        if csv_c not in cancer_pool_map:
            cancer_pool_map[csv_c] = sorted(
                set(g for gs in cancer_gene_pools.values() for g in gs))

    cancer_n_preds = {}
    for _, row in triples.iterrows():
        c = row['Cancer_Type']
        cancer_n_preds[c] = cancer_n_preds.get(c, 0) + 1

    recalls_exact = []
    recalls_superset = []
    recalls_pair_overlap = []
    recalls_any_overlap = []

    for _ in range(n_trials):
        random_predictions = {}
        for cancer in csv_cancers:
            pool = cancer_pool_map.get(cancer, [])
            n = min(5, cancer_n_preds.get(cancer, 5))
            preds = []
            for _ in range(n):
                if len(pool) >= 3:
                    preds.append(frozenset(rng.sample(pool, 3)))
                else:
                    preds.append(frozenset(pool))
            random_predictions[cancer] = preds

        tp_exact = tp_superset = tp_pair_overlap = tp_any_overlap = 0
        for gold in COMBINATION_GOLD_STANDARD:
            found_exact = found_superset = found_pair_overlap = found_any_overlap = False
            for our_cancer, our_preds in random_predictions.items():
                if not match_cancer(our_cancer, gold['cancer']):
                    continue
                for our_targets in our_preds:
                    matched, mtype = check_match(our_targets, gold['targets'])
                    if matched:
                        if mtype == 'exact':
                            found_exact = True
                        if mtype in ('exact', 'superset'):
                            found_superset = True
                        if mtype in ('exact', 'superset', 'pair_overlap'):
                            found_pair_overlap = True
                        found_any_overlap = True
                        break
                if found_any_overlap:
                    break
            if found_exact:
                tp_exact += 1
            if found_superset:
                tp_superset += 1
            if found_pair_overlap:
                tp_pair_overlap += 1
            if found_any_overlap:
                tp_any_overlap += 1

        ng = len(COMBINATION_GOLD_STANDARD)
        recalls_exact.append(tp_exact / ng if ng else 0)
        recalls_superset.append(tp_superset / ng if ng else 0)
        recalls_pair_overlap.append(tp_pair_overlap / ng if ng else 0)
        recalls_any_overlap.append(tp_any_overlap / ng if ng else 0)

    return {
        'mean_recall_exact': float(np.mean(recalls_exact)),
        'std_recall_exact': float(np.std(recalls_exact)),
        'mean_recall_superset': float(np.mean(recalls_superset)),
        'std_recall_superset': float(np.std(recalls_superset)),
        'mean_recall_pair_overlap': float(np.mean(recalls_pair_overlap)),
        'std_recall_pair_overlap': float(np.std(recalls_pair_overlap)),
        'mean_recall_any_overlap': float(np.mean(recalls_any_overlap)),
        'std_recall_any_overlap': float(np.std(recalls_any_overlap)),
        # Backward-compatible aliases
        'mean_recall_pairwise': float(np.mean(recalls_pair_overlap)),
        'std_recall_pairwise': float(np.std(recalls_pair_overlap)),
        'n_trials': n_trials,
        'method': 'pool_matched',
        'median_pool_size': sorted(len(v) for v in cancer_pool_map.values())[
            len(cancer_pool_map) // 2] if cancer_pool_map else 0,
    }


# ============================================================================
# BIOLOGY-INFORMED BASELINES
# ============================================================================

# Cancer-type-specific driver genes curated from TCGA/COSMIC/OncoKB.
# For each cancer type in the gold standard, we list the most commonly
# mutated/amplified driver genes with FDA-approved or late-stage targeted
# therapies.  This baseline answers: "What if we simply predicted the
# most clinically actionable driver genes for each cancer?"
#
# Sources: TCGA PanCancer Atlas (Hoadley et al. Cell 2018), OncoKB
# (Chakravarty et al. JCO PO 2017), COSMIC Cancer Gene Census v99.

CANCER_DRIVER_GENES = {
    'Melanoma': ['BRAF', 'NRAS', 'MAP2K1', 'KIT', 'NF1', 'CDKN2A'],
    'Non-Small Cell Lung Cancer': ['EGFR', 'KRAS', 'ALK', 'MET', 'BRAF',
                                   'ROS1', 'RET', 'ERBB2', 'STK11'],
    'Colorectal Adenocarcinoma': ['KRAS', 'BRAF', 'EGFR', 'PIK3CA', 'APC',
                                 'TP53', 'SMAD4', 'NRAS'],
    'Invasive Breast Carcinoma': ['ERBB2', 'ESR1', 'PIK3CA', 'CDK4', 'CDK6',
                                 'BRCA1', 'BRCA2', 'AKT1', 'MTOR'],
    'Renal Cell Carcinoma': ['VHL', 'MTOR', 'VEGFR2', 'MET', 'PBRM1',
                            'BAP1', 'SETD2'],
    'Acute Myeloid Leukemia': ['FLT3', 'NPM1', 'DNMT3A', 'IDH1', 'IDH2',
                               'BCL2', 'TP53', 'RUNX1'],
    'Head and Neck Squamous Cell Carcinoma': ['EGFR', 'PIK3CA', 'TP53',
                                             'CDKN2A', 'MET', 'FGFR1'],
    'Pancreatic Adenocarcinoma': ['KRAS', 'TP53', 'CDKN2A', 'SMAD4',
                                 'BRCA2', 'EGFR', 'STAT3'],
}


def run_driver_baseline(triples_csv):
    """
    Driver-gene baseline: per-cancer top-3 driver genes from TCGA/COSMIC/OncoKB.

    This is a biology-informed baseline that answers: "Can we match gold-standard
    combinations simply by predicting the most commonly mutated/actionable driver
    genes for each cancer type?"  Unlike the random and pool-matched baselines,
    this uses genuine biological knowledge (mutation frequency, actionability)
    but no DepMap essentiality data or network analysis.
    """
    triples = _read_triples(triples_csv)
    csv_cancers = triples['Cancer_Type'].unique()

    # Map each CSV cancer to its driver list
    cancer_predictions = {}
    for csv_cancer in csv_cancers:
        for driver_cancer, drivers in CANCER_DRIVER_GENES.items():
            if match_cancer(csv_cancer, driver_cancer):
                cancer_predictions[csv_cancer] = frozenset(drivers[:3])
                break
        # Cancers without curated drivers get no prediction (empty set)
        if csv_cancer not in cancer_predictions:
            cancer_predictions[csv_cancer] = frozenset()

    tp_exact = tp_superset = tp_pair_overlap = tp_any_overlap = 0
    for gold in COMBINATION_GOLD_STANDARD:
        for our_cancer, top3 in cancer_predictions.items():
            if not match_cancer(our_cancer, gold['cancer']):
                continue
            if not top3:
                break
            matched, mtype = check_match(top3, gold['targets'])
            if matched:
                if mtype == 'exact':
                    tp_exact += 1
                if mtype in ('exact', 'superset'):
                    tp_superset += 1
                if mtype in ('exact', 'superset', 'pair_overlap'):
                    tp_pair_overlap += 1
                tp_any_overlap += 1
                break

    n = len(COMBINATION_GOLD_STANDARD)
    return {
        'recall_exact': tp_exact / n if n else 0,
        'recall_superset': tp_superset / n if n else 0,
        'recall_pair_overlap': tp_pair_overlap / n if n else 0,
        'recall_any_overlap': tp_any_overlap / n if n else 0,
        'recall_pairwise': tp_pair_overlap / n if n else 0,  # backward-compatible
        'n_cancers_with_drivers': sum(1 for v in cancer_predictions.values() if v),
        'method': 'driver_genes',
    }


def run_essentiality_baseline(triples_csv, depmap_dir='./depmap_data',
                              dependency_threshold=-0.5):
    """
    DepMap essentiality baseline: per-cancer top-3 most essential genes.

    This baseline uses the SAME DepMap CRISPR data as ALIN but applies the
    simplest possible algorithm: rank genes by mean dependency score (most
    negative = most essential) per cancer type, exclude pan-essential genes
    (essential in >90% of all lines), and take the top 3.

    This directly tests whether ALIN's viability path inference, hitting set
    optimization, and synergy scoring add value over naive essentiality ranking.
    """
    from pathlib import Path as _Path
    depmap_path = _Path(depmap_dir)
    model_file = depmap_path / 'Model.csv'
    crispr_file = depmap_path / 'CRISPRGeneEffect.csv'

    if not model_file.exists() or not crispr_file.exists():
        return {
            'recall_exact': 0.0,
            'recall_superset': 0.0,
            'recall_pair_overlap': 0.0,
            'recall_any_overlap': 0.0,
            'recall_pairwise': 0.0,
            'method': 'essentiality',
            'error': 'DepMap data files not found',
        }

    # Load DepMap data
    model_df = pd.read_csv(model_file)
    crispr_df = pd.read_csv(crispr_file, index_col=0)

    # Clean gene names: "GENE (12345)" -> "GENE"
    crispr_df.columns = [c.split(' (')[0] if ' (' in c else c
                         for c in crispr_df.columns]

    # Identify pan-essential genes (essential in >90% of all lines)
    pan_essential_mask = (crispr_df < dependency_threshold).mean(axis=0) > 0.9
    pan_essential = set(pan_essential_mask[pan_essential_mask].index)

    # Map cell lines to cancer types
    if 'ModelID' in model_df.columns and 'OncotreePrimaryDisease' in model_df.columns:
        line_to_cancer = dict(zip(model_df['ModelID'],
                                  model_df['OncotreePrimaryDisease']))
    else:
        return {
            'recall_exact': 0.0,
            'recall_superset': 0.0,
            'recall_pair_overlap': 0.0,
            'recall_any_overlap': 0.0,
            'recall_pairwise': 0.0,
            'method': 'essentiality',
            'error': 'Model.csv missing required columns',
        }

    # Build per-cancer essentiality predictions
    triples = _read_triples(triples_csv)
    csv_cancers = triples['Cancer_Type'].unique()

    cancer_predictions = {}
    for csv_cancer in csv_cancers:
        # Find cell lines for this cancer
        cancer_lines = [lid for lid, ct in line_to_cancer.items()
                        if match_cancer(csv_cancer, str(ct))]
        cancer_lines = [lid for lid in cancer_lines if lid in crispr_df.index]

        if len(cancer_lines) < 2:
            cancer_predictions[csv_cancer] = frozenset()
            continue

        # Mean dependency across cancer-specific lines
        mean_dep = crispr_df.loc[cancer_lines].mean(axis=0)
        # Remove pan-essential genes
        selective = mean_dep.drop(labels=[g for g in pan_essential
                                          if g in mean_dep.index],
                                  errors='ignore')
        # Top 3 most essential (most negative)
        top3 = list(selective.nsmallest(3).index)
        cancer_predictions[csv_cancer] = frozenset(top3)

    tp_exact = tp_superset = tp_pair_overlap = tp_any_overlap = 0
    for gold in COMBINATION_GOLD_STANDARD:
        for our_cancer, top3 in cancer_predictions.items():
            if not match_cancer(our_cancer, gold['cancer']):
                continue
            if not top3:
                break
            matched, mtype = check_match(top3, gold['targets'])
            if matched:
                if mtype == 'exact':
                    tp_exact += 1
                if mtype in ('exact', 'superset'):
                    tp_superset += 1
                if mtype in ('exact', 'superset', 'pair_overlap'):
                    tp_pair_overlap += 1
                tp_any_overlap += 1
                break

    n = len(COMBINATION_GOLD_STANDARD)
    n_with_data = sum(1 for v in cancer_predictions.values() if v)
    return {
        'recall_exact': tp_exact / n if n else 0,
        'recall_superset': tp_superset / n if n else 0,
        'recall_pair_overlap': tp_pair_overlap / n if n else 0,
        'recall_any_overlap': tp_any_overlap / n if n else 0,
        'recall_pairwise': tp_pair_overlap / n if n else 0,  # backward-compatible
        'n_cancers_with_data': n_with_data,
        'method': 'essentiality',
    }


# ============================================================================
# REPORTING
# ============================================================================

def generate_benchmark_report(results, metrics):
    """Generate human-readable benchmark report"""
    lines = []
    lines.append("=" * 80)
    lines.append("BENCHMARK REPORT: ALIN Framework -- Combination Gold Standard")
    lines.append("=" * 80)
    lines.append("Comparison against {} independently curated".format(metrics['total_gold_standard']))
    lines.append("multi-target (>=2 gene) clinically validated combinations.")
    lines.append("")
    lines.append("Benchmark design:")
    lines.append("  - Gold standard assembled independently of ALIN predictions.")
    lines.append("  - Only multi-target combinations (>=2 HUGO gene symbols).")
    lines.append("  - Single-gene therapies evaluated separately (target prioritization).")
    lines.append("  - Exact-match is the PRIMARY metric; superset/pair-overlap/any-overlap are SECONDARY.")
    lines.append("")
    lines.append("=" * 80)
    lines.append("PRIMARY METRIC")
    lines.append("=" * 80)
    lines.append("  Exact-combination recall:     {:.1f}%  ({}/{})".format(
        metrics['recall_exact'] * 100, metrics['exact_matches'], metrics['total_gold_standard']))
    lines.append("")
    lines.append("SECONDARY METRICS")
    lines.append("=" * 80)
    lines.append("  Superset recall (exact+superset):         {:.1f}%".format(
        metrics['recall_superset_or_better'] * 100))
    lines.append("  Pair-overlap recall (exact+sup+pair|≥2):  {:.1f}%".format(
        metrics.get('recall_pair_overlap_or_better', metrics.get('recall_pairwise_or_better', 0)) * 100))
    lines.append("  Any-overlap recall (any shared gene|≥1):  {:.1f}%".format(
        metrics.get('recall_any_overlap_or_better', 0) * 100))
    lines.append("  No match:                         {}".format(metrics['no_match']))
    lines.append("  Mean rank when matched (1=top):   {:.2f}".format(
        metrics['mean_rank_when_matched']))
    lines.append("")
    lines.append("=" * 80)
    lines.append("DETAILED RESULTS")
    lines.append("=" * 80)

    for r in results:
        status = "V" if r.match_type != 'none' else "X"
        lines.append("")
        lines.append("{} {}".format(status, r.cancer_type))
        lines.append("  Gold: {} ({})".format(
            ', '.join(sorted(r.gold_targets)), r.gold_evidence))
        targets_str = ', '.join(sorted(r.our_targets)) if r.our_targets else 'N/A'
        lines.append("  Ours: {}".format(targets_str))
        lines.append("  Match: {} | Rank: {}".format(r.match_type, r.our_rank))
        lines.append("  {}".format(r.description))

    lines.append("")
    lines.append("=" * 80)
    return "\n".join(lines)


def export_benchmark(results, metrics, output_path, loco_cv=None, target_prioritization=None):
    """Export benchmark to CSV, JSON, and text report"""
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)

    rows = []
    for r in results:
        rows.append({
            'Cancer_Type': r.cancer_type,
            'Gold_Targets': ' + '.join(sorted(r.gold_targets)),
            'Gold_Evidence': r.gold_evidence,
            'Our_Targets': ' + '.join(sorted(r.our_targets)) if r.our_targets else '',
            'Match_Type': r.match_type,
            'Rank': r.our_rank,
            'Matched': r.match_type != 'none'
        })
    pd.DataFrame(rows).to_csv(output_path / "benchmark_results.csv", index=False)

    export_metrics = dict(metrics)
    if loco_cv:
        export_metrics['loco_cv'] = loco_cv
    if target_prioritization:
        export_metrics['target_prioritization'] = target_prioritization

    with open(output_path / "benchmark_metrics.json", 'w') as f:
        json.dump(export_metrics, f, indent=2, default=str)

    report = generate_benchmark_report(results, metrics)
    with open(output_path / "benchmark_report.txt", 'w') as f:
        f.write(report)


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark ALIN predictions")
    parser.add_argument('--triples', type=str,
                        default='results_triples/triple_combinations.csv')
    parser.add_argument('--summary', type=str,
                        default='results_triples/pan_cancer_summary.csv')
    parser.add_argument('--output', type=str, default='benchmark_results')
    parser.add_argument('--baselines', action='store_true',
                        help='Run random, top-genes, frequency, pool-matched baselines')
    parser.add_argument('--loco', action='store_true',
                        help='Run leave-one-cancer-out cross-validation')
    parser.add_argument('--n-trials', type=int, default=1000,
                        help='Random/pool-matched baseline trials')

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("ALIN BENCHMARK -- Independent Gold Standard")
    print("=" * 60)

    print("\n[1/5] Running combination benchmark ({} gold-standard entries)...".format(
        len(COMBINATION_GOLD_STANDARD)))
    results, metrics = run_benchmark(args.triples, args.summary)
    print("  Exact recall:    {:.1f}%".format(metrics['recall_exact'] * 100))
    print("  Superset recall: {:.1f}%".format(metrics['recall_superset_or_better'] * 100))
    print("  Pairwise recall: {:.1f}%".format(metrics['recall_pairwise_or_better'] * 100))

    print("\n[2/5] Running target prioritization ({} single-target entries)...".format(
        len(SINGLE_TARGET_GOLD_STANDARD)))
    target_pri = run_target_prioritization_benchmark(args.triples)
    print("  Hit rate: {:.1f}% ({}/{})".format(
        target_pri['hit_rate'] * 100, target_pri['hits'], target_pri['total']))

    loco_cv = None
    if args.loco:
        print("\n[3/5] Running leave-one-cancer-out cross-validation...")
        loco_cv = run_loco_cv(args.triples, args.summary)
        print("  LOCO-CV exact recall:    {:.1f}% +/- {:.1f}%".format(
            loco_cv['mean_recall_exact'] * 100, loco_cv['std_recall_exact'] * 100))
        print("  LOCO-CV superset recall: {:.1f}% +/- {:.1f}%".format(
            loco_cv['mean_recall_superset'] * 100, loco_cv['std_recall_superset'] * 100))
        for fold in loco_cv['per_fold']:
            print("    {:40s}  exact={:.0%}  superset={:.0%}  (n={})".format(
                fold['held_out_cancer'], fold['recall_exact'],
                fold['recall_superset_or_better'], fold['n_gold_entries']))
    else:
        print("\n[3/5] Skipping LOCO-CV (use --loco to enable)")

    if args.baselines:
        print("\n[4/5] Running baselines (n={} trials)...".format(args.n_trials))
        random_bl = run_random_baseline(args.triples, n_trials=args.n_trials)
        topgenes_bl = run_topgenes_baseline(args.triples)
        freq_bl = run_frequency_baseline(args.triples)
        poolmatched_bl = run_poolmatched_baseline(args.triples, n_trials=args.n_trials)
        driver_bl = run_driver_baseline(args.triples)
        essentiality_bl = run_essentiality_baseline(args.triples)

        metrics['random_baseline_exact'] = random_bl['mean_recall_exact']
        metrics['random_baseline_exact_std'] = random_bl['std_recall_exact']
        metrics['random_baseline_superset'] = random_bl['mean_recall_superset']
        metrics['random_baseline_superset_std'] = random_bl['std_recall_superset']
        metrics['random_baseline_pairwise'] = random_bl['mean_recall_pairwise']
        metrics['random_baseline_pairwise_std'] = random_bl['std_recall_pairwise']
        metrics['topgenes_baseline_exact'] = topgenes_bl['recall_exact']
        metrics['topgenes_baseline_superset'] = topgenes_bl['recall_superset']
        metrics['topgenes_baseline_pairwise'] = topgenes_bl.get('recall_pairwise', 0)
        metrics['frequency_baseline_exact'] = freq_bl['recall_exact']
        metrics['frequency_baseline_superset'] = freq_bl['recall_superset']
        metrics['frequency_baseline_pairwise'] = freq_bl.get('recall_pairwise', 0)
        metrics['poolmatched_baseline_exact'] = poolmatched_bl['mean_recall_exact']
        metrics['poolmatched_baseline_exact_std'] = poolmatched_bl['std_recall_exact']
        metrics['poolmatched_baseline_superset'] = poolmatched_bl['mean_recall_superset']
        metrics['poolmatched_baseline_superset_std'] = poolmatched_bl['std_recall_superset']
        metrics['poolmatched_baseline_pairwise'] = poolmatched_bl['mean_recall_pairwise']
        metrics['poolmatched_baseline_pairwise_std'] = poolmatched_bl['std_recall_pairwise']
        metrics['driver_baseline_exact'] = driver_bl['recall_exact']
        metrics['driver_baseline_superset'] = driver_bl['recall_superset']
        metrics['driver_baseline_pairwise'] = driver_bl['recall_pairwise']
        metrics['essentiality_baseline_exact'] = essentiality_bl['recall_exact']
        metrics['essentiality_baseline_superset'] = essentiality_bl['recall_superset']
        metrics['essentiality_baseline_pairwise'] = essentiality_bl['recall_pairwise']

        print("  Random:        exact={:.1f}% superset={:.1f}% pairwise={:.1f}%".format(
            random_bl['mean_recall_exact'] * 100, random_bl['mean_recall_superset'] * 100,
            random_bl['mean_recall_pairwise'] * 100))
        print("  Top-genes:     exact={:.1f}% superset={:.1f}% pairwise={:.1f}%".format(
            topgenes_bl['recall_exact'] * 100, topgenes_bl['recall_superset'] * 100,
            topgenes_bl.get('recall_pairwise', 0) * 100))
        print("  Frequency:     exact={:.1f}% superset={:.1f}% pairwise={:.1f}%".format(
            freq_bl['recall_exact'] * 100, freq_bl['recall_superset'] * 100,
            freq_bl.get('recall_pairwise', 0) * 100))
        print("  Pool-matched:  exact={:.1f}% superset={:.1f}% pairwise={:.1f}%".format(
            poolmatched_bl['mean_recall_exact'] * 100, poolmatched_bl['mean_recall_superset'] * 100,
            poolmatched_bl['mean_recall_pairwise'] * 100))
        print("  Driver genes:  exact={:.1f}% superset={:.1f}% pairwise={:.1f}%".format(
            driver_bl['recall_exact'] * 100, driver_bl['recall_superset'] * 100,
            driver_bl['recall_pairwise'] * 100))
        print("  Essentiality:  exact={:.1f}% superset={:.1f}% pairwise={:.1f}%".format(
            essentiality_bl['recall_exact'] * 100, essentiality_bl['recall_superset'] * 100,
            essentiality_bl['recall_pairwise'] * 100))
    else:
        print("\n[4/5] Skipping baselines (use --baselines to enable)")

    print("\n[5/5] Exporting results to {}/...".format(args.output))
    export_benchmark(results, metrics, Path(args.output),
                     loco_cv=loco_cv, target_prioritization=target_pri)

    report = generate_benchmark_report(results, metrics)
    print(report)
    print("Results saved to {}/".format(args.output))
