#!/usr/bin/env python3
"""
Translational Feasibility Analysis for ALIN Pan-Cancer X-Node Predictions.

Maps each predicted triple to:
  1. At least one drug per target (or explains 'no drug')
  2. Selectivity caveats (pan-kinase inhibitors, shared drugs, narrow TW)
  3. Combination toxicity red flags (overlapping organ toxicities, known DDIs)
  4. Clinical precedent (gold-standard subset overlap)
  5. Composite feasibility score that re-ranks predictions

Outputs: feasibility_results/ directory with CSV tables and summary stats.
"""

import csv
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# ---------------------------------------------------------------------------
# Import ALIN infrastructure
# ---------------------------------------------------------------------------
from alin.constants import (
    GENE_TO_DRUGS,
    GENE_CLINICAL_STAGE,
    GENE_TOXICITIES,
    GENE_TOXICITY_SCORES,
)
from alin.toxicity import (
    KNOWN_DDI,
    TOXICITY_CLASSES,
    DRUG_TOXICITY_PROFILE,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Stage → numeric score (higher = more clinically advanced = more feasible)
STAGE_SCORES: Dict[str, float] = {
    'approved': 1.0,
    'phase3': 0.80,
    'phase2': 0.55,
    'phase1': 0.30,
    'preclinical': 0.10,
}

# Selectivity caveats: drugs known to hit many off-targets
PAN_KINASE_DRUGS = {
    'dasatinib': 'SRC/ABL/KIT/PDGFR/FYN/YES1/LYN/EPHA2 (>40 kinases at clinical dose)',
    'bosutinib': 'SRC/ABL/TEC family (broad kinase profile)',
    'crizotinib': 'ALK/MET/ROS1/RON (multi-target)',
    'cabozantinib': 'MET/KDR/AXL/RET/KIT/FLT3 (multi-target)',
    'gilteritinib': 'FLT3/AXL (dual target)',
    'dinaciclib': 'CDK1/2/5/9 (pan-CDK; narrow therapeutic window)',
    'navitoclax': 'BCL2/BCL-XL/BCL-W (on-target thrombocytopenia from BCL-XL)',
    'ruxolitinib': 'JAK1/JAK2 (non-selective JAK)',
}

# Targets sharing the same drug (indicates non-independent drugging)
SHARED_DRUG_PAIRS = {
    frozenset({'CDK4', 'CDK6'}): 'palbociclib/ribociclib/abemaciclib (identical drugs)',
    frozenset({'MAP2K1', 'MAP2K2'}): 'trametinib/cobimetinib/binimetinib (identical drugs)',
    frozenset({'FGFR1', 'FGFR2'}): 'erdafitinib/pemigatinib (identical drugs)',
    frozenset({'IDH1', 'IDH2'}): 'different drugs (ivosidenib vs enasidenib)',  # OK
    frozenset({'SRC', 'FYN'}): 'dasatinib (same pan-kinase inhibitor)',
    frozenset({'SRC', 'YES1'}): 'dasatinib (same pan-kinase inhibitor)',
    frozenset({'SRC', 'LYN'}): 'dasatinib (same pan-kinase inhibitor)',
    frozenset({'FYN', 'YES1'}): 'dasatinib (same pan-kinase inhibitor)',
    frozenset({'FYN', 'LYN'}): 'dasatinib/saracatinib (same drugs)',
}

# Narrow therapeutic window targets
NARROW_TW_TARGETS = {
    'MCL1': 'Dose-limiting cardiotoxicity in Phase 1 (AMG-176, S64315)',
    'BCL2L1': 'On-target thrombocytopenia (navitoclax); limits dosing',
    'CDK2': 'Pan-CDK inhibitor dinaciclib: severe myelosuppression, narrow TW',
    'TP53': 'No validated direct inhibitor; undruggable',
    'MYC': 'Transcription factor; OMO-103 is Phase 1 only',
    'RB1': 'Tumor suppressor; no restoration therapy available',
}

# Overlapping organ-toxicity classes (high-severity red flags)
# Map from GENE_TOXICITIES entries → organ system
TOXICITY_TO_ORGAN = {
    'neutropenia': 'myelosuppression',
    'myelosuppression': 'myelosuppression',
    'anemia': 'myelosuppression',
    'thrombocytopenia': 'myelosuppression',
    'cardiotoxicity': 'cardiotoxicity',
    'hepatotoxicity': 'hepatotoxicity',
    'ILD': 'pulmonary',
    'pneumonitis': 'pulmonary',
    'rash': 'dermatologic',
    'photosensitivity': 'dermatologic',
    'diarrhea': 'gastrointestinal',
    'GI toxicity': 'gastrointestinal',
    'nausea': 'gastrointestinal',
    'mucositis': 'gastrointestinal',
    'hyperglycemia': 'metabolic',
    'edema': 'cardiovascular',
    'pleural effusion': 'pulmonary',
    'visual disturbances': 'ocular',
    'retinopathy': 'ocular',
    'infections': 'immunosuppression',
    'thrombosis': 'thrombotic',
    'tumor lysis syndrome': 'metabolic',
}

# Organ-toxicity severity weights for overlap penalty
ORGAN_SEVERITY = {
    'myelosuppression': 0.25,   # DLT in most combo trials
    'cardiotoxicity': 0.25,      # life-threatening
    'hepatotoxicity': 0.20,
    'pulmonary': 0.20,
    'gastrointestinal': 0.10,    # manageable
    'dermatologic': 0.05,        # manageable
    'metabolic': 0.10,
    'cardiovascular': 0.10,
    'ocular': 0.05,
    'immunosuppression': 0.15,
    'thrombotic': 0.15,
}

# Gold standard entries for clinical precedent scoring
# Format: {(cancer_type_keyword, frozenset_of_targets): evidence_level}
GOLD_STANDARD_PAIRS = {
    ('Melanoma', frozenset({'BRAF', 'MAP2K1'})): 'FDA_approved',
    ('NSCLC', frozenset({'BRAF', 'MAP2K1'})): 'FDA_approved',
    ('Thyroid', frozenset({'BRAF', 'MAP2K1'})): 'FDA_approved',
    ('Colorectal', frozenset({'BRAF', 'EGFR'})): 'FDA_approved',
    ('Colorectal', frozenset({'BRAF', 'EGFR', 'MAP2K1'})): 'Phase_3',
    ('Breast', frozenset({'CDK4', 'CDK6'})): 'FDA_approved',
    ('Breast', frozenset({'EGFR', 'ERBB2'})): 'FDA_approved',
    ('NSCLC', frozenset({'EGFR', 'MET'})): 'FDA_approved',
    ('NSCLC', frozenset({'EGFR', 'KRAS'})): 'Phase_2',
    ('Colorectal', frozenset({'EGFR', 'KRAS'})): 'Phase_2',
    ('HNSCC', frozenset({'EGFR', 'MET'})): 'Phase_2',
    ('AML', frozenset({'BCL2', 'FLT3'})): 'Phase_2',
    ('AML', frozenset({'BCL2', 'IDH1'})): 'Phase_2',
    ('AML', frozenset({'BCL2', 'IDH2'})): 'Phase_2',
    ('Ovarian', frozenset({'PARP1', 'PIK3CA'})): 'Phase_2',
    ('Pancreatic', frozenset({'EGFR', 'KRAS', 'STAT3'})): 'Preclinical',
    ('Bladder', frozenset({'FGFR1', 'MAP2K1'})): 'Phase_2',
    ('Endometrial', frozenset({'MTOR'})): 'Phase_2',
    ('HCC', frozenset({'MET'})): 'FDA_approved',
    ('Glioma', frozenset({'BRAF', 'MAP2K1'})): 'FDA_approved',
    ('Breast', frozenset({'PIK3CA'})): 'FDA_approved',
    ('Breast', frozenset({'AKT1'})): 'FDA_approved',
    ('NSCLC', frozenset({'ALK', 'MET'})): 'Phase_2',
    ('Liposarcoma', frozenset({'CDK4'})): 'Phase_2',
    ('Prostate', frozenset({'PARP1'})): 'FDA_approved',
    ('NSCLC', frozenset({'EGFR'})): 'FDA_approved',
    ('Ovarian', frozenset({'PARP1'})): 'FDA_approved',
    ('NSCLC', frozenset({'KRAS', 'MAP2K1'})): 'Phase_2',
}

EVIDENCE_SCORES = {
    'FDA_approved': 1.0,
    'Phase_3': 0.8,
    'Phase_2': 0.6,
    'Preclinical': 0.2,
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TargetDrugMapping:
    """Drug mapping for a single target within a triple."""
    gene: str
    drugs: List[str]
    clinical_stage: str
    drug_availability_score: float
    selectivity_caveats: List[str]
    narrow_tw: Optional[str]
    toxicities: List[str]


@dataclass
class ToxicityRedFlag:
    """A combination toxicity red flag."""
    category: str            # e.g., 'overlapping_organ_tox', 'known_ddi', 'narrow_tw'
    severity: str            # 'critical', 'high', 'moderate', 'low'
    description: str
    targets_involved: List[str]


@dataclass
class FeasibilityResult:
    """Complete feasibility assessment for one triple combination."""
    cancer_type: str
    targets: Tuple[str, ...]
    drugs: Tuple[str, ...]
    combined_score: float    # original ALIN combined_score
    
    # Component scores (0–1, higher = more feasible)
    drug_availability: float
    selectivity_score: float
    tox_feasibility: float
    clinical_precedent: float
    
    # Composite
    feasibility_score: float
    adjusted_score: float     # combined_score penalized by low feasibility
    
    # Details
    target_mappings: List[TargetDrugMapping] = field(default_factory=list)
    red_flags: List[ToxicityRedFlag] = field(default_factory=list)
    precedent_matches: List[str] = field(default_factory=list)
    
    # Rankings
    original_rank: int = 0
    feasibility_rank: int = 0
    rank_change: int = 0


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------

def score_drug_availability(gene: str) -> Tuple[float, TargetDrugMapping]:
    """Score drug availability for a single target. Returns (score, mapping)."""
    drugs = GENE_TO_DRUGS.get(gene, [])
    stage = GENE_CLINICAL_STAGE.get(gene, 'preclinical')
    toxicities = GENE_TOXICITIES.get(gene, [])
    
    # Base score from clinical stage
    base = STAGE_SCORES.get(stage, 0.10)
    
    # Drug count bonus
    n_drugs = len(drugs)
    if n_drugs == 0:
        score = 0.0
    elif n_drugs >= 3 and stage == 'approved':
        score = 1.0
    elif n_drugs >= 1 and stage == 'approved':
        score = 0.90
    else:
        # bonus for having more options
        bonus = min(0.15, (n_drugs - 1) * 0.05) if n_drugs > 1 else 0.0
        score = min(1.0, base + bonus)
    
    # Selectivity caveats for this target
    caveats = []
    for d in drugs:
        if d in PAN_KINASE_DRUGS:
            caveats.append(f"{d}: {PAN_KINASE_DRUGS[d]}")
    
    # Narrow therapeutic window
    ntw = NARROW_TW_TARGETS.get(gene)
    
    mapping = TargetDrugMapping(
        gene=gene,
        drugs=drugs if drugs else ['(no drug available)'],
        clinical_stage=stage,
        drug_availability_score=score,
        selectivity_caveats=caveats,
        narrow_tw=ntw,
        toxicities=toxicities,
    )
    return score, mapping


def score_selectivity(targets: Tuple[str, ...], mappings: List[TargetDrugMapping]) -> Tuple[float, List[str]]:
    """
    Score selectivity of the combination. Penalizes:
    - Shared drugs between targets (not truly independent)
    - Pan-kinase inhibitors
    - Narrow therapeutic window targets
    Returns (score, list_of_caveats).
    """
    penalties = 0.0
    caveats = []
    
    # Check shared-drug pairs
    for i, t1 in enumerate(targets):
        for j, t2 in enumerate(targets):
            if j <= i:
                continue
            pair = frozenset({t1, t2})
            if pair in SHARED_DRUG_PAIRS:
                desc = SHARED_DRUG_PAIRS[pair]
                if 'identical drugs' in desc or 'same pan-kinase' in desc or 'same drugs' in desc:
                    penalties += 0.25
                    caveats.append(f"Shared drug: {t1}+{t2} → {desc}")
    
    # Check pan-kinase inhibitors  
    pan_kinase_count = 0
    for m in mappings:
        for d in m.drugs:
            if d in PAN_KINASE_DRUGS and d != '(no drug available)':
                pan_kinase_count += 1
                break  # count per target, not per drug
    if pan_kinase_count >= 2:
        penalties += 0.20
        caveats.append(f"{pan_kinase_count} targets rely on pan-kinase inhibitors")
    elif pan_kinase_count == 1:
        penalties += 0.08
    
    # Check narrow therapeutic window
    ntw_count = 0
    for m in mappings:
        if m.narrow_tw:
            ntw_count += 1
            caveats.append(f"Narrow TW: {m.gene} — {m.narrow_tw}")
    if ntw_count >= 2:
        penalties += 0.30
    elif ntw_count == 1:
        penalties += 0.12
    
    score = max(0.0, 1.0 - penalties)
    return score, caveats


def score_combination_toxicity(
    targets: Tuple[str, ...],
    drugs: Tuple[str, ...],
) -> Tuple[float, List[ToxicityRedFlag]]:
    """
    Identify combination toxicity red flags and compute a tox feasibility score.
    Higher score = fewer red flags = more feasible.
    """
    red_flags: List[ToxicityRedFlag] = []
    penalty = 0.0
    
    # 1. Overlapping organ toxicities
    organ_map: Dict[str, List[str]] = {}  # organ → [genes with that toxicity]
    for gene in targets:
        gene_tox = GENE_TOXICITIES.get(gene, [])
        for tox in gene_tox:
            organ = TOXICITY_TO_ORGAN.get(tox, 'other')
            organ_map.setdefault(organ, []).append(gene)
    
    for organ, genes in organ_map.items():
        if len(genes) >= 2:
            unique_genes = list(set(genes))
            if len(unique_genes) >= 2:
                severity_weight = ORGAN_SEVERITY.get(organ, 0.10)
                overlap_count = len(unique_genes)
                
                if organ in ('myelosuppression', 'cardiotoxicity'):
                    sev = 'critical' if overlap_count >= 3 else 'high'
                elif organ in ('hepatotoxicity', 'pulmonary'):
                    sev = 'high'
                else:
                    sev = 'moderate'
                
                flag_penalty = severity_weight * (overlap_count - 1)
                penalty += flag_penalty
                
                red_flags.append(ToxicityRedFlag(
                    category='overlapping_organ_tox',
                    severity=sev,
                    description=f"Overlapping {organ}: {'+'.join(unique_genes)} "
                                f"({overlap_count} targets share {organ} risk)",
                    targets_involved=unique_genes,
                ))
    
    # 2. Known DDIs
    real_drugs = [d for d in drugs if not d.startswith('(')]
    for i, d1 in enumerate(real_drugs):
        for j, d2 in enumerate(real_drugs):
            if j <= i:
                continue
            pair = frozenset({d1, d2})
            if pair in KNOWN_DDI:
                ddi_info = KNOWN_DDI[pair]
                sev = ddi_info['severity']
                sev_weight = {'major': 0.25, 'moderate': 0.12, 'minor': 0.05}.get(sev, 0.05)
                penalty += sev_weight
                
                red_flags.append(ToxicityRedFlag(
                    category='known_ddi',
                    severity='critical' if sev == 'major' else ('high' if sev == 'moderate' else 'moderate'),
                    description=f"DDI ({sev}): {d1}+{d2} — {ddi_info['mechanism']}",
                    targets_involved=[d1, d2],
                ))
    
    # 3. Narrow therapeutic window drugs in combination
    ntw_drugs = [t for t in targets if t in NARROW_TW_TARGETS]
    if len(ntw_drugs) >= 2:
        penalty += 0.20
        red_flags.append(ToxicityRedFlag(
            category='narrow_tw_combo',
            severity='critical',
            description=f"Multiple narrow-TW targets: {'+'.join(ntw_drugs)}",
            targets_involved=ntw_drugs,
        ))
    
    score = max(0.0, 1.0 - min(1.0, penalty))
    return score, red_flags


def score_clinical_precedent(
    cancer_type: str,
    targets: Tuple[str, ...],
) -> Tuple[float, List[str]]:
    """
    Check if any subset of the predicted targets has clinical precedent.
    Returns (score, list_of_matching_descriptions).
    """
    target_set = set(targets)
    matches = []
    best_score = 0.0
    
    # Abbreviate cancer name for matching  
    cancer_lower = cancer_type.lower()
    cancer_aliases = {
        'non-small cell lung': 'nsclc',
        'nsclc': 'nsclc',
        'colorectal': 'colorectal',
        'melanoma': 'melanoma',
        'breast': 'breast',
        'aml': 'aml',
        'acute myeloid': 'aml',
        'head and neck': 'hnscc',
        'hnscc': 'hnscc',
        'ovarian': 'ovarian',
        'pancreatic': 'pancreatic',
        'bladder': 'bladder',
        'urothelial': 'bladder',
        'endometrial': 'endometrial',
        'hepatocellular': 'hcc',
        'hcc': 'hcc',
        'liposarcoma': 'liposarcoma',
        'glioma': 'glioma',
        'diffuse glioma': 'glioma',
        'prostate': 'prostate',
        'renal': 'renal',
        'rcc': 'renal',
        'esophagogastric': 'esophagogastric',
        'anaplastic thyroid': 'thyroid',
        'thyroid': 'thyroid',
    }
    
    # Find what cancer keyword in the gold standard matches
    matched_keywords = set()
    for key_fragment, alias in cancer_aliases.items():
        if key_fragment in cancer_lower:
            matched_keywords.add(alias)
    # Also try the exact cancer type words
    for word in cancer_lower.split():
        if word in cancer_aliases.values():
            matched_keywords.add(word)
    
    for (gs_cancer, gs_targets), evidence in GOLD_STANDARD_PAIRS.items():
        gs_key = gs_cancer.lower()
        # Check cancer match
        cancer_match = False
        for mk in matched_keywords:
            if mk == gs_key.lower() or mk in gs_key.lower():
                cancer_match = True
                break
        
        if not cancer_match:
            continue
        
        # Check target overlap
        if gs_targets.issubset(target_set):
            # Full subset match
            ev_score = EVIDENCE_SCORES.get(evidence, 0.2)
            if ev_score > best_score:
                best_score = ev_score
            matches.append(f"Full match: {'+'.join(sorted(gs_targets))} ({evidence})")
        elif gs_targets & target_set:
            # Partial overlap
            overlap = gs_targets & target_set
            ev_score = EVIDENCE_SCORES.get(evidence, 0.2) * 0.5  # partial credit
            if ev_score > best_score:
                best_score = ev_score
            matches.append(f"Partial: {'+'.join(sorted(overlap))} of {'+'.join(sorted(gs_targets))} ({evidence})")
    
    return best_score, matches


def compute_feasibility(
    cancer_type: str,
    targets: Tuple[str, ...],
    drugs: Tuple[str, ...],
    combined_score: float,
) -> FeasibilityResult:
    """Compute the full feasibility assessment for one triple."""
    
    # 1. Drug availability (per target, then average)
    target_mappings = []
    avail_scores = []
    for gene in targets:
        score, mapping = score_drug_availability(gene)
        target_mappings.append(mapping)
        avail_scores.append(score)
    drug_availability = sum(avail_scores) / len(avail_scores) if avail_scores else 0.0
    
    # 2. Selectivity
    selectivity_score, sel_caveats = score_selectivity(targets, target_mappings)
    
    # 3. Combination toxicity red flags
    tox_feasibility, red_flags = score_combination_toxicity(targets, drugs)
    
    # 4. Clinical precedent
    clinical_precedent, precedent_matches = score_clinical_precedent(cancer_type, targets)
    
    # 5. Composite feasibility score
    # Weights: drug availability most important for translation,
    # then toxicity (can kill a combo), selectivity, clinical precedent
    feasibility_score = (
        0.30 * drug_availability +
        0.25 * tox_feasibility +
        0.25 * selectivity_score +
        0.20 * clinical_precedent
    )
    
    # 6. Adjusted score: penalize combined_score by low feasibility
    # combined_score is lower = better, so we inflate poor-feasibility combos
    # adjusted_score = combined_score / feasibility_score (bounded)
    # This means low feasibility → higher adjusted_score → worse ranking
    bounded_feas = max(0.20, feasibility_score)  # floor to avoid division blowup
    adjusted_score = combined_score / bounded_feas
    
    return FeasibilityResult(
        cancer_type=cancer_type,
        targets=targets,
        drugs=drugs,
        combined_score=combined_score,
        drug_availability=drug_availability,
        selectivity_score=selectivity_score,
        tox_feasibility=tox_feasibility,
        clinical_precedent=clinical_precedent,
        feasibility_score=feasibility_score,
        adjusted_score=adjusted_score,
        target_mappings=target_mappings,
        red_flags=red_flags,
        precedent_matches=precedent_matches,
    )


# ---------------------------------------------------------------------------
# CSV parsing and main analysis
# ---------------------------------------------------------------------------

def parse_benchmark_csv(path: str) -> List[dict]:
    """Parse results/triple_combinations.csv."""
    rows = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            targets = tuple(
                row[k].strip() for k in ['Target 1', 'Target 2', 'Target 3']
                if row.get(k, '').strip()
            )
            drugs = tuple(
                row[k].strip() for k in ['Drug 1', 'Drug 2', 'Drug 3']
                if row.get(k, '').strip()
            )
            # Parse combined_score from Best_Combo_Score
            try:
                score = float(row.get('Best_Combo_Score', '0'))
            except ValueError:
                score = 0.0
            
            rows.append({
                'cancer_type': row.get('Cancer Type', '').strip(),
                'targets': targets,
                'drugs': drugs,
                'combined_score': score,
            })
    return rows


def parse_merged_csv(path: str) -> List[dict]:
    """Parse results/triple_combinations_merged.csv."""
    rows = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            targets = tuple(
                row[k].strip() for k in ['Target_1', 'Target_2', 'Target_3']
                if row.get(k, '').strip()
            )
            drugs = tuple(
                row[k].strip() for k in ['Drug_1', 'Drug_2', 'Drug_3']
                if row.get(k, '').strip()
            )
            try:
                score = float(row.get('Combined_Score', '0'))
            except ValueError:
                score = 0.0
            
            rows.append({
                'cancer_type': row.get('Cancer_Type', '').strip(),
                'targets': targets,
                'drugs': drugs,
                'combined_score': score,
            })
    return rows


def run_analysis():
    """Run the full feasibility analysis."""
    base = Path(__file__).parent
    outdir = base / 'feasibility_results'
    outdir.mkdir(exist_ok=True)
    
    # --- 1. Benchmark 17 cancers ---
    benchmark_path = base / 'results' / 'triple_combinations.csv'
    if not benchmark_path.exists():
        print(f"WARNING: {benchmark_path} not found, skipping benchmark analysis")
        benchmark_rows = []
    else:
        benchmark_rows = parse_benchmark_csv(str(benchmark_path))
    
    benchmark_results: List[FeasibilityResult] = []
    for row in benchmark_rows:
        result = compute_feasibility(
            row['cancer_type'], row['targets'], row['drugs'], row['combined_score']
        )
        benchmark_results.append(result)
    
    # Assign original ranks (by combined_score, lower = better)
    by_original = sorted(benchmark_results, key=lambda r: r.combined_score)
    for i, r in enumerate(by_original):
        r.original_rank = i + 1
    
    # Assign feasibility-adjusted ranks
    by_adjusted = sorted(benchmark_results, key=lambda r: r.adjusted_score)
    for i, r in enumerate(by_adjusted):
        r.feasibility_rank = i + 1
        r.rank_change = r.original_rank - r.feasibility_rank  # positive = improved
    
    # --- 2. Write detailed benchmark results ---
    write_detailed_csv(benchmark_results, outdir / 'benchmark_feasibility.csv')
    
    # --- 3. Write drug mapping table ---
    write_drug_mapping(benchmark_results, outdir / 'drug_mapping.csv')
    
    # --- 4. Write red flags table ---
    write_red_flags(benchmark_results, outdir / 'red_flags.csv')
    
    # --- 5. Write ranking comparison ---
    write_ranking_comparison(benchmark_results, outdir / 'ranking_comparison.csv')
    
    # --- 6. Summary statistics ---
    write_summary(benchmark_results, outdir / 'summary.txt')
    
    # --- 7. Merged 77-cancer analysis (if available) ---
    merged_path = base / 'results' / 'triple_combinations_merged.csv'
    if merged_path.exists():
        merged_rows = parse_merged_csv(str(merged_path))
        merged_results = []
        for row in merged_rows:
            result = compute_feasibility(
                row['cancer_type'], row['targets'], row['drugs'], row['combined_score']
            )
            merged_results.append(result)
        
        by_orig_m = sorted(merged_results, key=lambda r: r.combined_score)
        for i, r in enumerate(by_orig_m):
            r.original_rank = i + 1
        by_adj_m = sorted(merged_results, key=lambda r: r.adjusted_score)
        for i, r in enumerate(by_adj_m):
            r.feasibility_rank = i + 1
            r.rank_change = r.original_rank - r.feasibility_rank
        
        write_detailed_csv(merged_results, outdir / 'merged_feasibility.csv')
        write_ranking_comparison(merged_results, outdir / 'merged_ranking_comparison.csv')
        write_merged_summary(merged_results, outdir / 'merged_summary.txt')
    
    # --- 8. Paper-ready summary ---
    write_paper_summary(benchmark_results, outdir / 'paper_summary.txt')
    
    print(f"\n=== Feasibility analysis complete ===")
    print(f"Output directory: {outdir}")
    print(f"Benchmark cancers analyzed: {len(benchmark_results)}")
    if merged_path.exists():
        print(f"Pan-cancer predictions analyzed: {len(merged_results)}")
    
    return benchmark_results


def write_detailed_csv(results: List[FeasibilityResult], path: Path):
    """Write detailed feasibility results CSV."""
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Cancer_Type', 'Targets', 'Drugs',
            'Combined_Score', 'Drug_Availability', 'Selectivity',
            'Tox_Feasibility', 'Clinical_Precedent', 'Feasibility_Score',
            'Adjusted_Score', 'Original_Rank', 'Feasibility_Rank', 'Rank_Change',
            'Num_Red_Flags', 'Red_Flag_Summary', 'Precedent_Matches',
        ])
        for r in sorted(results, key=lambda x: x.adjusted_score):
            writer.writerow([
                r.cancer_type,
                '+'.join(r.targets),
                '+'.join(r.drugs),
                f"{r.combined_score:.3f}",
                f"{r.drug_availability:.3f}",
                f"{r.selectivity_score:.3f}",
                f"{r.tox_feasibility:.3f}",
                f"{r.clinical_precedent:.3f}",
                f"{r.feasibility_score:.3f}",
                f"{r.adjusted_score:.3f}",
                r.original_rank,
                r.feasibility_rank,
                r.rank_change,
                len(r.red_flags),
                '; '.join(rf.description for rf in r.red_flags) if r.red_flags else 'None',
                '; '.join(r.precedent_matches) if r.precedent_matches else 'None',
            ])


def write_drug_mapping(results: List[FeasibilityResult], path: Path):
    """Write per-target drug mapping table."""
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Cancer_Type', 'Gene', 'Available_Drugs', 'Clinical_Stage',
            'Availability_Score', 'Selectivity_Caveats', 'Narrow_TW', 'Toxicities',
        ])
        for r in results:
            for m in r.target_mappings:
                writer.writerow([
                    r.cancer_type,
                    m.gene,
                    ', '.join(m.drugs),
                    m.clinical_stage,
                    f"{m.drug_availability_score:.2f}",
                    '; '.join(m.selectivity_caveats) if m.selectivity_caveats else '-',
                    m.narrow_tw or '-',
                    ', '.join(m.toxicities) if m.toxicities else '-',
                ])


def write_red_flags(results: List[FeasibilityResult], path: Path):
    """Write all red flags across all predictions."""
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Cancer_Type', 'Targets', 'Flag_Category', 'Severity',
            'Description', 'Targets_Involved',
        ])
        for r in results:
            for rf in r.red_flags:
                writer.writerow([
                    r.cancer_type,
                    '+'.join(r.targets),
                    rf.category,
                    rf.severity,
                    rf.description,
                    '+'.join(rf.targets_involved),
                ])


def write_ranking_comparison(results: List[FeasibilityResult], path: Path):
    """Write ranking comparison table."""
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Cancer_Type', 'Targets', 'Combined_Score', 'Original_Rank',
            'Feasibility', 'Adjusted_Score', 'Feasibility_Rank', 'Rank_Change',
        ])
        for r in sorted(results, key=lambda x: x.feasibility_rank):
            writer.writerow([
                r.cancer_type,
                '+'.join(r.targets),
                f"{r.combined_score:.3f}",
                r.original_rank,
                f"{r.feasibility_score:.3f}",
                f"{r.adjusted_score:.3f}",
                r.feasibility_rank,
                f"{'+' if r.rank_change > 0 else ''}{r.rank_change}",
            ])


def write_summary(results: List[FeasibilityResult], path: Path):
    """Write human-readable summary statistics."""
    if not results:
        path.write_text("No results to summarize.\n")
        return
    
    lines = []
    lines.append("=" * 72)
    lines.append("TRANSLATIONAL FEASIBILITY ANALYSIS — BENCHMARK (17 CANCERS)")
    lines.append("=" * 72)
    
    # Overall stats
    feas_scores = [r.feasibility_score for r in results]
    lines.append(f"\nFeasibility score range: {min(feas_scores):.3f} – {max(feas_scores):.3f}")
    lines.append(f"Mean feasibility: {sum(feas_scores)/len(feas_scores):.3f}")
    
    # Tier classification
    high = [r for r in results if r.feasibility_score >= 0.60]
    mid = [r for r in results if 0.35 <= r.feasibility_score < 0.60]
    low = [r for r in results if r.feasibility_score < 0.35]
    lines.append(f"\nFeasibility tiers:")
    lines.append(f"  High (≥0.60): {len(high)} predictions")
    lines.append(f"  Medium (0.35–0.59): {len(mid)} predictions")
    lines.append(f"  Low (<0.35): {len(low)} predictions")
    
    # Drug availability
    all_avail = [r.drug_availability for r in results]
    lines.append(f"\nDrug availability: mean {sum(all_avail)/len(all_avail):.3f}")
    fully_druggable = sum(1 for r in results if r.drug_availability >= 0.90)
    lines.append(f"  Fully druggable (all 3 targets approved): {fully_druggable}/{len(results)}")
    
    # Red flags
    total_flags = sum(len(r.red_flags) for r in results)
    critical = sum(1 for r in results for rf in r.red_flags if rf.severity == 'critical')
    high_sev = sum(1 for r in results for rf in r.red_flags if rf.severity == 'high')
    lines.append(f"\nRed flags: {total_flags} total across {len(results)} predictions")
    lines.append(f"  Critical: {critical}")
    lines.append(f"  High: {high_sev}")
    
    # Clinical precedent
    with_precedent = sum(1 for r in results if r.clinical_precedent > 0)
    lines.append(f"\nClinical precedent: {with_precedent}/{len(results)} have ≥1 target pair in clinical trials")
    
    # Rank changes
    rank_changes = [abs(r.rank_change) for r in results]
    lines.append(f"\nRanking impact:")
    lines.append(f"  Mean absolute rank change: {sum(rank_changes)/len(rank_changes):.1f}")
    lines.append(f"  Max rank change: {max(rank_changes)}")
    movers = [r for r in results if abs(r.rank_change) >= 3]
    lines.append(f"  Predictions moving ≥3 ranks: {len(movers)}")
    
    # Top and bottom
    lines.append(f"\n{'—' * 72}")
    lines.append("TOP 5 BY FEASIBILITY-ADJUSTED RANKING:")
    for r in sorted(results, key=lambda x: x.adjusted_score)[:5]:
        lines.append(f"  {r.feasibility_rank:2d}. {r.cancer_type:<40s} "
                      f"{'+'.join(r.targets):<25s} feas={r.feasibility_score:.3f} "
                      f"adj={r.adjusted_score:.3f} (orig #{r.original_rank})")
    
    lines.append(f"\nBOTTOM 5 BY FEASIBILITY-ADJUSTED RANKING:")
    for r in sorted(results, key=lambda x: x.adjusted_score)[-5:]:
        lines.append(f"  {r.feasibility_rank:2d}. {r.cancer_type:<40s} "
                      f"{'+'.join(r.targets):<25s} feas={r.feasibility_score:.3f} "
                      f"adj={r.adjusted_score:.3f} (orig #{r.original_rank})")
    
    # Detailed per-cancer
    lines.append(f"\n{'=' * 72}")
    lines.append("DETAILED PER-CANCER BREAKDOWN")
    lines.append("=" * 72)
    
    for r in sorted(results, key=lambda x: x.adjusted_score):
        lines.append(f"\n{'—' * 72}")
        lines.append(f"Cancer: {r.cancer_type}")
        lines.append(f"Targets: {' + '.join(r.targets)}")
        lines.append(f"Combined score: {r.combined_score:.3f} (rank #{r.original_rank})")
        lines.append(f"Feasibility: {r.feasibility_score:.3f}")
        lines.append(f"Adjusted score: {r.adjusted_score:.3f} (rank #{r.feasibility_rank}, "
                      f"change: {'+' if r.rank_change > 0 else ''}{r.rank_change})")
        
        lines.append(f"\n  Components:")
        lines.append(f"    Drug availability: {r.drug_availability:.3f}")
        lines.append(f"    Selectivity:       {r.selectivity_score:.3f}")
        lines.append(f"    Tox feasibility:   {r.tox_feasibility:.3f}")
        lines.append(f"    Clinical precedent:{r.clinical_precedent:.3f}")
        
        lines.append(f"\n  Drug mapping:")
        for m in r.target_mappings:
            drugs_str = ', '.join(m.drugs)
            lines.append(f"    {m.gene:<10s} → {drugs_str} [{m.clinical_stage}] "
                          f"(avail={m.drug_availability_score:.2f})")
            if m.selectivity_caveats:
                for c in m.selectivity_caveats:
                    lines.append(f"      ⚠ Selectivity: {c}")
            if m.narrow_tw:
                lines.append(f"      ⚠ Narrow TW: {m.narrow_tw}")
        
        if r.red_flags:
            lines.append(f"\n  Red flags ({len(r.red_flags)}):")
            for rf in r.red_flags:
                lines.append(f"    [{rf.severity.upper():8s}] {rf.description}")
        else:
            lines.append(f"\n  Red flags: None")
        
        if r.precedent_matches:
            lines.append(f"\n  Clinical precedent:")
            for pm in r.precedent_matches:
                lines.append(f"    ✓ {pm}")
        else:
            lines.append(f"\n  Clinical precedent: None found")
    
    path.write_text('\n'.join(lines) + '\n')


def write_merged_summary(results: List[FeasibilityResult], path: Path):
    """Write summary for the 77-cancer merged analysis."""
    if not results:
        path.write_text("No merged results.\n")
        return
    
    lines = []
    lines.append("=" * 72)
    lines.append(f"TRANSLATIONAL FEASIBILITY — PAN-CANCER ({len(results)} TYPES)")
    lines.append("=" * 72)
    
    feas = [r.feasibility_score for r in results]
    lines.append(f"\nFeasibility range: {min(feas):.3f} – {max(feas):.3f}")
    lines.append(f"Mean: {sum(feas)/len(feas):.3f}, Median: {sorted(feas)[len(feas)//2]:.3f}")
    
    high = sum(1 for f in feas if f >= 0.60)
    mid = sum(1 for f in feas if 0.35 <= f < 0.60)
    low = sum(1 for f in feas if f < 0.35)
    lines.append(f"Tiers: High={high}, Medium={mid}, Low={low}")
    
    # Most common red flag categories
    flag_cats: Dict[str, int] = {}
    for r in results:
        for rf in r.red_flags:
            key = f"{rf.severity}:{rf.category}"
            flag_cats[key] = flag_cats.get(key, 0) + 1
    lines.append(f"\nMost common red flags:")
    for cat, count in sorted(flag_cats.items(), key=lambda x: -x[1])[:10]:
        lines.append(f"  {count:3d}× {cat}")
    
    # Rank change stats
    changes = [abs(r.rank_change) for r in results]
    lines.append(f"\nRank changes: mean={sum(changes)/len(changes):.1f}, max={max(changes)}")
    big_movers = sum(1 for c in changes if c >= 10)
    lines.append(f"Predictions moving ≥10 ranks: {big_movers}")
    
    path.write_text('\n'.join(lines) + '\n')


def write_paper_summary(results: List[FeasibilityResult], path: Path):
    """Write a concise summary suitable for the paper Results section."""
    if not results:
        path.write_text("No results.\n")
        return
    
    lines = []
    n = len(results)
    feas = [r.feasibility_score for r in results]
    mean_feas = sum(feas) / n
    
    high = [r for r in results if r.feasibility_score >= 0.60]
    mid = [r for r in results if 0.35 <= r.feasibility_score < 0.60]
    low = [r for r in results if r.feasibility_score < 0.35]
    
    total_flags = sum(len(r.red_flags) for r in results)
    critical_flags = sum(1 for r in results for rf in r.red_flags if rf.severity in ('critical', 'high'))
    
    with_precedent = sum(1 for r in results if r.clinical_precedent > 0)
    
    rank_changes = [abs(r.rank_change) for r in results]
    mean_rank_change = sum(rank_changes) / n
    big_movers = sum(1 for c in rank_changes if c >= 3)
    
    # Count fully-druggable
    fully_druggable = sum(1 for r in results if all(
        m.drug_availability_score >= 0.90 for m in r.target_mappings
    ))
    
    lines.append("PAPER-READY NUMBERS")
    lines.append("=" * 50)
    lines.append(f"N predictions: {n}")
    lines.append(f"Mean feasibility: {mean_feas:.3f}")
    lines.append(f"Range: {min(feas):.3f}–{max(feas):.3f}")
    lines.append(f"High tier (≥0.60): {len(high)}/{n}")
    lines.append(f"Medium tier (0.35–0.59): {len(mid)}/{n}")
    lines.append(f"Low tier (<0.35): {len(low)}/{n}")
    lines.append(f"All targets approved drugs: {fully_druggable}/{n}")
    lines.append(f"Total red flags: {total_flags}")
    lines.append(f"Critical/high red flags: {critical_flags}")
    lines.append(f"With clinical precedent: {with_precedent}/{n}")
    lines.append(f"Mean |rank change|: {mean_rank_change:.1f}")
    lines.append(f"Predictions moving ≥3: {big_movers}")
    
    lines.append(f"\nTOP 5 most feasible:")
    for r in sorted(results, key=lambda x: -x.feasibility_score)[:5]:
        lines.append(f"  {r.cancer_type}: {'+'.join(r.targets)} "
                      f"feas={r.feasibility_score:.3f}")
    
    lines.append(f"\nBOTTOM 5 least feasible:")
    for r in sorted(results, key=lambda x: x.feasibility_score)[:5]:
        flags = [rf.description for rf in r.red_flags if rf.severity in ('critical', 'high')]
        lines.append(f"  {r.cancer_type}: {'+'.join(r.targets)} "
                      f"feas={r.feasibility_score:.3f} flags={len(r.red_flags)}")
    
    lines.append(f"\nBIGGEST RANK CHANGES:")
    for r in sorted(results, key=lambda x: -abs(x.rank_change))[:5]:
        lines.append(f"  {r.cancer_type}: #{r.original_rank}→#{r.feasibility_rank} "
                      f"(change={'+' if r.rank_change > 0 else ''}{r.rank_change}) "
                      f"feas={r.feasibility_score:.3f}")
    
    path.write_text('\n'.join(lines) + '\n')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    results = run_analysis()
