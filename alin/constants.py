#!/usr/bin/env python3
"""
Canonical constants for ALIN Framework
=======================================
Single source of truth for gene-drug mappings, cancer type aliases,
pathway definitions, and shared utilities. All other modules should
import from here instead of maintaining their own copies.
"""

from typing import Dict, List, Set, FrozenSet

# ---------------------------------------------------------------------------
# tqdm safe import — use `from alin.constants import tqdm` everywhere
# ---------------------------------------------------------------------------
try:
    from tqdm import tqdm as _tqdm
    tqdm = _tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):  # type: ignore[misc]
        return iterable

# ---------------------------------------------------------------------------
# GENE → DRUG MAPPINGS (canonical)
# Sources: FDA labels, DGIdb, ChEMBL 34. Last updated 2025-12.
# ---------------------------------------------------------------------------

GENE_TO_DRUGS: Dict[str, List[str]] = {
    # EGFR family
    'EGFR':   ['erlotinib', 'gefitinib', 'osimertinib', 'afatinib', 'cetuximab'],
    'ERBB2':  ['trastuzumab', 'pertuzumab', 'lapatinib', 'tucatinib'],
    'ERBB3':  [],
    # KRAS
    'KRAS':   ['sotorasib', 'adagrasib'],
    # BRAF / MEK
    'BRAF':   ['vemurafenib', 'dabrafenib', 'encorafenib'],
    'MAP2K1': ['trametinib', 'cobimetinib', 'binimetinib'],
    'MAP2K2': ['trametinib', 'cobimetinib', 'binimetinib'],
    # PI3K / AKT / mTOR
    'PIK3CA': ['alpelisib', 'idelalisib', 'copanlisib'],
    'AKT1':   ['capivasertib', 'ipatasertib'],
    'MTOR':   ['everolimus', 'temsirolimus'],
    # CDK
    'CDK4':   ['palbociclib', 'ribociclib', 'abemaciclib'],
    'CDK6':   ['palbociclib', 'ribociclib', 'abemaciclib'],
    'CDK2':   ['dinaciclib'],
    # JAK / STAT
    'JAK1':   ['ruxolitinib', 'tofacitinib', 'baricitinib'],
    'JAK2':   ['ruxolitinib', 'fedratinib'],
    'STAT3':  ['napabucasin', 'TTI-101', 'stattic'],
    # SRC family
    'SRC':    ['dasatinib', 'bosutinib', 'saracatinib'],
    'FYN':    ['dasatinib', 'saracatinib'],
    'YES1':   ['dasatinib'],
    'LYN':    ['dasatinib', 'bafetinib'],
    # Other RTKs
    'MET':    ['capmatinib', 'tepotinib', 'crizotinib', 'cabozantinib'],
    'ALK':    ['crizotinib', 'alectinib', 'brigatinib', 'lorlatinib', 'ceritinib'],
    'ROS1':   ['crizotinib', 'entrectinib'],
    'RET':    ['selpercatinib', 'pralsetinib'],
    'FGFR1':  ['erdafitinib', 'pemigatinib'],
    'FGFR2':  ['erdafitinib', 'pemigatinib'],
    'AXL':    ['bemcentinib', 'gilteritinib'],
    # BCL-2 family
    'BCL2':   ['venetoclax', 'navitoclax'],
    'BCL2L1': ['navitoclax'],
    'MCL1':   ['AMG-176', 'S64315'],
    # PARP
    'PARP1':  ['olaparib', 'rucaparib', 'niraparib', 'talazoparib'],
    # IDH
    'IDH1':   ['ivosidenib'],
    'IDH2':   ['enasidenib'],
    # FLT3
    'FLT3':   ['midostaurin', 'gilteritinib'],
    # Difficult targets
    'TP53':   [],
    'MYC':    ['OMO-103'],
    'RB1':    [],
}

# Clinical stage per gene target
GENE_CLINICAL_STAGE: Dict[str, str] = {
    'EGFR': 'approved', 'ERBB2': 'approved', 'KRAS': 'approved',
    'BRAF': 'approved', 'MAP2K1': 'approved', 'MAP2K2': 'approved',
    'PIK3CA': 'approved', 'AKT1': 'phase3', 'MTOR': 'approved',
    'CDK4': 'approved', 'CDK6': 'approved', 'CDK2': 'phase2',
    'JAK1': 'approved', 'JAK2': 'approved', 'STAT3': 'phase2',
    'SRC': 'approved', 'FYN': 'approved', 'YES1': 'approved', 'LYN': 'approved',
    'MET': 'approved', 'ALK': 'approved', 'ROS1': 'approved', 'RET': 'approved',
    'FGFR1': 'approved', 'FGFR2': 'approved', 'AXL': 'phase2',
    'BCL2': 'approved', 'BCL2L1': 'phase2', 'MCL1': 'phase1',
    'PARP1': 'approved', 'IDH1': 'approved', 'IDH2': 'approved',
    'FLT3': 'approved',
    'TP53': 'preclinical', 'MYC': 'phase1', 'RB1': 'preclinical',
}

# Known toxicities per gene target
GENE_TOXICITIES: Dict[str, List[str]] = {
    'EGFR': ['rash', 'diarrhea', 'ILD'],
    'ERBB2': ['cardiotoxicity'],
    'KRAS': ['diarrhea', 'hepatotoxicity'],
    'BRAF': ['rash', 'photosensitivity'],
    'MAP2K1': ['rash', 'retinopathy'],
    'MAP2K2': ['rash'],
    'PIK3CA': ['hyperglycemia', 'rash', 'diarrhea'],
    'AKT1': ['hyperglycemia', 'rash'],
    'MTOR': ['mucositis', 'pneumonitis'],
    'CDK4': ['neutropenia'], 'CDK6': ['neutropenia'],
    'CDK2': ['myelosuppression'],
    'JAK1': ['infections', 'thrombosis'],
    'JAK2': ['anemia', 'thrombocytopenia'],
    'STAT3': ['GI toxicity'],
    'SRC': ['pleural effusion', 'myelosuppression'],
    'FYN': ['myelosuppression'],
    'MET': ['edema', 'nausea'],
    'ALK': ['visual disturbances'],
    'BCL2': ['tumor lysis syndrome', 'neutropenia'],
    'BCL2L1': ['thrombocytopenia'],
    'MCL1': ['cardiotoxicity'],
    'PARP1': ['anemia', 'nausea'],
}

# Base toxicity scores (0–1, higher = more toxic)
GENE_TOXICITY_SCORES: Dict[str, float] = {
    'EGFR': 0.6, 'ERBB2': 0.5, 'KRAS': 0.5, 'BRAF': 0.5,
    'MAP2K1': 0.5, 'MAP2K2': 0.5, 'PIK3CA': 0.6, 'AKT1': 0.5,
    'MTOR': 0.6, 'CDK4': 0.4, 'CDK6': 0.4, 'CDK2': 0.5,
    'JAK1': 0.5, 'JAK2': 0.6, 'STAT3': 0.3, 'SRC': 0.7,
    'FYN': 0.5, 'YES1': 0.6, 'LYN': 0.5, 'MET': 0.5,
    'ALK': 0.4, 'ROS1': 0.4, 'RET': 0.4, 'FGFR1': 0.5,
    'FGFR2': 0.5, 'AXL': 0.4, 'BCL2': 0.6, 'BCL2L1': 0.7,
    'MCL1': 0.6, 'PARP1': 0.5, 'IDH1': 0.4, 'IDH2': 0.4,
    'FLT3': 0.5, 'TP53': 0.9, 'MYC': 0.8, 'RB1': 0.9,
}


# ---------------------------------------------------------------------------
# CANCER TYPE ALIASES (abbreviation → full OncoTree name)
# ---------------------------------------------------------------------------

CANCER_TYPE_ALIASES: Dict[str, str] = {
    'PAAD': 'Pancreatic Adenocarcinoma',
    'PDAC': 'Pancreatic Adenocarcinoma',
    'LUAD': 'Lung Adenocarcinoma',
    'LUSC': 'Lung Squamous Cell Carcinoma',
    'NSCLC': 'Non-Small Cell Lung Cancer',
    'BRCA': 'Invasive Breast Carcinoma',
    'COAD': 'Colon Adenocarcinoma',
    'READ': 'Rectal Adenocarcinoma',
    'CRC': 'Colorectal Adenocarcinoma',
    'SKCM': 'Melanoma',
    'MEL': 'Melanoma',
    'GBM': 'Glioblastoma',
    'OV': 'Ovarian Epithelial Tumor',
    'HGSOC': 'High-Grade Serous Ovarian Cancer',
    'PRAD': 'Prostate Adenocarcinoma',
    'STAD': 'Stomach Adenocarcinoma',
    'ESCA': 'Esophageal Adenocarcinoma',
    'LIHC': 'Hepatocellular Carcinoma',
    'HCC': 'Hepatocellular Carcinoma',
    'BLCA': 'Bladder Urothelial Carcinoma',
    'KIRC': 'Renal Clear Cell Carcinoma',
    'CCRCC': 'Renal Clear Cell Carcinoma',
    'AML': 'Acute Myeloid Leukemia',
    'ALL': 'Acute Lymphoblastic Leukemia',
    'MM': 'Plasma Cell Myeloma',
    'PCM': 'Plasma Cell Myeloma',
}

# Cancer type → search terms for ClinicalTrials.gov
CANCER_SEARCH_TERMS: Dict[str, List[str]] = {
    'Pancreatic Adenocarcinoma': ['pancreatic cancer', 'pancreatic adenocarcinoma', 'PDAC'],
    'Non-Small Cell Lung Cancer': ['NSCLC', 'non-small cell lung cancer', 'lung adenocarcinoma'],
    'Melanoma': ['melanoma', 'metastatic melanoma'],
    'Colorectal Adenocarcinoma': ['colorectal cancer', 'colon cancer', 'CRC'],
    'Ovarian Epithelial Tumor': ['ovarian cancer', 'ovarian carcinoma'],
    'Invasive Breast Carcinoma': ['breast cancer', 'invasive breast carcinoma'],
    'Breast Invasive Carcinoma': ['breast cancer', 'invasive breast carcinoma'],
    'Acute Myeloid Leukemia': ['AML', 'acute myeloid leukemia'],
    'Anaplastic Thyroid Cancer': ['anaplastic thyroid', 'thyroid cancer'],
    'Pleural Mesothelioma': ['mesothelioma', 'pleural mesothelioma'],
    'Hepatocellular Carcinoma': ['hepatocellular carcinoma', 'HCC', 'liver cancer'],
}

# Cancer type aliases for benchmarking match (full name → list of variants)
CANCER_BENCHMARK_ALIASES: Dict[str, List[str]] = {
    'Melanoma': ['Melanoma', 'Cutaneous Melanoma', 'Mucosal Melanoma', 'Ocular Melanoma'],
    'Non-Small Cell Lung Cancer': ['Non-Small Cell Lung Cancer', 'NSCLC', 'Lung Adenocarcinoma', 'Lung'],
    'Pancreatic Adenocarcinoma': ['Pancreatic Adenocarcinoma', 'PDAC', 'Pancreas', 'Ampullary'],
    'Colorectal Adenocarcinoma': ['Colorectal Adenocarcinoma', 'Colon', 'CRC', 'Bowel'],
    'Breast Invasive Carcinoma': ['Invasive Breast Carcinoma', 'Breast Invasive Carcinoma', 'Breast'],
    'Acute Myeloid Leukemia': ['Acute Myeloid Leukemia', 'AML', 'Myeloid'],
    'Renal Cell Carcinoma': ['Renal Cell Carcinoma', 'Kidney'],
    'Head and Neck Squamous Cell Carcinoma': ['Head and Neck Squamous Cell Carcinoma', 'HNSCC', 'Head and Neck'],
    'Diffuse Glioma': ['Diffuse Glioma', 'Glioma', 'CNS'],
    'Liposarcoma': ['Liposarcoma', 'Sarcoma'],
    'Hepatocellular Carcinoma': ['Hepatocellular Carcinoma', 'HCC', 'Liver'],
}


# ---------------------------------------------------------------------------
# GENE EQUIVALENTS (for benchmarking — shared drug class)
# ---------------------------------------------------------------------------

GENE_EQUIVALENTS: Dict[str, Set[str]] = {
    'MAP2K2': {'MAP2K1'},
    'MAP2K1': {'MAP2K2'},
    'CDK4': {'CDK6'},
    'CDK6': {'CDK4'},
    'FGFR1': {'FGFR2'},
    'FGFR2': {'FGFR1'},
    'AKT1': {'AKT2'},
    'AKT2': {'AKT1'},
    'JAK1': {'JAK2'},
    'JAK2': {'JAK1'},
    'KDR': {'VEGFR2'},
    'VEGFR2': {'KDR'},
}


# ---------------------------------------------------------------------------
# DEFAULT SCORING WEIGHTS (used in pipeline combined_score)
# ---------------------------------------------------------------------------

DEFAULT_SCORING_WEIGHTS: Dict[str, float] = {
    'cost': 0.22,
    'synergy': 0.18,
    'resistance': 0.18,
    'combo_tox': 0.18,
    'coverage': 0.14,
}


# ---------------------------------------------------------------------------
# GENE → PATHWAY flat mapping (derived from PATHWAYS)
# ---------------------------------------------------------------------------

def _build_gene_to_pathway() -> Dict[str, str]:
    """Build flat gene→pathway lookup from PATHWAYS dict."""
    mapping: Dict[str, str] = {}
    for pathway, genes in PATHWAYS.items():
        for gene in genes:
            mapping[gene] = pathway
    return mapping


# ---------------------------------------------------------------------------
# PATHWAY DEFINITIONS
# ---------------------------------------------------------------------------

PATHWAYS: Dict[str, Set[str]] = {
    'RAS-MAPK': {'KRAS', 'NRAS', 'HRAS', 'BRAF', 'RAF1', 'MAP2K1', 'MAP2K2', 'MAPK1', 'MAPK3', 'ERK1', 'ERK2'},
    'PI3K-AKT': {'PIK3CA', 'PIK3CB', 'PIK3R1', 'AKT1', 'AKT2', 'MTOR', 'PTEN', 'TSC1', 'TSC2'},
    'JAK-STAT': {'JAK1', 'JAK2', 'TYK2', 'STAT3', 'STAT5A', 'STAT5B'},
    'Cell cycle': {'CDK4', 'CDK6', 'CDK2', 'CCND1', 'CCNE1', 'RB1', 'CDKN2A', 'E2F1'},
    'RTK signaling': {'EGFR', 'ERBB2', 'ERBB3', 'MET', 'FGFR1', 'FGFR2', 'ALK', 'RET', 'AXL', 'IGF1R'},
    'Apoptosis': {'BCL2', 'BCL2L1', 'MCL1', 'BAX', 'BAK1', 'BIM', 'BBC3', 'CASP3', 'CASP9'},
    'SRC family': {'SRC', 'FYN', 'YES1', 'LYN', 'LCK'},
    'Wnt': {'CTNNB1', 'APC', 'GSK3B', 'DVL1', 'WNT1', 'FZD1'},
    'p53': {'TP53', 'MDM2', 'CDKN1A', 'BAX', 'BBC3'},
    'NF-kB': {'NFKB1', 'RELA', 'IKBKB', 'CHUK'},
    'Hippo': {'YAP1', 'WWTR1', 'LATS1', 'LATS2', 'STK3', 'STK4'},
}

# Gene symbol → Ensembl ID mapping (for OpenTargets API)
GENE_TO_ENSEMBL: Dict[str, str] = {
    'EGFR': 'ENSG00000146648', 'KRAS': 'ENSG00000133703', 'BRAF': 'ENSG00000157764',
    'PIK3CA': 'ENSG00000121879', 'TP53': 'ENSG00000141510', 'ERBB2': 'ENSG00000141736',
    'CDK4': 'ENSG00000135446', 'CDK6': 'ENSG00000105810', 'STAT3': 'ENSG00000115415',
    'SRC': 'ENSG00000197122', 'FYN': 'ENSG00000010810', 'MET': 'ENSG00000005968',
    'BCL2': 'ENSG00000171791', 'MCL1': 'ENSG00000143384', 'MTOR': 'ENSG00000198793',
    'JAK1': 'ENSG00000162434', 'JAK2': 'ENSG00000096968', 'ALK': 'ENSG00000171094',
    'MAP2K1': 'ENSG00000169032', 'MAP2K2': 'ENSG00000126934', 'FGFR1': 'ENSG00000077782',
}

# Liaki et al. 2025 citation (used in figures and reports)
LIAKI_CITATION = (
    "Liaki V, Barrambana S, Kostopoulou M, Lechuga CG, et al. 2025. "
    "A targeted combination therapy achieves effective pancreatic cancer "
    "regression and prevents tumor resistance. bioRxiv doi: 10.1101/2025.08.04.668325."
)


def normalize_cancer_type(cancer_type: str) -> str:
    """Normalize cancer type string to full OncoTree name."""
    upper = cancer_type.upper().strip()
    if upper in CANCER_TYPE_ALIASES:
        return CANCER_TYPE_ALIASES[upper]
    return cancer_type.strip()


def get_drugs_for_gene(gene: str) -> List[str]:
    """Look up drugs targeting a gene. Returns empty list if unknown."""
    return GENE_TO_DRUGS.get(gene, [])


def get_druggability_score(gene: str) -> float:
    """Druggability score 0–1 based on clinical stage and drug count."""
    stage = GENE_CLINICAL_STAGE.get(gene, 'preclinical')
    stage_scores = {'approved': 1.0, 'phase3': 0.8, 'phase2': 0.6, 'phase1': 0.4, 'preclinical': 0.2}
    base = stage_scores.get(stage, 0.2)
    n_drugs = len(GENE_TO_DRUGS.get(gene, []))
    bonus = min(0.2, n_drugs * 0.05)
    return min(1.0, base + bonus)


def get_toxicity_score(gene: str) -> float:
    """Toxicity score 0–1 (higher = more toxic). Default 0.5 for unknown genes."""
    return GENE_TOXICITY_SCORES.get(gene, 0.5)


def get_pathway(gene: str) -> str:
    """Return the primary pathway a gene belongs to, or 'Other'."""
    for pathway, members in PATHWAYS.items():
        if gene in members:
            return pathway
    return 'Other'


def expand_with_equivalents(genes: Set[str]) -> Set[str]:
    """Expand a set of gene symbols with known equivalents."""
    expanded = set(genes)
    for g in list(expanded):
        if g in GENE_EQUIVALENTS:
            expanded.update(GENE_EQUIVALENTS[g])
    return expanded


def check_match(predicted: Set[str], gold: Set[str]) -> str:
    """
    Check match type between predicted and gold standard target sets.

    Match levels (highest to lowest):
      'exact'        — expanded predicted == expanded gold
      'superset'     — gold ⊆ predicted (predicted covers all gold targets)
      'pair_overlap' — |expanded ∩ gold_expanded| ≥ 2
      'any_overlap'  — |expanded ∩ gold_expanded| ≥ 1
      'none'         — no overlap
    """
    pred_expanded = expand_with_equivalents(set(predicted))
    gold_expanded = expand_with_equivalents(set(gold))
    overlap = pred_expanded & gold_expanded

    if gold_expanded <= pred_expanded and pred_expanded <= gold_expanded:
        return 'exact'
    if gold_expanded <= pred_expanded:
        return 'superset'
    if len(overlap) >= 2:
        return 'pair_overlap'
    if len(overlap) >= 1:
        return 'any_overlap'
    return 'none'
