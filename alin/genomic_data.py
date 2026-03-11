"""
Genomic data integration for ALIN pipeline.

Parses TCGA PanCanAtlas mc3 somatic mutation data to provide:
  - Per-gene mutation frequency per cancer type
  - Mutation-specific actionability (KRAS G12C, BRAF V600E, etc.)
  - Cancer-type–aware candidate boosting/filtering

Also provides a curated actionability table that replaces the need for
OncoKB (which requires a commercial/academic account).

Data source:
  mc3.v0.2.8.PUBLIC.maf.gz — TCGA Multi-Center Mutation Calling (MC3)
  consensus somatic variant calls across 33 cancer types, ~10K patients.
"""
from __future__ import annotations

import gzip
import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  TCGA study code → DepMap OncotreePrimaryDisease mapping
# ═══════════════════════════════════════════════════════════════════════════════

# Maps each TCGA study abbreviation to the corresponding DepMap
# OncotreePrimaryDisease name(s).  Some TCGA studies map to multiple
# DepMap names; others have no direct counterpart.
TCGA_TO_DEPMAP: Dict[str, List[str]] = {
    "ACC":      ["Adrenocortical Carcinoma"],
    "BLCA":     ["Bladder Urothelial Carcinoma"],
    "BRCA":     ["Breast Invasive Carcinoma", "Breast Cancer"],
    "CESC":     ["Cervical Squamous Cell Carcinoma"],
    "CHOL":     ["Cholangiocarcinoma"],
    "COAD":     ["Colon Adenocarcinoma", "Colorectal Adenocarcinoma", "Colon/Rectal Adenocarcinoma"],
    "DLBC":     ["Diffuse Large B-Cell Lymphoma", "B-Cell Non-Hodgkin Lymphoma"],
    "ESCA":     ["Esophageal Carcinoma", "Esophagogastric Adenocarcinoma", "Esophageal Squamous Cell Carcinoma"],
    "GBM":      ["Glioblastoma Multiforme", "Glioblastoma", "Diffuse Glioma"],
    "HNSC":     ["Head and Neck Squamous Cell Carcinoma"],
    "KICH":     ["Renal Non-Clear Cell Carcinoma", "Chromophobe Renal Cell Carcinoma"],
    "KIRC":     ["Renal Cell Carcinoma", "Clear Cell Renal Cell Carcinoma"],
    "KIRP":     ["Renal Non-Clear Cell Carcinoma", "Papillary Renal Cell Carcinoma"],
    "LAML":     ["Acute Myeloid Leukemia"],
    "LGG":      ["Low-Grade Glioma", "Diffuse Glioma"],
    "LIHC":     ["Hepatocellular Carcinoma", "Liver Hepatocellular Carcinoma"],
    "LUAD":     ["Lung Adenocarcinoma", "Non-Small Cell Lung Cancer"],
    "LUSC":     ["Lung Squamous Cell Carcinoma", "Non-Small Cell Lung Cancer"],
    "MESO":     ["Mesothelioma"],
    "OV":       ["Ovarian Epithelial Tumor", "High-Grade Serous Ovarian Cancer"],
    "PAAD":     ["Pancreatic Adenocarcinoma", "Pancreatic Cancer"],
    "PCPG":     ["Pheochromocytoma"],
    "PRAD":     ["Prostate Adenocarcinoma", "Prostate Cancer"],
    "READ":     ["Colon Adenocarcinoma", "Colorectal Adenocarcinoma", "Colon/Rectal Adenocarcinoma"],
    "SARC":     ["Soft Tissue Sarcoma", "Sarcoma", "Bone Cancer"],
    "SKCM":     ["Melanoma", "Cutaneous Melanoma", "Skin Cutaneous Melanoma"],
    "STAD":     ["Stomach Adenocarcinoma", "Gastric Cancer", "Esophagogastric Adenocarcinoma"],
    "TGCT":     ["Germ Cell Tumor", "Testicular Cancer"],
    "THCA":     ["Thyroid Cancer", "Papillary Thyroid Cancer"],
    "THYM":     ["Thymoma"],
    "UCEC":     ["Endometrial Carcinoma", "Uterine Corpus Endometrial Carcinoma"],
    "UCS":      ["Uterine Carcinosarcoma"],
    "UVM":      ["Uveal Melanoma"],
}

# Reverse map: DepMap disease name → list of TCGA study codes
# (built lazily and cached)
_DEPMAP_TO_TCGA: Optional[Dict[str, List[str]]] = None


def _build_reverse_map() -> Dict[str, List[str]]:
    """Build DepMap→TCGA reverse mapping (case-insensitive)."""
    global _DEPMAP_TO_TCGA
    if _DEPMAP_TO_TCGA is not None:
        return _DEPMAP_TO_TCGA

    result: Dict[str, List[str]] = defaultdict(list)
    for tcga_code, depmap_names in TCGA_TO_DEPMAP.items():
        for name in depmap_names:
            result[name.lower()].append(tcga_code)
    _DEPMAP_TO_TCGA = dict(result)
    return _DEPMAP_TO_TCGA


def depmap_to_tcga_codes(cancer_type: str) -> List[str]:
    """
    Given a DepMap OncotreePrimaryDisease name, return matching TCGA study codes.  
    Falls back to fuzzy substring matching if exact match fails.
    """
    rev = _build_reverse_map()
    key = cancer_type.lower().strip()

    # Exact match
    if key in rev:
        return rev[key]

    # Substring match
    matches = []
    for depmap_name, codes in rev.items():
        if key in depmap_name or depmap_name in key:
            matches.extend(codes)
    if matches:
        return list(set(matches))

    # Word-overlap match (≥2 shared words)
    key_words = set(key.split())
    for depmap_name, codes in rev.items():
        overlap = key_words & set(depmap_name.split())
        if len(overlap) >= 2:
            matches.extend(codes)
    return list(set(matches))


# ═══════════════════════════════════════════════════════════════════════════════
#  TCGA barcode → study code mapping
# ═══════════════════════════════════════════════════════════════════════════════

# The TCGA barcode format is: TCGA-{TSS}-{participant}-{sample}-{portion}-...
# TSS (Tissue Source Site) is a 2-character code (positions 5-6).
# The definitive TSS→study mapping is maintained by the GDC:
# https://gdc.cancer.gov/resources-tcga-users/tcga-code-tables/tissue-source-site-codes
# We embed the complete mapping here to avoid a network dependency.

TSS_TO_STUDY: Dict[str, str] = {
    # ACC
    "OR": "ACC", "OU": "ACC", "PK": "ACC",
    # BLCA
    "BL": "BLCA", "BT": "BLCA", "C4": "BLCA", "CF": "BLCA", "CU": "BLCA",
    "DK": "BLCA", "E7": "BLCA", "FD": "BLCA", "FJ": "BLCA", "FT": "BLCA",
    "G2": "BLCA", "GC": "BLCA", "GD": "BLCA", "GU": "BLCA", "GV": "BLCA",
    "HQ": "BLCA", "K4": "BLCA", "KQ": "BLCA", "LC": "BLCA", "LT": "BLCA",
    "MV": "BLCA", "PQ": "BLCA", "SY": "BLCA", "UY": "BLCA", "XF": "BLCA",
    "ZF": "BLCA", "2F": "BLCA", "4Z": "BLCA",
    # BRCA
    "3C": "BRCA", "4H": "BRCA", "5L": "BRCA", "5T": "BRCA",
    "A1": "BRCA", "A2": "BRCA", "A7": "BRCA", "A8": "BRCA",
    "AC": "BRCA", "AN": "BRCA", "AO": "BRCA", "AQ": "BRCA",
    "AR": "BRCA", "B6": "BRCA", "BH": "BRCA", "C8": "BRCA",
    "D8": "BRCA", "E2": "BRCA", "E9": "BRCA", "EW": "BRCA",
    "GM": "BRCA", "GI": "BRCA", "HN": "BRCA", "JL": "BRCA",
    "LD": "BRCA", "LL": "BRCA", "LQ": "BRCA", "MS": "BRCA",
    "OL": "BRCA", "PE": "BRCA", "PL": "BRCA", "S3": "BRCA",
    "UU": "BRCA", "WT": "BRCA",
    # CESC
    "2W": "CESC", "4J": "CESC", "BI": "CESC", "C5": "CESC",
    "DS": "CESC", "EA": "CESC", "EK": "CESC", "EX": "CESC",
    "FU": "CESC", "HC": "CESC", "HG": "CESC", "HM": "CESC",
    "IR": "CESC", "JW": "CESC", "LP": "CESC", "MA": "CESC",
    "MU": "CESC", "MY": "CESC", "Q1": "CESC", "RA": "CESC",
    "RU": "CESC", "UC": "CESC", "VS": "CESC", "WL": "CESC",
    # CHOL
    "3X": "CHOL", "4G": "CHOL", "W5": "CHOL", "ZH": "CHOL",
    # COAD
    "3L": "COAD", "4N": "COAD", "4T": "COAD", "5M": "COAD",
    "A6": "COAD", "AA": "COAD", "AD": "COAD", "AU": "COAD",
    "AY": "COAD", "AZ": "COAD", "CA": "COAD", "CK": "COAD",
    "CM": "COAD", "D5": "COAD", "DM": "COAD", "F4": "COAD",
    "F5": "COAD", "G4": "COAD", "NH": "COAD", "QG": "COAD",
    "RU": "COAD", "T9": "COAD", "WS": "COAD",
    # DLBC
    "FA": "DLBC", "FF": "DLBC", "FM": "DLBC", "FL": "DLBC",
    "FR": "DLBC", "FS": "DLBC", "GR": "DLBC", "GS": "DLBC",
    "RC": "DLBC", "RF": "DLBC", "RQ": "DLBC",
    # ESCA
    "IG": "ESCA", "L5": "ESCA", "L7": "ESCA", "LN": "ESCA",
    "M9": "ESCA", "Q9": "ESCA", "R6": "ESCA", "RE": "ESCA",
    "S8": "ESCA", "V5": "ESCA", "VR": "ESCA", "Z6": "ESCA",
    "2H": "ESCA",
    # GBM
    "02": "GBM", "06": "GBM", "12": "GBM", "14": "GBM",
    "16": "GBM", "19": "GBM", "26": "GBM", "27": "GBM",
    "28": "GBM", "32": "GBM", "41": "GBM", "76": "GBM",
    # HNSC
    "BA": "HNSC", "BB": "HNSC", "CN": "HNSC", "CR": "HNSC",
    "CV": "HNSC", "CQ": "HNSC", "D6": "HNSC", "DQ": "HNSC",
    "F7": "HNSC", "H7": "HNSC", "HD": "HNSC", "IQ": "HNSC",
    "KU": "HNSC", "MZ": "HNSC", "P3": "HNSC", "QK": "HNSC",
    "RS": "HNSC", "T2": "HNSC", "TN": "HNSC", "UP": "HNSC",
    "UF": "HNSC", "WA": "HNSC",
    # KICH
    "KL": "KICH", "KM": "KICH", "KN": "KICH", "KO": "KICH",
    # KIRC
    "A3": "KIRC", "AK": "KIRC", "AL": "KIRC", "AS": "KIRC",
    "B0": "KIRC", "B2": "KIRC", "B4": "KIRC", "B8": "KIRC",
    "BP": "KIRC", "CJ": "KIRC", "CW": "KIRC", "CZ": "KIRC",
    "DV": "KIRC", "EU": "KIRC", "G7": "KIRC", "HE": "KIRC",
    "IA": "KIRC", "LR": "KIRC", "MM": "KIRC", "MH": "KIRC",
    "P4": "KIRC", "SX": "KIRC", "T7": "KIRC", "UZ": "KIRC",
    # KIRP
    "A4": "KIRP", "AL": "KIRP", "B1": "KIRP", "B9": "KIRP",
    "BQ": "KIRP", "GL": "KIRP", "G6": "KIRP", "HE": "KIRP",
    "J7": "KIRP", "MR": "KIRP", "SB": "KIRP", "UN": "KIRP",
    "Y8": "KIRP",
    # LAML
    "AB": "LAML",
    # LGG
    "CS": "LGG", "DB": "LGG", "DH": "LGG", "DU": "LGG",
    "E1": "LGG", "FG": "LGG", "FN": "LGG", "HT": "LGG",
    "HW": "LGG", "IK": "LGG", "KT": "LGG", "P5": "LGG",
    "QH": "LGG", "R8": "LGG", "S9": "LGG", "TM": "LGG",
    "TQ": "LGG", "VM": "LGG", "VW": "LGG", "WH": "LGG",
    "WY": "LGG",
    # LIHC
    "2V": "LIHC", "2Y": "LIHC", "BC": "LIHC", "BD": "LIHC",
    "BW": "LIHC", "CC": "LIHC", "DD": "LIHC", "ED": "LIHC",
    "EP": "LIHC", "ES": "LIHC", "FV": "LIHC", "G3": "LIHC",
    "HP": "LIHC", "K7": "LIHC", "LG": "LIHC", "MI": "LIHC",
    "MR": "LIHC", "NI": "LIHC", "O8": "LIHC", "PD": "LIHC",
    "QA": "LIHC", "RC": "LIHC", "RN": "LIHC", "T1": "LIHC",
    "UB": "LIHC", "WX": "LIHC", "XR": "LIHC", "ZP": "LIHC",
    "ZS": "LIHC",
    # LUAD
    "05": "LUAD", "17": "LUAD", "35": "LUAD", "38": "LUAD",
    "44": "LUAD", "49": "LUAD", "4B": "LUAD", "50": "LUAD",
    "53": "LUAD", "55": "LUAD", "5P": "LUAD", "62": "LUAD",
    "64": "LUAD", "67": "LUAD", "69": "LUAD", "71": "LUAD",
    "73": "LUAD", "75": "LUAD", "78": "LUAD", "80": "LUAD",
    "83": "LUAD", "86": "LUAD", "91": "LUAD", "93": "LUAD",
    "95": "LUAD", "97": "LUAD", "99": "LUAD", "J2": "LUAD",
    "L4": "LUAD", "MN": "LUAD", "MP": "LUAD", "NJ": "LUAD",
    "NQ": "LUAD", "NK": "LUAD",
    # LUSC
    "18": "LUSC", "21": "LUSC", "22": "LUSC", "33": "LUSC",
    "34": "LUSC", "37": "LUSC", "39": "LUSC", "43": "LUSC",
    "46": "LUSC", "51": "LUSC", "52": "LUSC", "56": "LUSC",
    "58": "LUSC", "60": "LUSC", "63": "LUSC", "66": "LUSC",
    "68": "LUSC", "77": "LUSC", "85": "LUSC", "90": "LUSC",
    "92": "LUSC", "94": "LUSC", "96": "LUSC", "98": "LUSC",
    "NC": "LUSC", "NK": "LUSC", "O2": "LUSC",
    # MESO
    "3H": "MESO", "3U": "MESO", "LK": "MESO", "SC": "MESO",
    "T7": "MESO", "TS": "MESO", "UD": "MESO", "WF": "MESO",
    # OV
    "04": "OV", "09": "OV", "10": "OV", "13": "OV",
    "20": "OV", "23": "OV", "24": "OV", "25": "OV",
    "29": "OV", "30": "OV", "31": "OV", "36": "OV",
    "42": "OV", "57": "OV", "59": "OV", "61": "OV",
    # PAAD
    "2J": "PAAD", "2L": "PAAD", "3A": "PAAD", "3E": "PAAD",
    "F2": "PAAD", "FB": "PAAD", "HV": "PAAD", "HZ": "PAAD",
    "IB": "PAAD", "L1": "PAAD", "LB": "PAAD", "OE": "PAAD",
    "Q3": "PAAD", "RB": "PAAD", "S4": "PAAD", "US": "PAAD",
    "XD": "PAAD", "XN": "PAAD", "YB": "PAAD", "YH": "PAAD",
    "YY": "PAAD",
    # PCPG
    "P7": "PCPG", "P8": "PCPG", "QR": "PCPG", "RW": "PCPG",
    "SQ": "PCPG", "WB": "PCPG",
    # PRAD
    "2A": "PRAD", "CH": "PRAD", "EJ": "PRAD", "FC": "PRAD",
    "G9": "PRAD", "H9": "PRAD", "HC": "PRAD", "HI": "PRAD",
    "J4": "PRAD", "J9": "PRAD", "KC": "PRAD", "KK": "PRAD",
    "M7": "PRAD", "MF": "PRAD", "QU": "PRAD", "TK": "PRAD",
    "V1": "PRAD", "VN": "PRAD", "VP": "PRAD", "WC": "PRAD",
    "X4": "PRAD", "XJ": "PRAD", "YL": "PRAD",
    # READ
    "AF": "READ", "AG": "READ", "BM": "READ", "CI": "READ",
    "DC": "READ", "EI": "READ", "F5": "READ",
    # SARC
    "3B": "SARC", "DX": "SARC", "FH": "SARC", "GN": "SARC",
    "HB": "SARC", "IF": "SARC", "IW": "SARC", "J5": "SARC",
    "JS": "SARC", "K1": "SARC", "LI": "SARC", "MO": "SARC",
    "MX": "SARC", "PC": "SARC", "QQ": "SARC", "R2": "SARC",
    "SU": "SARC", "VT": "SARC", "WK": "SARC", "X6": "SARC",
    "X2": "SARC", "Y6": "SARC", "Z4": "SARC",
    # SKCM
    "3N": "SKCM", "BF": "SKCM", "D3": "SKCM", "D9": "SKCM",
    "DA": "SKCM", "EB": "SKCM", "ER": "SKCM", "EE": "SKCM",
    "FQ": "SKCM", "FR": "SKCM", "FS": "SKCM", "FW": "SKCM",
    "GF": "SKCM", "GN": "SKCM", "HR": "SKCM", "IH": "SKCM",
    "LH": "SKCM", "OD": "SKCM", "QB": "SKCM", "QE": "SKCM",
    "RP": "SKCM", "RX": "SKCM", "VD": "SKCM", "WE": "SKCM",
    "XV": "SKCM", "YG": "SKCM",
    # STAD
    "BR": "STAD", "CD": "STAD", "CG": "STAD", "D7": "STAD",
    "FP": "STAD", "HU": "STAD", "IN": "STAD", "KB": "STAD",
    "LM": "STAD", "MX": "STAD", "R5": "STAD", "RD": "STAD",
    "RE": "STAD", "VQ": "STAD", "W2": "STAD", "XM": "STAD",
    # TGCT
    "2G": "TGCT", "S5": "TGCT", "SM": "TGCT", "W4": "TGCT",
    "WZ": "TGCT", "X3": "TGCT",
    # THCA
    "4C": "THCA", "BJ": "THCA", "DE": "THCA", "DJ": "THCA",
    "DO": "THCA", "E3": "THCA", "E8": "THCA", "EM": "THCA",
    "ET": "THCA", "EY": "THCA", "FE": "THCA", "FX": "THCA",
    "FY": "THCA", "H2": "THCA", "IM": "THCA", "J8": "THCA",
    "KI": "THCA", "KS": "THCA", "MK": "THCA", "ML": "THCA",
    "N5": "THCA", "N9": "THCA",
    # THYM
    "3G": "THYM", "4V": "THYM", "X7": "THYM", "ZL": "THYM",
    "ZB": "THYM",
    # UCEC
    "2E": "UCEC", "AP": "UCEC", "AX": "UCEC", "B5": "UCEC",
    "BG": "UCEC", "BK": "UCEC", "BS": "UCEC", "D1": "UCEC",
    "DI": "UCEC", "DF": "UCEC", "E6": "UCEC", "EC": "UCEC",
    "EM": "UCEC", "EO": "UCEC", "EY": "UCEC", "FI": "UCEC",
    "FL": "UCEC", "GT": "UCEC", "HX": "UCEC", "N5": "UCEC",
    "N7": "UCEC", "N9": "UCEC", "PG": "UCEC", "QM": "UCEC",
    "SO": "UCEC", "SU": "UCEC",
    # UCS
    "N4": "UCS", "N5": "UCS", "N6": "UCS", "NA": "UCS",
    "NB": "UCS", "ND": "UCS", "NF": "UCS", "NG": "UCS",
    "NK": "UCS",
    # UVM
    "RZ": "UVM", "V3": "UVM", "V4": "UVM", "VD": "UVM",
    "WM": "UVM", "YZ": "UVM",
}


def barcode_to_study(barcode: str) -> Optional[str]:
    """
    Extract TCGA study code from a Tumor_Sample_Barcode.
    
    Barcode format: TCGA-{TSS}-{participant}-...
    TSS is positions 5-6 (0-indexed).
    """
    if not barcode.startswith("TCGA-"):
        return None
    parts = barcode.split("-")
    if len(parts) < 3:
        return None
    tss = parts[1]
    return TSS_TO_STUDY.get(tss)


# ═══════════════════════════════════════════════════════════════════════════════
#  Curated actionability table (replaces OncoKB)
# ═══════════════════════════════════════════════════════════════════════════════

# Maps (gene, variant_pattern) → actionability info.
# variant_pattern is matched against HGVSp_Short from the MAF.
# "any" means any non-silent mutation in that gene is relevant.
#
# actionability levels:
#   1 = FDA-approved targeted therapy exists
#   2 = strong clinical evidence (Phase 2/3)
#   3 = preclinical evidence / emerging
#
# conditional_drug: if present, the gene is only "druggable" when this
# specific mutation is present (e.g., KRAS is only druggable via
# sotorasib/adagrasib when the G12C mutation is present).

ACTIONABLE_VARIANTS: List[Dict] = [
    # ─── Level 1: FDA-approved ───────────────────────────────────
    {"gene": "BRAF",  "pattern": "V600E",      "level": 1, "drugs": ["vemurafenib", "dabrafenib", "encorafenib"],
     "cancers": ["SKCM", "LUAD", "COAD", "THCA"]},
    {"gene": "BRAF",  "pattern": "V600K",      "level": 1, "drugs": ["dabrafenib"],
     "cancers": ["SKCM"]},
    {"gene": "EGFR",  "pattern": "L858R",      "level": 1, "drugs": ["osimertinib", "erlotinib", "gefitinib"],
     "cancers": ["LUAD"]},
    {"gene": "EGFR",  "pattern": "exon19del",  "level": 1, "drugs": ["osimertinib"],
     "cancers": ["LUAD"]},  # matched via In_Frame_Del in exon 19
    {"gene": "EGFR",  "pattern": "T790M",      "level": 1, "drugs": ["osimertinib"],
     "cancers": ["LUAD"]},
    {"gene": "KRAS",  "pattern": "G12C",       "level": 1, "drugs": ["sotorasib", "adagrasib"],
     "cancers": ["LUAD", "COAD"], "conditional": True},
    {"gene": "ERBB2", "pattern": "any",        "level": 1, "drugs": ["trastuzumab", "pertuzumab", "T-DXd"],
     "cancers": ["BRCA", "STAD"]},
    {"gene": "ALK",   "pattern": "fusion",     "level": 1, "drugs": ["crizotinib", "alectinib", "lorlatinib"],
     "cancers": ["LUAD"]},
    {"gene": "KIT",   "pattern": "any",        "level": 1, "drugs": ["imatinib"],
     "cancers": ["SARC"]},  # GIST
    {"gene": "PDGFRA","pattern": "any",        "level": 1, "drugs": ["avapritinib"],
     "cancers": ["SARC"]},
    {"gene": "FLT3",  "pattern": "ITD",        "level": 1, "drugs": ["midostaurin", "gilteritinib"],
     "cancers": ["LAML"], "conditional": True},
    {"gene": "IDH1",  "pattern": "R132",       "level": 1, "drugs": ["ivosidenib"],
     "cancers": ["LAML", "CHOL"]},
    {"gene": "IDH2",  "pattern": "R140",       "level": 1, "drugs": ["enasidenib"],
     "cancers": ["LAML"]},
    {"gene": "IDH2",  "pattern": "R172",       "level": 1, "drugs": ["enasidenib"],
     "cancers": ["LAML"]},
    {"gene": "PIK3CA","pattern": "H1047",      "level": 1, "drugs": ["alpelisib"],
     "cancers": ["BRCA"]},
    {"gene": "PIK3CA","pattern": "E545K",      "level": 1, "drugs": ["alpelisib"],
     "cancers": ["BRCA"]},
    {"gene": "PIK3CA","pattern": "E542K",      "level": 1, "drugs": ["alpelisib"],
     "cancers": ["BRCA"]},
    {"gene": "NTRK1", "pattern": "fusion",     "level": 1, "drugs": ["larotrectinib", "entrectinib"],
     "cancers": []},  # tumor-agnostic
    {"gene": "NTRK2", "pattern": "fusion",     "level": 1, "drugs": ["larotrectinib", "entrectinib"],
     "cancers": []},
    {"gene": "NTRK3", "pattern": "fusion",     "level": 1, "drugs": ["larotrectinib", "entrectinib"],
     "cancers": []},
    {"gene": "FGFR2", "pattern": "fusion",     "level": 1, "drugs": ["pemigatinib", "futibatinib"],
     "cancers": ["CHOL"]},
    {"gene": "FGFR3", "pattern": "any",        "level": 1, "drugs": ["erdafitinib"],
     "cancers": ["BLCA"]},
    {"gene": "RET",   "pattern": "fusion",     "level": 1, "drugs": ["selpercatinib", "pralsetinib"],
     "cancers": ["LUAD", "THCA"]},
    {"gene": "RET",   "pattern": "any",        "level": 1, "drugs": ["selpercatinib"],
     "cancers": ["THCA"]},  # MTC with activating RET
    {"gene": "BRCA1", "pattern": "any",        "level": 1, "drugs": ["olaparib", "rucaparib"],
     "cancers": ["OV", "BRCA", "PAAD", "PRAD"]},
    {"gene": "BRCA2", "pattern": "any",        "level": 1, "drugs": ["olaparib", "rucaparib"],
     "cancers": ["OV", "BRCA", "PAAD", "PRAD"]},

    # ─── Level 2: Strong clinical evidence ───────────────────────
    {"gene": "KRAS",  "pattern": "G12D",       "level": 2, "drugs": ["MRTX1133"],
     "cancers": ["PAAD", "COAD", "LUAD"]},
    {"gene": "KRAS",  "pattern": "G12V",       "level": 2, "drugs": [],
     "cancers": ["PAAD"]},  # recognized driver, no approved drug yet
    {"gene": "KRAS",  "pattern": "G12R",       "level": 2, "drugs": [],
     "cancers": ["PAAD"]},
    {"gene": "KRAS",  "pattern": "G13D",       "level": 2, "drugs": [],
     "cancers": ["COAD"]},
    {"gene": "MET",   "pattern": "exon14skip", "level": 2, "drugs": ["capmatinib", "tepotinib"],
     "cancers": ["LUAD"]},
    {"gene": "NRAS",  "pattern": "Q61",        "level": 2, "drugs": ["binimetinib"],
     "cancers": ["SKCM"]},
    {"gene": "ARID1A","pattern": "any",        "level": 2, "drugs": ["EZH2i"],
     "cancers": ["OV", "UCEC", "STAD"]},
    {"gene": "TP53",  "pattern": "any",        "level": 2, "drugs": [],
     "cancers": []},  # universal driver, no targeted therapy yet
    {"gene": "PTEN",  "pattern": "any",        "level": 2, "drugs": ["AKT inhibitors"],
     "cancers": ["PRAD", "UCEC", "GBM"]},
    {"gene": "APC",   "pattern": "any",        "level": 2, "drugs": [],
     "cancers": ["COAD", "READ"]},
    {"gene": "SMAD4", "pattern": "any",        "level": 2, "drugs": [],
     "cancers": ["PAAD", "COAD"]},
    {"gene": "CDKN2A","pattern": "any",        "level": 2, "drugs": ["CDK4/6 inhibitors"],
     "cancers": ["SKCM", "GBM", "PAAD", "LUAD"]},
    {"gene": "RB1",   "pattern": "any",        "level": 2, "drugs": [],
     "cancers": ["LUSC", "BLCA"]},
    {"gene": "STK11", "pattern": "any",        "level": 2, "drugs": [],
     "cancers": ["LUAD"]},

    # ─── Level 3: Emerging / preclinical ─────────────────────────
    {"gene": "MAP2K1","pattern": "any",        "level": 3, "drugs": ["trametinib", "selumetinib"],
     "cancers": ["SKCM", "LUAD"]},
    {"gene": "STAT3", "pattern": "any",        "level": 3, "drugs": [],
     "cancers": ["LAML"]},  # activated but not commonly mutated
    {"gene": "CDK4",  "pattern": "amplification", "level": 2, "drugs": ["palbociclib", "ribociclib"],
     "cancers": ["SARC"]},  # well-differentiated liposarcoma
    {"gene": "MDM2",  "pattern": "amplification", "level": 3, "drugs": ["milademetan"],
     "cancers": ["SARC"]},
]


# Build lookup: gene → list of actionable variant entries
_ACTIONABLE_BY_GENE: Dict[str, List[Dict]] = defaultdict(list)
for _entry in ACTIONABLE_VARIANTS:
    _ACTIONABLE_BY_GENE[_entry["gene"]].append(_entry)
_ACTIONABLE_BY_GENE = dict(_ACTIONABLE_BY_GENE)


def get_actionable_info(gene: str) -> List[Dict]:
    """Return actionability entries for a gene, or empty list."""
    return _ACTIONABLE_BY_GENE.get(gene, [])


def is_conditionally_druggable(gene: str) -> bool:
    """
    Check if a gene's druggability is conditional on a specific mutation.
    
    For example, KRAS is only druggable (via sotorasib) if the G12C
    mutation is present.  If a cancer type doesn't have G12C mutations,
    KRAS should not get a druggability bonus.
    """
    entries = _ACTIONABLE_BY_GENE.get(gene, [])
    return any(e.get("conditional", False) for e in entries)


# ═══════════════════════════════════════════════════════════════════════════════
#  MC3 MAF Parser — builds per-gene mutation frequency matrix
# ═══════════════════════════════════════════════════════════════════════════════

# Variant classifications considered "non-silent" (functional impact)
NONSILENT_CLASSIFICATIONS = frozenset({
    "Missense_Mutation", "Nonsense_Mutation",
    "Frame_Shift_Del", "Frame_Shift_Ins",
    "Splice_Site", "In_Frame_Del", "In_Frame_Ins",
    "Nonstop_Mutation", "Translation_Start_Site",
})


class TCGAMutationLoader:
    """
    Loads and summarizes somatic mutation data from the TCGA MC3 MAF file.
    
    Provides:
      - mutation_freq(gene, tcga_code) → fraction of patients mutated
      - variant_present(gene, pattern, tcga_code) → bool
      - top_mutated_genes(tcga_code, n) → list of (gene, freq)
      - mutation_freq_for_cancer(gene, depmap_cancer) → float
        (translates DepMap cancer names to TCGA codes)
    
    The data is pre-aggregated and cached to avoid re-parsing the large
    MAF file on every pipeline run.
    """
    
    CACHE_FILE = "data/mc3_mutation_summary.json.gz"
    
    def __init__(self, maf_path: str = "data/mc3.v0.2.8.PUBLIC.maf.gz",
                 cache_dir: str = "data"):
        self.maf_path = Path(maf_path)
        self.cache_path = Path(cache_dir) / "mc3_mutation_summary.json.gz"
        
        # gene → study → fraction of patients with non-silent mutation
        self._freq: Dict[str, Dict[str, float]] = {}
        # gene → study → set of HGVSp_Short variants seen
        self._variants: Dict[str, Dict[str, Set[str]]] = {}
        # study → total patient count
        self._study_patients: Dict[str, int] = {}
        
        self._loaded = False
    
    def load(self) -> None:
        """Load mutation data from cache or parse from MAF."""
        if self._loaded:
            return
        
        if self.cache_path.exists():
            self._load_cache()
        elif self.maf_path.exists():
            self._parse_maf()
            self._save_cache()
        else:
            logger.warning("No MAF file or cache found at %s — mutation data unavailable",
                           self.maf_path)
            self._loaded = True
            return
        
        self._loaded = True
        n_studies = len(self._study_patients)
        n_genes = len(self._freq)
        total_patients = sum(self._study_patients.values())
        logger.info("Mutation data loaded: %d genes, %d studies, %d patients",
                    n_genes, n_studies, total_patients)
    
    def _parse_maf(self) -> None:
        """Parse the mc3 MAF file and build mutation frequency tables."""
        logger.info("Parsing MAF file %s (this may take 1-2 minutes)...", self.maf_path)
        
        # Intermediate: gene → study → set of patient barcodes
        gene_study_patients: Dict[str, Dict[str, set]] = defaultdict(lambda: defaultdict(set))
        # study → set of all patients  
        study_patients: Dict[str, set] = defaultdict(set)
        # gene → study → set of HGVSp_Short variants
        gene_study_variants: Dict[str, Dict[str, set]] = defaultdict(lambda: defaultdict(set))
        
        unmapped_tss = set()
        
        with gzip.open(self.maf_path, "rt") as f:
            header = f.readline().strip().split("\t")
            
            # Column indices
            idx_hugo = header.index("Hugo_Symbol")
            idx_var_class = header.index("Variant_Classification")
            idx_barcode = header.index("Tumor_Sample_Barcode")
            idx_hgvsp = header.index("HGVSp_Short")
            idx_filter = header.index("FILTER")
            
            for i, line in enumerate(f):
                if i % 1_000_000 == 0 and i > 0:
                    logger.debug("  ...parsed %d variants", i)
                
                cols = line.split("\t")
                
                # Only PASS variants
                filt = cols[idx_filter].strip()
                if filt != "PASS":
                    continue
                
                # Only non-silent
                var_class = cols[idx_var_class]
                if var_class not in NONSILENT_CLASSIFICATIONS:
                    continue
                
                gene = cols[idx_hugo]
                barcode = cols[idx_barcode]
                patient = barcode[:12]  # TCGA patient barcode
                
                # Map to study code
                study = barcode_to_study(barcode)
                if study is None:
                    tss = barcode.split("-")[1] if len(barcode.split("-")) > 1 else "??"
                    unmapped_tss.add(tss)
                    continue
                
                study_patients[study].add(patient)
                gene_study_patients[gene][study].add(patient)
                
                # Record variant
                hgvsp = cols[idx_hgvsp].strip()
                if hgvsp and hgvsp != ".":
                    gene_study_variants[gene][study].add(hgvsp)
        
        if unmapped_tss:
            logger.warning("Could not map %d TSS codes to studies: %s",
                           len(unmapped_tss), sorted(unmapped_tss)[:10])
        
        # Convert to frequencies
        self._study_patients = {s: len(pts) for s, pts in study_patients.items()}
        
        for gene, studies in gene_study_patients.items():
            self._freq[gene] = {}
            for study, pts in studies.items():
                total = self._study_patients.get(study, 1)
                self._freq[gene][study] = len(pts) / total
        
        self._variants = {}
        for gene, studies in gene_study_variants.items():
            self._variants[gene] = {s: vs for s, vs in studies.items()}
        
        logger.info("MAF parsing complete: %d genes, %d studies",
                    len(self._freq), len(self._study_patients))
    
    def _save_cache(self) -> None:
        """Save parsed data to compressed JSON cache."""
        cache_data = {
            "study_patients": self._study_patients,
            "freq": self._freq,
            "variants": {
                gene: {study: sorted(vs) for study, vs in studies.items()}
                for gene, studies in self._variants.items()
            },
        }
        
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(self.cache_path, "wt") as f:
            json.dump(cache_data, f)
        
        size_mb = self.cache_path.stat().st_size / (1024 * 1024)
        logger.info("Mutation cache saved to %s (%.1f MB)", self.cache_path, size_mb)
    
    def _load_cache(self) -> None:
        """Load pre-parsed data from compressed JSON cache."""
        logger.info("Loading mutation cache from %s", self.cache_path)
        with gzip.open(self.cache_path, "rt") as f:
            data = json.load(f)
        
        self._study_patients = data["study_patients"]
        self._freq = data["freq"]
        self._variants = {
            gene: {study: set(vs) for study, vs in studies.items()}
            for gene, studies in data.get("variants", {}).items()
        }
    
    # ── Public API ────────────────────────────────────────────────
    
    def mutation_freq(self, gene: str, tcga_code: str) -> float:
        """
        Fraction of patients with a non-silent mutation in `gene`
        within TCGA study `tcga_code`.  Returns 0.0 if gene/study unknown.
        """
        self.load()
        return self._freq.get(gene, {}).get(tcga_code, 0.0)
    
    def mutation_freq_for_cancer(self, gene: str, depmap_cancer: str) -> float:
        """
        Get mutation frequency for a DepMap cancer name.
        Aggregates across all matching TCGA studies (e.g., NSCLC → LUAD + LUSC).
        """
        self.load()
        tcga_codes = depmap_to_tcga_codes(depmap_cancer)
        if not tcga_codes:
            return 0.0
        
        total_patients = 0
        mutated_patients = 0
        for code in tcga_codes:
            n = self._study_patients.get(code, 0)
            total_patients += n
            mutated_patients += self._freq.get(gene, {}).get(code, 0.0) * n
        
        return mutated_patients / total_patients if total_patients > 0 else 0.0
    
    def variant_present(self, gene: str, pattern: str, tcga_code: str) -> bool:
        """
        Check if a specific variant pattern is seen in a TCGA study.
        `pattern` is matched as a substring of HGVSp_Short values.
        """
        self.load()
        variants = self._variants.get(gene, {}).get(tcga_code, set())
        if pattern == "any":
            return len(variants) > 0
        return any(pattern in v for v in variants)
    
    def variant_present_for_cancer(self, gene: str, pattern: str,
                                    depmap_cancer: str) -> bool:
        """Check variant presence using DepMap cancer name."""
        self.load()
        for code in depmap_to_tcga_codes(depmap_cancer):
            if self.variant_present(gene, pattern, code):
                return True
        return False
    
    def top_mutated_genes(self, tcga_code: str, n: int = 50) -> List[Tuple[str, float]]:
        """Return top N most-frequently-mutated genes for a TCGA study."""
        self.load()
        gene_freqs = []
        for gene, studies in self._freq.items():
            freq = studies.get(tcga_code, 0.0)
            if freq > 0:
                gene_freqs.append((gene, freq))
        gene_freqs.sort(key=lambda x: -x[1])
        return gene_freqs[:n]
    
    def top_mutated_for_cancer(self, depmap_cancer: str, n: int = 50) -> List[Tuple[str, float]]:
        """Return top N most-frequently-mutated genes using DepMap cancer name."""
        self.load()
        tcga_codes = depmap_to_tcga_codes(depmap_cancer)
        if not tcga_codes:
            return []
        
        gene_freqs: Dict[str, float] = {}
        for gene in self._freq:
            freq = self.mutation_freq_for_cancer(gene, depmap_cancer)
            if freq > 0:
                gene_freqs[gene] = freq
        
        return sorted(gene_freqs.items(), key=lambda x: -x[1])[:n]
    
    def get_actionable_mutations(self, depmap_cancer: str) -> List[Dict]:
        """
        Return actionable mutations present in this cancer type.
        Combines our curated actionability table with actual TCGA data.
        """
        self.load()
        tcga_codes = depmap_to_tcga_codes(depmap_cancer)
        if not tcga_codes:
            return []
        
        results = []
        for entry in ACTIONABLE_VARIANTS:
            gene = entry["gene"]
            pattern = entry["pattern"]
            
            # Check if variant is present in any matching TCGA study
            for code in tcga_codes:
                if entry.get("cancers") and code not in entry["cancers"]:
                    # This actionable variant is specific to other cancers
                    # Still check — it may be present as an off-label finding
                    pass
                
                if pattern in ("any", "fusion", "amplification"):
                    # For "any" — check if gene is mutated at all
                    freq = self.mutation_freq(gene, code)
                    if freq > 0.01:  # ≥1% of patients
                        results.append({
                            **entry,
                            "tcga_code": code,
                            "frequency": freq,
                            "in_expected_cancer": code in entry.get("cancers", []),
                        })
                else:
                    # Check for specific variant
                    if self.variant_present(gene, pattern, code):
                        freq = self.mutation_freq(gene, code)
                        results.append({
                            **entry,
                            "tcga_code": code,
                            "frequency": freq,
                            "in_expected_cancer": code in entry.get("cancers", []),
                        })
        
        return results
    
    @property
    def available_studies(self) -> List[str]:
        """List of TCGA study codes with data."""
        self.load()
        return sorted(self._study_patients.keys())
    
    @property
    def study_patient_counts(self) -> Dict[str, int]:
        """Study → patient count mapping."""
        self.load()
        return dict(self._study_patients)


# ═══════════════════════════════════════════════════════════════════════════════
#  Genomic-aware scoring helpers (used by TripleCombinationFinder)
# ═══════════════════════════════════════════════════════════════════════════════

# Minimum mutation frequency to consider a gene "genomically relevant"
# in a cancer type.  Genes below this threshold get no mutation bonus
# and may be penalized if they're conditionally druggable.
MIN_MUTATION_FREQ_RELEVANT: float = 0.02  # 2% of patients

# Minimum mutation frequency for a gene to be considered a "driver"
# in a cancer type (gets strong bonus).
MIN_MUTATION_FREQ_DRIVER: float = 0.10  # 10% of patients

# Weight for the genomic relevance bonus in combined_score.
# This must be large enough to compete with druggability (max 0.27)
# and selectivity (max 0.18).  At 0.50 with bonus range [0, 0.5],
# the max shift is 0.25 — comparable to druggability.
W_GENOMIC: float = 0.50


# ═══════════════════════════════════════════════════════════════════════════════
#  Curated CNV events (amplifications / deletions) per cancer type
# ═══════════════════════════════════════════════════════════════════════════════
# The mc3 MAF only captures point mutations.  Many oncogenic events are
# amplifications (ERBB2, MYC, CCND1, MDM2) or homozygous deletions
# (CDKN2A, RB1, PTEN).  We add curated prevalence estimates from
# TCGA PanCanAtlas GISTIC2 analyses (Beroukhim et al. 2010; Zack et al.
# 2013) and cancer-specific literature.
#
# Format: gene → {TCGA_code → estimated_frequency}
# These are ADDED to the point-mutation frequency from the MAF.

CURATED_CNV: Dict[str, Dict[str, float]] = {
    # HER2 amplification
    "ERBB2": {"BRCA": 0.15, "STAD": 0.15, "ESCA": 0.08, "BLCA": 0.06},
    # MYC amplification
    "MYC":   {"BRCA": 0.15, "OV": 0.30, "LUAD": 0.08, "LUSC": 0.10,
              "LIHC": 0.12, "ESCA": 0.10, "SKCM": 0.05},
    # CCND1 (Cyclin D1) amplification
    "CCND1": {"BRCA": 0.15, "HNSC": 0.25, "BLCA": 0.10, "ESCA": 0.15,
              "LIHC": 0.10},
    # CDKN2A homozygous deletion
    "CDKN2A":{"GBM": 0.55, "PAAD": 0.30, "LUAD": 0.15, "SKCM": 0.45,
              "MESO": 0.60, "ESCA": 0.15, "HNSC": 0.15, "BLCA": 0.10},
    # PTEN homozygous deletion / loss
    "PTEN":  {"GBM": 0.35, "PRAD": 0.15, "UCEC": 0.10, "SKCM": 0.10,
              "BRCA": 0.05},
    # RB1 loss
    "RB1":   {"LUSC": 0.15, "BLCA": 0.12, "SARC": 0.10, "OV": 0.08},
    # MDM2 amplification
    "MDM2":  {"SARC": 0.20, "GBM": 0.08, "LUAD": 0.05},
    # CDK4 amplification
    "CDK4":  {"SARC": 0.15, "GBM": 0.12},
    # EGFR amplification
    "EGFR":  {"GBM": 0.40, "LUAD": 0.12, "HNSC": 0.10, "ESCA": 0.08},
    # FGFR1 amplification
    "FGFR1": {"LUSC": 0.18, "HNSC": 0.10, "BRCA": 0.08},
    # PIK3CA amplification
    "PIK3CA":{"HNSC": 0.20, "LUSC": 0.15, "CESC": 0.12, "STAD": 0.08},
    # KRAS amplification (in addition to point mutations)
    "KRAS":  {"STAD": 0.06, "OV": 0.05},
    # CCNE1 amplification
    "CCNE1": {"OV": 0.20, "BRCA": 0.05, "UCEC": 0.05},
    # SOX2 amplification
    "SOX2":  {"LUSC": 0.20, "ESCA": 0.10},
    # NKX2-1/TITF1 amplification
    "NKX2-1":{"LUAD": 0.12},
    # MYCN amplification
    "MYCN":  {"GBM": 0.05},  # also neuroblastoma but that's paediatric
}


# ═══════════════════════════════════════════════════════════════════════════════
#  Known passenger / long-gene artifacts
# ═══════════════════════════════════════════════════════════════════════════════
# These genes are very large and accumulate many somatic mutations simply
# due to their coding length, NOT because they are cancer drivers.  They
# will appear as "top mutated" in any pan-cancer analysis but should never
# be recommended as therapeutic targets.
#
# Sources:
#  - Lawrence et al., Nature 2013 (background mutation rate model)
#  - Vogelstein et al., Science 2013 (cancer gene census)
#  - Known from TCGA analyses to be recurrently artefactual

KNOWN_PASSENGERS: FrozenSet[str] = frozenset({
    "TTN",      # titin (363 exons, largest human protein)
    "MUC16",    # mucin-16 / CA-125 (very large mucin)
    "MUC4",     # mucin-4
    "MUC17",    # mucin-17
    "CSMD1",    # CUB-sushi multiple domain 1
    "CSMD3",    # CUB-sushi multiple domain 3
    "DNAH5",    # dynein axonemal heavy chain 5
    "DNAH7",    # dynein axonemal heavy chain 7
    "DNAH9",    # dynein axonemal heavy chain 9
    "DNAH11",   # dynein axonemal heavy chain 11
    "DNAH17",   # dynein axonemal heavy chain 17
    "GPR98",    # adhesion GPCR V1 (ADGRV1)
    "LRP1B",    # LDL receptor related protein 1B
    "PCLO",     # piccolo presynaptic cytomatrix protein
    "RYR1",     # ryanodine receptor 1
    "RYR2",     # ryanodine receptor 2
    "RYR3",     # ryanodine receptor 3
    "OBSCN",    # obscurin
    "SYNE1",    # spectrin repeat nuclear envelope 1 (nesprin-1)
    "SYNE2",    # spectrin repeat nuclear envelope 2
    "HMCN1",    # hemicentin-1
    "USH2A",    # usherin (Usher syndrome)
    "FLG",      # filaggrin
    "XIRP2",    # xin actin binding repeat containing 2
    "FAT3",     # FAT atypical cadherin 3
    "FAT4",     # FAT atypical cadherin 4
    "ZFHX3",    # zinc finger homeobox 3
    "PKHD1",    # PKHD1 ciliary IPT domain containing fibrocystin
    "PKHD1L1",  # PKHD1 like 1
    "LAMA2",    # laminin subunit alpha 2
    "DST",      # dystonin
    "SPTA1",    # spectrin alpha
    "AHNAK2",   # AHNAK nucleoprotein 2
    "COL11A1",  # collagen type XI alpha 1
    "MACF1",    # microtubule actin crosslinking factor 1
    "FSIP2",    # fibrous sheath interacting protein 2
    # Additional genes frequently appearing as false positives in
    # high-TMB cancers (melanoma, lung) — large coding length or
    # repetitive structure leads to passenger accumulation
    "HYDIN",    # HYDIN axonemal central pair apparatus protein
    "ANK3",     # ankyrin 3 (very large, 4,377 aa)
    "APOB",     # apolipoprotein B (4,563 aa)
    "DSCAM",    # DS cell adhesion molecule (2,013 aa)
    "THSD7B",   # thrombospondin type 1 domain containing 7B
    "CSMD2",    # CUB-sushi multiple domain 2
    "MGAM",     # maltase-glucoamylase
    "RP1",      # RP1 axonemal microtubule associated
    "NEB",      # nebulin (very large, 8,563 aa)
    "UNC80",    # unc-80 homolog (3,258 aa)
    "MUC5B",    # mucin-5B
    "ABCA13",   # ATP binding cassette A13 (very large)
    "ZFHX4",    # zinc finger homeobox 4
    "LAMA1",    # laminin alpha 1 (very large)
    "LAMA3",    # laminin alpha 3
    "HERC2",    # HECT E3 ubiquitin ligase (4,834 aa)
    "SACS",     # sacsin (4,579 aa)
    "CUBN",     # cubilin (3,623 aa)
    "DNAH1",    # dynein axonemal heavy chain 1
    "DNAH2",    # dynein axonemal heavy chain 2
    "DNAH3",    # dynein axonemal heavy chain 3
    "DNAH6",    # dynein axonemal heavy chain 6
    "DNAH8",    # dynein axonemal heavy chain 8
    "DNAH10",   # dynein axonemal heavy chain 10
    "DNAH12",   # dynein axonemal heavy chain 12
    "DNAH14",   # dynein axonemal heavy chain 14
})


# ═══════════════════════════════════════════════════════════════════════════════
#  Curated AML (LAML) driver frequencies
# ═══════════════════════════════════════════════════════════════════════════════
# TCGA LAML was sequenced before the Pan-Cancer mc3 pipeline and is NOT
# included in the mc3 MAF.  We add literature-curated driver frequencies
# from TCGA AML publication (Cancer Genome Atlas Research Network, NEJM 2013)
# and large-scale sequencing studies (Papaemmanuil et al., NEJM 2016).
#
# Format: gene → frequency (fraction of patients)

CURATED_LAML_DRIVERS: Dict[str, float] = {
    "FLT3":   0.28,   # FLT3-ITD (~20%) + FLT3-TKD (~8%)
    "NPM1":   0.27,   # nucleophosmin insertion
    "DNMT3A": 0.26,   # DNA methyltransferase
    "IDH2":   0.12,   # isocitrate dehydrogenase 2
    "IDH1":   0.08,   # isocitrate dehydrogenase 1
    "RUNX1":  0.15,   # runt-related transcription factor
    "TET2":   0.12,   # tet methylcytosine dioxygenase 2
    "TP53":   0.08,   # tumor suppressor
    "NRAS":   0.12,   # neuroblastoma RAS
    "CEBPA":  0.10,   # CCAAT enhancer binding protein alpha
    "WT1":    0.06,   # Wilms tumor 1
    "KIT":    0.06,   # stem cell factor receptor
    "PTPN11": 0.05,   # SHP2 phosphatase
    "KRAS":   0.05,   # KRAS
    "EZH2":   0.03,   # enhancer of zeste 2
    "ASXL1":  0.17,   # additional sex combs like 1
    "SRSF2":  0.12,   # serine/arginine-rich splicing factor
    "SF3B1":  0.08,   # splicing factor 3b subunit 1
    "U2AF1":  0.06,   # U2 small nuclear RNA auxiliary factor 1
    "STAG2":  0.05,   # cohesin subunit
    "RAD21":  0.04,   # cohesin subunit
    "SMC1A":  0.03,   # cohesin subunit
    "SMC3":   0.03,   # cohesin subunit
    "PHF6":   0.05,   # PHD finger protein 6
    "BCOR":   0.04,   # BCL6 corepressor
}


def get_combined_alteration_freq(
    gene: str,
    cancer_type: str,
    mutation_loader: Optional[TCGAMutationLoader],
) -> float:
    """
    Get the combined alteration frequency for a gene in a cancer type,
    including point mutations (MAF), CNV events (curated), and fallback
    curated data for cancer types missing from mc3 (e.g. LAML).

    Known passenger genes (TTN, MUC16, etc.) are forced to return 0.0
    so they never appear as "drivers".
    """
    # Blacklist: large-gene artifacts should never score as drivers
    if gene in KNOWN_PASSENGERS:
        return 0.0

    freq = 0.0

    # Check curated LAML data if this maps to LAML
    tcga_codes = depmap_to_tcga_codes(cancer_type)
    if "LAML" in tcga_codes and gene in CURATED_LAML_DRIVERS:
        freq = max(freq, CURATED_LAML_DRIVERS[gene])

    # MAF-based frequency
    if mutation_loader is not None and mutation_loader._loaded:
        maf_freq = mutation_loader.mutation_freq_for_cancer(gene, cancer_type)
        freq = max(freq, maf_freq)

    # Add curated CNV frequency
    if gene in CURATED_CNV:
        for code in tcga_codes:
            cnv_freq = CURATED_CNV[gene].get(code, 0.0)
            # Combine: 1 - (1 - point_mut) * (1 - cnv)  ≈ point_mut + cnv
            freq = 1.0 - (1.0 - freq) * (1.0 - cnv_freq)
    return freq


def compute_genomic_bonus(
    combo: Tuple[str, ...],
    cancer_type: str,
    mutation_loader: TCGAMutationLoader,
) -> float:
    """
    Compute a genomic relevance bonus for a combination of targets.

    The bonus reflects how well the combo's targets match the actual
    mutational landscape of this cancer type.  It is strong enough to
    meaningfully re-rank combos (max ~0.50, scaled by W_GENOMIC = 0.50
    for an effective max shift of ~0.25).

    Scoring per gene:
      driver (≥10% altered): 0.35 + 0.30 * freq  (≈ 0.38–0.50)
      relevant (2–10%):      0.10 + freq          (≈ 0.12–0.20)
      irrelevant (<2%):      -0.10 (penalty)
      conditional w/o variant: -0.15 (stronger penalty)

    Returns a value in [-0.15, ~0.50] where higher = more relevant.
    The caller should SUBTRACT this from combined_score (lower = better).
    """
    if not mutation_loader._loaded:
        return 0.0

    scores = []
    for gene in combo:
        freq = get_combined_alteration_freq(gene, cancer_type, mutation_loader)

        # Check conditional druggability
        if is_conditionally_druggable(gene):
            entries = get_actionable_info(gene)
            has_actionable = False
            for entry in entries:
                if entry.get("conditional"):
                    pattern = entry["pattern"]
                    if mutation_loader.variant_present_for_cancer(
                        gene, pattern, cancer_type
                    ):
                        has_actionable = True
                        break
            if not has_actionable:
                # E.g. KRAS without G12C → strong penalty
                scores.append(-0.15)
                continue

        if freq >= MIN_MUTATION_FREQ_DRIVER:
            # Strong driver: large bonus, scaled by frequency
            scores.append(0.35 + 0.30 * min(freq, 0.5))
        elif freq >= MIN_MUTATION_FREQ_RELEVANT:
            # Relevant but not dominant driver
            scores.append(0.10 + freq)
        else:
            # Not altered in this cancer → penalty
            # This is the key difference from the old version:
            # genes with NO genomic footprint should be actively
            # disfavoured, not just ignored.
            scores.append(-0.10)

    return sum(scores) / len(scores) if scores else 0.0


def compute_genomic_candidate_boost(
    candidate_genes: List[str],
    cancer_type: str,
    mutation_loader: TCGAMutationLoader,
    boost_threshold: float = MIN_MUTATION_FREQ_DRIVER,
) -> Set[str]:
    """
    Identify candidate genes that should be boosted to priority status
    based on high mutation/alteration frequency in this cancer type.
    Uses combined point-mutation + CNV frequency.
    """
    if not mutation_loader._loaded:
        return set()

    boosted = set()
    for gene in candidate_genes:
        freq = get_combined_alteration_freq(gene, cancer_type, mutation_loader)
        if freq >= boost_threshold:
            boosted.add(gene)
    return boosted


def get_cancer_driver_genes(
    cancer_type: str,
    mutation_loader: TCGAMutationLoader,
    min_freq: float = MIN_MUTATION_FREQ_DRIVER,
    max_genes: int = 30,
) -> List[Tuple[str, float]]:
    """
    Get the top driver genes for a cancer type based on TCGA mutation data.
    Includes curated CNV events, curated LAML drivers, and filters out
    known passenger/long-gene artifacts (TTN, MUC16, etc.).
    """
    if not mutation_loader._loaded:
        # Still try curated LAML data even without MAF
        tcga_codes = depmap_to_tcga_codes(cancer_type)
        if "LAML" in tcga_codes:
            results = [(g, f) for g, f in CURATED_LAML_DRIVERS.items()
                       if f >= min_freq and g not in KNOWN_PASSENGERS]
            results.sort(key=lambda x: -x[1])
            return results[:max_genes]
        return []

    # Start with MAF-based top genes (request more to compensate for filtering)
    maf_top = mutation_loader.top_mutated_for_cancer(cancer_type, n=max_genes * 3)
    gene_freq: Dict[str, float] = {g: f for g, f in maf_top
                                    if g not in KNOWN_PASSENGERS}

    # Augment with curated LAML drivers if applicable
    tcga_codes = depmap_to_tcga_codes(cancer_type)
    if "LAML" in tcga_codes:
        for gene, freq in CURATED_LAML_DRIVERS.items():
            if gene not in KNOWN_PASSENGERS:
                gene_freq[gene] = max(gene_freq.get(gene, 0.0), freq)

    # Augment with CNV events
    for gene, code_freqs in CURATED_CNV.items():
        if gene in KNOWN_PASSENGERS:
            continue
        for code in tcga_codes:
            if code in code_freqs:
                old = gene_freq.get(gene, 0.0)
                cnv_f = code_freqs[code]
                gene_freq[gene] = 1.0 - (1.0 - old) * (1.0 - cnv_f)

    # Filter and sort
    results = [(g, f) for g, f in gene_freq.items() if f >= min_freq]
    results.sort(key=lambda x: -x[1])
    return results[:max_genes]


def get_driver_injection_set(
    cancer_type: str,
    mutation_loader: TCGAMutationLoader,
    max_inject: int = 10,
) -> Set[str]:
    """
    Return the top TCGA driver genes for direct injection into the
    candidate pool.  Unlike compute_genomic_candidate_boost (which
    only boosts genes already in the path-analysis pool), this returns
    genes from the FULL genome so true drivers always enter the pool.

    Returns:
        Set of gene symbols (top drivers with ≥5% alteration frequency)
    """
    drivers = get_cancer_driver_genes(
        cancer_type, mutation_loader,
        min_freq=0.05,  # lower bar for injection
        max_genes=max_inject,
    )
    return {g for g, _ in drivers}


def filter_genomically_irrelevant(
    candidate_genes: List[str],
    cancer_type: str,
    mutation_loader: TCGAMutationLoader,
    drug_db=None,
) -> List[str]:
    """
    Hard-filter candidates: remove genes that are NOT genomically relevant
    to this cancer type, UNLESS they have an approved drug for this cancer.

    A gene is considered genomically relevant if:
      - It has ≥1% combined alteration frequency (mutation + CNV), OR
      - It appears in the actionability table for a matching cancer, OR
      - It has an approved/phase3 drug (drug_db check)

    This replaces the soft-bonus-only approach: genes like CDK6 (0.4%
    mutation in melanoma) are REMOVED from candidates rather than just
    receiving zero bonus.
    """
    if not mutation_loader._loaded:
        return candidate_genes  # no data → no filtering

    MIN_FREQ_FOR_RELEVANCE = 0.01  # 1%
    tcga_codes = depmap_to_tcga_codes(cancer_type)
    # Build set of actionable genes for this cancer
    actionable_here = set()
    for entry in ACTIONABLE_VARIANTS:
        gene = entry["gene"]
        entry_cancers = entry.get("cancers", [])
        if not entry_cancers:  # tumor-agnostic
            actionable_here.add(gene)
        elif any(c in entry_cancers for c in tcga_codes):
            actionable_here.add(gene)

    kept = []
    for gene in candidate_genes:
        # Check alteration frequency
        freq = get_combined_alteration_freq(gene, cancer_type, mutation_loader)
        if freq >= MIN_FREQ_FOR_RELEVANCE:
            kept.append(gene)
            continue

        # Check actionability table
        if gene in actionable_here:
            kept.append(gene)
            continue

        # Check drug database (approved/phase3 for any cancer)
        if drug_db is not None:
            if hasattr(drug_db, 'has_approved_drug'):
                if drug_db.has_approved_drug(gene, cancer_type=cancer_type):
                    kept.append(gene)
                    continue
            else:
                # Legacy fallback: access DRUG_DB dict directly
                info = getattr(drug_db, 'DRUG_DB', {}).get(gene)
                if info and info.get('stage') in ('approved', 'phase3'):
                    kept.append(gene)
                    continue

        # Gene is not genomically relevant and has no approved drug → remove
        # (this is the hard filter)

    return kept
