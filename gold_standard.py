#!/usr/bin/env python3
"""
Gold Standard for ALIN Benchmarking
===================================

Tier 1: Curated clinical gold standard (40+ unique cancer+target entries)
         from FDA-approved and Phase 2/3-validated multi-target targeted
         therapy combinations in oncology.

Tier 2: Cell-line synergy data from DrugComb / NCI-ALMANAC / O'Neil
         (downloaded on demand, mapped drug→gene target).

Inclusion criteria (same as benchmarking_module.py):
 - Multi-target (≥2 distinct HUGO gene symbol targets)
 - Both/all agents molecularly targeted (no chemo, no immunotherapy)
 - FDA-approved or Phase 2+ positive efficacy data
 - Independently documented (PMID or NCT)
 - Deduplicated by (cancer_type, frozenset(targets))

Design: Entries ALIN cannot match are intentionally included to measure
        recall honestly — the gold standard is pipeline-agnostic.
"""

import json
import hashlib
import warnings
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict

import pandas as pd
import numpy as np

from alin.constants import GENE_EQUIVALENTS

# ============================================================================
# DRUG → GENE TARGET MAPPING (inverted from gene→drug, plus additional drugs)
# ============================================================================

DRUG_TO_GENE: Dict[str, str] = {
    # EGFR family
    'erlotinib': 'EGFR', 'gefitinib': 'EGFR', 'osimertinib': 'EGFR',
    'afatinib': 'EGFR', 'cetuximab': 'EGFR', 'panitumumab': 'EGFR',
    'lapatinib': 'EGFR',  # dual EGFR/ERBB2 but primary = EGFR for mapping
    'trastuzumab': 'ERBB2', 'pertuzumab': 'ERBB2', 'tucatinib': 'ERBB2',
    # BRAF / MEK
    'vemurafenib': 'BRAF', 'dabrafenib': 'BRAF', 'encorafenib': 'BRAF',
    'trametinib': 'MAP2K1', 'cobimetinib': 'MAP2K1', 'binimetinib': 'MAP2K1',
    'avutometinib': 'MAP2K1',  # dual RAF/MEK
    # KRAS
    'sotorasib': 'KRAS', 'adagrasib': 'KRAS',
    # CDK
    'palbociclib': 'CDK4', 'ribociclib': 'CDK4', 'abemaciclib': 'CDK4',
    # Note: palbociclib/ribociclib/abemaciclib target both CDK4 and CDK6
    # We list primary as CDK4, gold standard entries list {CDK4, CDK6} explicitly
    'dinaciclib': 'CDK2',
    # mTOR / PI3K / AKT
    'everolimus': 'MTOR', 'temsirolimus': 'MTOR',
    'alpelisib': 'PIK3CA', 'idelalisib': 'PIK3CD', 'copanlisib': 'PIK3CA',
    'gedatolisib': 'PIK3CA',  # dual PI3K/mTOR
    'capivasertib': 'AKT1', 'ipatasertib': 'AKT1',
    # BCL2 family
    'venetoclax': 'BCL2', 'navitoclax': 'BCL2',
    # FLT3
    'midostaurin': 'FLT3', 'gilteritinib': 'FLT3',
    # MET
    'capmatinib': 'MET', 'tepotinib': 'MET', 'savolitinib': 'MET',
    'crizotinib': 'ALK',  # multi-kinase: ALK, MET, ROS1
    # VEGFR
    'lenvatinib': 'KDR', 'sunitinib': 'KDR', 'axitinib': 'KDR',
    'cabozantinib': 'MET',  # multi-kinase: MET, VEGFR2, RET, AXL
    'cediranib': 'KDR', 'bevacizumab': 'VEGFA',
    # ALK / ROS1 / RET
    'alectinib': 'ALK', 'brigatinib': 'ALK', 'lorlatinib': 'ALK',
    'entrectinib': 'NTRK1',  # NTRK/ALK/ROS1
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
    # Hormone therapy (endocrine)
    'fulvestrant': 'ESR1', 'tamoxifen': 'ESR1',
    'letrozole': 'CYP19A1', 'anastrozole': 'CYP19A1', 'exemestane': 'CYP19A1',
    'abiraterone': 'CYP17A1', 'enzalutamide': 'AR',
    # FAK (focal adhesion kinase)
    'defactinib': 'PTK2',
    # Menin
    'revumenib': 'MEN1', 'ziftomenib': 'MEN1',
    # SHP2
    'TNO155': 'PTPN11', 'RMC-4630': 'PTPN11',
    # IDH
    'ivosidenib': 'IDH1', 'enasidenib': 'IDH2',
    # MDM2
    'milademetan': 'MDM2',
    # Amivantamab is bispecific EGFR+MET — handled as multi-target drug below
}

# Drugs that target MULTIPLE genes (bispecific antibodies, multi-kinase inhibitors)
MULTI_TARGET_DRUGS: Dict[str, Set[str]] = {
    'amivantamab': {'EGFR', 'MET'},       # bispecific
    'lapatinib': {'EGFR', 'ERBB2'},       # dual TKI
    'afatinib': {'EGFR', 'ERBB2', 'ERBB4'},  # pan-HER
    'lenvatinib': {'KDR', 'FGFR1'},       # VEGFR + FGFR
    'cabozantinib': {'MET', 'KDR', 'RET', 'AXL'},  # multi-kinase
    'crizotinib': {'ALK', 'MET', 'ROS1'}, # multi-kinase
    'avutometinib': {'BRAF', 'MAP2K1'},    # dual RAF/MEK
}


# ============================================================================
# TIER 1: CURATED CLINICAL GOLD STANDARD
# ============================================================================
# Deduplicated by (cancer_type, frozenset(targets)).
# Each entry has a unique cancer+target pair.
#
# Categories covered:
#   BRAF+MEK (7 indications), CDK4/6+endocrine (2), HER2 (2),
#   VEGFR+mTOR (1), BCL2 combos (2), KRAS+EGFR (2), EGFR+MET (2),
#   EGFR+RET (1), PARP combos (2), RAF/MEK+FAK (1), PI3K combos (2),
#   Tri-axial (1 Liaki), ALK+MET (1), AML combos (2), miscellaneous (5+)

GOLD_STANDARD = [
    # ===== BRAF + MEK pathway (7 cancer types) =====
    {
        'cancer': 'Melanoma',
        'targets': frozenset({'BRAF', 'MAP2K1'}),
        'evidence': 'FDA_approved',
        'description': 'Dabrafenib+trametinib (COMBI-d), vemurafenib+cobimetinib (coBRIM), '
                       'encorafenib+binimetinib (COLUMBUS) for BRAF V600 melanoma.',
        'pmid': '25399551',
        'trial': 'COMBI-d/coBRIM/COLUMBUS',
    },
    {
        'cancer': 'Non-Small Cell Lung Cancer',
        'targets': frozenset({'BRAF', 'MAP2K1'}),
        'evidence': 'FDA_approved',
        'description': 'Dabrafenib+trametinib for BRAF V600E metastatic NSCLC. '
                       'Phase 2: ORR 63%, mPFS 14.6 mo.',
        'pmid': '27809962',
        'trial': 'BRF113928 (NCT01336634)',
    },
    {
        'cancer': 'Anaplastic Thyroid Cancer',
        'targets': frozenset({'BRAF', 'MAP2K1'}),
        'evidence': 'FDA_approved',
        'description': 'Dabrafenib+trametinib for BRAF V600E anaplastic thyroid cancer. '
                       'ORR 69%, 12-mo OS 80%.',
        'pmid': '29072975',
        'trial': 'NCT02034110',
    },
    {
        'cancer': 'Low-Grade Glioma',
        'targets': frozenset({'BRAF', 'MAP2K1'}),
        'evidence': 'FDA_approved',
        'description': 'Dabrafenib+trametinib for pediatric BRAF V600E low-grade glioma. '
                       'ORR 47% vs 11% chemo, mPFS 20.1 vs 7.4 mo.',
        'pmid': '37084739',
        'trial': 'CDRB436G2201',
    },
    {
        'cancer': 'Biliary Tract Cancer',
        'targets': frozenset({'BRAF', 'MAP2K1'}),
        'evidence': 'FDA_approved',
        'description': 'Dabrafenib+trametinib for BRAF V600E biliary tract cancer '
                       '(pan-tumor basket trial). ORR 46%.',
        'pmid': '35373072',
        'trial': 'NCI-MATCH/ROAR',
    },
    {
        'cancer': 'High-Grade Glioma',
        'targets': frozenset({'BRAF', 'MAP2K1'}),
        'evidence': 'Phase_2',
        'description': 'Dabrafenib+trametinib for BRAF V600E high-grade glioma '
                       '(pan-tumor basket trial). ORR 33%.',
        'pmid': '35373072',
        'trial': 'NCI-MATCH',
    },
    {
        'cancer': 'Colorectal Adenocarcinoma',
        'targets': frozenset({'BRAF', 'EGFR'}),
        'evidence': 'FDA_approved',
        'description': 'Encorafenib+cetuximab for BRAF V600E mCRC. '
                       'BEACON Phase 3: OS HR 0.60, ORR 20%.',
        'pmid': '31566309',
        'trial': 'BEACON CRC (NCT02928224)',
    },
    {
        'cancer': 'Colorectal Adenocarcinoma',
        'targets': frozenset({'BRAF', 'MAP2K1', 'EGFR'}),
        'evidence': 'Phase_3',
        'description': 'Encorafenib+binimetinib+cetuximab CRC triplet (BEACON triple arm). '
                       'Also dabrafenib+trametinib+panitumumab Phase 2.',
        'pmid': '31566309',
        'trial': 'BEACON CRC triplet',
    },

    # ===== CDK4/6 + Endocrine therapy (2 distinct target sets) =====
    {
        'cancer': 'Invasive Breast Carcinoma',
        'targets': frozenset({'CDK4', 'CDK6', 'ESR1'}),
        'evidence': 'FDA_approved',
        'description': 'CDK4/6 inhibitor + fulvestrant (ESR1 degrader) for HR+/HER2- mBC. '
                       'PALOMA-3: PFS HR 0.46, MONARCH-2: PFS HR 0.55.',
        'pmid': '26394241',
        'trial': 'PALOMA-3/MONARCH-2/MONALEESA-3',
    },
    {
        'cancer': 'Invasive Breast Carcinoma',
        'targets': frozenset({'CDK4', 'CDK6', 'CYP19A1'}),
        'evidence': 'FDA_approved',
        'description': 'CDK4/6 inhibitor + aromatase inhibitor (CYP19A1) for HR+/HER2- mBC. '
                       '1L PALOMA-2: mPFS 24.8 vs 14.5 mo, MONALEESA-2: mPFS 25.3 vs 16.0.',
        'pmid': '27717303',
        'trial': 'PALOMA-2/MONALEESA-2/MONARCH-3',
    },

    # ===== HER2 dual-targeting =====
    {
        'cancer': 'Invasive Breast Carcinoma',
        'targets': frozenset({'ERBB2', 'EGFR'}),
        'evidence': 'FDA_approved',
        'description': 'Lapatinib (EGFR+ERBB2 dual TKI) + trastuzumab (ERBB2) for '
                       'HER2+ breast cancer. NeoALTTO: pCR RR 1.43.',
        'pmid': '22153890',
        'trial': 'NeoALTTO (NCT00553358)',
    },
    {
        'cancer': 'Invasive Breast Carcinoma',
        'targets': frozenset({'ERBB2', 'CDK4', 'CDK6'}),
        'evidence': 'Phase_3',
        'description': 'Palbociclib+trastuzumab+endocrine for HR+/HER2+ mBC. '
                       'PATINA Phase 3: PFS 44 vs 29 mo.',
        'pmid': '36631847',
        'trial': 'PATINA AFT-38 (NCT02947685)',
    },

    # ===== VEGFR + mTOR =====
    {
        'cancer': 'Renal Cell Carcinoma',
        'targets': frozenset({'KDR', 'MTOR'}),
        'evidence': 'FDA_approved',
        'description': 'Lenvatinib (VEGFR2/KDR) + everolimus (mTOR) for advanced RCC. '
                       'Phase 2: PFS HR 0.45 vs everolimus.',
        'pmid': '26116099',
        'trial': 'Study 205 (NCT01136733)',
    },

    # ===== BCL2 combinations =====
    {
        'cancer': 'Acute Myeloid Leukemia',
        'targets': frozenset({'FLT3', 'BCL2'}),
        'evidence': 'Phase_2',
        'description': 'Venetoclax (BCL2) + gilteritinib (FLT3) for R/R FLT3-mut AML. '
                       'High composite CR rate.',
        'pmid': '35443125',
        'trial': 'NCT03625505',
    },
    {
        'cancer': 'Chronic Lymphocytic Leukemia',
        'targets': frozenset({'BTK', 'BCL2'}),
        'evidence': 'Phase_3',
        'description': 'Ibrutinib (BTK) + venetoclax (BCL2) for CLL. '
                       'GLOW Phase 3: PFS HR 0.22 vs chlorambucil+obinutuzumab.',
        'pmid': '34543595',
        'trial': 'GLOW (NCT03462719)',
    },

    # ===== KRAS G12C + EGFR =====
    {
        'cancer': 'Colorectal Adenocarcinoma',
        'targets': frozenset({'KRAS', 'EGFR'}),
        'evidence': 'Phase_2',
        'description': 'Sotorasib+panitumumab or adagrasib+cetuximab for KRAS G12C mCRC. '
                       'CodeBreaK: ORR 26%, KRYSTAL-1: ORR 34%.',
        'pmid': '36546659',
        'trial': 'CodeBreaK 101/KRYSTAL-1',
    },
    {
        'cancer': 'Non-Small Cell Lung Cancer',
        'targets': frozenset({'KRAS', 'EGFR'}),
        'evidence': 'Phase_2',
        'description': 'Sotorasib+afatinib for KRAS G12C NSCLC. '
                       'CodeBreaK 101: ORR 35%.',
        'pmid': '38598642',
        'trial': 'CodeBreaK 101 (NCT04185883)',
    },

    # ===== EGFR + MET =====
    {
        'cancer': 'Non-Small Cell Lung Cancer',
        'targets': frozenset({'EGFR', 'MET'}),
        'evidence': 'FDA_approved',
        'description': 'Amivantamab (bispecific EGFR+MET) for EGFR exon 20 ins NSCLC. '
                       'CHRYSALIS ORR 40%. Also osimertinib+savolitinib TATTON ORR 52%.',
        'pmid': '34043995',
        'trial': 'CHRYSALIS/TATTON/MARIPOSA',
    },
    {
        'cancer': 'Head and Neck Squamous Cell Carcinoma',
        'targets': frozenset({'EGFR', 'MET'}),
        'evidence': 'Phase_2',
        'description': 'Ficlatuzumab (MET) + cetuximab (EGFR) for R/M HNSCC.',
        'pmid': '32416071',
        'trial': 'NCT02277197',
    },

    # ===== EGFR + RET =====
    {
        'cancer': 'Non-Small Cell Lung Cancer',
        'targets': frozenset({'EGFR', 'RET'}),
        'evidence': 'Phase_2',
        'description': 'Osimertinib+selpercatinib for EGFR-mutant NSCLC with acquired '
                       'RET fusion. ORR 50%, DCR 80%.',
        'pmid': '37582234',
        'trial': 'Multicenter cohort',
    },

    # ===== PARP + other targeted agent =====
    {
        'cancer': 'Ovarian Epithelial Tumor',
        'targets': frozenset({'PARP1', 'KDR'}),
        'evidence': 'Phase_2',
        'description': 'Olaparib (PARP) + cediranib (VEGFR2/KDR) for platinum-sensitive '
                       'BRCA-wt ovarian cancer. PFS 16.5 vs 8.2 mo.',
        'pmid': '30345851',
        'trial': 'NCT01116648',
    },
    {
        'cancer': 'Prostate Adenocarcinoma',
        'targets': frozenset({'PARP1', 'CYP17A1'}),
        'evidence': 'FDA_approved',
        'description': 'Niraparib+abiraterone (CYP17A1) for BRCA-mut mCSPC. '
                       'FDA approved 2023.',
        'pmid': '37406291',
        'trial': 'MAGNITUDE (NCT03748641)',
    },

    # ===== RAF/MEK + FAK =====
    {
        'cancer': 'Low-Grade Serous Ovarian Cancer',
        'targets': frozenset({'MAP2K1', 'PTK2'}),
        'evidence': 'FDA_approved',
        'description': 'Avutometinib (dual RAF/MEK) + defactinib (FAK/PTK2) for '
                       'KRAS-mut recurrent LGSOC. RAMP 201: ORR 44%. FDA May 2025.',
        'pmid': '38557903',
        'trial': 'RAMP 201 (NCT04625270)',
    },

    # ===== PI3K combinations =====
    {
        'cancer': 'Invasive Breast Carcinoma',
        'targets': frozenset({'PIK3CA', 'ESR1'}),
        'evidence': 'FDA_approved',
        'description': 'Alpelisib (PI3Kα) + fulvestrant (ESR1) for PIK3CA-mut HR+ mBC. '
                       'SOLAR-1: PFS 11.0 vs 5.7 mo.',
        'pmid': '31091374',
        'trial': 'SOLAR-1 (NCT02437318)',
    },
    {
        'cancer': 'Invasive Breast Carcinoma',
        'targets': frozenset({'AKT1', 'ESR1'}),
        'evidence': 'FDA_approved',
        'description': 'Capivasertib (AKT) + fulvestrant (ESR1) for HR+ mBC with '
                       'AKT/PTEN/PIK3CA alterations. CAPItello-291: PFS HR 0.60.',
        'pmid': '37256976',
        'trial': 'CAPItello-291 (NCT04305496)',
    },

    # ===== ALK/MET combination =====
    {
        'cancer': 'Non-Small Cell Lung Cancer',
        'targets': frozenset({'ALK', 'MET'}),
        'evidence': 'Phase_2',
        'description': 'Alectinib (ALK) + crizotinib (MET) for ALK+ NSCLC with '
                       'MET amplification-driven resistance.',
        'pmid': '39430327',
        'trial': 'Case series/Phase 2',
    },

    # ===== AML: IDH combinations =====
    {
        'cancer': 'Acute Myeloid Leukemia',
        'targets': frozenset({'IDH1', 'BCL2'}),
        'evidence': 'Phase_2',
        'description': 'Ivosidenib (IDH1) + venetoclax (BCL2) for R/R IDH1-mut AML. '
                       'Composite CR 71%.',
        'pmid': '36001327',
        'trial': 'NCT03471260',
    },
    {
        'cancer': 'Acute Myeloid Leukemia',
        'targets': frozenset({'IDH2', 'BCL2'}),
        'evidence': 'Phase_2',
        'description': 'Enasidenib (IDH2) + venetoclax (BCL2) for IDH2-mut AML.',
        'pmid': '36001327',
        'trial': 'NCT03471260',
    },

    # ===== Menin combinations (emerging) =====
    {
        'cancer': 'Acute Myeloid Leukemia',
        'targets': frozenset({'MEN1', 'BCL2'}),
        'evidence': 'Phase_2',
        'description': 'Revumenib (menin inhibitor) + venetoclax (BCL2) for NPM1-mut AML. '
                       'Phase 2 trials ongoing; revumenib FDA-approved as monotherapy Nov 2024.',
        'pmid': '38372855',
        'trial': 'NCT04065399/cAMeLot',
    },

    # ===== KRAS + MEK (different from BRAF+MEK) =====
    {
        'cancer': 'Non-Small Cell Lung Cancer',
        'targets': frozenset({'KRAS', 'MAP2K1'}),
        'evidence': 'Phase_2',
        'description': 'Sotorasib+trametinib for KRAS G12C NSCLC. '
                       'CodeBreaK 101 MEK arm.',
        'pmid': '38598642',
        'trial': 'CodeBreaK 101',
    },

    # ===== FGFR combinations =====
    {
        'cancer': 'Bladder Urothelial Carcinoma',
        'targets': frozenset({'FGFR1', 'MAP2K1'}),
        'evidence': 'Phase_2',
        'description': 'Erdafitinib (FGFR) + MEK inhibitor for FGFR-altered urothelial '
                       'cancer with acquired MAPK resistance. Preclinical+Phase 2, PFS data.',
        'pmid': '32968194',
        'trial': 'Phase 2 FGFR+MEK',
    },

    # ===== Endometrial cancer =====
    {
        'cancer': 'Endometrial Carcinoma',
        'targets': frozenset({'KDR', 'MTOR'}),
        'evidence': 'Phase_2',
        'description': 'Lenvatinib (VEGFR/KDR) + everolimus (mTOR) for advanced '
                       'endometrial cancer. Phase 2: ORR 32%.',
        'pmid': '25519769',
        'trial': 'Study 111 (NCT01136733)',
    },

    # ===== Thyroid cancer (MTC, non-anaplastic) =====
    {
        'cancer': 'Medullary Thyroid Carcinoma',
        'targets': frozenset({'RET', 'KDR'}),
        'evidence': 'FDA_approved',
        'description': 'Cabozantinib (RET+VEGFR2/KDR) or vandetanib (RET+KDR) for MTC. '
                       'EXAM Phase 3: PFS 11.2 vs 4.0 mo.',
        'pmid': '22256481',
        'trial': 'EXAM/ZETA',
    },

    # ===== HCC =====
    {
        'cancer': 'Hepatocellular Carcinoma',
        'targets': frozenset({'KDR', 'MET'}),
        'evidence': 'FDA_approved',
        'description': 'Cabozantinib (VEGFR2+MET) for HCC after sorafenib. '
                       'CELESTIAL Phase 3: OS HR 0.76.',
        'pmid': '29972759',
        'trial': 'CELESTIAL (NCT01908426)',
    },

    # ===== Multiple Myeloma =====
    {
        'cancer': 'Plasma Cell Myeloma',
        'targets': frozenset({'PSMB5', 'HDAC6'}),
        'evidence': 'Phase_3',
        'description': 'Bortezomib (proteasome/PSMB5) + panobinostat (HDAC) for R/R MM. '
                       'PANORAMA-1: PFS 12.0 vs 8.1 mo.',
        'pmid': '25034862',
        'trial': 'PANORAMA-1 (NCT01023308)',
    },

    # ===== CLL additional =====
    {
        'cancer': 'Chronic Lymphocytic Leukemia',
        'targets': frozenset({'PIK3CD', 'BCL2'}),
        'evidence': 'Phase_2',
        'description': 'Idelalisib (PI3Kd) + venetoclax (BCL2) for R/R CLL.',
        'pmid': '34879939',
        'trial': 'Phase 2 (NCT02756897)',
    },

    # ===== PDAC tri-axial (Liaki) =====
    {
        'cancer': 'Pancreatic Adenocarcinoma',
        'targets': frozenset({'KRAS', 'EGFR', 'STAT3'}),
        'evidence': 'Preclinical',
        'description': 'RAS inhibitor + EGFR inhibitor + STAT3 PROTAC for PDAC. '
                       'Tri-axial blockade: complete regression, no resistance >200 days.',
        'pmid': 'Liaki2025',
        'trial': 'Preclinical',
    },

    # ===== Cholangiocarcinoma =====
    {
        'cancer': 'Cholangiocarcinoma',
        'targets': frozenset({'FGFR2', 'MAP2K1'}),
        'evidence': 'Phase_2',
        'description': 'Futibatinib (FGFR2) + MEK inhibitor for FGFR2-fusion iCCA '
                       'with acquired MAPK resistance.',
        'pmid': '33127845',
        'trial': 'Phase 2 (NCT04093362)',
    },

    # ===== GIST =====
    {
        'cancer': 'Gastrointestinal Stromal Tumor',
        'targets': frozenset({'KIT', 'KDR'}),
        'evidence': 'FDA_approved',
        'description': 'Sunitinib (KIT+VEGFR) for imatinib-resistant GIST. '
                       'Phase 3: TTP 27.3 vs 6.4 weeks.',
        'pmid': '16397024',
        'trial': 'NCT00075218',
    },

    # ===== Gastric/GEJ =====
    {
        'cancer': 'Stomach Adenocarcinoma',
        'targets': frozenset({'ERBB2', 'KDR'}),
        'evidence': 'Phase_2',
        'description': 'Trastuzumab (HER2) + ramucirumab (VEGFR2/KDR) for HER2+ '
                       'gastric cancer. Phase 2 data.',
        'pmid': '29045837',
        'trial': 'Phase 2 (NCT02661971)',
    },

    # ===== Soft tissue sarcoma =====
    {
        'cancer': 'Liposarcoma',
        'targets': frozenset({'MDM2', 'CDK4'}),
        'evidence': 'Phase_2',
        'description': 'Milademetan (MDM2) + abemaciclib (CDK4/6) for dedifferentiated '
                       'liposarcoma with MDM2 amp + CDK4 amp. Phase 2.',
        'pmid': '36706352',
        'trial': 'Phase 2 (NCT04116541)',
    },

    # ===== Ovarian (BRCA) =====
    {
        'cancer': 'Ovarian Epithelial Tumor',
        'targets': frozenset({'PARP1', 'PIK3CA'}),
        'evidence': 'Phase_2',
        'description': 'Olaparib (PARP) + alpelisib (PI3Ka) for platinum-resistant '
                       'ovarian cancer. ORR 36%.',
        'pmid': '33514514',
        'trial': 'Phase 1b (NCT01623349)',
    },

    # ===== NSCLC: EGFR + HER3 =====
    {
        'cancer': 'Non-Small Cell Lung Cancer',
        'targets': frozenset({'EGFR', 'ERBB3'}),
        'evidence': 'Phase_3',
        'description': 'Osimertinib (EGFR) + patritumab deruxtecan (HER3/ERBB3 ADC) '
                       'for EGFR-mutant NSCLC. HERTHENA-Lung02 Phase 3.',
        'pmid': '37795877',
        'trial': 'HERTHENA-Lung02 (NCT05338970)',
    },
]


# ============================================================================
# CANCER BENCHMARK ALIASES
# ============================================================================
# Map gold standard cancer names → pipeline cancer type names.
# Must cover all cancer types in GOLD_STANDARD.

CANCER_ALIASES: Dict[str, List[str]] = {
    # Gold standard cancer type → EXACT pipeline cancer type names from CSV
    # NO substring matching — each entry must be an exact Cancer_Type value
    'Melanoma': ['Melanoma'],
    'Non-Small Cell Lung Cancer': ['Non-Small Cell Lung Cancer'],
    'Anaplastic Thyroid Cancer': ['Anaplastic Thyroid Cancer'],
    'Low-Grade Glioma': ['Diffuse Glioma'],  # closest match in pipeline
    'Biliary Tract Cancer': [],  # no exact match in pipeline
    'High-Grade Glioma': ['Diffuse Glioma'],
    'Colorectal Adenocarcinoma': ['Colorectal Adenocarcinoma'],
    'Invasive Breast Carcinoma': ['Invasive Breast Carcinoma'],
    'Renal Cell Carcinoma': ['Renal Cell Carcinoma'],
    'Acute Myeloid Leukemia': ['Acute Myeloid Leukemia'],
    'Chronic Lymphocytic Leukemia': [],  # no exact match in pipeline
    'Head and Neck Squamous Cell Carcinoma': ['Head and Neck Squamous Cell Carcinoma'],
    'Ovarian Epithelial Tumor': ['Ovarian Epithelial Tumor'],
    'Low-Grade Serous Ovarian Cancer': [],  # no exact match in pipeline
    'Prostate Adenocarcinoma': ['Prostate Adenocarcinoma'],
    'Pancreatic Adenocarcinoma': ['Pancreatic Adenocarcinoma'],
    'Bladder Urothelial Carcinoma': ['Bladder Urothelial Carcinoma'],
    'Endometrial Carcinoma': ['Endometrial Carcinoma'],
    'Medullary Thyroid Carcinoma': [],  # no exact match in pipeline
    'Hepatocellular Carcinoma': ['Hepatocellular Carcinoma'],
    'Plasma Cell Myeloma': [],  # no exact match in pipeline
    'Gastrointestinal Stromal Tumor': [],  # no exact match in pipeline
    'Stomach Adenocarcinoma': ['Esophagogastric Adenocarcinoma'],
    'Liposarcoma': ['Liposarcoma'],
    'Cholangiocarcinoma': [],  # no exact match in pipeline
}

# Gene equivalents imported from alin.constants above


# ============================================================================
# TESTABILITY FILTER
# ============================================================================
#
# Some gold-standard entries are structurally unmatchable by a CRISPR/DepMap
# pipeline for one of two reasons:
#
#   1. **No DepMap cell lines**: the cancer type maps to an empty alias list
#      in CANCER_ALIASES, so the pipeline never produces predictions
#      for that cancer (7 cancer types: Biliary Tract Cancer, CLL,
#      Low-Grade Serous Ovarian, Medullary Thyroid, Myeloma, GIST,
#      Cholangiocarcinoma).
#
#   2. **Non-CRISPR target modality**: one or more targets in the combination
#      act through a mechanism that CRISPR knockout screens cannot capture —
#      endocrine receptors (ESR1), metabolic enzymes (CYP19A1, CYP17A1),
#      secreted growth factors (VEGFA), anti-angiogenic targets (KDR),
#      proteasome subunits (PSMB5), and epigenetic erasers (HDAC6).
#
# We report recall on BOTH the full 43-entry gold standard (for honesty) and
# the testable subset (for a fair assessment of CRISPR-based pipeline power).

NON_CRISPR_TARGETS: Set[str] = {
    'ESR1',     # estrogen receptor — endocrine therapy (fulvestrant, tamoxifen)
    'CYP19A1',  # aromatase — endocrine (letrozole, anastrozole, exemestane)
    'CYP17A1',  # steroid 17α-hydroxylase — endocrine (abiraterone)
    'VEGFA',    # secreted VEGF ligand — anti-angiogenic antibody (bevacizumab)
    'KDR',      # VEGFR2 — anti-angiogenic (lenvatinib, sunitinib, cediranib)
    'PSMB5',    # proteasome β5 subunit (bortezomib)
    'HDAC6',    # histone deacetylase (panobinostat)
}


def is_testable(entry: Dict) -> bool:
    """Check if a gold standard entry is testable by a CRISPR/DepMap pipeline.

    Returns True if:
      (1) the cancer type has a non-empty alias mapping (DepMap cell lines), AND
      (2) none of the gold-standard targets are in NON_CRISPR_TARGETS.
    """
    # Criterion 1: cancer type must map to at least one pipeline cancer
    cancer = entry['cancer']
    if cancer in CANCER_ALIASES:
        if not CANCER_ALIASES[cancer]:  # empty list
            return False
    # Criterion 2: all targets must be plausible CRISPR essentiality targets
    if entry['targets'] & NON_CRISPR_TARGETS:
        return False
    return True


# Pre-compute counts for convenience
TESTABLE_ENTRIES = [e for e in GOLD_STANDARD if is_testable(e)]
N_TESTABLE = len(TESTABLE_ENTRIES)
N_UNTESTABLE = len(GOLD_STANDARD) - N_TESTABLE


# ============================================================================
# GOLD STANDARD INTEGRITY CHECKS
# ============================================================================

def validate_gold_standard(entries=None):
    """Validate gold standard for integrity and report statistics."""
    if entries is None:
        entries = GOLD_STANDARD

    cancers = set()
    unique_pairs = set()
    duplicates = []
    evidence_counts = defaultdict(int)

    for i, entry in enumerate(entries):
        assert 'cancer' in entry, f"Entry {i} missing 'cancer'"
        assert 'targets' in entry, f"Entry {i} missing 'targets'"
        assert isinstance(entry['targets'], frozenset), f"Entry {i} targets not frozenset"
        assert len(entry['targets']) >= 2, f"Entry {i} has <2 targets: {entry['targets']}"

        cancers.add(entry['cancer'])
        evidence_counts[entry.get('evidence', 'unknown')] += 1

        key = (entry['cancer'], entry['targets'])
        if key in unique_pairs:
            duplicates.append(key)
        unique_pairs.add(key)

    if duplicates:
        warnings.warn(f"Gold standard has {len(duplicates)} duplicate(s): {duplicates}")

    return {
        'total_entries': len(entries),
        'unique_pairs': len(unique_pairs),
        'cancer_types': len(cancers),
        'cancers': sorted(cancers),
        'evidence_breakdown': dict(evidence_counts),
        'duplicates': duplicates,
        'all_targets': sorted({g for e in entries for g in e['targets']}),
    }


# ============================================================================
# TIER 2: DRUGCOMB / ALMANAC SYNERGY DATA LOADER
# ============================================================================

def download_drugcomb_summary(output_dir: str = 'synergy_data') -> Optional[pd.DataFrame]:
    """
    Download DrugComb summary synergy data.
    Returns DataFrame with columns: drug1, drug2, cell_line, tissue,
    synergy_bliss, synergy_loewe, synergy_zip, synergy_hsa.
    Returns None if download fails.
    """
    import requests
    out = Path(output_dir)
    out.mkdir(exist_ok=True)
    cache_file = out / 'drugcomb_summary.csv'

    if cache_file.exists():
        print(f"Loading cached DrugComb data from {cache_file}")
        return pd.read_csv(cache_file)

    # Try DrugComb API
    api_urls = [
        'https://api.drugcomb.org/summary',
        'https://drugcomb.org/api/summary',
    ]

    for url in api_urls:
        try:
            print(f"Downloading from {url}...")
            r = requests.get(url, timeout=60)
            if r.ok:
                data = r.json()
                df = pd.DataFrame(data)
                df.to_csv(cache_file, index=False)
                print(f"Downloaded {len(df)} synergy entries")
                return df
        except Exception as e:
            print(f"  Failed: {e}")
            continue

    print("DrugComb API unreachable. Place 'drugcomb_summary.csv' in synergy_data/ manually.")
    print("Download from: https://drugcomb.org → Data → Download summary")
    return None


def map_drugs_to_gene_targets(drug_name: str) -> Set[str]:
    """Map a drug name to its gene target(s)."""
    drug_lower = drug_name.strip().lower()

    # Check multi-target drugs first
    for drug, targets in MULTI_TARGET_DRUGS.items():
        if drug_lower == drug.lower():
            return targets

    # Check single-target mapping
    for drug, gene in DRUG_TO_GENE.items():
        if drug_lower == drug.lower():
            return {gene}

    return set()


def build_synergy_gold_standard(
    df: pd.DataFrame,
    min_synergy_score: float = 10.0,
    synergy_metric: str = 'synergy_zip',
    min_cell_lines: int = 2,
) -> List[Dict]:
    """
    Build Tier 2 gold standard from cell-line synergy data.

    Args:
        df: DrugComb summary DataFrame
        min_synergy_score: Minimum synergy score threshold
        synergy_metric: Which synergy metric to filter on
        min_cell_lines: Minimum number of cell lines showing synergy

    Returns:
        List of gold standard entries from synergy data
    """
    if df is None or df.empty:
        return []

    entries = []

    # Filter for synergistic combinations
    if synergy_metric in df.columns:
        synergistic = df[df[synergy_metric] >= min_synergy_score].copy()
    else:
        print(f"Metric '{synergy_metric}' not in columns: {list(df.columns)}")
        return []

    print(f"Synergistic combinations (score >= {min_synergy_score}): {len(synergistic)}")

    # Group by drug pair and tissue
    if 'tissue' not in df.columns:
        print("No 'tissue' column — cannot group by cancer type")
        return []

    grouped = synergistic.groupby(['drug1', 'drug2', 'tissue']).agg(
        n_cell_lines=('cell_line', 'nunique'),
        mean_synergy=(synergy_metric, 'mean'),
    ).reset_index()

    # Filter for reproducible synergy
    reproducible = grouped[grouped['n_cell_lines'] >= min_cell_lines]
    print(f"Reproducible synergistic pairs (≥{min_cell_lines} cell lines): {len(reproducible)}")

    # Map drugs to gene targets
    for _, row in reproducible.iterrows():
        targets1 = map_drugs_to_gene_targets(row['drug1'])
        targets2 = map_drugs_to_gene_targets(row['drug2'])
        all_targets = targets1 | targets2

        if len(all_targets) >= 2:
            entries.append({
                'cancer': row['tissue'],
                'targets': frozenset(all_targets),
                'evidence': f'DrugComb_synergy_{synergy_metric}>{min_synergy_score}',
                'description': f"{row['drug1']}+{row['drug2']} in {row['tissue']} "
                               f"({row['n_cell_lines']} cell lines, "
                               f"mean {synergy_metric}={row['mean_synergy']:.1f})",
                'pmid': 'DrugComb',
                'trial': 'Cell-line synergy',
            })

    # Deduplicate by (cancer, targets)
    seen = set()
    deduped = []
    for e in entries:
        key = (e['cancer'], e['targets'])
        if key not in seen:
            seen.add(key)
            deduped.append(e)

    print(f"Tier 2 gold standard: {len(deduped)} unique cancer+target entries")
    return deduped


# ============================================================================
# UNIFIED BENCHMARK
# ============================================================================

def check_match(predicted: frozenset, gold: frozenset, gene_equivalents=None) -> str:
    """
    Check match type between predicted and gold standard target sets.

    Match levels (highest to lowest):
      'exact'        — expanded predicted == expanded gold
      'superset'     — gold ⊆ predicted (predicted covers all gold targets)
      'pair_overlap'  — |expanded ∩ gold_expanded| ≥ 2 (at least two shared genes)
      'any_overlap'  — |expanded ∩ gold_expanded| ≥ 1 (at least one shared gene)
      'none'         — no overlap
    """
    if gene_equivalents is None:
        gene_equivalents = GENE_EQUIVALENTS

    # Expand predicted with equivalents
    expanded = set(predicted)
    for g in predicted:
        if g in gene_equivalents:
            expanded.update(gene_equivalents[g])
    expanded = frozenset(expanded)

    # Expand gold with equivalents
    gold_expanded = set(gold)
    for g in gold:
        if g in gene_equivalents:
            gold_expanded.update(gene_equivalents[g])
    gold_expanded = frozenset(gold_expanded)

    # Check overlap
    overlap = expanded & gold_expanded

    if gold_expanded <= expanded and expanded <= gold_expanded:
        return 'exact'
    if gold_expanded <= expanded:
        return 'superset'
    if len(overlap) >= 2:
        return 'pair_overlap'
    if len(overlap) >= 1:
        return 'any_overlap'
    return 'none'


def _resolve_pipeline_cancers(gold_cancer: str) -> Set[str]:
    """Resolve a gold standard cancer type to exact pipeline Cancer_Type names.

    Returns the set of pipeline Cancer_Type values that should be matched.
    Uses CANCER_ALIASES with exact matching only — no substring.
    """
    result = set()
    # Check if gold_cancer itself is a key in aliases
    if gold_cancer in CANCER_ALIASES:
        result.update(CANCER_ALIASES[gold_cancer])
    # Always include the literal gold cancer name (exact match)
    result.add(gold_cancer)
    # Remove empty strings
    result.discard('')
    return result


def _expand_with_equivalents(targets: set) -> set:
    """Expand a set of gene symbols with known equivalents."""
    expanded = set(targets)
    for g in targets:
        if g in GENE_EQUIVALENTS:
            expanded.update(GENE_EQUIVALENTS[g])
    return expanded


def _compute_cancer_precision(
    df: pd.DataFrame,
    gold_entries: List[Dict],
    pred_targets_by_cancer: Optional[Dict[str, set]] = None,
) -> Dict:
    """Compute cancer-level precision for a set of predictions.

    Precision = (# evaluable cancers where predicted triple contains ≥1
                  gold-standard target) / (# evaluable cancers).

    An *evaluable cancer* is a pipeline Cancer_Type that appears in
    both the predictions and at least one gold-standard entry (via alias
    mapping).  Cancers with predictions but no gold-standard data are
    excluded — they cannot be assessed as correct or incorrect.

    Two versions are computed:
      - precision_all:      denominator = evaluable cancers from all GS entries
      - precision_testable: denominator = evaluable cancers from testable GS entries only

    Parameters
    ----------
    df : DataFrame with columns Cancer_Type, Target_1, Target_2, Target_3
    gold_entries : list of gold standard dicts
    pred_targets_by_cancer : optional override mapping cancer→predicted targets.
        If None, built from df (one row per cancer, taking first row).

    Returns
    -------
    dict with keys: precision, n_evaluable, n_prec_hits,
                    precision_testable, n_evaluable_testable, n_prec_hits_testable
    """
    # Build mapping: pipeline_cancer → union of GS targets (all & testable)
    cancer_gs_all = defaultdict(set)
    cancer_gs_testable = defaultdict(set)
    for entry in gold_entries:
        pipeline_cancers = _resolve_pipeline_cancers(entry['cancer'])
        for pc in pipeline_cancers:
            if pc:
                cancer_gs_all[pc].update(entry['targets'])
                if is_testable(entry):
                    cancer_gs_testable[pc].update(entry['targets'])

    # Build predicted targets for each cancer if not supplied
    if pred_targets_by_cancer is None:
        pred_targets_by_cancer = {}
        for _, row in df.iterrows():
            c = row['Cancer_Type']
            if c not in pred_targets_by_cancer:
                pred_targets_by_cancer[c] = {row['Target_1'], row['Target_2'], row['Target_3']}

    # Compute precision
    n_evaluable = 0
    n_prec_hits = 0
    n_evaluable_testable = 0
    n_prec_hits_testable = 0

    for cancer, pred in pred_targets_by_cancer.items():
        pred_exp = _expand_with_equivalents(pred)
        if cancer in cancer_gs_all and cancer_gs_all[cancer]:
            n_evaluable += 1
            gs_exp = _expand_with_equivalents(cancer_gs_all[cancer])
            if pred_exp & gs_exp:
                n_prec_hits += 1
        if cancer in cancer_gs_testable and cancer_gs_testable[cancer]:
            n_evaluable_testable += 1
            gs_t_exp = _expand_with_equivalents(cancer_gs_testable[cancer])
            if pred_exp & gs_t_exp:
                n_prec_hits_testable += 1

    return {
        'precision': n_prec_hits / n_evaluable if n_evaluable > 0 else 0,
        'n_evaluable': n_evaluable,
        'n_prec_hits': n_prec_hits,
        'precision_testable': n_prec_hits_testable / n_evaluable_testable if n_evaluable_testable > 0 else 0,
        'n_evaluable_testable': n_evaluable_testable,
        'n_prec_hits_testable': n_prec_hits_testable,
    }


def run_benchmark(
    predictions_csv: str = 'results/triple_combinations.csv',
    tier1: bool = True,
    tier2: bool = False,
    verbose: bool = True,
) -> Dict:
    """
    Run ALIN predictions against gold standard.

    Returns dict with recall metrics and per-entry results.
    """
    # Load predictions
    df = pd.read_csv(predictions_csv)
    # Normalize column names: accept both "Cancer Type" and "Cancer_Type" styles
    df.columns = [c.replace(' ', '_') for c in df.columns]

    # Build gold standard
    gold_entries = []
    if tier1:
        gold_entries.extend(GOLD_STANDARD)
    if tier2:
        synergy_df = download_drugcomb_summary()
        if synergy_df is not None:
            tier2_entries = build_synergy_gold_standard(synergy_df)
            gold_entries.extend(tier2_entries)

    if not gold_entries:
        return {'error': 'No gold standard entries'}

    # Validate
    stats = validate_gold_standard(gold_entries)
    if verbose:
        print(f"\nGold Standard: {stats['total_entries']} entries, "
              f"{stats['cancer_types']} cancer types")
        print(f"Evidence: {stats['evidence_breakdown']}")
        print(f"All target genes: {stats['all_targets']}")
        print()

    # Build cancer → aliases lookup
    aliases = CANCER_ALIASES

    # Evaluate each gold standard entry
    results = []
    for entry in gold_entries:
        gold_cancer = entry['cancer']
        gold_targets = entry['targets']

        # Find matching predictions by cancer type (exact matching only)
        pipeline_cancers = _resolve_pipeline_cancers(gold_cancer)
        matched_predictions = []
        for _, row in df.iterrows():
            if row['Cancer_Type'] in pipeline_cancers:
                # Check the traditional triple prediction
                pred_targets = frozenset({row['Target_1'], row['Target_2'], row['Target_3']})
                match_type = check_match(pred_targets, gold_targets)
                matched_predictions.append({
                    'cancer_type': row['Cancer_Type'],
                    'predicted': pred_targets,
                    'match_type': match_type,
                })
                
                # Also check best-combination columns (may be a doublet)
                bc1 = row.get('Best_Combo_1', '')
                bc2 = row.get('Best_Combo_2', '')
                bc3 = row.get('Best_Combo_3', '')
                if bc1 and str(bc1) != 'nan' and bc2 and str(bc2) != 'nan':
                    combo_set = {str(bc1), str(bc2)}
                    if bc3 and str(bc3) != 'nan' and str(bc3) != '':
                        combo_set.add(str(bc3))
                    combo_targets = frozenset(combo_set)
                    if combo_targets != pred_targets:  # avoid double-counting
                        combo_match = check_match(combo_targets, gold_targets)
                        matched_predictions.append({
                            'cancer_type': row['Cancer_Type'],
                            'predicted': combo_targets,
                            'match_type': combo_match,
                        })

        # Best match
        match_priority = {'exact': 4, 'superset': 3, 'pair_overlap': 2, 'any_overlap': 1, 'none': 0}
        if matched_predictions:
            best = max(matched_predictions, key=lambda x: match_priority[x['match_type']])
        else:
            best = {'cancer_type': 'NO_PREDICTION', 'predicted': frozenset(), 'match_type': 'none'}

        results.append({
            'gold_cancer': gold_cancer,
            'gold_targets': gold_targets,
            'evidence': entry.get('evidence', ''),
            'best_match': best['match_type'],
            'predicted': best['predicted'],
            'pipeline_cancer': best['cancer_type'],
            'n_predictions': len(matched_predictions),
        })

    # Compute recall
    n = len(results)
    n_exact = sum(1 for r in results if r['best_match'] == 'exact')
    n_superset = sum(1 for r in results if r['best_match'] in ('exact', 'superset'))
    n_pair_overlap = sum(1 for r in results if r['best_match'] in ('exact', 'superset', 'pair_overlap'))
    n_any_overlap = sum(1 for r in results if r['best_match'] in ('exact', 'superset', 'pair_overlap', 'any_overlap'))
    n_no_prediction = sum(1 for r in results if r['pipeline_cancer'] == 'NO_PREDICTION')

    # Testable-only recall (entries where cancer has DepMap lines AND targets are CRISPR-plausible)
    testable_results = [r for r, e in zip(results, gold_entries) if is_testable(e)]
    nt = len(testable_results)
    nt_exact = sum(1 for r in testable_results if r['best_match'] == 'exact')
    nt_superset = sum(1 for r in testable_results if r['best_match'] in ('exact', 'superset'))
    nt_pair_overlap = sum(1 for r in testable_results if r['best_match'] in ('exact', 'superset', 'pair_overlap'))
    nt_any_overlap = sum(1 for r in testable_results if r['best_match'] in ('exact', 'superset', 'pair_overlap', 'any_overlap'))

    recall = {
        'exact': n_exact / n if n > 0 else 0,
        'superset': n_superset / n if n > 0 else 0,
        'pair_overlap': n_pair_overlap / n if n > 0 else 0,
        'any_overlap': n_any_overlap / n if n > 0 else 0,
        'n_entries': n,
        'n_exact': n_exact,
        'n_superset': n_superset,
        'n_pair_overlap': n_pair_overlap,
        'n_any_overlap': n_any_overlap,
        'n_no_prediction': n_no_prediction,
        # Testable-only metrics
        'n_testable': nt,
        'n_untestable': n - nt,
        'testable_exact': nt_exact / nt if nt > 0 else 0,
        'testable_superset': nt_superset / nt if nt > 0 else 0,
        'testable_pair_overlap': nt_pair_overlap / nt if nt > 0 else 0,
        'testable_any_overlap': nt_any_overlap / nt if nt > 0 else 0,
        'nt_exact': nt_exact,
        'nt_superset': nt_superset,
        'nt_pair_overlap': nt_pair_overlap,
        'nt_any_overlap': nt_any_overlap,
    }

    if verbose:
        print("=" * 70)
        print(f"BENCHMARK RESULTS ({n} gold standard entries)")
        print("=" * 70)
        print(f"  All {n} entries:")
        print(f"    Exact recall:        {recall['exact']:.1%} ({n_exact}/{n})")
        print(f"    Superset recall:     {recall['superset']:.1%} ({n_superset}/{n})")
        print(f"    Pair-overlap recall: {recall['pair_overlap']:.1%} ({n_pair_overlap}/{n}) [|G∩T|≥2]")
        print(f"    Any-overlap recall:  {recall['any_overlap']:.1%} ({n_any_overlap}/{n}) [|G∩T|≥1]")
        print(f"    No prediction:       {n_no_prediction}/{n} entries had no matching cancer type")
        print(f"  Testable {nt} entries (excl. {n - nt} structurally unmatchable):")
        print(f"    Exact recall:        {recall['testable_exact']:.1%} ({nt_exact}/{nt})")
        print(f"    Superset recall:     {recall['testable_superset']:.1%} ({nt_superset}/{nt})")
        print(f"    Pair-overlap recall: {recall['testable_pair_overlap']:.1%} ({nt_pair_overlap}/{nt}) [|G∩T|≥2]")
        print(f"    Any-overlap recall:  {recall['testable_any_overlap']:.1%} ({nt_any_overlap}/{nt}) [|G∩T|≥1]")
        print()

        # Show matches
        print("MATCHES:")
        for r in results:
            if r['best_match'] != 'none':
                print(f"  [{r['best_match'].upper():>8}] {r['gold_cancer']}: "
                      f"gold={sorted(r['gold_targets'])} → "
                      f"pred={sorted(r['predicted'])} ({r['evidence']})")

        print("\nMISSES:")
        for r in results:
            if r['best_match'] == 'none':
                reason = 'no cancer match' if r['pipeline_cancer'] == 'NO_PREDICTION' \
                    else 'no target overlap'
                print(f"  {r['gold_cancer']}: gold={sorted(r['gold_targets'])} "
                      f"[{reason}] ({r['evidence']})")

    # Cancer-level precision
    prec = _compute_cancer_precision(df, gold_entries)
    recall.update(prec)

    if verbose:
        print(f"\n  Cancer-level precision (evaluable cancers with GS entries):")
        print(f"    All:      {prec['precision']:.1%} ({prec['n_prec_hits']}/{prec['n_evaluable']})")
        print(f"    Testable: {prec['precision_testable']:.1%} "
              f"({prec['n_prec_hits_testable']}/{prec['n_evaluable_testable']})")

    return {'recall': recall, 'results': results, 'gold_standard_stats': stats}


# ============================================================================
# FREQUENCY BASELINE
# ============================================================================

def run_frequency_baseline(
    predictions_csv: str = 'results/triple_combinations.csv',
    gold_entries: Optional[List] = None,
    verbose: bool = True,
) -> Dict:
    """
    Global frequency baseline: always predict the 3 most frequently selected
    genes across ALL cancer types.  This is the correct null hypothesis —
    "does ALIN's cancer-specific selection add value over a naive approach?"

    Per-cancer frequency is tautological when each cancer has a single
    prediction row (the top-3 per cancer IS the ALIN prediction).
    """
    if gold_entries is None:
        gold_entries = GOLD_STANDARD

    df = pd.read_csv(predictions_csv)
    df.columns = [c.replace(' ', '_') for c in df.columns]

    # Global gene frequency across all cancers
    from collections import Counter
    gene_freq = Counter()
    for _, row in df.iterrows():
        gene_freq[row['Target_1']] += 1
        gene_freq[row['Target_2']] += 1
        gene_freq[row['Target_3']] += 1
    global_top3 = frozenset(g for g, _ in gene_freq.most_common(3))

    if verbose:
        print(f"  Global top-3 genes: {sorted(global_top3)}")

    # Check which cancers exist in predictions
    pipeline_cancer_set = set(df['Cancer_Type'].unique())

    results = []
    for entry in gold_entries:
        gold_cancer = entry['cancer']
        gold_targets = entry['targets']

        # Check if cancer exists in pipeline
        resolved = _resolve_pipeline_cancers(gold_cancer)
        has_cancer = bool(resolved & pipeline_cancer_set)

        if not has_cancer:
            match_type = 'none'
        else:
            match_type = check_match(global_top3, gold_targets)

        results.append({
            'gold_cancer': gold_cancer,
            'gold_targets': gold_targets,
            'best_match': match_type,
            'predicted': global_top3 if has_cancer else frozenset(),
        })

    n = len(results)
    n_exact = sum(1 for r in results if r['best_match'] == 'exact')
    n_superset = sum(1 for r in results if r['best_match'] in ('exact', 'superset'))
    n_pair_overlap = sum(1 for r in results if r['best_match'] in ('exact', 'superset', 'pair_overlap'))
    n_any_overlap = sum(1 for r in results if r['best_match'] in ('exact', 'superset', 'pair_overlap', 'any_overlap'))

    # Testable-only
    testable_results = [r for r, e in zip(results, gold_entries) if is_testable(e)]
    nt = len(testable_results)
    nt_any_overlap = sum(1 for r in testable_results if r['best_match'] in ('exact', 'superset', 'pair_overlap', 'any_overlap'))
    nt_pair_overlap = sum(1 for r in testable_results if r['best_match'] in ('exact', 'superset', 'pair_overlap'))

    recall = {
        'exact': n_exact / n if n > 0 else 0,
        'superset': n_superset / n if n > 0 else 0,
        'pair_overlap': n_pair_overlap / n if n > 0 else 0,
        'any_overlap': n_any_overlap / n if n > 0 else 0,
        'n_testable': nt,
        'testable_any_overlap': nt_any_overlap / nt if nt > 0 else 0,
        'testable_pair_overlap': nt_pair_overlap / nt if nt > 0 else 0,
    }

    if verbose:
        print(f"\nGLOBAL FREQUENCY BASELINE ({n} entries):")
        print(f"  Exact:        {recall['exact']:.1%} ({n_exact}/{n})")
        print(f"  Superset:     {recall['superset']:.1%} ({n_superset}/{n})")
        print(f"  Pair-overlap: {recall['pair_overlap']:.1%} ({n_pair_overlap}/{n})")
        print(f"  Any-overlap:  {recall['any_overlap']:.1%} ({n_any_overlap}/{n})")
        print(f"  Testable ({nt}): pair-overlap {recall['testable_pair_overlap']:.1%}, "
              f"any-overlap {recall['testable_any_overlap']:.1%}")

    # Cancer-level precision: global top-3 predicted for every cancer in the CSV
    pred_targets_by_cancer = {c: set(global_top3) for c in df['Cancer_Type'].unique()}
    prec = _compute_cancer_precision(df, gold_entries, pred_targets_by_cancer)
    recall.update(prec)

    if verbose:
        print(f"  Precision (evaluable): {prec['precision']:.1%} "
              f"({prec['n_prec_hits']}/{prec['n_evaluable']}), "
              f"testable {prec['precision_testable']:.1%} "
              f"({prec['n_prec_hits_testable']}/{prec['n_evaluable_testable']})")

    return {'recall': recall, 'results': results}


# ============================================================================
# DRIVER BASELINE
# ============================================================================

def run_driver_baseline(
    gold_entries: Optional[List] = None,
    verbose: bool = True,
) -> Dict:
    """
    Run driver-gene baseline (TCGA/COSMIC top-3 per cancer) against
    gold standard.
    """
    from benchmarking_module import CANCER_DRIVER_GENES

    if gold_entries is None:
        gold_entries = GOLD_STANDARD

    aliases = CANCER_ALIASES

    results = []
    for entry in gold_entries:
        gold_cancer = entry['cancer']
        gold_targets = entry['targets']

        driver_pred = frozenset()
        # Try exact match against CANCER_DRIVER_GENES keys
        # (which use the same naming as pipeline cancer types)
        pipeline_cancers = _resolve_pipeline_cancers(gold_cancer)
        for driver_cancer, drivers in CANCER_DRIVER_GENES.items():
            if driver_cancer in pipeline_cancers or driver_cancer == gold_cancer:
                driver_pred = frozenset(drivers[:3])
                break

        match_type = check_match(driver_pred, gold_targets) if driver_pred else 'none'
        results.append({
            'gold_cancer': gold_cancer,
            'gold_targets': gold_targets,
            'best_match': match_type,
            'predicted': driver_pred,
        })

    n = len(results)
    n_exact = sum(1 for r in results if r['best_match'] == 'exact')
    n_superset = sum(1 for r in results if r['best_match'] in ('exact', 'superset'))
    n_pair_overlap = sum(1 for r in results if r['best_match'] in ('exact', 'superset', 'pair_overlap'))
    n_any_overlap = sum(1 for r in results if r['best_match'] in ('exact', 'superset', 'pair_overlap', 'any_overlap'))

    # Testable-only
    testable_results = [r for r, e in zip(results, gold_entries) if is_testable(e)]
    nt = len(testable_results)
    nt_any_overlap = sum(1 for r in testable_results if r['best_match'] in ('exact', 'superset', 'pair_overlap', 'any_overlap'))
    nt_pair_overlap = sum(1 for r in testable_results if r['best_match'] in ('exact', 'superset', 'pair_overlap'))

    recall = {
        'exact': n_exact / n if n > 0 else 0,
        'superset': n_superset / n if n > 0 else 0,
        'pair_overlap': n_pair_overlap / n if n > 0 else 0,
        'any_overlap': n_any_overlap / n if n > 0 else 0,
        'n_testable': nt,
        'testable_any_overlap': nt_any_overlap / nt if nt > 0 else 0,
        'testable_pair_overlap': nt_pair_overlap / nt if nt > 0 else 0,
    }

    if verbose:
        print(f"\nDRIVER BASELINE ({n} entries):")
        print(f"  Exact:        {recall['exact']:.1%} ({n_exact}/{n})")
        print(f"  Superset:     {recall['superset']:.1%} ({n_superset}/{n})")
        print(f"  Pair-overlap: {recall['pair_overlap']:.1%} ({n_pair_overlap}/{n})")
        print(f"  Any-overlap:  {recall['any_overlap']:.1%} ({n_any_overlap}/{n})")
        print(f"  Testable ({nt}): pair-overlap {recall['testable_pair_overlap']:.1%}, "
              f"any-overlap {recall['testable_any_overlap']:.1%}")

    # Cancer-level precision: per-cancer driver genes as predictions
    pred_targets_by_cancer = {}
    for driver_cancer, drivers in CANCER_DRIVER_GENES.items():
        pred_targets_by_cancer[driver_cancer] = set(drivers[:3])
    prec = _compute_cancer_precision(
        pd.DataFrame({'Cancer_Type': list(pred_targets_by_cancer.keys())}),
        gold_entries, pred_targets_by_cancer,
    )
    recall.update(prec)

    if verbose:
        print(f"  Precision (evaluable): {prec['precision']:.1%} "
              f"({prec['n_prec_hits']}/{prec['n_evaluable']}), "
              f"testable {prec['precision_testable']:.1%} "
              f"({prec['n_prec_hits_testable']}/{prec['n_evaluable_testable']})")

    return {'recall': recall, 'results': results}


# ============================================================================
# RANDOM BASELINE
# ============================================================================

def run_random_baseline(
    predictions_csv: str = 'results/triple_combinations.csv',
    gold_entries: Optional[List] = None,
    n_trials: int = 1000,
    verbose: bool = True,
) -> Dict:
    """Random baseline: draw 3 genes from global gene pool for each cancer."""
    if gold_entries is None:
        gold_entries = GOLD_STANDARD

    df = pd.read_csv(predictions_csv)
    df.columns = [c.replace(' ', '_') for c in df.columns]
    all_genes = sorted({g for _, r in df.iterrows()
                        for g in [r['Target_1'], r['Target_2'], r['Target_3']]})

    aliases = CANCER_ALIASES
    rng = np.random.RandomState(42)

    n = len(gold_entries)
    exact_counts = []
    superset_counts = []
    pair_overlap_counts = []
    any_overlap_counts = []

    # Pre-compute testability mask
    testable_mask = [is_testable(e) for e in gold_entries]
    nt = sum(testable_mask)
    testable_pair_overlap_counts = []
    testable_any_overlap_counts = []

    # Pre-compute GS targets by cancer for precision (once, outside loop)
    cancer_gs_all = defaultdict(set)
    cancer_gs_testable_map = defaultdict(set)
    pipeline_cancer_set = set(df['Cancer_Type'].unique())
    for entry in gold_entries:
        pipeline_cancers = _resolve_pipeline_cancers(entry['cancer'])
        for pc in pipeline_cancers:
            if pc:
                cancer_gs_all[pc].update(entry['targets'])
                if is_testable(entry):
                    cancer_gs_testable_map[pc].update(entry['targets'])
    evaluable_cancers = sorted(c for c in pipeline_cancer_set if c in cancer_gs_all)
    evaluable_cancers_testable = sorted(c for c in pipeline_cancer_set if c in cancer_gs_testable_map)

    precision_counts = []
    precision_testable_counts = []

    for _ in range(n_trials):
        n_e = n_s = n_po = n_ao = 0
        nt_po = nt_ao = 0
        for idx, entry in enumerate(gold_entries):
            gold_cancer = entry['cancer']
            gold_targets = entry['targets']

            # Check if pipeline has this cancer type
            has_cancer = False
            pipeline_cancers = _resolve_pipeline_cancers(gold_cancer)
            for _, row in df.iterrows():
                if row['Cancer_Type'] in pipeline_cancers:
                    has_cancer = True
                    break

            if not has_cancer:
                continue

            random_pred = frozenset(rng.choice(all_genes, size=3, replace=False))
            mt = check_match(random_pred, gold_targets)
            if mt == 'exact':
                n_e += 1; n_s += 1; n_po += 1; n_ao += 1
                if testable_mask[idx]:
                    nt_po += 1; nt_ao += 1
            elif mt == 'superset':
                n_s += 1; n_po += 1; n_ao += 1
                if testable_mask[idx]:
                    nt_po += 1; nt_ao += 1
            elif mt == 'pair_overlap':
                n_po += 1; n_ao += 1
                if testable_mask[idx]:
                    nt_po += 1; nt_ao += 1
            elif mt == 'any_overlap':
                n_ao += 1
                if testable_mask[idx]:
                    nt_ao += 1

        exact_counts.append(n_e / n)
        superset_counts.append(n_s / n)
        pair_overlap_counts.append(n_po / n)
        any_overlap_counts.append(n_ao / n)
        testable_pair_overlap_counts.append(nt_po / nt if nt > 0 else 0)
        testable_any_overlap_counts.append(nt_ao / nt if nt > 0 else 0)

        # Precision for this trial: random triple per evaluable cancer
        n_prec = 0
        for c in evaluable_cancers:
            rand_pred = set(rng.choice(all_genes, size=3, replace=False))
            pred_exp = _expand_with_equivalents(rand_pred)
            gs_exp = _expand_with_equivalents(cancer_gs_all[c])
            if pred_exp & gs_exp:
                n_prec += 1
        precision_counts.append(n_prec / len(evaluable_cancers) if evaluable_cancers else 0)

        n_prec_t = 0
        for c in evaluable_cancers_testable:
            rand_pred = set(rng.choice(all_genes, size=3, replace=False))
            pred_exp = _expand_with_equivalents(rand_pred)
            gs_exp = _expand_with_equivalents(cancer_gs_testable_map[c])
            if pred_exp & gs_exp:
                n_prec_t += 1
        precision_testable_counts.append(
            n_prec_t / len(evaluable_cancers_testable) if evaluable_cancers_testable else 0
        )

    recall = {
        'exact': float(np.mean(exact_counts)),
        'superset': float(np.mean(superset_counts)),
        'pair_overlap': float(np.mean(pair_overlap_counts)),
        'any_overlap': float(np.mean(any_overlap_counts)),
        'n_testable': nt,
        'testable_pair_overlap': float(np.mean(testable_pair_overlap_counts)),
        'testable_any_overlap': float(np.mean(testable_any_overlap_counts)),
        'precision': float(np.mean(precision_counts)),
        'n_evaluable': len(evaluable_cancers),
        'n_prec_hits': round(float(np.mean(precision_counts)) * len(evaluable_cancers)),
        'precision_testable': float(np.mean(precision_testable_counts)),
        'n_evaluable_testable': len(evaluable_cancers_testable),
        'n_prec_hits_testable': round(float(np.mean(precision_testable_counts)) * len(evaluable_cancers_testable)),
    }

    if verbose:
        print(f"\nRANDOM BASELINE ({n} entries, {n_trials} trials):")
        print(f"  Exact:        {recall['exact']:.1%}")
        print(f"  Superset:     {recall['superset']:.1%}")
        print(f"  Pair-overlap: {recall['pair_overlap']:.1%}")
        print(f"  Any-overlap:  {recall['any_overlap']:.1%}")
        print(f"  Testable ({nt}): pair-overlap {recall['testable_pair_overlap']:.1%}, "
              f"any-overlap {recall['testable_any_overlap']:.1%}")
        print(f"  Precision (evaluable): {recall['precision']:.1%}, "
              f"testable {recall['precision_testable']:.1%}")

    return {'recall': recall}


# ============================================================================
# CANDIDATE-POOL RANDOM BASELINE
# ============================================================================
# Instead of sampling from the global gene pool (~20k genes), sample from the
# same cancer-specific candidate pool the pipeline uses: selective essential
# genes (CRISPR dep < -0.5 in ≥30% of cancer cell lines, excluding
# pan-essential genes essential in >90% of ALL lines).
# ============================================================================

def _build_candidate_pools(
    depmap_dir: str = './depmap_data',
    dependency_threshold: float = -0.5,
    min_selectivity_fraction: float = 0.3,
    pan_essential_threshold: float = 0.9,
) -> Dict[str, List[str]]:
    """Build per-cancer candidate gene pools mirroring the pipeline's filtering.

    Returns
    -------
    dict mapping cancer_type (OncotreePrimaryDisease) → sorted list of
    selective essential genes after pan-essential removal.
    """
    from pan_cancer_xnode import DepMapLoader

    depmap = DepMapLoader(depmap_dir)
    crispr = depmap.load_crispr_dependencies()
    pan_essential = depmap.get_pan_essential_genes(threshold=pan_essential_threshold)

    # Get all cancer types available in DepMap
    cancer_types = depmap.get_available_cancer_types()
    pools: Dict[str, List[str]] = {}

    for cancer_type, _count in cancer_types:
        cell_lines = depmap.get_cell_lines_for_cancer(cancer_type)
        available = [cl for cl in cell_lines if cl in crispr.index]
        if len(available) < 3:
            continue

        crispr_sub = crispr.loc[available]
        n_lines = len(available)
        min_lines = max(1, int(n_lines * min_selectivity_fraction))

        # Essential if dep < threshold in each line
        essential_matrix = crispr_sub < dependency_threshold
        essential_count = essential_matrix.sum(axis=0)

        # Selective: essential in ≥ min_lines AND not pan-essential
        selective = set(essential_count[essential_count >= min_lines].index) - pan_essential
        if len(selective) < 2:
            # Fallback: mean dependency < threshold, minus pan-essential
            mean_dep = crispr_sub.mean(axis=0)
            selective = set(mean_dep[mean_dep < dependency_threshold].index) - pan_essential
        if len(selective) >= 3:
            pools[cancer_type] = sorted(selective)

    return pools


def run_candidate_pool_random_baseline(
    predictions_csv: str = 'results/triple_combinations.csv',
    gold_entries: Optional[List] = None,
    n_trials: int = 1000,
    depmap_dir: str = './depmap_data',
    verbose: bool = True,
) -> Dict:
    """Random baseline sampling from cancer-specific candidate pools.

    For each gold-standard entry's cancer type the function draws 3 genes
    from the selective-essential pool that the pipeline would consider for
    that cancer type.  This is the *right null*: if random picks from the
    candidate pool match as often as ALIN, the scoring adds no value
    beyond the pre-filtering step.
    """
    if gold_entries is None:
        gold_entries = GOLD_STANDARD

    df = pd.read_csv(predictions_csv)
    df.columns = [c.replace(' ', '_') for c in df.columns]

    # Build cancer-specific candidate pools
    pools = _build_candidate_pools(depmap_dir)
    if verbose:
        print(f"\nBuilt candidate pools for {len(pools)} cancer types "
              f"(median {np.median([len(v) for v in pools.values()]):.0f} genes)")

    aliases = CANCER_ALIASES
    rng = np.random.RandomState(42)

    n = len(gold_entries)
    exact_counts = []
    superset_counts = []
    pair_overlap_counts = []
    any_overlap_counts = []

    testable_mask = [is_testable(e) for e in gold_entries]
    nt = sum(testable_mask)
    testable_pair_overlap_counts = []
    testable_any_overlap_counts = []

    # Pre-compute GS targets by cancer for precision
    cancer_gs_all = defaultdict(set)
    cancer_gs_testable_map = defaultdict(set)
    pipeline_cancer_set = set(df['Cancer_Type'].unique())
    for entry in gold_entries:
        pipeline_cancers = _resolve_pipeline_cancers(entry['cancer'])
        for pc in pipeline_cancers:
            if pc:
                cancer_gs_all[pc].update(entry['targets'])
                if is_testable(entry):
                    cancer_gs_testable_map[pc].update(entry['targets'])
    evaluable_cancers = sorted(c for c in pipeline_cancer_set if c in cancer_gs_all)
    evaluable_cancers_testable = sorted(c for c in pipeline_cancer_set if c in cancer_gs_testable_map)

    precision_counts = []
    precision_testable_counts = []

    # Map gold-standard cancer types to candidate-pool keys
    def _get_pool_for_gold_cancer(gold_cancer: str) -> Optional[List[str]]:
        """Return candidate pool for a gold standard cancer type."""
        # Try direct match first
        if gold_cancer in pools:
            return pools[gold_cancer]
        # Try via alias mapping
        pipeline_names = _resolve_pipeline_cancers(gold_cancer)
        for pn in pipeline_names:
            if pn in pools:
                return pools[pn]
        return None

    for _ in range(n_trials):
        n_e = n_s = n_po = n_ao = 0
        nt_po = nt_ao = 0
        for idx, entry in enumerate(gold_entries):
            gold_cancer = entry['cancer']
            gold_targets = entry['targets']

            # Check if pipeline has this cancer type
            pipeline_cancers = _resolve_pipeline_cancers(gold_cancer)
            has_cancer = any(
                row['Cancer_Type'] in pipeline_cancers
                for _, row in df.iterrows()
            )
            if not has_cancer:
                continue

            pool = _get_pool_for_gold_cancer(gold_cancer)
            if pool is None or len(pool) < 3:
                continue  # skip if no candidate pool available

            random_pred = frozenset(rng.choice(pool, size=3, replace=False))
            mt = check_match(random_pred, gold_targets)
            if mt == 'exact':
                n_e += 1; n_s += 1; n_po += 1; n_ao += 1
                if testable_mask[idx]:
                    nt_po += 1; nt_ao += 1
            elif mt == 'superset':
                n_s += 1; n_po += 1; n_ao += 1
                if testable_mask[idx]:
                    nt_po += 1; nt_ao += 1
            elif mt == 'pair_overlap':
                n_po += 1; n_ao += 1
                if testable_mask[idx]:
                    nt_po += 1; nt_ao += 1
            elif mt == 'any_overlap':
                n_ao += 1
                if testable_mask[idx]:
                    nt_ao += 1

        exact_counts.append(n_e / n)
        superset_counts.append(n_s / n)
        pair_overlap_counts.append(n_po / n)
        any_overlap_counts.append(n_ao / n)
        testable_pair_overlap_counts.append(nt_po / nt if nt > 0 else 0)
        testable_any_overlap_counts.append(nt_ao / nt if nt > 0 else 0)

        # Precision per evaluable cancer
        n_prec = 0
        for c in evaluable_cancers:
            c_pool = pools.get(c)
            if c_pool is None or len(c_pool) < 3:
                continue
            rand_pred = set(rng.choice(c_pool, size=3, replace=False))
            pred_exp = _expand_with_equivalents(rand_pred)
            gs_exp = _expand_with_equivalents(cancer_gs_all[c])
            if pred_exp & gs_exp:
                n_prec += 1
        precision_counts.append(n_prec / len(evaluable_cancers) if evaluable_cancers else 0)

        n_prec_t = 0
        for c in evaluable_cancers_testable:
            c_pool = pools.get(c)
            if c_pool is None or len(c_pool) < 3:
                continue
            rand_pred = set(rng.choice(c_pool, size=3, replace=False))
            pred_exp = _expand_with_equivalents(rand_pred)
            gs_exp = _expand_with_equivalents(cancer_gs_testable_map[c])
            if pred_exp & gs_exp:
                n_prec_t += 1
        precision_testable_counts.append(
            n_prec_t / len(evaluable_cancers_testable) if evaluable_cancers_testable else 0
        )

    recall = {
        'exact': float(np.mean(exact_counts)),
        'superset': float(np.mean(superset_counts)),
        'pair_overlap': float(np.mean(pair_overlap_counts)),
        'any_overlap': float(np.mean(any_overlap_counts)),
        'n_testable': nt,
        'testable_pair_overlap': float(np.mean(testable_pair_overlap_counts)),
        'testable_any_overlap': float(np.mean(testable_any_overlap_counts)),
        'precision': float(np.mean(precision_counts)),
        'n_evaluable': len(evaluable_cancers),
        'n_prec_hits': round(float(np.mean(precision_counts)) * len(evaluable_cancers)),
        'precision_testable': float(np.mean(precision_testable_counts)),
        'n_evaluable_testable': len(evaluable_cancers_testable),
        'n_prec_hits_testable': round(float(np.mean(precision_testable_counts)) * len(evaluable_cancers_testable)),
    }

    if verbose:
        print(f"\nCANDIDATE-POOL RANDOM BASELINE ({n} entries, {n_trials} trials):")
        print(f"  Exact:        {recall['exact']:.1%}")
        print(f"  Superset:     {recall['superset']:.1%}")
        print(f"  Pair-overlap: {recall['pair_overlap']:.1%}")
        print(f"  Any-overlap:  {recall['any_overlap']:.1%}")
        print(f"  Testable ({nt}): pair-overlap {recall['testable_pair_overlap']:.1%}, "
              f"any-overlap {recall['testable_any_overlap']:.1%}")
        print(f"  Precision (evaluable): {recall['precision']:.1%}, "
              f"testable {recall['precision_testable']:.1%}")

    return {'recall': recall}


# ============================================================================
# CLI
# ============================================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Gold Standard Benchmark')
    parser.add_argument('--predictions', default='results/triple_combinations.csv')
    parser.add_argument('--predictions-no-ks', default=None,
                        help='Predictions CSV generated WITHOUT KNOWN_SYNERGIES '
                             '(for circularity test)')
    parser.add_argument('--tier2', action='store_true', help='Include Tier 2 synergy data')
    parser.add_argument('--validate-only', action='store_true', help='Only validate gold standard')
    args = parser.parse_args()

    # Validate gold standard
    print("=" * 70)
    print("GOLD STANDARD VALIDATION")
    print("=" * 70)
    stats = validate_gold_standard()
    print(f"Total entries: {stats['total_entries']}")
    print(f"Unique cancer+target pairs: {stats['unique_pairs']}")
    print(f"Cancer types: {stats['cancer_types']}")
    print(f"Evidence breakdown: {stats['evidence_breakdown']}")
    print(f"Duplicates: {stats['duplicates']}")
    print(f"Unique gene targets: {len(stats['all_targets'])} — {stats['all_targets']}")

    if args.validate_only:
        exit(0)

    print()

    # Run ALIN benchmark
    alin_results = run_benchmark(args.predictions, tier1=True, tier2=args.tier2)

    # Run baselines
    freq_results = run_frequency_baseline(args.predictions)
    driver_results = run_driver_baseline()
    random_results = run_random_baseline(args.predictions)
    cpool_results = run_candidate_pool_random_baseline(args.predictions)

    # Summary comparison
    print()
    print("=" * 70)
    print("COMPARISON: ALIN vs BASELINES")
    print("=" * 70)
    print(f"{'Method':<30} {'Exact':>8} {'Superset':>10} {'PairOvlp':>10} {'AnyOvlp':>10} {'Precision':>10}")
    print("-" * 80)
    for name, res in [
        ('ALIN', alin_results['recall']),
        ('Global frequency', freq_results['recall']),
        ('Driver genes', driver_results['recall']),
        ('Random-global (1000×)', random_results['recall']),
        ('Random-candidate (1000×)', cpool_results['recall']),
    ]:
        prec_str = f"{res.get('precision', 0):>9.1%}"
        print(f"{name:<30} {res['exact']:>7.1%} {res['superset']:>9.1%} "
              f"{res.get('pair_overlap', 0):>9.1%} {res.get('any_overlap', res.get('pairwise', 0)):>9.1%} "
              f"{prec_str}")

    # Testable-only comparison
    nt = alin_results['recall'].get('n_testable', N_TESTABLE)
    print()
    print(f"TESTABLE-ONLY ({nt}/{len(GOLD_STANDARD)} entries):")
    print(f"{'Method':<30} {'PairOvlp':>10} {'AnyOvlp':>10} {'Precision':>10}")
    print("-" * 62)
    for name, res in [
        ('ALIN', alin_results['recall']),
        ('Global frequency', freq_results['recall']),
        ('Driver genes', driver_results['recall']),
        ('Random-global (1000×)', random_results['recall']),
        ('Random-candidate (1000×)', cpool_results['recall']),
    ]:
        prec_t_str = f"{res.get('precision_testable', 0):>9.1%}"
        print(f"{name:<30} {res.get('testable_pair_overlap', 0):>9.1%} "
              f"{res.get('testable_any_overlap', 0):>9.1%} {prec_t_str}")

    # Key question: does ALIN beat baselines?
    print()
    alin_ao = alin_results['recall'].get('any_overlap', alin_results['recall'].get('pairwise', 0))
    freq_ao = freq_results['recall'].get('any_overlap', freq_results['recall'].get('pairwise', 0))
    cpool_ao = cpool_results['recall'].get('any_overlap', 0)
    rand_ao = random_results['recall'].get('any_overlap', 0)

    if abs(alin_ao - freq_ao) < 0.001:
        print("⚠  ALIN TIES global frequency baseline on any-overlap recall.")
    elif alin_ao > freq_ao:
        print(f"✓  ALIN beats global frequency by {alin_ao - freq_ao:.1%} on any-overlap recall.")
    else:
        print(f"✗  Global frequency beats ALIN by {freq_ao - alin_ao:.1%} on any-overlap recall.")

    # Candidate-pool random vs ALIN
    if abs(alin_ao - cpool_ao) < 0.03:
        print(f"⚠  Candidate-pool random ({cpool_ao:.1%}) ≈ ALIN ({alin_ao:.1%}) — "
              f"scoring may add little value beyond pre-filtering.")
    elif alin_ao > cpool_ao:
        print(f"✓  ALIN ({alin_ao:.1%}) beats candidate-pool random ({cpool_ao:.1%}) "
              f"by {alin_ao - cpool_ao:.1%} — scoring adds value beyond filtering.")
    else:
        print(f"✗  Candidate-pool random ({cpool_ao:.1%}) beats ALIN ({alin_ao:.1%}) "
              f"by {cpool_ao - alin_ao:.1%} — scoring adds no value beyond pre-filtering.")

    # ── KNOWN_SYNERGIES circularity test ──────────────────────────────
    if args.predictions_no_ks:
        print()
        print("=" * 70)
        print("CIRCULARITY TEST: KNOWN_SYNERGIES ablation")
        print("=" * 70)
        noks_results = run_benchmark(
            args.predictions_no_ks, tier1=True, tier2=args.tier2
        )
        noks = noks_results['recall']
        ks = alin_results['recall']

        print(f"{'Metric':<25} {'With KS':>10} {'Without KS':>12} {'Δ':>8}")
        print("-" * 57)
        for label, key in [
            ('Exact recall', 'exact'),
            ('Superset recall', 'superset'),
            ('Pair-overlap recall', 'pair_overlap'),
            ('Any-overlap recall', 'any_overlap'),
            ('Precision', 'precision'),
            ('Testable any-overlap', 'testable_any_overlap'),
            ('Testable precision', 'precision_testable'),
        ]:
            v_ks = ks.get(key, 0)
            v_no = noks.get(key, 0)
            delta = v_no - v_ks
            sign = '+' if delta >= 0 else ''
            print(f"{label:<25} {v_ks:>9.1%} {v_no:>11.1%} {sign}{delta:>6.1%}")

        # Interpretation
        ao_ks = ks.get('any_overlap', 0)
        ao_no = noks.get('any_overlap', 0)
        if abs(ao_ks - ao_no) < 0.03:
            print("\n✓  Recall is STABLE without KNOWN_SYNERGIES — no circularity concern.")
        elif ao_no >= ao_ks * 0.8:
            print(f"\n✓  Recall drops modestly ({ao_ks:.1%} → {ao_no:.1%}); "
                  f"pipeline retains most value without curated synergies.")
        else:
            print(f"\n⚠  Recall drops substantially ({ao_ks:.1%} → {ao_no:.1%}); "
                  f"curated synergies contribute meaningfully to predictions.")
