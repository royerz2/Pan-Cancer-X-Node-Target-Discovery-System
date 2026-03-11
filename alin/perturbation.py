#!/usr/bin/env python3
"""
Perturbation Response Module
============================
Enriches viability paths with manually curated perturbation-response knowledge.

IMPORTANT LIMITATIONS
---------------------
This module encodes expert-curated summaries of ~15 published inhibitor
studies (phosphoproteomics and transcriptomics) as Python dictionaries.
It does NOT integrate systematic high-throughput perturbation databases
such as LINCS L1000 (~1.3 million gene-expression profiles across
thousands of perturbagens; Subramanian et al. 2017, Cell) or the
Connectivity Map (Lamb et al. 2006, Science).

Key caveats:
- Confidence scores (0.85--0.95) are author-assigned heuristics reflecting
  perceived study quality and replication, NOT formal statistical measures.
- Several aliases share identical perturbation profiles (FYN→SRC, CDK6→CDK4,
  JAK2→JAK1, MAP2K1/MAP2K2→MEK1), which is a biologically dubious
  simplification given distinct substrate specificities of these kinases.
- Coverage is biased toward well-studied oncogenic kinases with clinically
  approved inhibitors; rare cancers and under-studied targets have no or
  minimal representation.
- The perturbation bonus β_pert rewards combinations that match this curated
  prior knowledge, not independently measured experimental signatures.

Data sources (curated, not systematic):
1. Curated phosphoproteomics summaries from ~15 published kinase-inhibitor
   studies (see individual signature entries for PMIDs/sources).
2. Curated transcriptional response summaries from the same studies.

Future work should integrate L1000/CMap signatures at scale, replacing
hand-curated dictionaries with data-driven perturbation profiles.
"""

import logging
import copy
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Named constants
# ---------------------------------------------------------------------------
#: Weights for the fallback (curated) perturbation composite score.
FALLBACK_W_EFFECTOR: float = 0.6
FALLBACK_W_FEEDBACK: float = 0.4
#: Base and variable fractions for the direct-hit confidence adjustment.
#: confidence = sig.confidence * (CONF_ADJ_BASE + CONF_ADJ_VARIABLE * ratio)
CONF_ADJ_BASE: float = 0.5
CONF_ADJ_VARIABLE: float = 0.5


@dataclass
class PerturbationSignature:
    """Signature of downstream changes after inhibiting a target."""
    target: str  # The inhibited target (e.g., KRAS, EGFR)
    perturbation_type: str  # 'inhibitor', 'knockdown', 'knockout'
    
    # Phosphoproteomics: proteins with changed phosphorylation
    phospho_decreased: Set[str] = field(default_factory=set)  # Less phosphorylated
    phospho_increased: Set[str] = field(default_factory=set)  # More phosphorylated
    
    # Transcriptional: genes with changed expression
    expression_decreased: Set[str] = field(default_factory=set)  # Downregulated
    expression_increased: Set[str] = field(default_factory=set)  # Upregulated
    
    # Confidence and source
    # NOTE: confidence values are author-assigned heuristics (0.0--1.0)
    # reflecting perceived study quality and replication level.
    # They are NOT formal statistical measures (e.g., p-values or FDR).
    confidence: float = 0.8
    source: str = "curated"
    pmid: Optional[str] = None
    
    @property
    def all_responders(self) -> Set[str]:
        """All genes/proteins that respond to this perturbation."""
        return (self.phospho_decreased | self.phospho_increased | 
                self.expression_decreased | self.expression_increased)
    
    @property
    def direct_effectors(self) -> Set[str]:
        """Genes that decrease when target is inhibited (likely in pathway)."""
        return self.phospho_decreased | self.expression_decreased


# =============================================================================
# CURATED PERTURBATION SIGNATURES
# =============================================================================
# Based on published phosphoproteomics and transcriptomics studies

PERTURBATION_SIGNATURES: Dict[str, PerturbationSignature] = {
    
    # KRAS inhibition (sotorasib/adagrasib in KRAS G12C tumors)
    # Sources: Hallin et al. 2020 (sotorasib), Canon et al. 2019
    'KRAS': PerturbationSignature(
        target='KRAS',
        perturbation_type='inhibitor',
        phospho_decreased={
            'ERK1', 'ERK2', 'MAPK1', 'MAPK3',  # MAPK pathway
            'RSK1', 'RSK2', 'RPS6KA1', 'RPS6KA3',  # RSK kinases
            'ELK1', 'CREB1',  # Transcription factors
            'S6', 'RPS6',  # Ribosomal protein
            'AKT1', 'AKT2',  # Cross-talk with PI3K
            'MEK1', 'MEK2', 'MAP2K1', 'MAP2K2',  # MEK
            'BRAF', 'CRAF', 'RAF1',  # RAF
        },
        phospho_increased={
            'EGFR',  # Feedback activation (resistance mechanism)
            'ERBB2', 'ERBB3',  # HER family feedback
            'MET',  # RTK feedback
        },
        expression_decreased={
            'MYC', 'CCND1', 'CCNE1',  # Cell cycle
            'DUSP4', 'DUSP6',  # Phosphatases (MAPK targets)
            'ETV1', 'ETV4', 'ETV5',  # ETS transcription factors
            'SPRY2', 'SPRY4',  # Sprouty (feedback)
            'FOSL1', 'JUNB',  # AP-1 components
        },
        expression_increased={
            'BIM', 'BCL2L11',  # Pro-apoptotic (pathway off)
            'CDKN1A', 'CDKN1B',  # Cell cycle inhibitors
        },
        confidence=0.95,
        source='Canon et al. 2019, Hallin et al. 2020',
        pmid='31658955',
    ),
    
    # EGFR inhibition (erlotinib, gefitinib, osimertinib)
    # Sources: Bivona et al. 2011, multiple phosphoproteomics studies
    'EGFR': PerturbationSignature(
        target='EGFR',
        perturbation_type='inhibitor',
        phospho_decreased={
            'EGFR',  # Autophosphorylation
            'ERBB2', 'ERBB3',  # HER family
            'GRB2', 'SOS1', 'SHC1',  # Adaptor proteins
            'GAB1', 'GAB2',  # Scaffolds
            'ERK1', 'ERK2', 'MAPK1', 'MAPK3',
            'AKT1', 'AKT2', 'AKT3',
            'STAT3', 'STAT5A', 'STAT5B',
            'PLCg1', 'PLCG1',
            'SRC', 'FAK', 'PTK2',
        },
        phospho_increased={
            'MET',  # Bypass/resistance
            'AXL',  # EMT/resistance
            'FGFR1',  # Alternative RTK
        },
        expression_decreased={
            'MYC', 'CCND1',
            'VEGFA',  # Angiogenesis
            'HIF1A',  # Hypoxia
            'BCL2', 'BCLXL', 'BCL2L1',  # Anti-apoptotic
        },
        expression_increased={
            'BIM', 'BCL2L11', 'PUMA', 'BBC3',  # Pro-apoptotic
            'CDKN1A',
        },
        confidence=0.95,
        source='Bivona et al. 2011, EGFR TKI phosphoproteomics',
        pmid='21430269',
    ),
    
    # BRAF inhibition (vemurafenib, dabrafenib in BRAF V600E)
    # Sources: Bollag et al. 2010, Lito et al. 2012
    'BRAF': PerturbationSignature(
        target='BRAF',
        perturbation_type='inhibitor',
        phospho_decreased={
            'MEK1', 'MEK2', 'MAP2K1', 'MAP2K2',
            'ERK1', 'ERK2', 'MAPK1', 'MAPK3',
            'RSK1', 'RSK2',
            'ELK1', 'CREB1',
            'MYC',
        },
        phospho_increased={
            'EGFR',  # Feedback (paradoxical activation in WT)
            'CRAF', 'RAF1',  # RAF dimerization
            'ERBB2', 'ERBB3',
        },
        expression_decreased={
            'CCND1', 'CCNE1',
            'MYC',
            'DUSP4', 'DUSP6',
            'ETV1', 'ETV4',
            'MITF',  # Melanoma-specific
        },
        expression_increased={
            'BIM', 'BCL2L11',
            'CDKN1A',
            'PTEN',  # Negative regulator
        },
        confidence=0.95,
        source='Bollag et al. 2010, Lito et al. 2012',
        pmid='20823850',
    ),
    
    # MEK inhibition (trametinib, cobimetinib)
    'MEK1': PerturbationSignature(
        target='MEK1',
        perturbation_type='inhibitor',
        phospho_decreased={
            'ERK1', 'ERK2', 'MAPK1', 'MAPK3',
            'RSK1', 'RSK2', 'RSK3',
            'MNK1', 'MNK2', 'MKNK1', 'MKNK2',
            'ELK1', 'CREB1', 'ATF2',
            'MYC',
        },
        phospho_increased={
            'MEK1', 'MEK2',  # Feedback via RAF
            'BRAF', 'CRAF',  # RAF activation
            'AKT1', 'AKT2',  # Crosstalk
            'EGFR', 'ERBB2', 'ERBB3',  # RTK reactivation
        },
        expression_decreased={
            'MYC', 'CCND1', 'CCNE1',
            'FOSL1', 'JUNB',
            'DUSP4', 'DUSP6',
        },
        confidence=0.90,
        source='MEK inhibitor phosphoproteomics',
    ),
    'MAP2K1': None,  # Alias for MEK1 (set below)
    
    # CDK4/6 inhibition (palbociclib, ribociclib, abemaciclib)
    # Sources: Finn et al. 2016, multiple studies
    'CDK4': PerturbationSignature(
        target='CDK4',
        perturbation_type='inhibitor',
        phospho_decreased={
            'RB1',  # Retinoblastoma protein
            'FOXM1',  # Cell cycle TF
            'E2F1', 'E2F2', 'E2F3',  # E2F targets (functional)
            'CCNE1', 'CCNE2',  # Cyclin E (not directly, but cascade)
        },
        phospho_increased={
            'CDK2',  # Compensatory (resistance)
            'CCNE1',  # Cyclin E (resistance)
        },
        expression_decreased={
            'E2F1', 'E2F2',  # E2F targets
            'CCNE1', 'CCNE2',
            'CDC6', 'ORC1', 'MCM2', 'MCM7',  # Replication
            'PCNA',
            'MYC',  # Proliferation
            'CCNA2',  # Cyclin A
        },
        expression_increased={
            'CDKN1A', 'CDKN1B',  # p21, p27
            'CDKN2A',  # p16 (feedback)
        },
        confidence=0.90,
        source='CDK4/6 inhibitor studies, Finn et al. 2016',
    ),
    'CDK6': None,  # Alias (set below)
    
    # PI3K inhibition (alpelisib, copanlisib)
    'PIK3CA': PerturbationSignature(
        target='PIK3CA',
        perturbation_type='inhibitor',
        phospho_decreased={
            'AKT1', 'AKT2', 'AKT3',
            'MTOR', 'RPTOR',  # mTORC1
            'S6K1', 'RPS6KB1',
            'S6', 'RPS6',
            '4EBP1', 'EIF4EBP1',
            'GSK3A', 'GSK3B',
            'FOXO1', 'FOXO3',
            'PRAS40', 'AKT1S1',
        },
        phospho_increased={
            'EGFR', 'ERBB2', 'ERBB3',  # RTK feedback
            'IRS1', 'IRS2',  # Insulin signaling feedback
            'ERK1', 'ERK2',  # MAPK crosstalk
        },
        expression_decreased={
            'MYC', 'CCND1',
            'HIF1A', 'VEGFA',
            'GLUT1', 'SLC2A1',  # Glucose metabolism
        },
        expression_increased={
            'FOXO1', 'FOXO3',  # Active (dephosphorylated)
            'BIM', 'BCL2L11',
            'PTEN',
            'CDKN1B',
        },
        confidence=0.90,
        source='PI3K inhibitor phosphoproteomics',
    ),
    
    # mTOR inhibition (everolimus, temsirolimus)
    'MTOR': PerturbationSignature(
        target='MTOR',
        perturbation_type='inhibitor',
        phospho_decreased={
            'S6K1', 'RPS6KB1', 'S6K2', 'RPS6KB2',
            'S6', 'RPS6',
            '4EBP1', 'EIF4EBP1',
            'ULK1',  # Autophagy
            'LIPIN1',
        },
        phospho_increased={
            'AKT1', 'AKT2',  # Feedback (loss of S6K-IRS negative feedback)
            'EGFR', 'ERBB2',
            'ERK1', 'ERK2',  # MAPK reactivation
        },
        expression_decreased={
            'MYC', 'CCND1',
            'HIF1A', 'VEGFA',
            'SREBF1',  # Lipid synthesis
        },
        expression_increased={
            'LC3', 'MAP1LC3B',  # Autophagy
            'SQSTM1',  # p62
            'ATG5', 'ATG7',
        },
        confidence=0.90,
        source='mTOR inhibitor studies',
    ),
    
    # SRC inhibition (dasatinib)
    'SRC': PerturbationSignature(
        target='SRC',
        perturbation_type='inhibitor',
        phospho_decreased={
            'SRC',  # Autophosphorylation
            'FAK', 'PTK2',
            'STAT3', 'STAT5A', 'STAT5B',
            'p130CAS', 'BCAR1',
            'paxillin', 'PXN',
            'cortactin', 'CTTN',
            'EGFR',  # Transactivation
        },
        phospho_increased={
            'FYN', 'YES1', 'LYN',  # Compensatory SFKs
        },
        expression_decreased={
            'MYC', 'CCND1',
            'MMP2', 'MMP9',  # Matrix metalloproteinases
            'VEGFA',
        },
        confidence=0.85,
        source='Dasatinib phosphoproteomics',
    ),
    
    # STAT3 inhibition
    'STAT3': PerturbationSignature(
        target='STAT3',
        perturbation_type='inhibitor',
        phospho_decreased={
            'STAT3',  # Y705 phosphorylation
        },
        expression_decreased={
            'BCL2', 'BCLXL', 'BCL2L1', 'MCL1',  # Survival
            'MYC', 'CCND1',
            'VEGFA',
            'BIRC5',  # Survivin
            'MMP2', 'MMP9',
            'PD-L1', 'CD274',  # Immune evasion
            'IL6', 'IL10',  # Cytokines
        },
        expression_increased={
            'TP53',  # Tumor suppressor
            'CDKN1A',
            'CASP3', 'CASP9',  # Apoptosis
        },
        confidence=0.85,
        source='STAT3 inhibitor studies',
    ),
    
    # BCL2 inhibition (venetoclax)
    'BCL2': PerturbationSignature(
        target='BCL2',
        perturbation_type='inhibitor',
        phospho_decreased=set(),  # BCL2 inhibition primarily affects protein interactions
        phospho_increased={
            'MCL1',  # Compensatory (resistance)
            'BCL2L1',  # BCL-XL (resistance)
        },
        expression_decreased=set(),  # Venetoclax causes apoptosis; many genes decrease as cells die
        expression_increased={
            'BIM', 'BCL2L11',  # Freed from BCL2
            'BAX', 'BAK1',  # Activated
            'NOXA', 'PMAIP1',
        },
        confidence=0.85,
        source='Venetoclax studies',
    ),
    
    # JAK1/2 inhibition (ruxolitinib)
    'JAK1': PerturbationSignature(
        target='JAK1',
        perturbation_type='inhibitor',
        phospho_decreased={
            'JAK1', 'JAK2',
            'STAT1', 'STAT3', 'STAT5A', 'STAT5B',
            'STAT6',
        },
        expression_decreased={
            'BCL2', 'BCLXL', 'BCL2L1',
            'MYC', 'CCND1',
            'PIM1', 'PIM2',
            'SOCS1', 'SOCS3',  # Negative feedback
        },
        expression_increased={
            'BIM', 'BCL2L11',
            'CDKN1A',
        },
        confidence=0.85,
        source='JAK inhibitor studies',
    ),
    'JAK2': None,  # Alias (set below)
    
    # MET inhibition (capmatinib, tepotinib)
    'MET': PerturbationSignature(
        target='MET',
        perturbation_type='inhibitor',
        phospho_decreased={
            'MET',  # Autophosphorylation
            'GAB1',
            'ERK1', 'ERK2', 'MAPK1', 'MAPK3',
            'AKT1', 'AKT2',
            'STAT3',
            'FAK', 'PTK2',
            'SRC',
        },
        phospho_increased={
            'EGFR',  # Bypass
            'ERBB2', 'ERBB3',
            'AXL',  # EMT bypass
        },
        expression_decreased={
            'MYC', 'CCND1',
            'VEGFA',
            'MMP2', 'MMP9',
            'SNAI1', 'SNAI2',  # EMT
        },
        confidence=0.85,
        source='MET inhibitor phosphoproteomics',
    ),
    
    # ALK inhibition (alectinib, lorlatinib)
    'ALK': PerturbationSignature(
        target='ALK',
        perturbation_type='inhibitor',
        phospho_decreased={
            'ALK',
            'ERK1', 'ERK2', 'MAPK1', 'MAPK3',
            'AKT1', 'AKT2',
            'STAT3', 'STAT5A',
            'PLCg1', 'PLCG1',
        },
        expression_decreased={
            'MYC', 'CCND1',
            'BCL2',
        },
        expression_increased={
            'BIM', 'BCL2L11',
        },
        confidence=0.90,
        source='ALK inhibitor studies',
    ),

    # NRAS inhibition (no approved direct inhibitors; MEK/ERK are targets downstream)
    # Same RTK feedback rebound as KRAS — RTK upregulation upon RAS inhibition is
    # a class effect.  Source: Nazarian et al. 2010, Kemper et al. 2016.
    'NRAS': PerturbationSignature(
        target='NRAS',
        perturbation_type='inhibitor',
        phospho_decreased={
            'ERK1', 'ERK2', 'MAPK1', 'MAPK3',
            'RSK1', 'RPS6KA1',
            'MEK1', 'MEK2', 'MAP2K1', 'MAP2K2',
        },
        phospho_increased={
            'EGFR', 'ERBB2', 'ERBB3',  # RTK rebound (same mechanism as KRAS)
            'MET',
        },
        expression_decreased={
            'MYC', 'CCND1', 'CCNE1', 'FOSL1', 'JUNB',
        },
        expression_increased={
            'BIM', 'BCL2L11', 'CDKN1A', 'CDKN1B',
        },
        confidence=0.88,
        source='Nazarian et al. 2010; RTK rebound analogy from KRAS inhibitor data',
    ),

    # HRAS inhibition (farnesyl-transferase inhibitors; bladder, HNSCC)
    'HRAS': PerturbationSignature(
        target='HRAS',
        perturbation_type='inhibitor',
        phospho_decreased={
            'ERK1', 'ERK2', 'MAPK1', 'MAPK3',
            'AKT1', 'AKT2',
            'MEK1', 'MEK2', 'MAP2K1', 'MAP2K2',
        },
        phospho_increased={
            'EGFR', 'ERBB2', 'ERBB3',
            'MET',
        },
        expression_decreased={
            'MYC', 'CCND1',
        },
        expression_increased={
            'CDKN1A', 'CDKN1B',
        },
        confidence=0.82,
        source='RAS inhibitor analogy; Berndt et al. 2011',
    ),

    # ERBB2/HER2 inhibition (trastuzumab, lapatinib, pertuzumab, T-DM1)
    # Sources: Sergina et al. 2007 (HER3 feedback), Nahta et al. 2005 (IGF1R),
    #          Garrett et al. 2011 (MET upregulation)
    'ERBB2': PerturbationSignature(
        target='ERBB2',
        perturbation_type='inhibitor',
        phospho_decreased={
            'ERBB2', 'ERBB3', 'ERBB4',
            'AKT1', 'AKT2',
            'ERK1', 'ERK2', 'MAPK1', 'MAPK3',
            'SRC',
        },
        phospho_increased={
            'EGFR',   # EGFR homodimers compensate for HER2 loss
            'ERBB3',  # HER3 upregulation — major resistance node (Sergina 2007)
            'IGF1R',  # IGF1R-driven bypass (Nahta 2005)
            'MET',    # MET amplification/upregulation
            'AXL',    # AXL-driven EMT/resistance
        },
        expression_decreased={
            'MYC', 'CCND1', 'BCL2',
        },
        expression_increased={
            'BIM', 'BCL2L11', 'FOXO3',
        },
        confidence=0.92,
        source='Sergina et al. 2007 (HER3), Nahta et al. 2005 (IGF1R)',
        pmid='17206155',
    ),

    # FLT3 inhibition (midostaurin, gilteritinib, quizartinib) — AML
    # Sources: Park et al. 2019 (AXL), Smith et al. 2012 (FLT3-ITD resistance)
    'FLT3': PerturbationSignature(
        target='FLT3',
        perturbation_type='inhibitor',
        phospho_decreased={
            'FLT3',
            'STAT5A', 'STAT5B',
            'AKT1',
            'ERK1', 'ERK2', 'MAPK1', 'MAPK3',
            'SRC',
        },
        phospho_increased={
            'AXL',   # AXL upregulation — primary resistance RTK (Park 2019)
            'SRC',   # SRC-family kinase activation
            'EGFR',  # EGFR as escape RTK in some AML contexts
        },
        expression_decreased={
            'MYC', 'CCND1', 'MCL1',
        },
        expression_increased={
            'BIM', 'BCL2L11', 'CDKN1A',
        },
        confidence=0.85,
        source='Park et al. 2019 (AXL feedback), Smith et al. 2012',
        pmid='31558488',
    ),

    # RET inhibition (selpercatinib, pralsetinib, vandetanib) — thyroid, NSCLC
    # Sources: Yoh et al. 2020 (resistance mechanisms), Mulligan 2014 (RET biology)
    'RET': PerturbationSignature(
        target='RET',
        perturbation_type='inhibitor',
        phospho_decreased={
            'RET',
            'ERK1', 'ERK2', 'MAPK1', 'MAPK3',
            'AKT1', 'STAT3',
        },
        phospho_increased={
            'EGFR',   # RTK feedback
            'MET',    # MET upregulation (resistance)
            'AXL',    # AXL-driven resistance
            'FGFR1',  # FGFR autocrine upregulation in thyroid
        },
        expression_decreased={
            'MYC', 'CCND1',
        },
        expression_increased={
            'CDKN1A', 'BIM', 'BCL2L11',
        },
        confidence=0.83,
        source='Yoh et al. 2020; Mulligan et al. 2014 (Nat Rev Cancer)',
    ),

    # FGFR1 inhibition (erdafitinib, pemigatinib, infigratinib)
    # Sources: Formisano et al. 2019 (AXL/EGFR feedback), Harbinski et al. 2012
    'FGFR1': PerturbationSignature(
        target='FGFR1',
        perturbation_type='inhibitor',
        phospho_decreased={
            'FGFR1',
            'ERK1', 'ERK2', 'MAPK1', 'MAPK3',
            'AKT1', 'STAT3', 'STAT1',
            'PLCg1', 'PLCG1',
        },
        phospho_increased={
            'EGFR',   # EGFR bypass (Formisano 2019)
            'MET',    # MET upregulation
            'AXL',    # AXL — primary resistance for FGFR inhibitors
            'IGF1R',  # IGF1R bypass signaling
        },
        expression_decreased={
            'MYC', 'CCND1',
        },
        expression_increased={
            'CDKN1A',
        },
        confidence=0.85,
        source='Formisano et al. 2019 (FGFR-i resistance)',
        pmid='31160358',
    ),

    # FGFR2 inhibition — gastric, endometrial, cholangiocarcinoma
    'FGFR2': PerturbationSignature(
        target='FGFR2',
        perturbation_type='inhibitor',
        phospho_decreased={
            'FGFR2',
            'ERK1', 'ERK2', 'MAPK1', 'MAPK3',
            'AKT1', 'STAT3',
        },
        phospho_increased={
            'EGFR', 'MET', 'AXL', 'IGF1R',
        },
        expression_decreased={
            'MYC', 'CCND1',
        },
        expression_increased={
            'CDKN1A',
        },
        confidence=0.83,
        source='Inferred from FGFR1 analogy; FGFR-i resistance literature',
    ),

    # FGFR3 inhibition — bladder, myeloma, cervical
    'FGFR3': PerturbationSignature(
        target='FGFR3',
        perturbation_type='inhibitor',
        phospho_decreased={
            'FGFR3',
            'ERK1', 'ERK2', 'MAPK1', 'MAPK3',
            'AKT1',
        },
        phospho_increased={
            'EGFR', 'MET', 'AXL', 'IGF1R',
        },
        expression_decreased={
            'MYC', 'CCND1',
        },
        expression_increased={
            'CDKN1A',
        },
        confidence=0.82,
        source='Inferred from FGFR1 analogy; erdafitinib (bladder) resistance data',
    ),

    # AKT1 inhibition (ipatasertib, capivasertib) — pan-cancer PI3K pathway
    # Sources: Chandarlapaty et al. 2011 (RTK upregulation upon AKT-i)
    'AKT1': PerturbationSignature(
        target='AKT1',
        perturbation_type='inhibitor',
        phospho_decreased={
            'AKT1', 'AKT2', 'AKT3',
            'MTOR', 'RPTOR',
            'S6K1', 'RPS6KB1',
            'S6', 'RPS6',
            'FOXO1', 'FOXO3',
            'GSK3A', 'GSK3B',
        },
        phospho_increased={
            'EGFR',   # RTK rebound (Chandarlapaty 2011)
            'ERBB3',  # HER3 — primary resistance node for AKT-i
            'IGF1R',  # IGF1R feedback activation
            'ERK1', 'ERK2',  # RAS/MAPK reactivation via released feedback
        },
        expression_decreased={
            'MYC', 'CCND1',
        },
        expression_increased={
            'FOXO1', 'FOXO3',
            'CDKN1B', 'BIM', 'BCL2L11',
        },
        confidence=0.88,
        source='Chandarlapaty et al. 2011 (PNAS): RTK upregulation upon AKT-i',
        pmid='21427227',
    ),

    # KIT inhibition (imatinib, sunitinib) — GIST, AML (CBF), melanoma
    # Sources: Mahadevan et al. 2007, Ramos et al. 2015 (RTK crosstalk)
    'KIT': PerturbationSignature(
        target='KIT',
        perturbation_type='inhibitor',
        phospho_decreased={
            'KIT',
            'AKT1', 'AKT2',
            'ERK1', 'ERK2', 'MAPK1', 'MAPK3',
            'STAT3', 'STAT5A',
            'SRC',
        },
        phospho_increased={
            'EGFR',   # Compensatory RTK activation
            'MET',    # MET upregulation
            'AXL',    # AXL-driven resistance
            'FGFR1',  # FGF/FGFR autocrine upregulation
        },
        expression_decreased={
            'MYC', 'CCND1', 'BCL2',
        },
        expression_increased={
            'BIM', 'BCL2L11', 'CDKN1A',
        },
        confidence=0.83,
        source='Mahadevan et al. 2007; GIST imatinib-resistance mechanisms',
    ),

    # PDGFRA inhibition (imatinib, avapritinib) — GIST, glioma
    'PDGFRA': PerturbationSignature(
        target='PDGFRA',
        perturbation_type='inhibitor',
        phospho_decreased={
            'PDGFRA',
            'AKT1',
            'ERK1', 'ERK2', 'MAPK1', 'MAPK3',
            'STAT3', 'SRC',
        },
        phospho_increased={
            'EGFR',   # Compensatory RTK
            'FGFR1',  # FGF/FGFR autocrine feedback
            'MET',
        },
        expression_decreased={
            'MYC', 'CCND1',
        },
        expression_increased={
            'CDKN1A', 'BIM',
        },
        confidence=0.80,
        source='PDGFR inhibitor resistance analogy; imatinib resistance literature',
    ),

    # CCND1 cycling — palbociclib/ribociclib target CDK4/CCND1 complex;
    # CCND1 KO or CDK4/6-i causes CDK2/CCNE compensatory escape.
    # Sources: O'Leary et al. 2018 (CCND1 in breast Ca), Finn et al. 2016.
    'CCND1': PerturbationSignature(
        target='CCND1',
        perturbation_type='inhibitor',
        phospho_decreased={
            'RB1',
            'E2F1', 'E2F2', 'E2F3',
        },
        phospho_increased={
            'CDK2',   # CDK2/CCNE compensatory (major resistance)
            'CCNE1',  # Cyclin E upregulation
            'EGFR',   # EGFR-mediated G1 bypass
            'ERBB2',  # HER2-mediated bypass
        },
        expression_decreased={
            'CCND1', 'MYC', 'CCNE1', 'CDC6', 'MCM2',
        },
        expression_increased={
            'CDKN1A', 'CDKN1B', 'CCNE1',
        },
        confidence=0.88,
        source="O'Leary et al. 2018; Finn et al. 2016; CDK4/6-i resistance via CCNE/CDK2",
        pmid='29420467',
    ),

    # ABL1 inhibition (imatinib, dasatinib, nilotinib) — CML, Ph+ ALL, sarcomas
    # Sources: Donato et al. 2003 (SFK bypass), Gorre et al. 2001 (BCR-ABL resistance)
    'ABL1': PerturbationSignature(
        target='ABL1',
        perturbation_type='inhibitor',
        phospho_decreased={
            'ABL1', 'BCR',
            'CRKL', 'CRK',
            'SHC1', 'GAB2',
            'AKT1',
            'ERK1', 'ERK2', 'MAPK1', 'MAPK3',
            'STAT5A', 'STAT5B',
        },
        phospho_increased={
            'SRC',    # SRC-family kinase activation — primary resistance (Donato 2003)
            'LYN',    # LYN (SFK member, especially in CML)
            'EGFR',   # EGFR-mediated bypass signaling
            'FGFR1',  # FGF/FGFR autocrine upregulation
        },
        expression_decreased={
            'MYC', 'CCND1', 'BCL2',
        },
        expression_increased={
            'BIM', 'BCL2L11', 'CDKN1A',
        },
        confidence=0.90,
        source='Donato et al. 2003 (SFK bypass); Gorre et al. 2001',
        pmid='14559813',
    ),
}

# --------------------------------------------------------------------------
# Aliases: kinases that share the copied perturbation profile of a related
# kinase.  This is a known simplification — these kinases have distinct
# substrate specificities (e.g., FYN and SRC phosphorylate overlapping but
# non-identical substrates).  Where target-specific perturbation data becomes
# available, these aliases should be replaced with independent signatures.
# --------------------------------------------------------------------------
_ALIAS_MAP: Dict[str, str] = {
    'MAP2K1': 'MEK1',
    'MAP2K2': 'MEK1',
    'CDK6':   'CDK4',
    'JAK2':   'JAK1',
    'FYN':    'SRC',
    'AKT2':   'AKT1',
    'AKT3':   'AKT1',
    'PDGFRB': 'PDGFRA',
    'KRAS2':  'KRAS',
}

for _alias, _source in _ALIAS_MAP.items():
    _sig = copy.deepcopy(PERTURBATION_SIGNATURES[_source])
    _sig.target = _alias  # update target name
    _sig.source = f'alias of {_source} (shared profile — distinct substrate specificity not modeled)'
    PERTURBATION_SIGNATURES[_alias] = _sig
del _alias, _source, _sig  # clean up module namespace

# --------------------------------------------------------------------------
# Formal alias limitation registry (Supplementary Table S-ALIAS)
# --------------------------------------------------------------------------
# Each entry documents the biological approximation made when an alias
# gene inherits the perturbation profile of its source gene.  This table
# is exported by generate_alias_supplementary_table() for inclusion in
# the paper's supplementary materials.
KNOWN_ALIAS_LIMITATIONS: Dict[str, Dict[str, str]] = {
    'MAP2K1': {
        'source': 'MEK1',
        'shared_family': 'MAP2K (dual-specificity MAPKK)',
        'limitation': 'MAP2K1 and MEK1 are synonymous (HUGO alias); no approximation.',
    },
    'MAP2K2': {
        'source': 'MEK1',
        'shared_family': 'MAP2K (dual-specificity MAPKK)',
        'limitation': 'MAP2K2 (MEK2) shares ~80 % kinase-domain identity with MEK1 '
                      'but has distinct scaffolding interactions (KSR1 vs KSR2) and '
                      'divergent feedback sensitivity to ERK-mediated phosphorylation.',
    },
    'CDK6': {
        'source': 'CDK4',
        'shared_family': 'Cyclin D-dependent kinase',
        'limitation': 'CDK6 has non-catalytic transcription-factor roles (STAT3, AP-1) '
                      'not shared by CDK4; palbociclib/ribociclib IC50 ratios differ ~3×.',
    },
    'JAK2': {
        'source': 'JAK1',
        'shared_family': 'Janus kinase',
        'limitation': 'JAK2 uniquely phosphorylates STAT5 via the EPO/TPO receptor; '
                      'JAK1 signals predominantly through STAT1/STAT3 in Type-I/II '
                      'interferon pathways.  Shared profile underestimates haematological '
                      'specificity of JAK2.',
    },
    'FYN': {
        'source': 'SRC',
        'shared_family': 'SRC-family kinase (SFK)',
        'limitation': 'FYN is enriched in T-cell receptor signalling and myelin '
                      'maintenance (CNS); SRC is enriched in integrin/FAK signalling.  '
                      'Shared profile masks tissue-specific phosphoproteomics.',
    },
}


def generate_alias_supplementary_table() -> 'pd.DataFrame':
    """Generate Supplementary Table documenting alias approximations.

    Returns a DataFrame suitable for inclusion in supplementary materials.
    Columns: alias, source, shared_family, limitation, profile_fields_copied.
    """
    import pandas as pd

    rows = []
    for alias, info in KNOWN_ALIAS_LIMITATIONS.items():
        source = info['source']
        sig = PERTURBATION_SIGNATURES.get(alias)
        fields_copied = []
        if sig:
            for attr in ('phospho_increased', 'phospho_decreased',
                         'expression_increased', 'expression_decreased'):
                val = getattr(sig, attr, None)
                if val:
                    fields_copied.append(f'{attr}({len(val)})')
        rows.append({
            'alias': alias,
            'source': source,
            'shared_family': info['shared_family'],
            'limitation': info['limitation'],
            'profile_fields_copied': ', '.join(fields_copied) if fields_copied else 'none',
        })
    return pd.DataFrame(rows)


# =============================================================================
# LINCS L1000 integration (optional, data-driven)
# =============================================================================
# When LINCS data is present in lincs_data/, the module-level functions below
# will prefer LINCS-derived consensus signatures over the curated dictionary.
# This gives genome-scale coverage (~thousands of targets) with measured
# transcriptomic responses.  The curated dictionary is kept as a fallback
# for targets not covered by LINCS or when LINCS data is unavailable.
#
# To enable: download LINCS data, then either:
#   (a) run LINCSSignatureDB("lincs_data").build_index()  once, or
#   (b) simply call any perturbation function — it auto-initialises.
# =============================================================================

_LINCS_DB = None  # lazy singleton
_LINCS_INIT_ATTEMPTED = False
_LINCS_LOCK = __import__('threading').Lock()


def _get_lincs_db():
    """Lazily initialise the LINCS signature database (singleton).

    Thread-safe: uses a lock to prevent race conditions where multiple
    threads see ``_LINCS_INIT_ATTEMPTED=True`` but ``_LINCS_DB`` is
    still ``None`` because loading hasn't finished yet.
    """
    global _LINCS_DB, _LINCS_INIT_ATTEMPTED
    if _LINCS_INIT_ATTEMPTED:
        return _LINCS_DB
    with _LINCS_LOCK:
        # Double-checked locking: re-test inside the lock in case
        # another thread completed init while we were waiting.
        if _LINCS_INIT_ATTEMPTED:
            return _LINCS_DB
        try:
            from alin.lincs import get_default_db
            _LINCS_DB = get_default_db()  # returns None if dir missing
            if _LINCS_DB is not None:
                logger.info("LINCS L1000 database available — will prefer data-driven signatures")
        except Exception as exc:
            logger.debug("LINCS module not available: %s", exc)
        _LINCS_INIT_ATTEMPTED = True
    return _LINCS_DB


def get_perturbation_signature(target: str) -> Optional[PerturbationSignature]:
    """Get perturbation signature for a target gene.

    Prefers LINCS L1000 data-driven signatures when available,
    falls back to the curated dictionary otherwise.

    When both are available, **merges** them: LINCS provides genome-scale
    transcriptional coverage, while curated provides phosphoproteomics
    data that L1000 cannot measure (L1000 measures mRNA, not phospho).
    This prevents the silent loss of phospho data when LINCS replaces
    a curated entry.
    """
    curated = PERTURBATION_SIGNATURES.get(target)

    # Try LINCS (data-driven, genome-scale)
    db = _get_lincs_db()
    if db is not None:
        lincs_sig = db.get_perturbation_signature(target)
        if lincs_sig is not None:
            if curated is not None:
                # MERGE: keep LINCS transcriptional + curated phospho
                # LINCS has better expression data (genome-scale, measured);
                # curated has phosphoproteomics (LINCS can't measure phospho).
                lincs_sig.phospho_decreased = curated.phospho_decreased.copy()
                lincs_sig.phospho_increased = curated.phospho_increased.copy()
                # Union expression sets for completeness (LINCS is primary)
                lincs_sig.expression_decreased |= curated.expression_decreased
                lincs_sig.expression_increased |= curated.expression_increased
                lincs_sig.source = (
                    f"{lincs_sig.source} + curated_phospho"
                )
            return lincs_sig

    # Fallback to curated
    return curated


def get_perturbation_responders(target: str) -> Set[str]:
    """Get all genes that respond to perturbation of target."""
    sig = get_perturbation_signature(target)
    if sig is None:
        return set()
    return sig.all_responders


def get_direct_effectors(target: str) -> Set[str]:
    """Get genes that decrease when target is inhibited (likely downstream)."""
    sig = get_perturbation_signature(target)
    if sig is None:
        return set()
    return sig.direct_effectors


def get_feedback_genes(target: str) -> Set[str]:
    """Get genes that increase when target is inhibited (feedback/resistance)."""
    sig = get_perturbation_signature(target)
    if sig is None:
        return set()
    return sig.phospho_increased | sig.expression_increased


def compute_perturbation_path_enrichment(
    path_genes: Set[str],
    target: str,
) -> Dict:
    """
    Compute how much a path overlaps with perturbation response.
    
    Returns dict with:
    - overlap_score: fraction of path genes in perturbation signature
    - effector_overlap: genes in both path and direct effectors
    - feedback_overlap: genes in both path and feedback genes
    """
    sig = get_perturbation_signature(target)
    if sig is None:
        return {
            'overlap_score': 0.0,
            'effector_overlap': set(),
            'feedback_overlap': set(),
            'has_signature': False,
        }
    
    all_responders = sig.all_responders
    direct_effectors = sig.direct_effectors
    feedback_genes = sig.phospho_increased | sig.expression_increased
    
    effector_overlap = path_genes & direct_effectors
    feedback_overlap = path_genes & feedback_genes
    total_overlap = path_genes & all_responders
    
    overlap_score = len(total_overlap) / max(len(path_genes), 1)
    
    return {
        'overlap_score': round(overlap_score, 3),
        'effector_overlap': effector_overlap,
        'feedback_overlap': feedback_overlap,
        'n_effectors': len(effector_overlap),
        'n_feedback': len(feedback_overlap),
        'has_signature': True,
    }


def build_perturbation_response_paths(
    essential_genes: Set[str],
    targets: List[str] = None,
    min_overlap: int = 2,
) -> List[Tuple[str, Set[str], float]]:
    """
    Build viability paths from perturbation response signatures.
    
    For each target with a signature, find essential genes that are
    in the perturbation response. These form a 'perturbation-response path'.
    
    When LINCS data is available, this covers thousands of targets
    (data-driven).  Otherwise falls back to the 13 curated signatures.
    
    Args:
        essential_genes: Set of essential genes (from DepMap)
        targets: Specific targets to check (default: all with signatures)
        min_overlap: Minimum overlap to form a path
    
    Returns:
        List of (target, path_genes, confidence) tuples
    """
    # Try LINCS first for broader coverage
    db = _get_lincs_db()
    if db is not None:
        lincs_paths = db.build_perturbation_response_paths(
            essential_genes, targets=targets, min_overlap=min_overlap
        )
        if lincs_paths:
            # Supplement with curated paths for targets not in LINCS
            lincs_targets = {t for t, _, _ in lincs_paths}
            curated_targets = (
                list(PERTURBATION_SIGNATURES.keys()) if targets is None
                else targets
            )
            for ct in curated_targets:
                if ct in lincs_targets:
                    continue
                sig = PERTURBATION_SIGNATURES.get(ct)
                if sig is None:
                    continue
                responders = sig.all_responders
                essential_responders = essential_genes & responders
                if len(essential_responders) >= min_overlap:
                    direct = essential_responders & sig.direct_effectors
                    n_direct = len(direct)
                    confidence = sig.confidence * (
                        CONF_ADJ_BASE + CONF_ADJ_VARIABLE * n_direct / max(len(essential_responders), 1)
                    )
                    lincs_paths.append((
                        ct,
                        essential_responders | {ct},
                        round(confidence, 2),
                    ))
            return lincs_paths

    # Fallback: curated only
    if targets is None:
        targets = list(PERTURBATION_SIGNATURES.keys())
    
    paths = []
    
    for target in targets:
        sig = get_perturbation_signature(target)
        if sig is None:
            continue
        
        # Find essential genes that are in the perturbation response
        responders = sig.all_responders
        essential_responders = essential_genes & responders
        
        if len(essential_responders) >= min_overlap:
            # Weight by how many are direct effectors vs feedback
            direct = essential_responders & sig.direct_effectors
            n_direct = len(direct)
            confidence = sig.confidence * (CONF_ADJ_BASE + CONF_ADJ_VARIABLE * n_direct / max(len(essential_responders), 1))
            
            paths.append((
                target,
                essential_responders | {target},  # Include target
                round(confidence, 2),
            ))
    
    return paths


def get_resistance_mechanism_genes(targets: List[str]) -> Set[str]:
    """
    Get genes that are upregulated when targets are inhibited.
    These represent potential resistance mechanisms.
    """
    resistance_genes = set()
    
    for target in targets:
        sig = get_perturbation_signature(target)
        if sig:
            resistance_genes.update(sig.phospho_increased)
            resistance_genes.update(sig.expression_increased)
    
    return resistance_genes


def score_combination_by_perturbation(
    targets: List[str],
    essential_genes: Set[str],
) -> Dict:
    """
    Score a combination by how well it covers perturbation responses.
    
    Uses LINCS data-driven signatures when available, otherwise curated.
    
    A good combination should:
    1. Cover downstream effectors of each target
    2. Cover resistance/feedback genes that emerge when targets are inhibited
    """
    # Try LINCS database first for genome-scale scoring
    db = _get_lincs_db()
    if db is not None:
        # Check if LINCS has consensus for ANY of these targets
        has_lincs = any(db.get_consensus(t) is not None for t in targets)
        if has_lincs:
            return db.score_combination_by_perturbation(targets, essential_genes)

    # Fallback: curated signatures
    all_effectors = set()
    all_feedback = set()
    total_responders = set()
    
    for target in targets:
        sig = get_perturbation_signature(target)
        if sig:
            all_effectors.update(sig.direct_effectors)
            all_feedback.update(sig.phospho_increased | sig.expression_increased)
            total_responders.update(sig.all_responders)
    
    # Check if combination targets feedback genes (resistance prevention)
    feedback_targeted = set(targets) & all_feedback
    
    # Essential genes covered by this combination's perturbation network
    essential_covered = essential_genes & total_responders
    
    # Score: higher if targets hit feedback loops
    feedback_coverage = len(feedback_targeted) / max(len(all_feedback), 1)
    effector_coverage = len(essential_covered) / max(len(essential_genes), 1)
    
    return {
        'feedback_coverage': round(feedback_coverage, 3),
        'effector_coverage': round(effector_coverage, 3),
        'feedback_genes_targeted': feedback_targeted,
        'essential_effectors': essential_covered,
        'resistance_genes_untargeted': all_feedback - set(targets),
        'perturbation_score': round(FALLBACK_W_EFFECTOR * effector_coverage + FALLBACK_W_FEEDBACK * feedback_coverage, 3),
    }
