#!/usr/bin/env python3
"""
Perturbation Response Module
============================
Enriches viability paths with perturbation-induced signaling changes.

Data sources:
1. Phosphoproteomics signatures (proteins whose phosphorylation changes after target inhibition)
2. Transcriptional response modules (genes that change expression after perturbation)
3. LINCS L1000 signatures (drug-induced gene expression changes)

This captures dynamic/functional pathway relationships that static co-essentiality
and network topology miss.
"""

import logging
import copy
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


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
}

# Set aliases (deep copy to avoid shared mutable references)
PERTURBATION_SIGNATURES['MAP2K1'] = copy.deepcopy(PERTURBATION_SIGNATURES['MEK1'])
PERTURBATION_SIGNATURES['MAP2K2'] = copy.deepcopy(PERTURBATION_SIGNATURES['MEK1'])
PERTURBATION_SIGNATURES['CDK6'] = copy.deepcopy(PERTURBATION_SIGNATURES['CDK4'])
PERTURBATION_SIGNATURES['JAK2'] = copy.deepcopy(PERTURBATION_SIGNATURES['JAK1'])
PERTURBATION_SIGNATURES['FYN'] = copy.deepcopy(PERTURBATION_SIGNATURES['SRC'])


def get_perturbation_signature(target: str) -> Optional[PerturbationSignature]:
    """Get perturbation signature for a target gene."""
    return PERTURBATION_SIGNATURES.get(target)


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
    
    Args:
        essential_genes: Set of essential genes (from DepMap)
        targets: Specific targets to check (default: all with signatures)
        min_overlap: Minimum overlap to form a path
    
    Returns:
        List of (target, path_genes, confidence) tuples
    """
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
            confidence = sig.confidence * (0.5 + 0.5 * n_direct / max(len(essential_responders), 1))
            
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
    
    A good combination should:
    1. Cover downstream effectors of each target
    2. Cover resistance/feedback genes that emerge when targets are inhibited
    """
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
        'perturbation_score': round(0.6 * effector_coverage + 0.4 * feedback_coverage, 3),
    }
