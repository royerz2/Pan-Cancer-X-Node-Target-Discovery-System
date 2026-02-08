#!/usr/bin/env python3
"""
Patient Stratification Module for ALIN Framework (Adaptive Lethal Intersection Network)
==========================================================
Identifies patient subgroups most likely to benefit from predicted combinations.

Features:
- Mutation-based stratification (KRAS G12C, BRAF V600E, etc.)
- Expression-based biomarkers
- Multi-gene signature scoring
- Clinical outcome correlation
- Companion diagnostic prediction

Data Sources:
- TCGA via cBioPortal API
- DepMap cell line annotations
- ClinVar/COSMIC mutation data
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass, field
from pathlib import Path
from scipy import stats
from collections import defaultdict
import logging
import requests
import json
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
logger = logging.getLogger(__name__)

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class PatientSubgroup:
    """Defined patient subgroup for stratification"""
    name: str
    description: str
    biomarkers: Dict[str, Any]  # Gene -> criterion (mutation, expression level, etc.)
    estimated_prevalence: float  # % of cancer type
    evidence_level: str  # 'FDA-approved', 'NCCN', 'Clinical trial', 'Preclinical'
    rationale: str
    
@dataclass
class MutationProfile:
    """Mutation profile for a gene"""
    gene: str
    mutation_frequency: float  # % of patients with mutation
    common_mutations: List[Tuple[str, float]]  # (mutation, frequency)
    is_oncogene: bool
    is_druggable: bool
    actionable_mutations: List[str]
    
@dataclass
class ExpressionBiomarker:
    """Expression-based biomarker"""
    gene: str
    high_expression_threshold: float  # Z-score or percentile
    low_expression_threshold: float
    association: str  # 'sensitivity', 'resistance', 'prognosis'
    evidence: str
    
@dataclass
class StratificationResult:
    """Complete patient stratification result"""
    targets: List[str]
    drugs: List[str]
    cancer_type: str
    
    # Patient subgroups
    recommended_subgroups: List[PatientSubgroup] = field(default_factory=list)
    
    # Mutation profiles
    mutation_profiles: List[MutationProfile] = field(default_factory=list)
    
    # Expression biomarkers
    expression_biomarkers: List[ExpressionBiomarker] = field(default_factory=list)
    
    # Estimated benefit
    total_addressable_patients: float = 0.0  # % of cancer type
    high_response_subgroup: float = 0.0  # % expected to respond well
    
    # Companion diagnostic
    companion_diagnostic: Optional[str] = None
    cdt_genes: List[str] = field(default_factory=list)
    
    # Exclusion criteria
    exclusion_criteria: List[str] = field(default_factory=list)
    
    # Scores
    stratification_score: float = 0.0
    confidence: str = "unknown"

# ============================================================================
# MUTATION DATA
# ============================================================================

class MutationDatabase:
    """Database of cancer mutations and their frequencies"""
    
    # Mutation frequencies by cancer type (approximate, from TCGA/COSMIC)
    MUTATION_FREQUENCIES = {
        'Pancreatic Adenocarcinoma': {
            'KRAS': {'freq': 0.90, 'common': [('G12D', 0.40), ('G12V', 0.30), ('G12R', 0.15), ('G12C', 0.02)]},
            'TP53': {'freq': 0.70, 'common': [('R175H', 0.10), ('R248Q', 0.08)]},
            'CDKN2A': {'freq': 0.60, 'common': []},
            'SMAD4': {'freq': 0.50, 'common': []},
        },
        'Non-Small Cell Lung Cancer': {
            'KRAS': {'freq': 0.30, 'common': [('G12C', 0.13), ('G12V', 0.06), ('G12D', 0.04)]},
            'EGFR': {'freq': 0.15, 'common': [('L858R', 0.06), ('Exon19del', 0.05), ('T790M', 0.03)]},
            'ALK': {'freq': 0.05, 'common': [('EML4-ALK', 0.05)]},
            'TP53': {'freq': 0.50, 'common': []},
            'STK11': {'freq': 0.20, 'common': []},
            'MET': {'freq': 0.05, 'common': [('MET amplification', 0.03), ('MET exon14 skip', 0.02)]},
        },
        'Melanoma': {
            'BRAF': {'freq': 0.50, 'common': [('V600E', 0.40), ('V600K', 0.06)]},
            'NRAS': {'freq': 0.20, 'common': [('Q61R', 0.10), ('Q61K', 0.05)]},
            'NF1': {'freq': 0.15, 'common': []},
        },
        'Colorectal Adenocarcinoma': {
            'KRAS': {'freq': 0.45, 'common': [('G12D', 0.15), ('G12V', 0.10), ('G13D', 0.08)]},
            'APC': {'freq': 0.80, 'common': []},
            'TP53': {'freq': 0.55, 'common': []},
            'PIK3CA': {'freq': 0.20, 'common': [('E545K', 0.06), ('H1047R', 0.05)]},
            'BRAF': {'freq': 0.10, 'common': [('V600E', 0.08)]},
        },
        'Breast Invasive Carcinoma': {
            'PIK3CA': {'freq': 0.35, 'common': [('H1047R', 0.15), ('E545K', 0.10)]},
            'TP53': {'freq': 0.35, 'common': []},
            'ERBB2': {'freq': 0.15, 'common': [('ERBB2 amplification', 0.15)]},
            'ESR1': {'freq': 0.70, 'common': []},  # Expression, not mutation
            'CDH1': {'freq': 0.15, 'common': []},
        },
        'Acute Myeloid Leukemia': {
            'FLT3': {'freq': 0.30, 'common': [('FLT3-ITD', 0.25), ('D835Y', 0.05)]},
            'NPM1': {'freq': 0.30, 'common': []},
            'DNMT3A': {'freq': 0.25, 'common': []},
            'IDH1': {'freq': 0.10, 'common': [('R132H', 0.06)]},
            'IDH2': {'freq': 0.15, 'common': [('R140Q', 0.08)]},
        },
    }
    
    # Actionable mutations (FDA-approved drugs available)
    ACTIONABLE_MUTATIONS = {
        'KRAS': {'G12C': ['sotorasib', 'adagrasib']},
        'BRAF': {'V600E': ['vemurafenib', 'dabrafenib', 'encorafenib'], 'V600K': ['dabrafenib']},
        'EGFR': {'L858R': ['erlotinib', 'gefitinib', 'osimertinib'], 
                 'Exon19del': ['erlotinib', 'gefitinib', 'osimertinib'],
                 'T790M': ['osimertinib']},
        'ALK': {'EML4-ALK': ['crizotinib', 'alectinib', 'lorlatinib']},
        'MET': {'MET exon14 skip': ['capmatinib', 'tepotinib']},
        'PIK3CA': {'H1047R': ['alpelisib'], 'E545K': ['alpelisib']},
        'FLT3': {'FLT3-ITD': ['midostaurin', 'gilteritinib']},
        'IDH1': {'R132H': ['ivosidenib']},
        'IDH2': {'R140Q': ['enasidenib']},
    }
    
    def get_mutation_profile(self, gene: str, cancer_type: str) -> Optional[MutationProfile]:
        """Get mutation profile for a gene in a cancer type"""
        cancer_data = self.MUTATION_FREQUENCIES.get(cancer_type, {})
        gene_data = cancer_data.get(gene)
        
        if not gene_data:
            return None
        
        # Check if gene has actionable mutations
        actionable = list(self.ACTIONABLE_MUTATIONS.get(gene, {}).keys())
        is_druggable = len(actionable) > 0 or gene in [
            'CDK4', 'CDK6', 'MTOR', 'BCL2', 'JAK1', 'JAK2', 'SRC', 'STAT3'
        ]
        
        # Oncogene vs tumor suppressor
        oncogenes = {'KRAS', 'BRAF', 'EGFR', 'ERBB2', 'MET', 'ALK', 'PIK3CA', 'MYC', 'CCND1'}
        is_oncogene = gene in oncogenes
        
        return MutationProfile(
            gene=gene,
            mutation_frequency=gene_data['freq'],
            common_mutations=gene_data['common'],
            is_oncogene=is_oncogene,
            is_druggable=is_druggable,
            actionable_mutations=actionable
        )


# ============================================================================
# cBIOPORTAL API
# ============================================================================

class cBioPortalClient:
    """Client for cBioPortal API to get TCGA patient data"""
    
    # Cancer type to TCGA study mapping
    STUDY_MAPPING = {
        'Pancreatic Adenocarcinoma': 'paad_tcga',
        'Non-Small Cell Lung Cancer': 'luad_tcga',
        'Lung Adenocarcinoma': 'luad_tcga',
        'Melanoma': 'skcm_tcga',
        'Colorectal Adenocarcinoma': 'coadread_tcga',
        'Breast Invasive Carcinoma': 'brca_tcga',
        'Acute Myeloid Leukemia': 'laml_tcga',
        'Ovarian Epithelial Tumor': 'ov_tcga',
        'Hepatocellular Carcinoma': 'lihc_tcga',
        'Renal Cell Carcinoma': 'kirc_tcga',
    }
    
    def __init__(self, cache_dir: str = "./cbioportal_cache"):
        self.base_url = "https://www.cbioportal.org/api"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
    def get_study_id(self, cancer_type: str) -> Optional[str]:
        """Map cancer type to TCGA study"""
        return self.STUDY_MAPPING.get(cancer_type)
    
    def get_mutation_data(self, genes: List[str], cancer_type: str) -> Optional[pd.DataFrame]:
        """
        Get mutation data for genes in a cancer type
        
        Returns DataFrame with columns: [gene, mutation, sample_count, frequency]
        """
        study_id = self.get_study_id(cancer_type)
        if not study_id:
            logger.warning(f"No TCGA study mapping for {cancer_type}")
            return None
        
        try:
            # Get molecular profile ID
            url = f"{self.base_url}/studies/{study_id}/molecular-profiles"
            response = requests.get(url, timeout=10)
            
            if response.status_code != 200:
                return None
            
            profiles = response.json()
            mutation_profile = None
            for p in profiles:
                if 'mutation' in p.get('molecularAlterationType', '').lower():
                    mutation_profile = p.get('molecularProfileId')
                    break
            
            if not mutation_profile:
                return None
            
            # This is a simplified example - full implementation would query mutations
            # For now, return data from our built-in database
            logger.info(f"Using built-in mutation data for {cancer_type}")
            return None
            
        except Exception as e:
            logger.warning(f"cBioPortal query failed: {e}")
            return None
    
    def get_survival_data(self, cancer_type: str) -> Optional[pd.DataFrame]:
        """Get survival data from TCGA"""
        study_id = self.get_study_id(cancer_type)
        if not study_id:
            return None
        
        try:
            url = f"{self.base_url}/studies/{study_id}/clinical-data"
            params = {
                'clinicalDataType': 'PATIENT',
                'projection': 'DETAILED'
            }
            
            response = requests.get(url, params=params, timeout=15)
            
            if response.status_code != 200:
                return None
            
            # Parse clinical data
            data = response.json()
            # This would return survival information
            return None
            
        except Exception as e:
            logger.warning(f"Survival data query failed: {e}")
            return None


# ============================================================================
# PATIENT STRATIFIER
# ============================================================================

class PatientStratifier:
    """
    Stratifies patients based on molecular profiles for targeted therapy
    
    Approaches:
    1. Mutation-based selection (actionable mutations)
    2. Expression-based biomarkers
    3. Multi-gene signature scoring
    4. Synthetic lethality-based selection
    """
    
    # Known synthetic lethality relationships
    SYNTHETIC_LETHALITY = {
        'BRCA1': ['PARP1', 'PARP2'],  # BRCA-PARP SL
        'BRCA2': ['PARP1', 'PARP2'],
        'ATM': ['PARP1'],
        'ARID1A': ['EZH2'],
        'KRAS': ['STK33', 'PLK1'],  # Reported KRAS SL partners
        'MYC': ['AURKA', 'AURKB'],
        'TP53': ['ATR', 'CHK1'],
    }
    
    # Resistance markers
    RESISTANCE_MARKERS = {
        'EGFR_inhibitors': ['T790M', 'C797S', 'MET_amp'],
        'BRAF_inhibitors': ['NRAS_mut', 'MAP2K1_mut', 'BRAF_amp'],
        'CDK4_inhibitors': ['RB1_loss', 'CDK6_amp'],
        'PI3K_inhibitors': ['PTEN_loss'],
    }
    
    def __init__(self, cache_dir: str = "./stratification_cache"):
        self.mutation_db = MutationDatabase()
        self.cbioportal = cBioPortalClient(cache_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
    def define_patient_subgroups(self, 
                                 targets: List[str], 
                                 drugs: List[str],
                                 cancer_type: str) -> List[PatientSubgroup]:
        """
        Define patient subgroups most likely to benefit from the combination
        """
        subgroups = []
        
        # 1. Mutation-based subgroups (primary)
        for target in targets:
            mutation_profile = self.mutation_db.get_mutation_profile(target, cancer_type)
            
            if mutation_profile and mutation_profile.actionable_mutations:
                for mutation in mutation_profile.actionable_mutations[:2]:  # Top 2
                    # Find mutation frequency
                    mut_freq = next(
                        (f for m, f in mutation_profile.common_mutations if m == mutation),
                        mutation_profile.mutation_frequency * 0.3
                    )
                    
                    subgroups.append(PatientSubgroup(
                        name=f"{target} {mutation}",
                        description=f"Patients with {target} {mutation} mutation",
                        biomarkers={target: {'mutation': mutation}},
                        estimated_prevalence=mut_freq,
                        evidence_level='FDA-approved' if mutation in ['V600E', 'G12C', 'L858R'] else 'Clinical trial',
                        rationale=f"Direct target mutation predicts response to {target} inhibitors"
                    ))
        
        # 2. Pathway-based subgroups
        pathway_targets = self._get_pathway_groups(targets)
        for pathway, genes in pathway_targets.items():
            if len(genes) >= 2:
                subgroups.append(PatientSubgroup(
                    name=f"{pathway} pathway active",
                    description=f"Patients with {pathway} pathway activation (mutations in {', '.join(genes)})",
                    biomarkers={g: {'pathway': pathway} for g in genes},
                    estimated_prevalence=min(0.5, sum(
                        self.mutation_db.get_mutation_profile(g, cancer_type).mutation_frequency 
                        if self.mutation_db.get_mutation_profile(g, cancer_type) else 0.1
                        for g in genes
                    )),
                    evidence_level='Preclinical',
                    rationale=f"Combined {pathway} pathway inhibition may overcome single-agent resistance"
                ))
        
        # 3. Expression-based subgroups
        for target in targets:
            # High expression might indicate dependency
            subgroups.append(PatientSubgroup(
                name=f"{target} high expression",
                description=f"Patients with high {target} expression (top quartile)",
                biomarkers={target: {'expression': 'high', 'threshold': 75}},
                estimated_prevalence=0.25,
                evidence_level='Preclinical',
                rationale=f"High {target} expression may indicate dependency on {target} signaling"
            ))
        
        # 4. Synthetic lethality-based subgroups
        for target in targets:
            sl_partners = self.SYNTHETIC_LETHALITY.get(target, [])
            for partner in sl_partners:
                partner_profile = self.mutation_db.get_mutation_profile(partner, cancer_type)
                if partner_profile and partner_profile.mutation_frequency > 0.05:
                    subgroups.append(PatientSubgroup(
                        name=f"{partner} deficient",
                        description=f"Patients with {partner} loss/mutation (synthetic lethal with {target})",
                        biomarkers={partner: {'status': 'loss'}},
                        estimated_prevalence=partner_profile.mutation_frequency,
                        evidence_level='Clinical trial' if partner in ['BRCA1', 'BRCA2'] else 'Preclinical',
                        rationale=f"{partner} deficiency creates synthetic lethality with {target} inhibition"
                    ))
        
        return subgroups
    
    def _get_pathway_groups(self, targets: List[str]) -> Dict[str, List[str]]:
        """Group targets by pathway (using canonical PATHWAYS from alin.constants)"""
        from alin.constants import PATHWAYS
        
        result = defaultdict(list)
        for target in targets:
            for pathway, genes in PATHWAYS.items():
                if target in genes:
                    result[pathway].append(target)
        
        return dict(result)
    
    def identify_expression_biomarkers(self, 
                                       targets: List[str],
                                       cancer_type: str) -> List[ExpressionBiomarker]:
        """Identify expression-based biomarkers for the combination"""
        biomarkers = []
        
        for target in targets:
            # High expression as sensitivity marker
            biomarkers.append(ExpressionBiomarker(
                gene=target,
                high_expression_threshold=1.5,  # Z-score
                low_expression_threshold=-0.5,
                association='sensitivity',
                evidence=f"High {target} expression may indicate dependency"
            ))
        
        # Add known resistance markers
        resistance_genes = ['MYC', 'BCL2L1', 'ABCB1']  # Common resistance genes
        for gene in resistance_genes:
            biomarkers.append(ExpressionBiomarker(
                gene=gene,
                high_expression_threshold=2.0,
                low_expression_threshold=0,
                association='resistance',
                evidence=f"High {gene} associated with drug resistance"
            ))
        
        return biomarkers
    
    def define_exclusion_criteria(self, 
                                  targets: List[str],
                                  drugs: List[str]) -> List[str]:
        """Define patient exclusion criteria based on known resistance mechanisms"""
        exclusions = []
        
        for drug in drugs:
            # Check for known resistance markers
            drug_class = self._get_drug_class(drug)
            if drug_class in self.RESISTANCE_MARKERS:
                for marker in self.RESISTANCE_MARKERS[drug_class]:
                    exclusions.append(f"Patients with {marker} (resistance to {drug})")
        
        # General exclusions
        exclusions.extend([
            "Prior treatment with same drug class (potential cross-resistance)",
            "RB1 loss (for CDK4/6 inhibitor combinations)",
        ])
        
        return exclusions
    
    def _get_drug_class(self, drug: str) -> str:
        """Map drug to class for resistance lookup"""
        drug_classes = {
            'erlotinib': 'EGFR_inhibitors', 'gefitinib': 'EGFR_inhibitors',
            'osimertinib': 'EGFR_inhibitors',
            'vemurafenib': 'BRAF_inhibitors', 'dabrafenib': 'BRAF_inhibitors',
            'palbociclib': 'CDK4_inhibitors', 'ribociclib': 'CDK4_inhibitors',
            'alpelisib': 'PI3K_inhibitors',
        }
        return drug_classes.get(drug.lower(), 'other')
    
    def generate_companion_diagnostic(self, 
                                      targets: List[str],
                                      subgroups: List[PatientSubgroup]) -> Tuple[str, List[str]]:
        """
        Generate companion diagnostic recommendation
        
        Returns:
            (CDx description, list of genes to test)
        """
        genes_to_test = set()
        
        # Add all target genes
        genes_to_test.update(targets)
        
        # Add genes from subgroup biomarkers
        for subgroup in subgroups:
            genes_to_test.update(subgroup.biomarkers.keys())
        
        # Add common resistance markers
        genes_to_test.update(['TP53', 'RB1', 'MYC'])
        
        genes_list = sorted(genes_to_test)
        
        # Generate CDx description
        if len(genes_list) <= 5:
            cdx = f"NGS panel testing for {', '.join(genes_list)}"
        else:
            cdx = f"Comprehensive genomic profiling panel ({len(genes_list)} genes)"
        
        return cdx, genes_list
    
    def stratify(self, 
                 targets: List[str],
                 drugs: List[str],
                 cancer_type: str) -> StratificationResult:
        """
        Complete patient stratification analysis
        """
        result = StratificationResult(
            targets=targets,
            drugs=drugs,
            cancer_type=cancer_type
        )
        
        # Get mutation profiles
        for target in targets:
            profile = self.mutation_db.get_mutation_profile(target, cancer_type)
            if profile:
                result.mutation_profiles.append(profile)
        
        # Define patient subgroups
        result.recommended_subgroups = self.define_patient_subgroups(targets, drugs, cancer_type)
        
        # Get expression biomarkers
        result.expression_biomarkers = self.identify_expression_biomarkers(targets, cancer_type)
        
        # Define exclusion criteria
        result.exclusion_criteria = self.define_exclusion_criteria(targets, drugs)
        
        # Generate companion diagnostic
        result.companion_diagnostic, result.cdt_genes = self.generate_companion_diagnostic(
            targets, result.recommended_subgroups
        )
        
        # Calculate addressable patient population
        if result.recommended_subgroups:
            # Use highest prevalence subgroup as primary
            result.high_response_subgroup = max(
                s.estimated_prevalence for s in result.recommended_subgroups
            )
            # Total addressable includes overlap
            result.total_addressable_patients = min(1.0, sum(
                s.estimated_prevalence for s in result.recommended_subgroups[:3]
            ))
        
        # Calculate stratification score
        scores = []
        
        # Score from mutation profiles
        if result.mutation_profiles:
            actionable_count = sum(1 for p in result.mutation_profiles if p.actionable_mutations)
            scores.append(actionable_count / len(result.mutation_profiles))
        
        # Score from subgroups
        if result.recommended_subgroups:
            fda_approved = sum(1 for s in result.recommended_subgroups if 'FDA' in s.evidence_level)
            scores.append(min(1.0, fda_approved * 0.3 + len(result.recommended_subgroups) * 0.1))
        
        # Score from patient population
        scores.append(result.total_addressable_patients)
        
        result.stratification_score = np.mean(scores) if scores else 0.0
        
        # Confidence
        if result.stratification_score >= 0.6:
            result.confidence = "HIGH"
        elif result.stratification_score >= 0.3:
            result.confidence = "MEDIUM"
        else:
            result.confidence = "LOW"
        
        return result


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_stratification_report(result: StratificationResult) -> str:
    """Generate detailed patient stratification report"""
    
    report = f"""
{'='*80}
PATIENT STRATIFICATION REPORT
{'='*80}
Combination: {' + '.join(result.targets)}
Drugs: {' + '.join(result.drugs)}
Cancer Type: {result.cancer_type}
Stratification Score: {result.stratification_score:.3f}
Confidence: {result.confidence}
{'='*80}

PATIENT POPULATION ESTIMATES:
{'-'*80}
Total Addressable Patients: {result.total_addressable_patients*100:.1f}% of {result.cancer_type}
High-Response Subgroup: {result.high_response_subgroup*100:.1f}%

{'='*80}
RECOMMENDED PATIENT SUBGROUPS:
{'-'*80}
"""
    
    for i, subgroup in enumerate(result.recommended_subgroups[:5], 1):
        report += f"""
{i}. {subgroup.name}
   Description: {subgroup.description}
   Estimated prevalence: {subgroup.estimated_prevalence*100:.1f}%
   Evidence level: {subgroup.evidence_level}
   Rationale: {subgroup.rationale}
"""
    
    report += f"""
{'='*80}
MUTATION PROFILES:
{'-'*80}
"""
    
    for profile in result.mutation_profiles:
        report += f"""
{profile.gene}:
  Mutation frequency: {profile.mutation_frequency*100:.1f}%
  Common mutations: {', '.join([f'{m} ({f*100:.1f}%)' for m, f in profile.common_mutations[:3]])}
  Druggable: {'Yes' if profile.is_druggable else 'No'}
  Actionable mutations: {', '.join(profile.actionable_mutations) if profile.actionable_mutations else 'None'}
"""
    
    report += f"""
{'='*80}
COMPANION DIAGNOSTIC RECOMMENDATION:
{'-'*80}
{result.companion_diagnostic}

Genes to test: {', '.join(result.cdt_genes[:15])}{'...' if len(result.cdt_genes) > 15 else ''}

{'='*80}
EXCLUSION CRITERIA:
{'-'*80}
"""
    
    for i, criterion in enumerate(result.exclusion_criteria, 1):
        report += f"{i}. {criterion}\n"
    
    report += f"""
{'='*80}
EXPRESSION BIOMARKERS:
{'-'*80}
"""
    
    for biomarker in result.expression_biomarkers[:5]:
        report += f"""
{biomarker.gene}: {biomarker.association.upper()}
  High threshold: Z > {biomarker.high_expression_threshold}
  Evidence: {biomarker.evidence}
"""
    
    return report


def export_stratification_results(results: List[StratificationResult], output_path: Path):
    """Export stratification results to files"""
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Summary CSV
    rows = []
    for r in results:
        rows.append({
            'Targets': ' + '.join(r.targets),
            'Drugs': ' + '.join(r.drugs),
            'Cancer_Type': r.cancer_type,
            'Score': f"{r.stratification_score:.3f}",
            'Confidence': r.confidence,
            'Addressable_Patients': f"{r.total_addressable_patients*100:.1f}%",
            'High_Response': f"{r.high_response_subgroup*100:.1f}%",
            'N_Subgroups': len(r.recommended_subgroups),
            'Companion_Diagnostic': r.companion_diagnostic
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path / "stratification_summary.csv", index=False)
    
    # Individual reports
    for r in results:
        cancer_str = r.cancer_type.replace(' ', '_').replace('/', '_')
        targets_str = '_'.join(r.targets)
        filename = f"stratification_{cancer_str}_{targets_str}.txt"
        
        report = generate_stratification_report(r)
        with open(output_path / filename, 'w') as f:
            f.write(report)
    
    logger.info(f"Stratification results exported to {output_path}")


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Patient stratification analysis")
    parser.add_argument('--targets', nargs='+', default=['KRAS', 'CDK6', 'STAT3'])
    parser.add_argument('--drugs', nargs='+', default=['sotorasib', 'palbociclib', 'napabucasin'])
    parser.add_argument('--cancer', type=str, default='Pancreatic Adenocarcinoma')
    parser.add_argument('--output', type=str, default='./stratification_results')
    
    args = parser.parse_args()
    
    print(f"\nPatient Stratification Analysis")
    print(f"Targets: {args.targets}")
    print(f"Drugs: {args.drugs}")
    print(f"Cancer: {args.cancer}\n")
    
    stratifier = PatientStratifier()
    result = stratifier.stratify(args.targets, args.drugs, args.cancer)
    
    report = generate_stratification_report(result)
    print(report)
    
    # Export
    export_stratification_results([result], Path(args.output))
    print(f"\nResults exported to {args.output}/")
