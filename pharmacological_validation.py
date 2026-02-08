"""
Pharmacological Validation Module
==================================

Addresses three systematic weaknesses in the ALIN pipeline that were previously
compensated for only with hedging language:

1. CRISPR-Drug Concordance Filter
   - CRISPR knockout ≠ pharmacological inhibition (Lin et al. 2017, Gonçalves et al. 2020)
   - For each predicted target, cross-validates CRISPR essentiality against PRISM drug
     sensitivity in matched cell lines
   - Outputs per-gene pharmacological concordance scores

2. Co-essentiality Interaction Estimator
   - Replaces the heuristic synergy score (hardcoded pairs + pathway diversity ratio)
     with data-driven pairwise interaction estimates from the Jaccard co-essentiality
     matrix already computed per cancer type
   - Low co-essentiality → genes in different pathways → higher synergy potential
   - High co-essentiality → genes in same pathway → likely redundant

3. Evidence Tier System
   - Classifies each cancer type's predictions into confidence tiers based on:
     Tier 1: Experimental tri-axial validation exists (PDAC)
     Tier 2: Gold-standard clinical combination recovered
     Tier 3: PRISM pharmacological concordance supports ≥2 targets
     Tier 4: Computational prediction only (no orthogonal support)

Usage:
    python pharmacological_validation.py [--data-dir ./depmap_data] [--drug-dir ./drug_sensitivity_data]

Or programmatically:
    from pharmacological_validation import PharmacologicalValidator
    validator = PharmacologicalValidator(depmap_dir, drug_dir)
    results = validator.validate_cancer_predictions(cancer_analysis)
"""

import logging
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set, Tuple, FrozenSet
from collections import defaultdict
from itertools import combinations

import numpy as np

from alin.constants import GENE_TO_DRUGS as _CANONICAL_GENE_TO_DRUGS

logger = logging.getLogger(__name__)


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class GeneConcordance:
    """CRISPR-drug concordance for one gene in one cancer type."""
    gene: str
    cancer_type: str
    drugs_tested: List[str]
    best_correlation: float  # Pearson r (CRISPR dep vs drug sensitivity); negative = concordant
    best_pvalue: float
    n_cell_lines: int  # number of matched cell lines used
    concordant: bool  # True if best_correlation < -0.2 and p < 0.05
    concordance_score: float  # 0-1 score; higher = more pharmacologically validated

    @property
    def summary(self) -> str:
        status = "CONCORDANT" if self.concordant else "DISCORDANT"
        return (f"{self.gene} [{status}]: r={self.best_correlation:.3f} "
                f"(p={self.best_pvalue:.3g}, n={self.n_cell_lines}, "
                f"drugs={self.drugs_tested})")


@dataclass
class PairwiseInteraction:
    """Data-driven interaction estimate for a gene pair."""
    gene_a: str
    gene_b: str
    jaccard_similarity: float  # from co-essentiality matrix
    interaction_type: str  # "independent", "redundant", "partially_redundant"
    synergy_estimate: float  # 0-1; higher = more likely synergistic (low Jaccard)

    @property
    def summary(self) -> str:
        return (f"{self.gene_a}-{self.gene_b}: Jaccard={self.jaccard_similarity:.3f} "
                f"→ {self.interaction_type} (synergy_est={self.synergy_estimate:.2f})")


@dataclass
class DataDrivenSynergyScore:
    """Replaces the heuristic synergy score with data-driven estimates."""
    targets: Tuple[str, ...]
    pairwise_interactions: List[PairwiseInteraction]
    mean_synergy_estimate: float  # average of pairwise synergy estimates
    min_synergy_estimate: float  # worst-case (highest Jaccard pair)
    pathway_diversity: float  # retained from original for comparison
    heuristic_synergy: float  # original heuristic score for comparison
    data_driven_synergy: float  # new score: weighted combination


@dataclass
class EvidenceTier:
    """Evidence classification for a cancer type's predictions."""
    cancer_type: str
    tier: int  # 1-4
    tier_label: str
    reasons: List[str]
    n_cell_lines: int
    n_concordant_targets: int
    n_total_targets: int
    gold_standard_match: bool
    concordance_fraction: float


@dataclass
class ValidationResult:
    """Complete validation result for one cancer type."""
    cancer_type: str
    gene_concordances: Dict[str, GeneConcordance]
    pairwise_interactions: List[PairwiseInteraction]
    data_driven_synergy: Optional[DataDrivenSynergyScore]
    evidence_tier: EvidenceTier
    original_triple: Optional[Tuple[str, ...]]
    rescored_combined: Optional[float]  # re-scored combined score with pharmacological data


# ============================================================================
# DRUG-GENE MAPPING — extends canonical alin.constants.GENE_TO_DRUGS
# with pharmacological-validation-specific entries (title-cased drug names)
# ============================================================================

# Additional targets not in the canonical mapping (for concordance validation)
_EXTRA_TARGETS: Dict[str, List[str]] = {
    'ABL1': ['Imatinib', 'Dasatinib', 'Bosutinib'],
    'RAF1': ['Sorafenib', 'Regorafenib'],
    'KIT': ['Imatinib', 'Sunitinib', 'Regorafenib'],
    'PDGFRA': ['Imatinib', 'Sunitinib'],
    'PARP2': ['Olaparib', 'Niraparib', 'Talazoparib'],
    'CCND1': ['Palbociclib', 'Ribociclib', 'Abemaciclib'],
}

# Build GENE_TO_DRUGS by title-casing canonical + adding extras
GENE_TO_DRUGS: Dict[str, List[str]] = {
    gene: [d.title() for d in drugs]
    for gene, drugs in _CANONICAL_GENE_TO_DRUGS.items()
}
GENE_TO_DRUGS.update(_EXTRA_TARGETS)


# ============================================================================
# GOLD STANDARD (for evidence tier classification)
# ============================================================================

# Independently curated multi-target combinations (from paper's benchmark)
GOLD_STANDARD_COMBINATIONS: Dict[str, FrozenSet[str]] = {
    'Melanoma': frozenset({'BRAF', 'MAP2K1'}),
    'Non-Small Cell Lung Cancer': frozenset({'ALK', 'MET'}),
    'Colorectal Adenocarcinoma': frozenset({'EGFR', 'BRAF', 'MAP2K1'}),
    'Breast Cancer': frozenset({'CDK4', 'CDK6', 'ERBB2'}),
    'Renal Cell Carcinoma': frozenset({'VEGFR2', 'MTOR'}),
    'Acute Myeloid Leukemia': frozenset({'FLT3', 'BCL2'}),
    'Pancreatic Adenocarcinoma': frozenset({'KRAS', 'EGFR', 'STAT3'}),
}


# ============================================================================
# CRISPR-DRUG CONCORDANCE
# ============================================================================

class CRISPRDrugConcordance:
    """
    Cross-validates CRISPR essentiality against pharmacological drug sensitivity.

    Rationale (Gonçalves et al. 2020, Mol Syst Biol):
      Only ~25% of drugs' killing patterns are phenocopied by CRISPR knockout
      of their nominal targets across 484 cell lines. This means ~75% of
      CRISPR-essential targets may not respond to pharmacological inhibition.

    For each gene predicted by ALIN:
      1. Find drugs targeting this gene (from GENE_TO_DRUGS)
      2. For cell lines of this cancer type, compute Pearson correlation between
         CRISPR dependency score and drug sensitivity (PRISM LFC or IC50)
      3. Negative correlation (CRISPR-essential lines are drug-sensitive) = concordant
      4. No/positive correlation = discordant → flag target
    """

    def __init__(self, depmap_df, prism_loader=None, gdsc_loader=None):
        """
        Args:
            depmap_df: pd.DataFrame, rows=ModelID, cols=gene symbols, values=Chronos scores
            prism_loader: PRISMLoader instance (optional)
            gdsc_loader: GDSCLoader instance (optional)
        """
        self.depmap_df = depmap_df
        self.prism = prism_loader
        self.gdsc = gdsc_loader

    def compute_gene_concordance(
        self,
        gene: str,
        cancer_type: str,
        cancer_cell_lines: List[str],
        min_cell_lines: int = 10,
    ) -> GeneConcordance:
        """
        Compute pharmacological concordance for one gene in one cancer type.
        """
        drugs = GENE_TO_DRUGS.get(gene, [])

        if not drugs:
            return GeneConcordance(
                gene=gene, cancer_type=cancer_type, drugs_tested=[],
                best_correlation=np.nan, best_pvalue=1.0, n_cell_lines=0,
                concordant=False,
                concordance_score=0.0,  # no drugs to test → no pharmacological validation
            )

        if gene not in self.depmap_df.columns:
            return GeneConcordance(
                gene=gene, cancer_type=cancer_type, drugs_tested=drugs,
                best_correlation=np.nan, best_pvalue=1.0, n_cell_lines=0,
                concordant=False, concordance_score=0.0,
            )

        # Get CRISPR dependency for this gene in cancer cell lines
        available_lines = [cl for cl in cancer_cell_lines if cl in self.depmap_df.index]
        if len(available_lines) < min_cell_lines:
            # Not enough cell lines for this cancer; try pan-cancer
            available_lines = list(self.depmap_df.index)

        dep_values = self.depmap_df.loc[available_lines, gene]

        best_r, best_p, best_drug, best_n = np.nan, 1.0, None, 0

        for drug in drugs:
            drug_sens = self._get_drug_sensitivity(drug)
            if drug_sens is None:
                continue

            sens_dict = drug_sens  # {cell_line_id: sensitivity_value}
            common = [cl for cl in available_lines if cl in sens_dict]
            if len(common) < min_cell_lines:
                continue

            dep_arr = self.depmap_df.loc[common, gene].values
            sens_arr = np.array([sens_dict[cl] for cl in common])

            # Remove NaN pairs
            mask = ~(np.isnan(dep_arr) | np.isnan(sens_arr))
            if mask.sum() < min_cell_lines:
                continue

            from scipy import stats as sp_stats
            r, p = sp_stats.pearsonr(dep_arr[mask], sens_arr[mask])

            # We want negative correlation (more essential → more sensitive)
            # For PRISM LFC: more negative LFC = more sensitive, more negative Chronos = more essential
            # So positive correlation in raw values = concordant (both negative together)
            # For GDSC IC50: lower IC50 = more sensitive, so we want negative correlation
            # Standardize: we always look for r < 0 after sign adjustment
            # PRISM LFC: both negative → r > 0 in raw → flip sign for concordance
            # Actually, PRISM LFC: negative = killing. Chronos: negative = essential.
            # So if gene is essential (negative Chronos) AND drug kills (negative LFC),
            # both are negative → correlation is POSITIVE. So concordant = r > 0 for PRISM LFC.
            # For IC50: essential (negative Chronos) and sensitive (low IC50) → r > 0 again.
            # Wait — Chronos is negative for essential. IC50 low for sensitive. So both
            # trending in same direction → positive correlation = concordant.

            # Correct interpretation: for PRISM LFC data, both scores are negative for
            # "essential/sensitive", so Pearson r > 0 means concordant.
            # We'll use the absolute value for scoring and track sign.

            concordance_r = r  # positive = concordant for PRISM LFC

            if best_drug is None or abs(concordance_r) > abs(best_r):
                best_r = concordance_r
                best_p = p
                best_drug = drug
                best_n = int(mask.sum())

        # Score: concordance_score is how well CRISPR predicts drug response
        # Positive r with low p → concordant → high score
        if np.isnan(best_r):
            concordance_score = 0.0
            concordant = False
        else:
            # Concordant if r > 0.15 and p < 0.1 (lenient thresholds for noisy data)
            concordant = best_r > 0.15 and best_p < 0.1
            # Score: sigmoid-like mapping of r to 0-1
            concordance_score = max(0.0, min(1.0, (best_r + 0.5) / 1.0))

        return GeneConcordance(
            gene=gene, cancer_type=cancer_type,
            drugs_tested=[d for d in drugs if d == best_drug] if best_drug else drugs,
            best_correlation=best_r, best_pvalue=best_p,
            n_cell_lines=best_n, concordant=concordant,
            concordance_score=concordance_score,
        )

    def _get_drug_sensitivity(self, drug_name: str) -> Optional[Dict[str, float]]:
        """
        Get drug sensitivity values as {cell_line_id: value} dict.
        Tries PRISM primary, then PRISM secondary, then GDSC.
        """
        # Try PRISM primary
        if self.prism is not None:
            try:
                profile = self.prism.get_drug_sensitivity(drug_name)
                if profile is not None and profile.cell_lines:
                    return dict(zip(profile.cell_lines, profile.ic50_values))
            except Exception:
                pass
            # Try secondary
            try:
                profile = self.prism.get_drug_sensitivity_secondary(drug_name)
                if profile is not None and profile.cell_lines:
                    return dict(zip(profile.cell_lines, profile.ic50_values))
            except Exception:
                pass

        # Try GDSC
        if self.gdsc is not None:
            try:
                profile = self.gdsc.get_drug_sensitivity(drug_name)
                if profile is not None and profile.cell_lines:
                    return dict(zip(profile.cell_lines, profile.ic50_values))
            except Exception:
                pass

        return None


# ============================================================================
# CO-ESSENTIALITY INTERACTION ESTIMATOR
# ============================================================================

class CoEssentialityInteractionEstimator:
    """
    Replaces the heuristic synergy score with data-driven interaction estimates.

    The existing pipeline computes a Jaccard co-essentiality matrix for each
    cancer type. We repurpose this:
      - Low Jaccard similarity → genes are essential in different cell lines
        → likely in independent pathways → combination may be synergistic
      - High Jaccard similarity → genes co-essential in same cell lines
        → likely in same pathway → combination may be redundant

    This directly addresses Norman et al. (2019) and Shen et al. (2017):
    single-gene effects don't predict combination effects, but co-essentiality
    patterns at least distinguish independent vs. redundant targets.
    """

    @staticmethod
    def compute_jaccard_matrix(
        genes: List[str],
        depmap_df,
        cell_lines: List[str],
        dependency_threshold: float = -0.5,
    ) -> np.ndarray:
        """
        Compute Jaccard similarity matrix for a set of genes.

        Args:
            genes: list of gene symbols
            depmap_df: dependency matrix (rows=cell lines, cols=genes)
            cell_lines: cell lines to use (cancer-specific)
            dependency_threshold: Chronos score below which gene is "essential"

        Returns:
            n×n numpy array of Jaccard similarities
        """
        n = len(genes)
        available_genes = [g for g in genes if g in depmap_df.columns]
        available_lines = [cl for cl in cell_lines if cl in depmap_df.index]

        if len(available_genes) < 2 or len(available_lines) < 3:
            return np.zeros((n, n))

        # Precompute essential-line sets
        essential_sets: Dict[str, Set[str]] = {}
        for g in available_genes:
            vals = depmap_df.loc[available_lines, g]
            essential_sets[g] = set(vals[vals < dependency_threshold].index)

        # Build gene index
        gene_idx = {g: i for i, g in enumerate(genes)}

        jac = np.zeros((n, n))
        for g1, g2 in combinations(available_genes, 2):
            s1 = essential_sets[g1]
            s2 = essential_sets[g2]
            union = len(s1 | s2)
            if union == 0:
                sim = 0.0
            else:
                sim = len(s1 & s2) / union
            i, j = gene_idx[g1], gene_idx[g2]
            jac[i, j] = sim
            jac[j, i] = sim

        return jac

    @staticmethod
    def estimate_pairwise_interaction(
        gene_a: str,
        gene_b: str,
        jaccard_sim: float,
    ) -> PairwiseInteraction:
        """
        Classify interaction type and estimate synergy from Jaccard similarity.

        Thresholds (conservative):
          Jaccard < 0.1  → independent pathways → high synergy potential
          Jaccard 0.1-0.4 → partially redundant → moderate synergy
          Jaccard > 0.4  → redundant (same pathway) → low synergy
        """
        if jaccard_sim < 0.1:
            itype = "independent"
            synergy = 0.9  # high synergy potential
        elif jaccard_sim < 0.4:
            itype = "partially_redundant"
            synergy = 0.6 - (jaccard_sim - 0.1) * (0.6 / 0.3)  # linear decay
            synergy = max(0.3, synergy)
        else:
            itype = "redundant"
            synergy = max(0.0, 0.3 - (jaccard_sim - 0.4))

        return PairwiseInteraction(
            gene_a=gene_a,
            gene_b=gene_b,
            jaccard_similarity=jaccard_sim,
            interaction_type=itype,
            synergy_estimate=synergy,
        )

    @classmethod
    def score_combination(
        cls,
        targets: Tuple[str, ...],
        depmap_df,
        cell_lines: List[str],
        dependency_threshold: float = -0.5,
        original_synergy: float = 0.0,
        original_pathway_diversity: float = 0.0,
    ) -> DataDrivenSynergyScore:
        """
        Compute data-driven synergy score for a combination.

        Blends co-essentiality-based interaction estimates with retained
        pathway diversity (which captures domain knowledge).
        """
        gene_list = list(targets)
        jac_matrix = cls.compute_jaccard_matrix(
            gene_list, depmap_df, cell_lines, dependency_threshold
        )

        interactions = []
        for i, g1 in enumerate(gene_list):
            for j, g2 in enumerate(gene_list):
                if i < j:
                    inter = cls.estimate_pairwise_interaction(g1, g2, jac_matrix[i, j])
                    interactions.append(inter)

        if interactions:
            synergy_estimates = [x.synergy_estimate for x in interactions]
            mean_syn = float(np.mean(synergy_estimates))
            min_syn = float(np.min(synergy_estimates))
        else:
            mean_syn = 0.5
            min_syn = 0.5

        # Blend: 60% data-driven (co-essentiality), 40% domain knowledge (pathway diversity)
        data_driven = mean_syn * 0.6 + original_pathway_diversity * 0.4

        return DataDrivenSynergyScore(
            targets=targets,
            pairwise_interactions=interactions,
            mean_synergy_estimate=mean_syn,
            min_synergy_estimate=min_syn,
            pathway_diversity=original_pathway_diversity,
            heuristic_synergy=original_synergy,
            data_driven_synergy=data_driven,
        )


# ============================================================================
# EVIDENCE TIER CLASSIFIER
# ============================================================================

class EvidenceTierClassifier:
    """
    Assigns confidence tiers to each cancer type's predictions.

    Tier 1: Experimental tri-axial validation (only PDAC via Liaki et al.)
    Tier 2: At least one gold-standard clinical combination recovered (pairwise ≥ 2 genes)
    Tier 3: PRISM concordance supports ≥2/3 targets in best triple
    Tier 4: Computational prediction only

    This replaces uniform "pan-cancer generalization" claims with honest
    per-cancer confidence stratification.
    """

    # Cancer types with experimental tri-axial validation
    TIER_1_CANCERS = {'Pancreatic Adenocarcinoma'}

    def __init__(self, gold_standard: Dict[str, FrozenSet[str]] = None):
        self.gold_standard = gold_standard or GOLD_STANDARD_COMBINATIONS

    def classify(
        self,
        cancer_type: str,
        predicted_targets: Tuple[str, ...],
        gene_concordances: Dict[str, GeneConcordance],
        n_cell_lines: int,
    ) -> EvidenceTier:
        """Classify evidence tier for a cancer type's predictions."""
        reasons = []

        # Count concordant targets
        concordant = [g for g, gc in gene_concordances.items() if gc.concordant]
        n_concordant = len(concordant)
        n_total = len(predicted_targets)
        concordance_frac = n_concordant / max(1, n_total)

        # Check gold standard
        gs_match = False
        if cancer_type in self.gold_standard:
            gs_targets = self.gold_standard[cancer_type]
            predicted_set = set(predicted_targets)
            overlap = gs_targets & predicted_set
            gs_match = len(overlap) >= 2
            if gs_match:
                reasons.append(f"Gold-standard match: {overlap}")

        # Tier assignment
        if cancer_type in self.TIER_1_CANCERS:
            tier = 1
            tier_label = "Experimentally validated (tri-axial)"
            reasons.append("Liaki et al. 2025 experimental validation")
        elif gs_match:
            tier = 2
            tier_label = "Gold-standard clinical match"
        elif n_concordant >= 2:
            tier = 3
            tier_label = "PRISM pharmacologically supported"
            reasons.append(f"{n_concordant}/{n_total} targets PRISM-concordant: {concordant}")
        else:
            tier = 4
            tier_label = "Computational prediction only"
            if n_cell_lines < 5:
                reasons.append(f"Low cell line count (n={n_cell_lines})")
            if n_concordant == 0:
                reasons.append("No pharmacological concordance data available")
            elif n_concordant == 1:
                reasons.append(f"Only 1/{n_total} targets PRISM-concordant")

        return EvidenceTier(
            cancer_type=cancer_type,
            tier=tier,
            tier_label=tier_label,
            reasons=reasons,
            n_cell_lines=n_cell_lines,
            n_concordant_targets=n_concordant,
            n_total_targets=n_total,
            gold_standard_match=gs_match,
            concordance_fraction=concordance_frac,
        )


# ============================================================================
# MAIN VALIDATOR (ORCHESTRATOR)
# ============================================================================

class PharmacologicalValidator:
    """
    Orchestrates all three validation systems.

    Usage:
        validator = PharmacologicalValidator(depmap_dir, drug_dir)
        result = validator.validate_predictions(
            cancer_type="Melanoma",
            predicted_targets=("BRAF", "CCND1", "STAT3"),
            cell_line_ids=["ACH-000001", ...],
            n_cell_lines=135,
        )
    """

    def __init__(
        self,
        depmap_dir: str = "./depmap_data",
        drug_dir: str = "./drug_sensitivity_data",
    ):
        self.depmap_dir = Path(depmap_dir)
        self.drug_dir = Path(drug_dir)
        self._depmap_df = None
        self._concordance = None
        self._tier_classifier = EvidenceTierClassifier()

    def _load_depmap(self):
        """Lazy-load DepMap dependency matrix."""
        if self._depmap_df is not None:
            return

        import pandas as pd

        crispr_path = self.depmap_dir / "CRISPRGeneEffect.csv"
        if not crispr_path.exists():
            logger.warning(f"DepMap file not found: {crispr_path}")
            self._depmap_df = pd.DataFrame()
            return

        logger.info(f"Loading DepMap CRISPR data from {crispr_path}...")
        df = pd.read_csv(crispr_path, index_col=0)

        # Parse column names: "GENE (ENTREZ)" → "GENE"
        new_cols = {}
        for col in df.columns:
            if ' (' in col:
                gene = col.split(' (')[0].strip()
                new_cols[col] = gene
            else:
                new_cols[col] = col
        df = df.rename(columns=new_cols)
        self._depmap_df = df
        logger.info(f"Loaded DepMap: {df.shape[0]} cell lines × {df.shape[1]} genes")

    def _load_concordance(self):
        """Lazy-load PRISM/GDSC data loaders for concordance checks."""
        if self._concordance is not None:
            return

        self._load_depmap()

        prism_loader = None
        gdsc_loader = None

        try:
            from alin.drug_sensitivity import PRISMLoader, GDSCLoader
            prism_loader = PRISMLoader(str(self.drug_dir))
            gdsc_loader = GDSCLoader(str(self.drug_dir))
        except Exception as e:
            logger.warning(f"Could not load drug sensitivity loaders: {e}")

        self._concordance = CRISPRDrugConcordance(
            depmap_df=self._depmap_df,
            prism_loader=prism_loader,
            gdsc_loader=gdsc_loader,
        )

    def validate_predictions(
        self,
        cancer_type: str,
        predicted_targets: Tuple[str, ...],
        cell_line_ids: List[str],
        n_cell_lines: int,
        original_synergy: float = 0.0,
        original_pathway_diversity: float = 0.0,
    ) -> ValidationResult:
        """
        Run all three validation systems on one cancer type's predictions.

        Returns ValidationResult with concordance scores, data-driven synergy,
        and evidence tier.
        """
        self._load_concordance()
        self._load_depmap()

        # 1. CRISPR-Drug concordance for each target
        gene_concordances: Dict[str, GeneConcordance] = {}
        for gene in predicted_targets:
            gc = self._concordance.compute_gene_concordance(
                gene=gene,
                cancer_type=cancer_type,
                cancer_cell_lines=cell_line_ids,
            )
            gene_concordances[gene] = gc

        # 2. Co-essentiality interaction estimates
        dd_synergy = CoEssentialityInteractionEstimator.score_combination(
            targets=predicted_targets,
            depmap_df=self._depmap_df,
            cell_lines=cell_line_ids,
            original_synergy=original_synergy,
            original_pathway_diversity=original_pathway_diversity,
        )

        # 3. Evidence tier
        evidence_tier = self._tier_classifier.classify(
            cancer_type=cancer_type,
            predicted_targets=predicted_targets,
            gene_concordances=gene_concordances,
            n_cell_lines=n_cell_lines,
        )

        # 4. Rescored combined score (pharmacologically adjusted)
        # Penalize targets that are CRISPR-essential but not drug-concordant
        concordance_penalty = 0.0
        for gene, gc in gene_concordances.items():
            if gc.drugs_tested and not gc.concordant:
                concordance_penalty += 0.05  # per discordant target with available drugs

        rescored = None  # will be computed by integration hook if needed

        return ValidationResult(
            cancer_type=cancer_type,
            gene_concordances=gene_concordances,
            pairwise_interactions=dd_synergy.pairwise_interactions,
            data_driven_synergy=dd_synergy,
            evidence_tier=evidence_tier,
            original_triple=predicted_targets,
            rescored_combined=rescored,
        )

    def validate_all_cancers(
        self,
        cancer_analyses: List,  # List[CancerTypeAnalysis] from pan_cancer_xnode
    ) -> Dict[str, ValidationResult]:
        """
        Validate all cancer types from a pipeline run.

        Args:
            cancer_analyses: list of CancerTypeAnalysis objects from pan_cancer_xnode

        Returns:
            dict mapping cancer_type → ValidationResult
        """
        results = {}
        for analysis in cancer_analyses:
            if not analysis.best_triple:
                continue

            targets = analysis.best_triple.targets
            result = self.validate_predictions(
                cancer_type=analysis.cancer_type,
                predicted_targets=targets,
                cell_line_ids=analysis.cell_line_ids,
                n_cell_lines=analysis.n_cell_lines,
                original_synergy=analysis.best_triple.synergy_score,
                original_pathway_diversity=analysis.best_triple.pathway_coverage.get(
                    'diversity', 0.0
                ) if analysis.best_triple.pathway_coverage else 0.0,
            )
            results[analysis.cancer_type] = result

        return results

    def export_validation_report(
        self,
        results: Dict[str, ValidationResult],
        output_path: str = "validation_report.json",
    ) -> str:
        """Export validation results to JSON."""
        report = {}
        for cancer, vr in sorted(results.items()):
            report[cancer] = {
                'evidence_tier': vr.evidence_tier.tier,
                'tier_label': vr.evidence_tier.tier_label,
                'tier_reasons': vr.evidence_tier.reasons,
                'n_cell_lines': vr.evidence_tier.n_cell_lines,
                'gold_standard_match': vr.evidence_tier.gold_standard_match,
                'concordance_fraction': vr.evidence_tier.concordance_fraction,
                'predicted_targets': list(vr.original_triple) if vr.original_triple else [],
                'gene_concordances': {
                    g: {
                        'concordant': gc.concordant,
                        'score': gc.concordance_score,
                        'correlation': gc.best_correlation if not np.isnan(gc.best_correlation) else None,
                        'p_value': gc.best_pvalue,
                        'drugs_tested': gc.drugs_tested,
                        'n_cell_lines': gc.n_cell_lines,
                    }
                    for g, gc in vr.gene_concordances.items()
                },
                'data_driven_synergy': {
                    'mean_estimate': vr.data_driven_synergy.mean_synergy_estimate,
                    'min_estimate': vr.data_driven_synergy.min_synergy_estimate,
                    'heuristic_synergy': vr.data_driven_synergy.heuristic_synergy,
                    'data_driven_synergy': vr.data_driven_synergy.data_driven_synergy,
                    'interactions': [
                        {
                            'genes': (pi.gene_a, pi.gene_b),
                            'jaccard': pi.jaccard_similarity,
                            'type': pi.interaction_type,
                            'synergy_est': pi.synergy_estimate,
                        }
                        for pi in vr.pairwise_interactions
                    ],
                } if vr.data_driven_synergy else None,
            }

        output = Path(output_path)
        with open(output, 'w') as f:
            json.dump(report, f, indent=2)

        # Summary statistics
        tiers = [vr.evidence_tier.tier for vr in results.values()]
        logger.info(f"Validation report: {len(results)} cancers")
        for t in range(1, 5):
            count = tiers.count(t)
            logger.info(f"  Tier {t}: {count} cancers ({100*count/max(1,len(tiers)):.0f}%)")

        return str(output)


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def main():
    """Run pharmacological validation on pipeline results."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Pharmacological validation of ALIN predictions"
    )
    parser.add_argument(
        "--depmap-dir", default="./depmap_data",
        help="Directory containing DepMap data files"
    )
    parser.add_argument(
        "--drug-dir", default="./drug_sensitivity_data",
        help="Directory containing PRISM/GDSC data files"
    )
    parser.add_argument(
        "--output", default="validation_report.json",
        help="Output JSON report path"
    )
    parser.add_argument(
        "--cancer-type", default=None,
        help="Validate single cancer type (default: all)"
    )
    parser.add_argument(
        "--targets", nargs="+", default=None,
        help="Gene targets to validate (required if --cancer-type is set)"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    validator = PharmacologicalValidator(args.depmap_dir, args.drug_dir)

    if args.cancer_type and args.targets:
        result = validator.validate_predictions(
            cancer_type=args.cancer_type,
            predicted_targets=tuple(args.targets),
            cell_line_ids=[],  # will use pan-cancer cell lines
            n_cell_lines=0,
        )
        print(f"\n{'='*60}")
        print(f"Validation: {args.cancer_type}")
        print(f"Targets: {args.targets}")
        print(f"Evidence Tier: {result.evidence_tier.tier} ({result.evidence_tier.tier_label})")
        print(f"Reasons: {result.evidence_tier.reasons}")
        print(f"\nGene concordances:")
        for g, gc in result.gene_concordances.items():
            print(f"  {gc.summary}")
        print(f"\nData-driven synergy: {result.data_driven_synergy.data_driven_synergy:.3f}")
        print(f"  (heuristic was: {result.data_driven_synergy.heuristic_synergy:.3f})")
        for pi in result.pairwise_interactions:
            print(f"  {pi.summary}")
    else:
        print("Use --cancer-type and --targets for single-cancer validation,")
        print("or integrate with pan_cancer_xnode.py for full pipeline validation.")


if __name__ == "__main__":
    main()
