"""
Tests for pharmacological_validation.py

Tests the three programmatic improvements:
1. CRISPR-Drug concordance scoring
2. Co-essentiality interaction estimation
3. Evidence tier classification

Plus integration tests with the pipeline.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from pharmacological_validation import (
    CRISPRDrugConcordance,
    CoEssentialityInteractionEstimator,
    EvidenceTierClassifier,
    GeneConcordance,
    PairwiseInteraction,
    DataDrivenSynergyScore,
    EvidenceTier,
    GENE_TO_DRUGS,
    GOLD_STANDARD_COMBINATIONS,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_depmap_df():
    """Create a small mock DepMap dependency matrix."""
    np.random.seed(42)
    cell_lines = [f"ACH-{i:06d}" for i in range(50)]
    genes = ["STAT3", "CDK4", "BRAF", "KRAS", "MCL1", "EGFR", "CCND1", "FLI1"]
    data = np.random.randn(50, 8) * 0.3 - 0.3  # mostly negative (essential)
    # Make STAT3 and CDK4 co-essential (correlated patterns)
    data[:, 1] = data[:, 0] * 0.8 + np.random.randn(50) * 0.1
    # Make BRAF independent of STAT3
    data[:, 2] = np.random.randn(50) * 0.3 - 0.2
    return pd.DataFrame(data, index=cell_lines, columns=genes)


# ============================================================================
# Co-essentiality Interaction Estimator
# ============================================================================

class TestCoEssentialityInteractionEstimator:

    def test_jaccard_matrix_shape(self, mock_depmap_df):
        genes = ["STAT3", "CDK4", "BRAF"]
        cell_lines = list(mock_depmap_df.index)
        jac = CoEssentialityInteractionEstimator.compute_jaccard_matrix(
            genes, mock_depmap_df, cell_lines
        )
        assert jac.shape == (3, 3)
        assert np.allclose(np.diag(jac), 0)  # diagonal is 0

    def test_jaccard_symmetry(self, mock_depmap_df):
        genes = ["STAT3", "CDK4", "BRAF", "KRAS"]
        cell_lines = list(mock_depmap_df.index)
        jac = CoEssentialityInteractionEstimator.compute_jaccard_matrix(
            genes, mock_depmap_df, cell_lines
        )
        assert np.allclose(jac, jac.T)

    def test_jaccard_range(self, mock_depmap_df):
        genes = ["STAT3", "CDK4", "BRAF"]
        cell_lines = list(mock_depmap_df.index)
        jac = CoEssentialityInteractionEstimator.compute_jaccard_matrix(
            genes, mock_depmap_df, cell_lines
        )
        assert np.all(jac >= 0) and np.all(jac <= 1)

    def test_coessential_genes_high_jaccard(self, mock_depmap_df):
        """STAT3 and CDK4 have correlated dependency → should have higher Jaccard."""
        genes = ["STAT3", "CDK4", "BRAF"]
        cell_lines = list(mock_depmap_df.index)
        jac = CoEssentialityInteractionEstimator.compute_jaccard_matrix(
            genes, mock_depmap_df, cell_lines
        )
        # STAT3-CDK4 should have higher Jaccard than STAT3-BRAF
        assert jac[0, 1] > jac[0, 2], (
            f"STAT3-CDK4 Jaccard ({jac[0,1]:.3f}) should exceed "
            f"STAT3-BRAF ({jac[0,2]:.3f}) due to correlated essentiality"
        )

    def test_independent_interaction(self):
        inter = CoEssentialityInteractionEstimator.estimate_pairwise_interaction(
            "A", "B", jaccard_sim=0.05
        )
        assert inter.interaction_type == "independent"
        assert inter.synergy_estimate == 0.9

    def test_redundant_interaction(self):
        inter = CoEssentialityInteractionEstimator.estimate_pairwise_interaction(
            "A", "B", jaccard_sim=0.6
        )
        assert inter.interaction_type == "redundant"
        assert inter.synergy_estimate < 0.3

    def test_partially_redundant_interaction(self):
        inter = CoEssentialityInteractionEstimator.estimate_pairwise_interaction(
            "A", "B", jaccard_sim=0.25
        )
        assert inter.interaction_type == "partially_redundant"
        assert 0.3 <= inter.synergy_estimate <= 0.6

    def test_score_combination(self, mock_depmap_df):
        result = CoEssentialityInteractionEstimator.score_combination(
            targets=("STAT3", "CDK4", "BRAF"),
            depmap_df=mock_depmap_df,
            cell_lines=list(mock_depmap_df.index),
            original_synergy=0.5,
            original_pathway_diversity=0.8,
        )
        assert isinstance(result, DataDrivenSynergyScore)
        assert 0.0 <= result.data_driven_synergy <= 1.0
        assert len(result.pairwise_interactions) == 3  # C(3,2)
        assert result.heuristic_synergy == 0.5

    def test_data_driven_vs_heuristic_differ(self, mock_depmap_df):
        """Data-driven score should generally differ from the heuristic."""
        result = CoEssentialityInteractionEstimator.score_combination(
            targets=("STAT3", "CDK4", "BRAF"),
            depmap_df=mock_depmap_df,
            cell_lines=list(mock_depmap_df.index),
            original_synergy=0.5,
            original_pathway_diversity=0.8,
        )
        # With co-essential STAT3-CDK4, synergy should differ from heuristic
        assert result.data_driven_synergy != result.heuristic_synergy

    def test_empty_genes(self, mock_depmap_df):
        """Edge case: fewer than 2 genes."""
        result = CoEssentialityInteractionEstimator.score_combination(
            targets=("STAT3",),
            depmap_df=mock_depmap_df,
            cell_lines=list(mock_depmap_df.index),
        )
        assert result.mean_synergy_estimate == 0.5  # default for no pairs


# ============================================================================
# CRISPR-Drug Concordance
# ============================================================================

class TestCRISPRDrugConcordance:

    def test_gene_without_drugs(self, mock_depmap_df):
        """Gene with no known drugs → no pharmacological validation."""
        concordance = CRISPRDrugConcordance(mock_depmap_df)
        result = concordance.compute_gene_concordance(
            "FLI1", "Ewing Sarcoma", list(mock_depmap_df.index)
        )
        assert not result.concordant
        assert result.concordance_score == 0.0
        assert result.drugs_tested == []

    def test_gene_not_in_depmap(self, mock_depmap_df):
        """Gene not in DepMap → cannot compute concordance."""
        concordance = CRISPRDrugConcordance(mock_depmap_df)
        result = concordance.compute_gene_concordance(
            "NONEXISTENT", "Any Cancer", list(mock_depmap_df.index)
        )
        assert not result.concordant
        assert result.concordance_score == 0.0

    def test_gene_with_drugs_no_loaders(self, mock_depmap_df):
        """Gene has drugs in GENE_TO_DRUGS but no PRISM/GDSC loader → graceful."""
        concordance = CRISPRDrugConcordance(mock_depmap_df)
        result = concordance.compute_gene_concordance(
            "BRAF", "Melanoma", list(mock_depmap_df.index)
        )
        # No drug data available, so not concordant but doesn't crash
        assert isinstance(result, GeneConcordance)
        assert result.gene == "BRAF"

    def test_gene_to_drugs_mapping_coverage(self):
        """Verify key pipeline targets have drug mappings."""
        key_targets = ["STAT3", "CDK4", "BRAF", "KRAS", "MCL1", "EGFR", "BCL2"]
        for gene in key_targets:
            assert gene in GENE_TO_DRUGS, f"{gene} missing from GENE_TO_DRUGS"
            assert len(GENE_TO_DRUGS[gene]) >= 1


# ============================================================================
# Evidence Tier Classifier
# ============================================================================

class TestEvidenceTierClassifier:

    def test_tier_1_pdac(self):
        classifier = EvidenceTierClassifier()
        tier = classifier.classify(
            cancer_type="Pancreatic Adenocarcinoma",
            predicted_targets=("KRAS", "CCND1", "STAT3"),
            gene_concordances={},
            n_cell_lines=64,
        )
        assert tier.tier == 1
        assert "Liaki" in tier.tier_label or "Experimental" in tier.tier_label

    def test_tier_2_gold_standard_match(self):
        classifier = EvidenceTierClassifier()
        # Melanoma gold standard is {BRAF, MAP2K1}
        concordances = {
            "BRAF": GeneConcordance("BRAF", "Melanoma", ["Vemurafenib"], -0.3, 0.01, 50, True, 0.7),
            "MAP2K1": GeneConcordance("MAP2K1", "Melanoma", ["Trametinib"], -0.25, 0.02, 50, True, 0.65),
            "STAT3": GeneConcordance("STAT3", "Melanoma", ["Napabucasin"], 0.0, 0.5, 50, False, 0.3),
        }
        tier = classifier.classify(
            cancer_type="Melanoma",
            predicted_targets=("BRAF", "MAP2K1", "STAT3"),
            gene_concordances=concordances,
            n_cell_lines=135,
        )
        assert tier.tier == 2
        assert tier.gold_standard_match

    def test_tier_3_prism_concordant(self):
        classifier = EvidenceTierClassifier()
        concordances = {
            "CDK4": GeneConcordance("CDK4", "Breast", ["Palbociclib"], -0.35, 0.001, 80, True, 0.85),
            "STAT3": GeneConcordance("STAT3", "Breast", ["Napabucasin"], -0.2, 0.05, 80, True, 0.6),
            "EGFR": GeneConcordance("EGFR", "Breast", ["Erlotinib"], 0.05, 0.6, 80, False, 0.3),
        }
        tier = classifier.classify(
            cancer_type="Some Rare Cancer",  # not in gold standard
            predicted_targets=("CDK4", "STAT3", "EGFR"),
            gene_concordances=concordances,
            n_cell_lines=30,
        )
        assert tier.tier == 3
        assert tier.n_concordant_targets == 2

    def test_tier_4_computational_only(self):
        classifier = EvidenceTierClassifier()
        concordances = {
            "CHMP4B": GeneConcordance("CHMP4B", "Glioma", [], np.nan, 1.0, 0, False, 0.0),
            "DNM2": GeneConcordance("DNM2", "Glioma", [], np.nan, 1.0, 0, False, 0.0),
            "STAT3": GeneConcordance("STAT3", "Glioma", ["Napabucasin"], 0.02, 0.8, 20, False, 0.3),
        }
        tier = classifier.classify(
            cancer_type="Some Very Rare Cancer",
            predicted_targets=("CHMP4B", "DNM2", "STAT3"),
            gene_concordances=concordances,
            n_cell_lines=3,
        )
        assert tier.tier == 4
        assert "Low cell line count" in str(tier.reasons)

    def test_tier_ordering(self):
        """Lower tier number = higher confidence."""
        classifier = EvidenceTierClassifier()
        # Tier 1 is highest confidence
        t1 = classifier.classify("Pancreatic Adenocarcinoma", ("KRAS",), {}, 64)
        t4 = classifier.classify("Unknown Cancer", ("X",), {}, 1)
        assert t1.tier < t4.tier

    def test_gold_standard_entries_exist(self):
        """Verify gold standard has expected cancer types."""
        assert "Melanoma" in GOLD_STANDARD_COMBINATIONS
        assert "Pancreatic Adenocarcinoma" in GOLD_STANDARD_COMBINATIONS
        assert len(GOLD_STANDARD_COMBINATIONS) >= 5


# ============================================================================
# Integration with CancerTypeAnalysis
# ============================================================================

class TestPipelineIntegration:

    def test_cancer_type_analysis_has_validation_field(self):
        """CancerTypeAnalysis dataclass should have pharmacological_validation field."""
        from core.data_structures import CancerTypeAnalysis
        import dataclasses
        field_names = [f.name for f in dataclasses.fields(CancerTypeAnalysis)]
        assert 'pharmacological_validation' in field_names

    def test_synergy_scorer_still_works(self):
        """Heuristic synergy scorer should still work (backwards compatibility)."""
        from pan_cancer_xnode import SynergyScorer
        # SynergyScorer needs OmniPath but we test the static KNOWN_SYNERGIES
        assert frozenset({'BRAF', 'MAP2K1'}) in SynergyScorer.KNOWN_SYNERGIES

    def test_data_driven_synergy_overrides_heuristic(self, mock_depmap_df):
        """Data-driven synergy should produce different values than heuristic."""
        # The heuristic for BRAF+CDK4+STAT3 would give:
        #   known_pair_score: BRAF-CDK4=0.0, BRAF-STAT3=0.0, CDK4-STAT3=0.0 → 0.0
        #   pathway_diversity: MAPK, CELL_CYCLE, JAK_STAT → 3/3 = 1.0
        #   synergy = 0.0*0.4 + 1.0*0.6 = 0.6
        # Data-driven should differ based on actual co-essentiality
        dd = CoEssentialityInteractionEstimator.score_combination(
            targets=("BRAF", "CDK4", "STAT3"),
            depmap_df=mock_depmap_df,
            cell_lines=list(mock_depmap_df.index),
            original_synergy=0.6,
            original_pathway_diversity=1.0,
        )
        # Assert it's not identical to heuristic
        assert dd.data_driven_synergy != 0.6 or dd.mean_synergy_estimate != 0.6


# ============================================================================
# Paper consistency tests
# ============================================================================

class TestPaperConsistency:

    @pytest.fixture(autouse=True)
    def load_paper(self):
        from pathlib import Path
        paper_path = Path(__file__).parent.parent / "paper.tex"
        if paper_path.exists():
            self.paper_text = paper_path.read_text()
        else:
            pytest.skip("paper.tex not found")

    def test_paper_mentions_prism_concordance(self):
        assert "PRISM" in self.paper_text or "pharmacological concordance" in self.paper_text

    def test_paper_mentions_evidence_tiers(self):
        assert "evidence tier" in self.paper_text.lower() or "tier" in self.paper_text.lower()

    def test_paper_mentions_coessentiality_synergy(self):
        assert "co-essentiality" in self.paper_text.lower() or "data-driven synergy" in self.paper_text.lower()

    def test_paper_has_limitation_14(self):
        assert "14." in self.paper_text and "CRISPR" in self.paper_text

    def test_paper_has_limitation_15(self):
        assert "15." in self.paper_text and "combination" in self.paper_text.lower()
