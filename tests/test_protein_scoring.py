"""
Tests for Protein-Level Druggability Scoring
=============================================

Unit tests for alin.protein_scoring module covering:
    - Sub-score computation (structural, abundance, degradability, PPI)
    - Composite blending (gene + protein)
    - Cache behaviour
    - Fallback / graceful degradation
    - NodeCost integration (effective_druggability, total_cost)
    - Report generation
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from alin.protein_scoring import (
    ALPHA_GENE_WEIGHT,
    KNOWN_DEGRADER_TARGETS,
    GENE_TO_UNIPROT,
    DegradabilityScore,
    PPIAccessibility,
    ProteinAbundance,
    ProteinAPICache,
    ProteinDruggabilityScore,
    ProteinDruggabilityScorer,
    StructuralDruggability,
    compute_abundance_score,
    compute_degradability_score,
    compute_ppi_score,
    compute_structural_score,
    generate_protein_scoring_report,
)
from core.data_structures import NodeCost


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def tmp_cache(tmp_path):
    """Temporary cache directory."""
    return ProteinAPICache(str(tmp_path / "protein_cache"))


@pytest.fixture
def mock_alphafold():
    """Mock AlphaFold pLDDT response."""
    return {
        "mean_plddt": 82.5,
        "sequence_length": 400,
        "n_high_conf": 310,
        "n_very_high": 180,
        "n_disordered": 30,
        "fraction_structured": 0.775,
        "fraction_disordered": 0.075,
    }


@pytest.fixture
def proteomics_data():
    """Fake CCLE proteomics DataFrame."""
    genes = ["EGFR", "STAT3", "CDK4", "OBSCUREGENE"]
    cell_lines = [f"CL_{i}" for i in range(20)]
    rng = np.random.default_rng(42)
    data = rng.normal(loc=1.5, scale=0.8, size=(20, 4))
    data[:, 3] = -1  # OBSCUREGENE: below threshold
    df = pd.DataFrame(data, index=cell_lines, columns=genes)
    cancer_map = {cl: "Lung" for cl in cell_lines[:15]}
    cancer_map.update({cl: "Breast" for cl in cell_lines[15:]})
    return df, cancer_map


# ============================================================================
# CACHE TESTS
# ============================================================================

class TestProteinAPICache:
    def test_put_and_get(self, tmp_cache):
        tmp_cache.put("test", "key1", {"score": 0.9})
        result = tmp_cache.get("test", "key1")
        assert result == {"score": 0.9}

    def test_miss(self, tmp_cache):
        assert tmp_cache.get("test", "nonexistent") is None

    def test_different_keys(self, tmp_cache):
        tmp_cache.put("pdb", "P00533", 42)
        tmp_cache.put("pdb", "P04626", 99)
        assert tmp_cache.get("pdb", "P00533") == 42
        assert tmp_cache.get("pdb", "P04626") == 99


# ============================================================================
# STRUCTURAL DRUGGABILITY TESTS
# ============================================================================

class TestStructuralDruggability:
    @patch("alin.protein_scoring.fetch_pdb_ligand_count")
    @patch("alin.protein_scoring.fetch_alphafold_plddt")
    def test_well_characterized_target(self, mock_af, mock_pdb, tmp_cache):
        """EGFR: lots of structures + ligands → high score."""
        mock_af.return_value = {
            "mean_plddt": 88.0,
            "fraction_structured": 0.85,
            "fraction_disordered": 0.05,
        }
        mock_pdb.return_value = {"n_structures": 250, "n_ligand_bound": 180}

        result = compute_structural_score("EGFR", "P00533", tmp_cache)

        assert result.has_pocket is True
        assert result.structural_score >= 0.8
        assert result.n_ligand_bound == 180

    @patch("alin.protein_scoring.fetch_pdb_ligand_count")
    @patch("alin.protein_scoring.fetch_alphafold_plddt")
    def test_no_structures(self, mock_af, mock_pdb, tmp_cache):
        """Hypothetical gene with no PDB → low pocket score."""
        mock_af.return_value = {
            "mean_plddt": 60.0,
            "fraction_structured": 0.40,
            "fraction_disordered": 0.35,
        }
        mock_pdb.return_value = {"n_structures": 0, "n_ligand_bound": 0}

        result = compute_structural_score("FAKEGENE", "Q99999", tmp_cache)

        assert result.has_pocket is False
        assert result.structural_score < 0.5

    @patch("alin.protein_scoring.fetch_pdb_ligand_count")
    @patch("alin.protein_scoring.fetch_alphafold_plddt")
    def test_structures_but_no_ligands(self, mock_af, mock_pdb, tmp_cache):
        """Structures exist but no ligand co-crystals → moderate."""
        mock_af.return_value = {
            "mean_plddt": 80.0,
            "fraction_structured": 0.70,
            "fraction_disordered": 0.10,
        }
        mock_pdb.return_value = {"n_structures": 10, "n_ligand_bound": 0}

        result = compute_structural_score("TP53", "P04637", tmp_cache)

        assert result.has_pocket is False
        assert 0.3 <= result.structural_score <= 0.6


# ============================================================================
# PROTEIN ABUNDANCE TESTS
# ============================================================================

class TestProteinAbundance:
    def test_high_detection(self, proteomics_data):
        df, cmap = proteomics_data
        result = compute_abundance_score("EGFR", "Lung", df, cmap)
        assert result is not None
        assert result.abundance_score >= 0.7
        assert result.detection_fraction > 0.5

    def test_low_detection(self, proteomics_data):
        df, cmap = proteomics_data
        result = compute_abundance_score("OBSCUREGENE", "Lung", df, cmap)
        assert result is not None
        assert result.abundance_score <= 0.4

    def test_no_data(self):
        result = compute_abundance_score("EGFR", "Lung", None, None)
        assert result is None

    def test_gene_not_in_dataframe(self, proteomics_data):
        df, cmap = proteomics_data
        result = compute_abundance_score("NONEXISTENT", "Lung", df, cmap)
        assert result is None

    def test_cancer_type_not_found(self, proteomics_data):
        df, cmap = proteomics_data
        result = compute_abundance_score("EGFR", "NoSuchCancer", df, cmap)
        assert result is None


# ============================================================================
# DEGRADABILITY TESTS
# ============================================================================

class TestDegradability:
    def test_known_degrader(self, mock_alphafold):
        """STAT3 has a known PROTAC (SD-36) → high score."""
        result = compute_degradability_score("STAT3", "P40763", mock_alphafold)
        assert result.has_known_degrader is True
        assert result.degradability_score >= 0.7
        assert result.degrader_exemplar == "SD-36"

    def test_no_degrader_high_lysines(self, mock_alphafold):
        """Gene with many surface lysines but no known degrader."""
        # Tweak AlphaFold data to give many structured residues
        af = {**mock_alphafold, "n_high_conf": 600, "sequence_length": 800}
        result = compute_degradability_score("PIK3CA", "P42336", af)
        assert result.has_known_degrader is False
        assert result.degradability_score >= 0.3

    def test_no_data(self):
        """No AlphaFold data → minimal score."""
        result = compute_degradability_score("UNKNOWNGENE", "Q99999", None)
        assert result.degradability_score == 0.1

    def test_known_degrader_targets_dict(self):
        """All curated degrader targets have required keys."""
        for gene, info in KNOWN_DEGRADER_TARGETS.items():
            assert "status" in info
            assert "exemplar" in info
            assert "source" in info


# ============================================================================
# PPI ACCESSIBILITY TESTS
# ============================================================================

class TestPPIAccessibility:
    @patch("alin.protein_scoring.fetch_pdb_complex_count")
    def test_well_characterized_ppi(self, mock_cpx, tmp_cache, mock_alphafold):
        mock_cpx.return_value = 12
        result = compute_ppi_score("BCL2L1", "Q07817", mock_alphafold, tmp_cache)
        assert result.ppi_score == 1.0
        assert result.has_interface_data is True

    @patch("alin.protein_scoring.fetch_pdb_complex_count")
    def test_few_complexes(self, mock_cpx, tmp_cache, mock_alphafold):
        mock_cpx.return_value = 2
        result = compute_ppi_score("CDK4", "P11802", mock_alphafold, tmp_cache)
        assert result.ppi_score == 0.6

    @patch("alin.protein_scoring.fetch_pdb_complex_count")
    def test_no_complexes_structured(self, mock_cpx, tmp_cache, mock_alphafold):
        mock_cpx.return_value = 0
        result = compute_ppi_score("KRAS", "P01116", mock_alphafold, tmp_cache)
        assert result.ppi_score == 0.3

    @patch("alin.protein_scoring.fetch_pdb_complex_count")
    def test_disordered_no_complexes(self, mock_cpx, tmp_cache):
        mock_cpx.return_value = 0
        af = {"fraction_disordered": 0.6}
        result = compute_ppi_score("MYC", "P01106", af, tmp_cache)
        assert result.ppi_score == 0.1


# ============================================================================
# COMPOSITE SCORER TESTS
# ============================================================================

class TestProteinDruggabilityScorer:
    @patch("alin.protein_scoring.fetch_pdb_complex_count")
    @patch("alin.protein_scoring.fetch_pdb_ligand_count")
    @patch("alin.protein_scoring.fetch_alphafold_plddt")
    def test_score_gene_returns_valid(self, mock_af, mock_pdb, mock_cpx, tmp_cache):
        mock_af.return_value = {
            "mean_plddt": 85.0,
            "sequence_length": 500,
            "n_high_conf": 400,
            "n_very_high": 200,
            "n_disordered": 20,
            "fraction_structured": 0.8,
            "fraction_disordered": 0.04,
        }
        mock_pdb.return_value = {"n_structures": 50, "n_ligand_bound": 30}
        mock_cpx.return_value = 8

        scorer = ProteinDruggabilityScorer(
            genes=["EGFR"],
            cancer_type="Lung",
            gene_druggability_fn=lambda g: 0.9,
            cache_dir=str(tmp_cache.cache_dir),
        )
        result = scorer.score_gene("EGFR")

        assert 0 <= result.protein_score <= 1
        assert 0 <= result.blended_score <= 1
        assert result.gene == "EGFR"
        assert result.gene_druggability == 0.9

    @patch("alin.protein_scoring.fetch_pdb_complex_count")
    @patch("alin.protein_scoring.fetch_pdb_ligand_count")
    @patch("alin.protein_scoring.fetch_alphafold_plddt")
    def test_score_all(self, mock_af, mock_pdb, mock_cpx, tmp_cache):
        mock_af.return_value = {
            "mean_plddt": 80.0, "sequence_length": 400, "n_high_conf": 300,
            "n_very_high": 150, "n_disordered": 30, "fraction_structured": 0.75,
            "fraction_disordered": 0.075,
        }
        mock_pdb.return_value = {"n_structures": 10, "n_ligand_bound": 5}
        mock_cpx.return_value = 3

        scorer = ProteinDruggabilityScorer(
            genes=["EGFR", "STAT3", "CDK4"],
            gene_druggability_fn=lambda g: 0.5,
            cache_dir=str(tmp_cache.cache_dir),
        )
        results = scorer.score_all(progress=False)

        assert len(results) == 3
        assert all(0 <= r.blended_score <= 1 for r in results.values())

    def test_fallback_for_unknown_gene(self, tmp_cache):
        scorer = ProteinDruggabilityScorer(
            genes=["NONEXISTENT_GENE"],
            gene_druggability_fn=lambda g: 0.2,
            cache_dir=str(tmp_cache.cache_dir),
        )
        result = scorer.score_gene("NONEXISTENT_GENE")

        assert result.protein_score == 0.3  # fallback
        expected_blended = 0.6 * 0.2 + 0.4 * 0.3
        assert abs(result.blended_score - expected_blended) < 0.01

    def test_blending_formula(self, tmp_cache):
        """Verify blended = α·gene + (1-α)·protein."""
        scorer = ProteinDruggabilityScorer(
            genes=["FAKE"],
            gene_druggability_fn=lambda g: 0.8,
            alpha=0.7,
            cache_dir=str(tmp_cache.cache_dir),
        )
        result = scorer._fallback_score("FAKE")
        expected = 0.7 * 0.8 + 0.3 * 0.3
        assert abs(result.blended_score - expected) < 0.01


# ============================================================================
# NODECOST INTEGRATION TESTS
# ============================================================================

class TestNodeCostProteinIntegration:
    def test_effective_druggability_with_protein(self):
        """effective_druggability blends gene + protein when protein is set."""
        nc = NodeCost(
            gene="EGFR",
            toxicity_score=0.3,
            tumor_specificity=0.7,
            druggability_score=0.9,
            protein_druggability_score=0.6,
        )
        # α=0.6: 0.6*0.9 + 0.4*0.6 = 0.54 + 0.24 = 0.78
        assert abs(nc.effective_druggability - 0.78) < 0.001

    def test_effective_druggability_without_protein(self):
        """Without protein score, effective = gene-level."""
        nc = NodeCost(
            gene="EGFR",
            toxicity_score=0.3,
            tumor_specificity=0.7,
            druggability_score=0.9,
        )
        assert nc.effective_druggability == 0.9

    def test_total_cost_uses_effective(self):
        """total_cost() should use effective_druggability (blended)."""
        nc_gene = NodeCost(
            gene="X", toxicity_score=0.5, tumor_specificity=0.5,
            druggability_score=0.8,
        )
        nc_blended = NodeCost(
            gene="X", toxicity_score=0.5, tumor_specificity=0.5,
            druggability_score=0.8, protein_druggability_score=0.4,
        )
        # The blended version has lower effective druggability (0.6*0.8+0.4*0.4=0.64)
        # so its cost should be higher (less druggable → less reward)
        assert nc_blended.total_cost() > nc_gene.total_cost()

    def test_backward_compatible(self):
        """Existing code that doesn't pass protein_druggability_score still works."""
        nc = NodeCost(
            gene="STAT3",
            toxicity_score=0.2,
            tumor_specificity=0.8,
            druggability_score=0.6,
            pan_essential_penalty=0.0,
            base_penalty=1.0,
        )
        assert nc.protein_druggability_score is None
        cost = nc.total_cost()
        expected = 1.0 * 0.2 - 0.5 * 0.8 - 0.3 * 0.6 + 0.0 + 1.0
        assert abs(cost - expected) < 0.001


# ============================================================================
# GENE TO UNIPROT MAPPING TESTS
# ============================================================================

class TestGeneMapping:
    def test_all_signaling_genes_covered(self):
        """All major signaling pathway genes should have UniProt IDs."""
        critical_genes = [
            "EGFR", "KRAS", "BRAF", "PIK3CA", "AKT1", "MTOR",
            "JAK2", "STAT3", "CDK4", "CDK6", "BCL2", "MCL1",
            "TP53", "MDM2", "MYC", "PARP1",
        ]
        for gene in critical_genes:
            assert gene in GENE_TO_UNIPROT, f"{gene} missing from GENE_TO_UNIPROT"

    def test_uniprot_ids_format(self):
        """UniProt IDs should match the canonical format (6+ chars, alphanumeric)."""
        for gene, uid in GENE_TO_UNIPROT.items():
            assert len(uid) >= 6, f"Short UniProt ID for {gene}: {uid}"
            assert uid[0].isalpha(), f"Invalid UniProt ID for {gene}: {uid}"


# ============================================================================
# REPORT GENERATION TEST
# ============================================================================

class TestReportGeneration:
    def test_generates_csv_and_json(self, tmp_path):
        """generate_protein_scoring_report produces CSV + JSON."""
        structural = StructuralDruggability(
            gene="EGFR", uniprot_id="P00533", mean_plddt=85.0,
            domain_plddt={}, n_pdb_structures=100, n_ligand_bound=50,
            has_pocket=True, structural_score=0.9,
        )
        degradability = DegradabilityScore(
            gene="EGFR", has_known_degrader=True, degrader_status="preclinical",
            degrader_exemplar="PROTAC-EGFR", n_surface_lysines=10,
            degradability_score=0.7,
        )
        ppi = PPIAccessibility(
            gene="EGFR", n_pdb_complexes=15, has_interface_data=True,
            disordered_fraction=0.05, ppi_score=1.0,
        )
        result = ProteinDruggabilityScore(
            gene="EGFR", structural=structural, abundance=None,
            degradability=degradability, ppi=ppi,
            protein_score=0.85, blended_score=0.88,
            gene_druggability=0.9,
        )

        csv_path = generate_protein_scoring_report(
            {"EGFR": result}, output_dir=str(tmp_path)
        )

        assert Path(csv_path).exists()
        df = pd.read_csv(csv_path)
        assert len(df) == 1
        assert df.iloc[0]["gene"] == "EGFR"
        assert df.iloc[0]["structural_score"] == 0.9

        json_path = tmp_path / "protein_druggability_scores.json"
        assert json_path.exists()
        jdata = json.loads(json_path.read_text())
        assert "EGFR" in jdata


# ============================================================================
# TO_DICT SERIALISATION TESTS
# ============================================================================

class TestSerialization:
    def test_protein_score_to_dict(self):
        structural = StructuralDruggability(
            gene="CDK4", uniprot_id="P11802", mean_plddt=78.0,
            domain_plddt={}, n_pdb_structures=30, n_ligand_bound=12,
            has_pocket=True, structural_score=0.75,
        )
        degradability = DegradabilityScore(
            gene="CDK4", has_known_degrader=True, degrader_status="preclinical",
            degrader_exemplar="BSJ-03-123", n_surface_lysines=8,
            degradability_score=0.7,
        )
        ppi = PPIAccessibility(
            gene="CDK4", n_pdb_complexes=5, has_interface_data=True,
            disordered_fraction=0.08, ppi_score=1.0,
        )
        score = ProteinDruggabilityScore(
            gene="CDK4", structural=structural, abundance=None,
            degradability=degradability, ppi=ppi,
            protein_score=0.8, blended_score=0.76,
            gene_druggability=0.7,
        )
        d = score.to_dict()
        assert d["gene"] == "CDK4"
        assert "structural" in d
        assert "abundance" not in d  # None → omitted
        assert d["protein_score"] == 0.8
        # Serializable to JSON
        json_str = json.dumps(d)
        assert "CDK4" in json_str
