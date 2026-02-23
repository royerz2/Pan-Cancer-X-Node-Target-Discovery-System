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
    RNAExpression,
    RNAProteinConcordance,
    StructuralDruggability,
    compute_abundance_score,
    compute_degradability_score,
    compute_ppi_score,
    compute_rna_expression_score,
    compute_rna_protein_concordance,
    compute_structural_score,
    generate_protein_scoring_report,
    load_protacdb,
    load_gygi_correlations,
    load_gygi_replicate_cv,
    load_gygi_raw_replicate_cv,
    load_gygi_mutations,
    _load_proteomics_uniprot_map,
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
# RNA EXPRESSION SCORING TESTS
# ============================================================================

class TestRNAExpressionScoring:
    @pytest.fixture
    def rnaseq_data(self):
        """Fake CCLE RNA-seq DataFrame (log2(TPM+1) values)."""
        genes = ["EGFR", "STAT3", "CDK4", "SILENTGENE"]
        cell_lines = [f"ACH-{i:04d}" for i in range(30)]
        rng = np.random.default_rng(99)
        data = rng.normal(loc=4.0, scale=1.0, size=(30, 4))
        data[:, 3] = 0.0  # SILENTGENE not expressed
        df = pd.DataFrame(data, index=cell_lines, columns=genes)
        cancer_map = {cl: "Lung" for cl in cell_lines[:20]}
        cancer_map.update({cl: "Breast" for cl in cell_lines[20:]})
        return df, cancer_map

    def test_high_expression(self, rnaseq_data):
        df, cmap = rnaseq_data
        result = compute_rna_expression_score("EGFR", "Lung", df, cmap)
        assert result is not None
        assert result.expression_score >= 0.6
        assert result.expression_fraction > 0.5
        assert result.n_cell_lines > 0

    def test_silent_gene(self, rnaseq_data):
        df, cmap = rnaseq_data
        result = compute_rna_expression_score("SILENTGENE", "Lung", df, cmap)
        assert result is not None
        assert result.expression_score <= 0.3
        assert result.expression_fraction < 0.2

    def test_no_data(self):
        result = compute_rna_expression_score("EGFR", "Lung", None, None)
        assert result is None

    def test_gene_not_in_dataframe(self, rnaseq_data):
        df, cmap = rnaseq_data
        result = compute_rna_expression_score("NONEXISTENT", "Lung", df, cmap)
        assert result is None

    def test_cancer_type_not_found(self, rnaseq_data):
        df, cmap = rnaseq_data
        result = compute_rna_expression_score("EGFR", "NoSuchCancer", df, cmap)
        assert result is None


# ============================================================================
# RNA/PROTEIN CONCORDANCE TESTS
# ============================================================================

class TestRNAProteinConcordance:
    @pytest.fixture
    def matched_data(self):
        """Create RNA + protein DataFrames with overlapping cell lines."""
        rng = np.random.default_rng(77)
        cell_lines = [f"ACH-{i:04d}" for i in range(20)]
        # Correlated RNA and protein for EGFR
        rna_vals = rng.normal(4.0, 1.0, 20)
        prot_vals = rna_vals * 0.8 + rng.normal(0, 0.2, 20)  # high correlation
        rna_df = pd.DataFrame({"EGFR": rna_vals, "MYC": rng.normal(3.0, 2.0, 20)},
                              index=cell_lines)
        prot_df = pd.DataFrame({"EGFR": prot_vals, "MYC": rng.uniform(-1, 1, 20)},
                               index=cell_lines)
        return rna_df, prot_df

    def test_high_concordance(self, matched_data):
        rna_df, prot_df = matched_data
        result = compute_rna_protein_concordance("EGFR", rna_df, prot_df)
        assert result is not None
        assert result.spearman_rho > 0.5
        assert result.concordance_tier == "high"
        assert result.concordance_score == 1.0

    def test_low_concordance(self, matched_data):
        rna_df, prot_df = matched_data
        result = compute_rna_protein_concordance("MYC", rna_df, prot_df)
        assert result is not None
        # MYC has random protein values → low/moderate concordance
        assert result.concordance_tier in ("low", "moderate")

    def test_no_rna_data(self, matched_data):
        _, prot_df = matched_data
        result = compute_rna_protein_concordance("EGFR", None, prot_df)
        assert result is None

    def test_no_protein_data(self, matched_data):
        rna_df, _ = matched_data
        result = compute_rna_protein_concordance("EGFR", rna_df, None)
        assert result is None

    def test_gene_missing_from_one(self, matched_data):
        rna_df, prot_df = matched_data
        result = compute_rna_protein_concordance("NONEXISTENT", rna_df, prot_df)
        assert result is None

    def test_too_few_cell_lines(self):
        """Fewer than CONCORDANCE_MIN_CELL_LINES → insufficient."""
        rng = np.random.default_rng(42)
        cells = [f"ACH-{i}" for i in range(5)]
        rna = pd.DataFrame({"EGFR": rng.normal(4, 1, 5)}, index=cells)
        prot = pd.DataFrame({"EGFR": rng.normal(1, 0.5, 5)}, index=cells)
        result = compute_rna_protein_concordance("EGFR", rna, prot)
        assert result is not None
        assert result.concordance_tier == "insufficient"
        assert result.concordance_score == 0.0


# ============================================================================
# PROTAC-DB LOADER TESTS
# ============================================================================

class TestPROTACDBLoader:
    def test_load_protacdb_no_file(self, tmp_path):
        """Without a PROTAC-DB CSV, should return built-in targets."""
        result = load_protacdb(str(tmp_path))
        assert isinstance(result, dict)
        # Should have at least the built-in curated targets
        assert "BRD4" in result or "STAT3" in result

    def test_load_protacdb_with_csv(self, tmp_path):
        """With a PROTAC-DB CSV, should parse and merge."""
        csv_content = (
            "Target_Name,PROTAC_Name,Clinical_Phase,E3_Ligase,DC50_nM\n"
            "NOVELGENE,PROTAC-NOVEL1,Phase I,CRBN,5.0\n"
            "NOVELGENE,PROTAC-NOVEL2,Preclinical,VHL,10.0\n"
            "BRD4,ARV-771,Phase II,CRBN,1.0\n"
        )
        (tmp_path / "protac_data.csv").write_text(csv_content)
        result = load_protacdb(str(tmp_path))

        assert "NOVELGENE" in result
        assert result["NOVELGENE"]["status"] == "phase1"
        assert result["NOVELGENE"]["exemplar"] == "PROTAC-NOVEL1"
        # BRD4 should be upgraded if Phase II > built-in status
        assert "BRD4" in result

    def test_degradability_with_custom_targets(self, mock_alphafold):
        """compute_degradability_score uses custom degrader targets."""
        custom = {"TESTGENE": {"status": "approved", "exemplar": "DRUG-1", "source": "test"}}
        result = compute_degradability_score("TESTGENE", "Q99999", mock_alphafold, custom)
        assert result.has_known_degrader is True
        assert result.degradability_score == 0.9


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
        """Verify adaptive blending: geometric mean when discordant."""
        scorer = ProteinDruggabilityScorer(
            genes=["FAKE"],
            gene_druggability_fn=lambda g: 0.8,
            alpha=0.7,
            cache_dir=str(tmp_cache.cache_dir),
        )
        result = scorer._fallback_score("FAKE")
        # gene=0.8, protein=0.3, discordance=0.5 > 0.4 → geometric mean
        # protein < 0.35 → alpha = min(0.75, 0.7 + 0.05) = 0.75
        # blended = 0.8^0.75 * 0.3^0.25 ≈ 0.623
        assert 0.55 < result.blended_score < 0.70
        # Key property: with old arithmetic (0.7*0.8 + 0.3*0.3 = 0.65),
        # geometric mean pulls it DOWN because protein is low.
        assert result.blended_score < 0.65


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
            rna_expression=None, rna_protein_concordance=None,
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
            rna_expression=None, rna_protein_concordance=None,
            protein_score=0.8, blended_score=0.76,
            gene_druggability=0.7,
        )
        d = score.to_dict()
        assert d["gene"] == "CDK4"
        assert "structural" in d
        assert "abundance" not in d  # None → omitted


# ============================================================================
# V2 SCORING MECHANISM TESTS
# ============================================================================

class TestSigmoidAbundance:
    """Tests for continuous sigmoid abundance scoring (Fix 1)."""

    def test_high_detection_gives_high_score(self):
        from alin.protein_scoring import _sigmoid_abundance
        # det_frac=0.9 → sigmoid near 1.0
        score = _sigmoid_abundance(0.9, 1.0)
        assert score > 0.9

    def test_low_detection_gives_low_score(self):
        from alin.protein_scoring import _sigmoid_abundance
        # det_frac=0.1 → sigmoid near 0
        score = _sigmoid_abundance(0.1, 0.5)
        assert score < 0.15

    def test_monotonic(self):
        from alin.protein_scoring import _sigmoid_abundance
        # Score must increase monotonically with detection fraction
        scores = [_sigmoid_abundance(f / 10, 0.5) for f in range(1, 10)]
        for i in range(len(scores) - 1):
            assert scores[i] <= scores[i + 1] + 1e-9

    def test_intensity_bonus(self):
        from alin.protein_scoring import _sigmoid_abundance
        # Same detection, higher abundance → higher score
        low = _sigmoid_abundance(0.5, 0.2)
        high = _sigmoid_abundance(0.5, 2.0)
        assert high > low

    def test_bounded(self):
        from alin.protein_scoring import _sigmoid_abundance
        for f in [0.0, 0.5, 1.0]:
            for m in [0.0, 1.0, 5.0]:
                s = _sigmoid_abundance(f, m)
                assert 0.0 <= s <= 1.0


class TestRNAImputation:
    """Tests for RNA-based abundance imputation (Fix 1)."""

    def test_imputation_rescues_sparse_proteomics(self, proteomics_data):
        """When protein detection is sparse but RNA is high, score should increase."""
        df, cmap = proteomics_data
        # Make gene almost undetectable in proteomics
        df["SPARSE_GENE"] = -0.5  # below threshold
        df.iloc[0, df.columns.get_loc("SPARSE_GENE")] = 0.5  # only 1 detected

        # Without RNA imputation
        score_no_rna = compute_abundance_score(
            "SPARSE_GENE", "Lung", df, cmap,
        )

        # With RNA imputation (high expression, high concordance)
        rna_expr = RNAExpression(
            gene="SPARSE_GENE", cancer_type="Lung",
            n_cell_lines=15, n_expressed=14,
            mean_expression=5.0, expression_fraction=0.93,
            expression_score=1.0,
        )
        concordance = RNAProteinConcordance(
            gene="SPARSE_GENE", n_matched_lines=100,
            spearman_rho=0.6, spearman_pvalue=1e-10,
            concordance_tier='high', concordance_score=1.0,
        )
        score_with_rna = compute_abundance_score(
            "SPARSE_GENE", "Lung", df, cmap,
            rna_expression=rna_expr, concordance=concordance,
        )

        assert score_with_rna is not None
        assert score_no_rna is not None
        # Imputation should increase the score
        assert score_with_rna.abundance_score > score_no_rna.abundance_score
        assert score_with_rna.rna_imputed is True
        assert score_no_rna.rna_imputed is False

    def test_no_imputation_when_rna_low(self, proteomics_data):
        """RNA imputation should NOT happen when RNA expression is low."""
        df, cmap = proteomics_data
        df["SPARSE2"] = -0.5
        df.iloc[0, df.columns.get_loc("SPARSE2")] = 0.5

        rna_expr = RNAExpression(
            gene="SPARSE2", cancer_type="Lung",
            n_cell_lines=15, n_expressed=2,
            mean_expression=0.5, expression_fraction=0.13,
            expression_score=0.1,  # below 0.6 threshold
        )
        concordance = RNAProteinConcordance(
            gene="SPARSE2", n_matched_lines=100,
            spearman_rho=0.6, spearman_pvalue=1e-10,
            concordance_tier='high', concordance_score=1.0,
        )
        score = compute_abundance_score(
            "SPARSE2", "Lung", df, cmap,
            rna_expression=rna_expr, concordance=concordance,
        )
        assert score is not None
        assert score.rna_imputed is False

    def test_confidence_weight_increases_with_samples(self, proteomics_data):
        """More detected samples → higher confidence."""
        df, cmap = proteomics_data
        # EGFR is well-expressed in the fake data (many detected)
        score = compute_abundance_score("EGFR", "Lung", df, cmap)
        assert score is not None
        assert score.confidence_weight > 0.5  # at least moderate confidence


class TestAdaptiveBlending:
    """Tests for adaptive blending with discordance penalty (Fix 3)."""

    def test_concordant_scores_use_arithmetic(self):
        from alin.protein_scoring import _adaptive_blend
        # gene=0.7, protein=0.6: discordance=0.1 < 0.4 → arithmetic
        result = _adaptive_blend(0.7, 0.6)
        # protein=0.6, not > 0.7 and not < 0.35 → alpha=0.6
        expected = 0.6 * 0.7 + 0.4 * 0.6  # = 0.66
        assert abs(result - expected) < 0.01

    def test_discordant_scores_use_geometric(self):
        from alin.protein_scoring import _adaptive_blend
        # gene=1.0, protein=0.2: discordance=0.8 > 0.4 → geometric
        result = _adaptive_blend(1.0, 0.2)
        # Geometric mean with adaptive alpha pulls result DOWN sharply
        arithmetic = 0.6 * 1.0 + 0.4 * 0.2  # = 0.68 old
        assert result < arithmetic  # geometric < arithmetic when discordant

    def test_low_protein_veto(self):
        from alin.protein_scoring import _adaptive_blend
        # gene=1.0, protein=0.1: veto caps blended at 0.5
        result = _adaptive_blend(1.0, 0.1)
        assert result <= 0.5

    def test_high_protein_increases_protein_weight(self):
        from alin.protein_scoring import _adaptive_blend
        # protein=0.9: should increase protein weight (lower alpha)
        result = _adaptive_blend(0.5, 0.9)
        # With lower alpha, protein 0.9 pulls UP more
        high_protein = result
        # Compare with mid-protein (alpha stays at 0.6)
        mid_result = _adaptive_blend(0.5, 0.55)
        # High protein result should be noticeably above mid-range
        assert high_protein > 0.65

    def test_bounded_output(self):
        from alin.protein_scoring import _adaptive_blend
        for g in [0.0, 0.2, 0.5, 0.8, 1.0]:
            for p in [0.0, 0.2, 0.5, 0.8, 1.0]:
                r = _adaptive_blend(g, p)
                assert 0.0 <= r <= 1.0

    def test_ros1_like_case(self):
        """ROS1 scenario: high gene druggability, very low protein."""
        from alin.protein_scoring import _adaptive_blend
        # gene=1.0 (approved drug), protein=0.4 (barely detectable in PDAC)
        result = _adaptive_blend(1.0, 0.4)
        # Old system: 0.6*1.0 + 0.4*0.4 = 0.76
        # New system should pull this down with geometric mean
        old_arithmetic = 0.6 * 1.0 + 0.4 * 0.4
        assert result < old_arithmetic


class TestCalibration:
    """Tests for weight calibration function (Fix 2)."""

    def test_calibration_basic(self):
        """Calibration runs and returns expected keys."""
        from alin.protein_scoring import calibrate_layer_weights

        # Build minimal mock results
        mock_results = {}
        for i, gene in enumerate(['GENE_A', 'GENE_B', 'GENE_C', 'GENE_D',
                                    'GENE_E', 'GENE_F', 'GENE_G', 'GENE_H',
                                    'GENE_I', 'GENE_J', 'GENE_K', 'GENE_L']):
            gene_d = 0.9 if i < 6 else 0.1
            struct = StructuralDruggability(
                gene=gene, uniprot_id="", mean_plddt=80.0,
                domain_plddt={}, n_pdb_structures=10+i,
                n_ligand_bound=5+i, has_pocket=True,
                structural_score=0.5 + (0.05 * i if i < 6 else 0.0),
            )
            degrad = DegradabilityScore(
                gene=gene, has_known_degrader=(i < 6),
                degrader_status="preclinical" if i < 6 else "none",
                degrader_exemplar="", n_surface_lysines=5,
                degradability_score=0.7 if i < 6 else 0.2,
            )
            ppi_sc = PPIAccessibility(
                gene=gene, n_pdb_complexes=5+i,
                has_interface_data=True, disordered_fraction=0.1,
                ppi_score=0.8 if i < 6 else 0.3,
            )
            rna = RNAExpression(
                gene=gene, cancer_type="Test",
                n_cell_lines=20, n_expressed=18,
                mean_expression=5.0, expression_fraction=0.9,
                expression_score=0.9 if i < 6 else 0.4,
            )
            conc = RNAProteinConcordance(
                gene=gene, n_matched_lines=50,
                spearman_rho=0.5, spearman_pvalue=0.001,
                concordance_tier='high', concordance_score=1.0,
            )
            abd = ProteinAbundance(
                gene=gene, cancer_type="Test",
                n_cell_lines=20, n_detected=15,
                mean_abundance=1.0, detection_fraction=0.75,
                abundance_score=0.7 if i < 6 else 0.3,
            )
            mock_results[gene] = ProteinDruggabilityScore(
                gene=gene, structural=struct, abundance=abd,
                degradability=degrad, ppi=ppi_sc,
                rna_expression=rna, rna_protein_concordance=conc,
                protein_score=0.7 if i < 6 else 0.3,
                blended_score=0.7, gene_druggability=gene_d,
            )

        result = calibrate_layer_weights(mock_results, n_bootstrap=50)
        assert 'nominal_auc' in result
        assert 'optimized_auc' in result
        assert 'rank_stability_mean_rho' in result
        assert 'weight_sensitivity' in result
        assert result['optimized_auc'] >= result['nominal_auc'] - 0.01  # should not degrade much


# ============================================================================
# DYNAMIC UNIPROT RESOLVER TESTS
# ============================================================================

class TestUniProtResolver:
    """Tests for the dynamic UniProt ID resolution system."""

    def test_static_dict_resolution(self):
        """Known genes in GENE_TO_UNIPROT resolve without API call."""
        from alin.protein_scoring import resolve_uniprot_id, GENE_TO_UNIPROT
        # EGFR is definitely in the static dict
        uid = resolve_uniprot_id('EGFR')
        assert uid == GENE_TO_UNIPROT['EGFR']
        assert uid == 'P00533'

    def test_static_dict_all_79_genes(self):
        """All 79 signaling genes in static dict should resolve."""
        from alin.protein_scoring import resolve_uniprot_id, GENE_TO_UNIPROT
        for gene, expected in GENE_TO_UNIPROT.items():
            uid = resolve_uniprot_id(gene)
            assert uid == expected, f"{gene}: expected {expected}, got {uid}"

    def test_dynamic_resolution_common_gene(self):
        """SOX9 (a known cancer gene not in static dict) resolves via API."""
        from alin.protein_scoring import resolve_uniprot_id, GENE_TO_UNIPROT
        assert 'SOX9' not in GENE_TO_UNIPROT  # not in static dict
        uid = resolve_uniprot_id('SOX9')
        # SOX9 Swiss-Prot accession is P48436
        assert uid == 'P48436', f"SOX9 expected P48436, got {uid}"

    def test_dynamic_resolution_with_cache(self):
        """Resolved IDs are cached and reused."""
        from alin.protein_scoring import (
            resolve_uniprot_id, _DYNAMIC_UNIPROT_CACHE, ProteinAPICache
        )
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ProteinAPICache(cache_dir=tmpdir)
            # First call: queries API
            uid1 = resolve_uniprot_id('ACLY', cache)
            assert uid1 is not None  # ACLY has Swiss-Prot entry P53396
            # Second call: should come from memory cache
            uid2 = resolve_uniprot_id('ACLY', cache)
            assert uid1 == uid2

    def test_nonsense_gene_returns_none(self):
        """A nonsense gene name returns None."""
        from alin.protein_scoring import resolve_uniprot_id
        uid = resolve_uniprot_id('ZZZZNOTAREALGENE12345')
        assert uid is None

    def test_unresolved_tracking(self):
        """Unresolved genes are tracked in _UNRESOLVED_GENES."""
        from alin.protein_scoring import resolve_uniprot_id, get_unresolved_genes
        fake_gene = 'FAKEGENE_TEST_XYZZY'
        resolve_uniprot_id(fake_gene)
        assert fake_gene in get_unresolved_genes()

    def test_batch_pre_resolve(self):
        """ProteinDruggabilityScorer.pre_resolve_genes resolves a batch."""
        from alin.protein_scoring import ProteinDruggabilityScorer
        scorer = ProteinDruggabilityScorer(
            genes=['EGFR', 'KRAS'],
            gene_druggability_fn=lambda g: 0.5,
        )
        result = scorer.pre_resolve_genes(['EGFR', 'SOX9', 'NOTAREALGENE'])
        assert result['EGFR'] == 'P00533'
        assert result['SOX9'] is not None  # Should resolve via API
        assert result['NOTAREALGENE'] is None  # Truly unmappable


# ============================================================================
# V3 FEATURES: PROTEOMICS UNIPROT MAP, GYGI DATA, PARTIAL SCORING
# ============================================================================

class TestProteomicsUniProtMap:
    """Test the proteomics-file-based UniProt resolver."""

    def test_load_from_proteomics_file(self, tmp_path):
        """Should extract Gene_Symbol → Uniprot_Acc from CSV."""
        csv_path = tmp_path / "protein_quant_current_normalized.csv"
        csv_path.write_text(
            "Gene_Symbol,Uniprot_Acc,CL1_TISSUE\n"
            "EGFR,P00533,1.5\n"
            "STAT3,P40763,2.1\n"
            "MYGENE,Q99999,0.5\n"
        )
        # Reset the global state for clean test
        import alin.protein_scoring as ps
        old_loaded = ps._PROTEOMICS_UNIPROT_LOADED
        old_map = ps._PROTEOMICS_UNIPROT_MAP.copy()
        try:
            ps._PROTEOMICS_UNIPROT_LOADED = False
            ps._PROTEOMICS_UNIPROT_MAP = {}
            result = _load_proteomics_uniprot_map(str(tmp_path))
            assert len(result) == 3
            assert result['EGFR'] == 'P00533'
            assert result['MYGENE'] == 'Q99999'
        finally:
            ps._PROTEOMICS_UNIPROT_LOADED = old_loaded
            ps._PROTEOMICS_UNIPROT_MAP = old_map

    def test_isoform_suffix_stripped(self, tmp_path):
        """Isoform suffixes (e.g. P21675-9) should be stripped to canonical."""
        csv_path = tmp_path / "protein_quant_current_normalized.csv"
        csv_path.write_text(
            "Gene_Symbol,Uniprot_Acc,CL1_TISSUE\n"
            "NF1,P21675-9,1.0\n"
            "BRCA1,P38398,2.0\n"
            "MAPK1,P28482-2,1.5\n"
        )
        import alin.protein_scoring as ps
        old_loaded = ps._PROTEOMICS_UNIPROT_LOADED
        old_map = ps._PROTEOMICS_UNIPROT_MAP.copy()
        try:
            ps._PROTEOMICS_UNIPROT_LOADED = False
            ps._PROTEOMICS_UNIPROT_MAP = {}
            result = _load_proteomics_uniprot_map(str(tmp_path))
            assert result['NF1'] == 'P21675', f"Expected P21675, got {result['NF1']}"
            assert result['BRCA1'] == 'P38398'    # no suffix, unchanged
            assert result['MAPK1'] == 'P28482', f"Expected P28482, got {result['MAPK1']}"
        finally:
            ps._PROTEOMICS_UNIPROT_LOADED = old_loaded
            ps._PROTEOMICS_UNIPROT_MAP = old_map

    def test_empty_dir(self, tmp_path):
        """Should return empty dict when no file exists."""
        import alin.protein_scoring as ps
        old_loaded = ps._PROTEOMICS_UNIPROT_LOADED
        old_map = ps._PROTEOMICS_UNIPROT_MAP.copy()
        try:
            ps._PROTEOMICS_UNIPROT_LOADED = False
            ps._PROTEOMICS_UNIPROT_MAP = {}
            result = _load_proteomics_uniprot_map(str(tmp_path))
            assert result == {}
        finally:
            ps._PROTEOMICS_UNIPROT_LOADED = old_loaded
            ps._PROTEOMICS_UNIPROT_MAP = old_map


class TestGygiDataLoaders:
    """Test the three Gygi lab data loaders."""

    def test_load_correlations(self, tmp_path):
        """Should parse Table S4 format with Spearman/Pearson columns."""
        xlsx_path = tmp_path / "Table_S4_Protein_RNA_Correlation_and_Enrichments.xlsx"
        df = pd.DataFrame({
            'Gene Symbol': ['EGFR', 'STAT3', 'CDK4'],
            'Pearson': [0.7, 0.4, 0.2],
            'Spearman': [0.65, 0.35, 0.15],
        })
        with pd.ExcelWriter(str(xlsx_path), engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Protein RNA Correlation', index=False)

        result = load_gygi_correlations(str(tmp_path))
        assert result is not None
        assert len(result) == 3
        assert abs(result['EGFR']['spearman'] - 0.65) < 1e-6
        assert abs(result['STAT3']['pearson'] - 0.4) < 1e-6

    def test_load_correlations_missing_file(self, tmp_path):
        """Should return None when file is missing."""
        result = load_gygi_correlations(str(tmp_path))
        assert result is None

    def test_load_replicate_cv(self, tmp_path):
        """Should compute per-gene CV from triplicate data."""
        xlsx_path = tmp_path / "Table_S3_Biological_Replicates_Protein_Quant_Normalized.xlsx"
        # 2 genes × 2 cell lines × 3 replicates
        df = pd.DataFrame({
            'Gene_Symbol': ['EGFR', 'STAT3'],
            'CL1_R1': [1.0, 2.0],
            'CL1_R2': [1.1, 2.2],
            'CL1_R3': [0.9, 1.8],
            'CL2_R1': [2.0, 3.0],
            'CL2_R2': [2.1, 3.1],
            'CL2_R3': [1.9, 2.9],
        })
        with pd.ExcelWriter(str(xlsx_path), engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Sheet1', index=False)

        result = load_gygi_replicate_cv(str(tmp_path))
        assert result is not None
        assert 'EGFR' in result
        assert result['EGFR'] > 0  # Should have non-zero CV

    def test_load_replicate_cv_missing(self, tmp_path):
        """Should return None when file is missing."""
        result = load_gygi_replicate_cv(str(tmp_path))
        assert result is None

    def test_load_raw_replicate_cv(self, tmp_path):
        """Should compute per-gene CV from non-normalized replicate data."""
        csv_path = tmp_path / "ccle_biological_replicates_nonnormalized.csv"
        # 2 genes × 2 cell lines × R1/R2/R3 per cell line
        df = pd.DataFrame({
            'Protein.Id': ['sp|P00533|EGFR_HUMAN', 'sp|P04049|BRAF_HUMAN'],
            'Gene.Symbol': ['EGFR', 'BRAF'],
            'Description': ['EGFR_HUMAN', 'BRAF_HUMAN'],
            'TenPx01.R1.Peptides': [10, 15],
            'TenPx01.R2.Peptides': [11, 14],
            'TenPx01.R3.Peptides': [9, 16],
            'CL1_BREAST.TenPx01.R1': [1000.0, 2000.0],
            'CL1_BREAST.TenPx01.R2': [1100.0, 2200.0],
            'CL1_BREAST.TenPx01.R3': [900.0, 1800.0],
            'CL2_LUNG.TenPx02.R1': [2000.0, 3000.0],
            'CL2_LUNG.TenPx02.R2': [2100.0, 3100.0],
            'CL2_LUNG.TenPx02.R3': [1900.0, 2900.0],
            'bridge.TenPx01.R1': [500.0, 800.0],
            'bridge.TenPx01.R2': [510.0, 810.0],
            'bridge.TenPx01.R3': [490.0, 790.0],
        })
        df.to_csv(str(csv_path), index=False)

        result = load_gygi_raw_replicate_cv(str(tmp_path))
        assert result is not None
        assert 'EGFR' in result
        assert 'BRAF' in result
        assert result['EGFR'] > 0
        # Bridge samples should be excluded from cell-line CV computation
        # (only CL1_BREAST and CL2_LUNG should be used)

    def test_load_raw_replicate_cv_missing(self, tmp_path):
        """Should return None when file is missing."""
        result = load_gygi_raw_replicate_cv(str(tmp_path))
        assert result is None

    def test_load_mutations(self, tmp_path):
        """Should parse Table S7 mutation associations."""
        xlsx_path = tmp_path / "Table_S7_Mutation_Associations.xlsx"
        df = pd.DataFrame({
            'Mutant Gene': ['KRAS', 'KRAS', 'TP53'],
            'Gene Symbol': ['KRAS', 'MDM2', 'TP53'],
            'Coefficient': [1.5, -0.8, 0.3],
            'P-Value': [0.001, 0.01, 0.5],
            'FDR': [0.005, 0.05, 0.8],
            'LFDR': [0.01, 0.03, 0.7],
        })
        with pd.ExcelWriter(str(xlsx_path), engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Sheet1', index=False)

        result = load_gygi_mutations(str(tmp_path))
        assert result is not None
        assert 'KRAS' in result
        assert len(result['KRAS']) >= 1

    def test_load_mutations_missing(self, tmp_path):
        """Should return None when file is missing."""
        result = load_gygi_mutations(str(tmp_path))
        assert result is None


class TestPartialScoring:
    """Test v3 partial scoring for genes without UniProt IDs."""

    def _make_scorer(self, genes=None, cancer_type='Lung'):
        """Create a scorer with mocked data for partial scoring tests."""
        if genes is None:
            genes = ['EGFR', 'SOX9', 'FAKEGENE123']
        scorer = ProteinDruggabilityScorer(
            genes=genes,
            cancer_type=cancer_type,
            gene_druggability_fn=lambda g: 0.5,
        )
        return scorer

    @patch('alin.protein_scoring.resolve_uniprot_id')
    def test_no_uniprot_still_scores(self, mock_resolve):
        """Gene without UniProt should get non-flat scores from RNA/proteomics."""
        mock_resolve.return_value = None
        scorer = self._make_scorer(genes=['FAKEGENE'])
        # Inject mock data
        scorer._proteomics_loaded = True
        scorer._expression_loaded = True
        scorer._gygi_correlations_loaded = True
        scorer._gygi_replicate_cv_loaded = True
        scorer._gygi_mutations_loaded = True
        scorer._degrader_targets = {}

        result = scorer.score_gene('FAKEGENE')
        # Should get a score even without UniProt
        assert result is not None
        assert result.gene == 'FAKEGENE'
        # Structural and PPI should be defaults (0.3)
        assert result.structural.structural_score == 0.3
        assert result.ppi.ppi_score == 0.3
        # But we should still get a degradability score
        assert result.degradability is not None

    @patch('alin.protein_scoring.resolve_uniprot_id')
    def test_partial_with_known_degrader(self, mock_resolve):
        """Gene without UniProt but with known degrader should score higher."""
        mock_resolve.return_value = None
        scorer = self._make_scorer(genes=['BRD4'])
        scorer._proteomics_loaded = True
        scorer._expression_loaded = True
        scorer._gygi_correlations_loaded = True
        scorer._gygi_replicate_cv_loaded = True
        scorer._gygi_mutations_loaded = True
        scorer._degrader_targets = KNOWN_DEGRADER_TARGETS.copy()

        result = scorer.score_gene('BRD4')
        # BRD4 has a known phase1 degrader (ARV-825)
        assert result.degradability.has_known_degrader is True
        assert result.degradability.degradability_score >= 0.7

    @patch('alin.protein_scoring.resolve_uniprot_id')
    def test_partial_with_mutation_boost(self, mock_resolve):
        """Mutation associations should boost degradability."""
        mock_resolve.return_value = None
        scorer = self._make_scorer(genes=['TESTGENE'])
        scorer._proteomics_loaded = True
        scorer._expression_loaded = True
        scorer._gygi_correlations_loaded = True
        scorer._gygi_replicate_cv_loaded = True
        scorer._gygi_mutations_loaded = True
        scorer._gygi_mutations = {
            'TESTGENE': [
                {'mutant_gene': 'KRAS', 'coefficient': 2.0, 'fdr': 0.01},
            ]
        }
        scorer._degrader_targets = {}

        result = scorer.score_gene('TESTGENE')
        # Should get mutation boost (base 0.3 + boost)
        assert result.degradability.degradability_score > 0.3

    @patch('alin.protein_scoring.resolve_uniprot_id')
    def test_partial_structural_ppi_reduced_confidence(self, mock_resolve):
        """Without UniProt, structural/PPI should have reduced confidence."""
        mock_resolve.return_value = None
        scorer = self._make_scorer(genes=['NOPROTEIN'])
        scorer._proteomics_loaded = True
        scorer._expression_loaded = True
        scorer._gygi_correlations_loaded = True
        scorer._gygi_replicate_cv_loaded = True
        scorer._gygi_mutations_loaded = True
        scorer._degrader_targets = {}

        result = scorer.score_gene('NOPROTEIN')
        # Should produce a valid score
        assert 0.0 <= result.protein_score <= 1.0
        assert 0.0 <= result.blended_score <= 1.0


class TestLabGradeConcordance:
    """Test integration of Gygi Table S4 lab correlations."""

    @patch('alin.protein_scoring.resolve_uniprot_id')
    def test_lab_concordance_used_when_available(self, mock_resolve):
        """Lab-grade Spearman ρ should be used instead of computed value."""
        mock_resolve.return_value = None
        scorer = ProteinDruggabilityScorer(
            genes=['MYGENE'],
            cancer_type='Lung',
            gene_druggability_fn=lambda g: 0.5,
        )
        scorer._proteomics_loaded = True
        scorer._expression_loaded = True
        scorer._gygi_replicate_cv_loaded = True
        scorer._gygi_mutations_loaded = True
        scorer._degrader_targets = {}
        # Inject lab correlations
        scorer._gygi_correlations_loaded = True
        scorer._gygi_correlations = {
            'MYGENE': {'spearman': 0.72, 'pearson': 0.68}
        }

        result = scorer.score_gene('MYGENE')
        # Should use lab concordance
        assert result.rna_protein_concordance is not None
        assert abs(result.rna_protein_concordance.spearman_rho - 0.72) < 1e-4
        assert result.rna_protein_concordance.concordance_tier == 'high'
        assert result.rna_protein_concordance.n_matched_lines == 375

    @patch('alin.protein_scoring.resolve_uniprot_id')
    def test_lab_concordance_low(self, mock_resolve):
        """Low Spearman ρ from lab should give low tier."""
        mock_resolve.return_value = None
        scorer = ProteinDruggabilityScorer(
            genes=['LOWGENE'],
            cancer_type='Lung',
            gene_druggability_fn=lambda g: 0.5,
        )
        scorer._proteomics_loaded = True
        scorer._expression_loaded = True
        scorer._gygi_replicate_cv_loaded = True
        scorer._gygi_mutations_loaded = True
        scorer._degrader_targets = {}
        scorer._gygi_correlations_loaded = True
        scorer._gygi_correlations = {
            'LOWGENE': {'spearman': 0.15}
        }

        result = scorer.score_gene('LOWGENE')
        assert result.rna_protein_concordance.concordance_tier == 'low'
        assert result.rna_protein_concordance.concordance_score == 0.3


class TestReplicateConfidence:
    """Test integration of Gygi Table S3 replicate CV."""

    @patch('alin.protein_scoring.resolve_uniprot_id')
    @patch('alin.protein_scoring.compute_abundance_score')
    def test_high_cv_reduces_confidence(self, mock_abundance, mock_resolve):
        """High replicate CV should reduce abundance confidence weight."""
        mock_resolve.return_value = None
        mock_abundance.return_value = ProteinAbundance(
            gene='TESTGENE', cancer_type='Lung',
            n_cell_lines=20, n_detected=15,
            mean_abundance=2.0, detection_fraction=0.75,
            abundance_score=0.8, confidence_weight=1.0,
        )

        scorer = ProteinDruggabilityScorer(
            genes=['TESTGENE'],
            cancer_type='Lung',
            gene_druggability_fn=lambda g: 0.5,
        )
        scorer._proteomics_loaded = True
        scorer._expression_loaded = True
        scorer._gygi_correlations_loaded = True
        scorer._gygi_mutations_loaded = True
        scorer._degrader_targets = {}
        # High CV = noisy measurement
        scorer._gygi_replicate_cv_loaded = True
        scorer._gygi_replicate_cv = {'TESTGENE': 1.0}

        result = scorer.score_gene('TESTGENE')
        # With CV=1.0, rep_confidence = 0.5, combined with base 1.0:
        # sqrt(1.0 * 0.5) ≈ 0.707
        if result.abundance is not None:
            assert result.abundance.confidence_weight < 1.0

    @patch('alin.protein_scoring.resolve_uniprot_id')
    @patch('alin.protein_scoring.compute_abundance_score')
    def test_zero_cv_full_confidence(self, mock_abundance, mock_resolve):
        """Zero replicate CV should maintain full confidence."""
        mock_resolve.return_value = None
        mock_abundance.return_value = ProteinAbundance(
            gene='PRECISE', cancer_type='Lung',
            n_cell_lines=20, n_detected=15,
            mean_abundance=2.0, detection_fraction=0.75,
            abundance_score=0.8, confidence_weight=1.0,
        )

        scorer = ProteinDruggabilityScorer(
            genes=['PRECISE'],
            cancer_type='Lung',
            gene_druggability_fn=lambda g: 0.5,
        )
        scorer._proteomics_loaded = True
        scorer._expression_loaded = True
        scorer._gygi_correlations_loaded = True
        scorer._gygi_mutations_loaded = True
        scorer._degrader_targets = {}
        scorer._gygi_replicate_cv_loaded = True
        scorer._gygi_replicate_cv = {'PRECISE': 0.0}

        result = scorer.score_gene('PRECISE')
        if result.abundance is not None:
            assert result.abundance.confidence_weight == 1.0
