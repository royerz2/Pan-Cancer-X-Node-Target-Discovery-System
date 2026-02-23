#!/usr/bin/env python3
"""
Tests for the LINCS L1000 integration module (alin.lincs).

All tests use synthetic in-memory data — no real GCTX files needed.
"""

import os
import sys
import tempfile
import textwrap

import numpy as np
import pandas as pd
import pytest

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alin.lincs import (
    ConsensusSignature,
    LINCSSignatureDB,
    _compute_confidence,
    compute_consensus,
    lincs_available,
    load_gene_info,
    load_sig_info,
)
from alin.perturbation import PerturbationSignature


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def gene_id_to_symbol():
    """Simple gene_id → symbol mapping."""
    return {
        1: "EGFR",
        2: "KRAS",
        3: "BRAF",
        4: "MYC",
        5: "TP53",
        6: "AKT1",
        7: "MTOR",
        8: "CDK4",
    }


@pytest.fixture
def z_matrix():
    """
    Synthetic z-score matrix: 8 genes × 6 signatures.

    Gene 1 (EGFR) → consistently down (z ≈ -3)
    Gene 4 (MYC)  → consistently up (z ≈ +2.5)
    Others → mixed / near zero
    """
    np.random.seed(42)
    data = np.random.normal(0, 0.5, (8, 6))
    # Make gene 1 strongly down across all sigs
    data[0, :] = np.array([-3.1, -2.8, -3.5, -2.9, -3.3, -3.0])
    # Make gene 4 strongly up across all sigs
    data[3, :] = np.array([2.5, 2.8, 2.3, 2.6, 2.4, 2.7])
    # Make gene 6 weakly up (should NOT pass threshold)
    data[5, :] = np.array([1.2, 1.5, 1.0, 1.1, 0.8, 1.3])

    return pd.DataFrame(data, index=[1, 2, 3, 4, 5, 6, 7, 8])


@pytest.fixture
def tmp_lincs_dir():
    """Create a minimal LINCS metadata directory (no GCTX)."""
    with tempfile.TemporaryDirectory(prefix="lincs_test_") as tmpdir:
        # geneinfo_beta.txt
        gene_info = textwrap.dedent("""\
            gene_id\tgene_symbol\tfeature_space
            1\tEGFR\tlandmark
            2\tKRAS\tlandmark
            3\tBRAF\tlandmark
            4\tMYC\tlandmark
            5\tTP53\tlandmark
            6\tAKT1\tbest_inferred
            7\tMTOR\tbest_inferred
            8\tCDK4\tlandmark
        """)
        with open(os.path.join(tmpdir, "geneinfo_beta.txt"), "w") as f:
            f.write(gene_info)

        # siginfo_beta.txt
        sig_info = textwrap.dedent("""\
            sig_id\tpert_iname\tpert_type\tis_exemplar_sig\tcell_iname
            SIG001\tEGFR\ttrt_xpr\t1\tA549
            SIG002\tEGFR\ttrt_xpr\t1\tMCF7
            SIG003\tEGFR\ttrt_xpr\t1\tA549
            SIG004\tKRAS\ttrt_xpr\t1\tA549
            SIG005\tKRAS\ttrt_xpr\t1\tHT29
            SIG006\tKRAS\ttrt_xpr\t1\tMCF7
            SIG007\tBRAF\ttrt_xpr\t0\tA549
            SIG008\tSINGLE\ttrt_xpr\t1\tA549
        """)
        with open(os.path.join(tmpdir, "siginfo_beta.txt"), "w") as f:
            f.write(sig_info)

        # cellinfo_beta.txt
        cell_info = textwrap.dedent("""\
            cell_iname\tcell_lineage
            A549\tlung
            MCF7\tbreast
            HT29\tcolon
        """)
        with open(os.path.join(tmpdir, "cellinfo_beta.txt"), "w") as f:
            f.write(cell_info)

        yield tmpdir


# ============================================================================
# Tests: consensus computation
# ============================================================================


class TestComputeConsensus:
    """Tests for compute_consensus()."""

    def test_basic_up_down(self, z_matrix, gene_id_to_symbol):
        mean_z, up_genes, down_genes = compute_consensus(
            z_matrix, gene_id_to_symbol
        )
        # EGFR should be in down_genes (z ≈ -3)
        assert "EGFR" in down_genes, f"EGFR not in down_genes: {down_genes}"
        # MYC should be in up_genes (z ≈ +2.5)
        assert "MYC" in up_genes, f"MYC not in up_genes: {up_genes}"
        # AKT1 should NOT be in up_genes (z ≈ 1.1, below threshold)
        assert "AKT1" not in up_genes

    def test_mean_z_values(self, z_matrix, gene_id_to_symbol):
        mean_z, _, _ = compute_consensus(z_matrix, gene_id_to_symbol)
        assert "EGFR" in mean_z
        assert mean_z["EGFR"] < -2.5
        assert "MYC" in mean_z
        assert mean_z["MYC"] > 2.0

    def test_empty_matrix(self, gene_id_to_symbol):
        empty = pd.DataFrame(
            np.array([]).reshape(0, 0), index=[], columns=[]
        )
        mean_z, up, down = compute_consensus(empty, gene_id_to_symbol)
        assert mean_z == {}
        assert up == frozenset()
        assert down == frozenset()

    def test_single_signature(self, gene_id_to_symbol):
        data = np.array([[-3.0], [0.1], [0.0], [2.5], [0.0], [0.0], [0.0], [0.0]])
        z_mat = pd.DataFrame(data, index=[1, 2, 3, 4, 5, 6, 7, 8])
        mean_z, up, down = compute_consensus(z_mat, gene_id_to_symbol)
        # With only 1 sig, min_freq=0.5 is met trivially
        assert "EGFR" in down
        assert "MYC" in up

    def test_custom_thresholds(self, z_matrix, gene_id_to_symbol):
        # With very strict threshold, fewer genes should pass
        _, up, down = compute_consensus(
            z_matrix,
            gene_id_to_symbol,
            up_threshold=4.0,
            down_threshold=-4.0,
        )
        # Nothing should pass at z=4.0 with our data
        assert len(up) == 0
        assert len(down) == 0


# ============================================================================
# Tests: confidence scoring
# ============================================================================


class TestComputeConfidence:
    def test_knockout_high_quality(self):
        conf = _compute_confidence(n_sigs=20, n_cell_lines=5, pert_type="knockout")
        assert 0.8 <= conf <= 0.98

    def test_compound_low_quality(self):
        conf = _compute_confidence(n_sigs=2, n_cell_lines=1, pert_type="compound")
        assert conf < 0.5

    def test_knockdown_medium(self):
        conf = _compute_confidence(n_sigs=10, n_cell_lines=3, pert_type="knockdown")
        assert 0.4 < conf < 0.9

    def test_never_exceeds_cap(self):
        conf = _compute_confidence(n_sigs=1000, n_cell_lines=100, pert_type="knockout")
        assert conf <= 0.98

    def test_minimum_is_positive(self):
        conf = _compute_confidence(n_sigs=1, n_cell_lines=0, pert_type="compound")
        assert conf > 0


# ============================================================================
# Tests: ConsensusSignature
# ============================================================================


class TestConsensusSignature:
    def test_to_perturbation_signature(self):
        cs = ConsensusSignature(
            target_gene="EGFR",
            pert_type="knockout",
            n_signatures=10,
            mean_z={"MYC": 2.5, "BRAF": -3.0},
            up_genes=frozenset({"MYC"}),
            down_genes=frozenset({"BRAF"}),
            cell_lines=frozenset({"A549", "MCF7"}),
            confidence=0.85,
        )
        sig = cs.to_perturbation_signature()
        assert isinstance(sig, PerturbationSignature)
        assert sig.target == "EGFR"
        assert sig.perturbation_type == "knockout"
        assert "BRAF" in sig.expression_decreased
        assert "MYC" in sig.expression_increased
        assert sig.confidence == 0.85
        assert "LINCS" in sig.source
        assert sig.pmid == "28678552"

    def test_to_perturbation_signature_responders(self):
        cs = ConsensusSignature(
            target_gene="KRAS",
            pert_type="knockout",
            n_signatures=5,
            mean_z={},
            up_genes=frozenset({"A", "B"}),
            down_genes=frozenset({"C", "D"}),
            confidence=0.7,
        )
        sig = cs.to_perturbation_signature()
        assert sig.all_responders == {"A", "B", "C", "D"}
        assert sig.direct_effectors == {"C", "D"}


# ============================================================================
# Tests: metadata loaders
# ============================================================================


class TestMetadataLoaders:
    def test_load_gene_info(self, tmp_lincs_dir):
        df = load_gene_info(tmp_lincs_dir)
        assert len(df) == 8
        assert "gene_symbol" in df.columns
        assert df.loc[1, "gene_symbol"] == "EGFR"

    def test_load_sig_info(self, tmp_lincs_dir):
        df = load_sig_info(tmp_lincs_dir)
        assert len(df) == 8
        assert "pert_iname" in df.columns

    def test_load_gene_info_missing(self):
        with pytest.raises(FileNotFoundError):
            load_gene_info("/nonexistent/path")

    def test_load_sig_info_missing(self):
        with pytest.raises(FileNotFoundError):
            load_sig_info("/nonexistent/path")


# ============================================================================
# Tests: LINCSSignatureDB
# ============================================================================


class TestLINCSSignatureDB:
    def test_init(self, tmp_lincs_dir):
        db = LINCSSignatureDB(tmp_lincs_dir)
        assert db.lincs_dir == tmp_lincs_dir
        assert not db._loaded

    def test_ensure_gene_info(self, tmp_lincs_dir):
        db = LINCSSignatureDB(tmp_lincs_dir)
        db._ensure_gene_info()
        assert "EGFR" in db._gene_id_to_symbol.values()
        assert "KRAS" in db._gene_id_to_symbol.values()
        assert db._symbol_to_gene_id["EGFR"] == 1

    def test_ensure_sig_info(self, tmp_lincs_dir):
        db = LINCSSignatureDB(tmp_lincs_dir)
        db._ensure_sig_info()
        assert db._sig_info is not None
        assert len(db._sig_info) == 8

    def test_find_gctx_missing(self, tmp_lincs_dir):
        db = LINCSSignatureDB(tmp_lincs_dir)
        assert db._find_gctx("trt_xpr") is None

    def test_build_index_no_gctx_warns(self, tmp_lincs_dir, caplog):
        """Build index with no GCTX files: should warn but not crash."""
        db = LINCSSignatureDB(tmp_lincs_dir)
        # Manually inject h5py as available to avoid ImportError
        import alin.lincs as lincs_mod
        orig = lincs_mod._HAS_H5PY
        lincs_mod._HAS_H5PY = True
        try:
            db.build_index()
            assert db._loaded
            assert len(db._consensus) == 0
        finally:
            lincs_mod._HAS_H5PY = orig

    def test_get_perturbation_signature_empty(self, tmp_lincs_dir):
        """No consensus built → returns None."""
        db = LINCSSignatureDB(tmp_lincs_dir)
        db._loaded = True  # Pretend loaded (empty)
        assert db.get_perturbation_signature("EGFR") is None
        assert db.get_perturbation_responders("EGFR") == set()
        assert db.get_direct_effectors("EGFR") == set()
        assert db.get_feedback_genes("EGFR") == set()

    def test_get_perturbation_signature_with_data(self, tmp_lincs_dir):
        """Inject a consensus and verify API returns correct signature."""
        db = LINCSSignatureDB(tmp_lincs_dir)
        db._loaded = True
        db._consensus["EGFR"] = ConsensusSignature(
            target_gene="EGFR",
            pert_type="knockout",
            n_signatures=15,
            mean_z={"MYC": 2.5, "BRAF": -3.0, "AKT1": -2.1},
            up_genes=frozenset({"MYC"}),
            down_genes=frozenset({"BRAF", "AKT1"}),
            cell_lines=frozenset({"A549", "MCF7"}),
            confidence=0.85,
        )

        sig = db.get_perturbation_signature("EGFR")
        assert sig is not None
        assert sig.target == "EGFR"
        assert "MYC" in sig.expression_increased
        assert "BRAF" in sig.expression_decreased

        responders = db.get_perturbation_responders("EGFR")
        assert responders == {"MYC", "BRAF", "AKT1"}

        effectors = db.get_direct_effectors("EGFR")
        assert effectors == {"BRAF", "AKT1"}

        feedback = db.get_feedback_genes("EGFR")
        assert feedback == {"MYC"}

    def test_get_top_responders(self, tmp_lincs_dir):
        db = LINCSSignatureDB(tmp_lincs_dir)
        db._loaded = True
        db._consensus["EGFR"] = ConsensusSignature(
            target_gene="EGFR",
            pert_type="knockout",
            n_signatures=10,
            mean_z={"MYC": 2.5, "BRAF": -3.0, "AKT1": -2.1, "TP53": 0.5},
            up_genes=frozenset({"MYC"}),
            down_genes=frozenset({"BRAF", "AKT1"}),
            confidence=0.85,
        )

        top_all = db.get_top_responders("EGFR", n=3)
        assert len(top_all) == 3
        # BRAF should be first (|z| = 3.0)
        assert top_all[0][0] == "BRAF"

        top_up = db.get_top_responders("EGFR", n=2, direction="up")
        assert top_up[0][0] == "MYC"

        top_down = db.get_top_responders("EGFR", n=2, direction="down")
        assert top_down[0][0] == "BRAF"

    def test_score_combination(self, tmp_lincs_dir):
        db = LINCSSignatureDB(tmp_lincs_dir)
        db._loaded = True
        db._consensus["EGFR"] = ConsensusSignature(
            target_gene="EGFR",
            pert_type="knockout",
            n_signatures=10,
            mean_z={},
            up_genes=frozenset({"MYC", "KRAS"}),
            down_genes=frozenset({"BRAF", "AKT1"}),
            confidence=0.85,
        )
        db._consensus["KRAS"] = ConsensusSignature(
            target_gene="KRAS",
            pert_type="knockout",
            n_signatures=8,
            mean_z={},
            up_genes=frozenset({"MTOR"}),
            down_genes=frozenset({"ERK1", "ERK2"}),
            confidence=0.80,
        )

        result = db.score_combination_by_perturbation(
            targets=["EGFR", "KRAS"],
            essential_genes={"BRAF", "AKT1", "ERK1", "MYC", "CDK4"},
        )
        assert "feedback_coverage" in result
        assert "effector_coverage" in result
        assert "perturbation_score" in result
        assert result["effector_coverage"] > 0  # BRAF, AKT1, ERK1 are covered
        # KRAS is a feedback gene of EGFR, so feedback_coverage > 0
        assert result["feedback_coverage"] > 0

    def test_build_perturbation_response_paths(self, tmp_lincs_dir):
        db = LINCSSignatureDB(tmp_lincs_dir)
        db._loaded = True
        db._consensus["EGFR"] = ConsensusSignature(
            target_gene="EGFR",
            pert_type="knockout",
            n_signatures=10,
            mean_z={},
            up_genes=frozenset({"MYC"}),
            down_genes=frozenset({"BRAF", "AKT1"}),
            confidence=0.85,
        )

        essential = {"BRAF", "AKT1", "CDK4", "MTOR"}
        paths = db.build_perturbation_response_paths(essential, min_overlap=2)
        assert len(paths) >= 1
        target, genes, conf = paths[0]
        assert target == "EGFR"
        assert "BRAF" in genes
        assert "AKT1" in genes
        assert 0 < conf <= 1.0

    def test_case_insensitive_lookup(self, tmp_lincs_dir):
        db = LINCSSignatureDB(tmp_lincs_dir)
        db._loaded = True
        db._consensus["EGFR"] = ConsensusSignature(
            target_gene="EGFR",
            pert_type="knockout",
            n_signatures=10,
            mean_z={},
            up_genes=frozenset({"MYC"}),
            down_genes=frozenset({"BRAF"}),
            confidence=0.85,
        )
        # lowercase should still work
        assert db.get_consensus("egfr") is not None
        assert db.get_perturbation_signature("Egfr") is not None

    def test_summary(self, tmp_lincs_dir):
        db = LINCSSignatureDB(tmp_lincs_dir)
        db._loaded = True
        db._consensus["EGFR"] = ConsensusSignature(
            target_gene="EGFR",
            pert_type="knockout",
            n_signatures=10,
            mean_z={"MYC": 2.5},
            up_genes=frozenset({"MYC"}),
            down_genes=frozenset({"BRAF"}),
            confidence=0.85,
        )
        s = db.summary()
        assert s["n_targets"] == 1
        assert s["source"] == "LINCS_L1000"
        assert s["median_confidence"] == 0.85

    def test_summary_empty(self, tmp_lincs_dir):
        db = LINCSSignatureDB(tmp_lincs_dir)
        db._loaded = True
        s = db.summary()
        assert s["n_targets"] == 0

    def test_available_targets(self, tmp_lincs_dir):
        db = LINCSSignatureDB(tmp_lincs_dir)
        db._loaded = True
        db._consensus["EGFR"] = ConsensusSignature(
            target_gene="EGFR", pert_type="knockout", n_signatures=10,
            mean_z={}, confidence=0.85,
        )
        db._consensus["KRAS"] = ConsensusSignature(
            target_gene="KRAS", pert_type="knockout", n_signatures=5,
            mean_z={}, confidence=0.80,
        )
        assert db.available_targets == ["EGFR", "KRAS"]
        assert db.n_targets == 2


# ============================================================================
# Tests: index caching (pickle round-trip)
# ============================================================================


class TestIndexCaching:
    def test_save_and_load(self, tmp_lincs_dir):
        db = LINCSSignatureDB(tmp_lincs_dir)
        db._loaded = True
        db._consensus["EGFR"] = ConsensusSignature(
            target_gene="EGFR",
            pert_type="knockout",
            n_signatures=10,
            mean_z={"MYC": 2.5},
            up_genes=frozenset({"MYC"}),
            down_genes=frozenset({"BRAF"}),
            confidence=0.85,
        )
        db._save_index()

        # Verify file exists
        idx_path = os.path.join(tmp_lincs_dir, "lincs_index.pkl")
        assert os.path.isfile(idx_path)

        # Load into a fresh DB
        db2 = LINCSSignatureDB(tmp_lincs_dir)
        loaded = db2._load_cached_index()
        assert loaded
        assert db2._loaded
        assert "EGFR" in db2._consensus
        assert db2._consensus["EGFR"].confidence == 0.85


# ============================================================================
# Tests: lincs_available()
# ============================================================================


class TestLincsAvailable:
    def test_no_dir(self):
        assert lincs_available("/nonexistent/path") is False

    def test_dir_no_gctx(self, tmp_lincs_dir):
        # Has metadata but no GCTX
        assert lincs_available(tmp_lincs_dir) is False

    def test_dir_with_gctx(self, tmp_lincs_dir):
        # Create a dummy GCTX file
        dummy = os.path.join(tmp_lincs_dir, "level5_beta_trt_xpr_test.gctx")
        with open(dummy, "w") as f:
            f.write("dummy")
        assert lincs_available(tmp_lincs_dir) is True
