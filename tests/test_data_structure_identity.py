"""
Tests for Fix #12: Production Code / Test Code Class Identity
=============================================================

Ensures that the data-structure classes used by the production pipeline
(pan_cancer_xnode.py) are **the exact same class objects** as those in
core/data_structures.py, preventing silent divergence where tests pass
against a richer API that production code never exposes.

Regression guard: if anyone re-introduces a duplicate class definition
inside pan_cancer_xnode.py, these tests will fail immediately.
"""

import pytest


# ============================================================================
# Identity tests — same object, not just same name
# ============================================================================

class TestClassIdentity:
    """Verify production code re-exports core classes, not copies."""

    def test_viability_path_identity(self):
        from core.data_structures import ViabilityPath as CoreVP
        from pan_cancer_xnode import ViabilityPath as ProdVP
        assert CoreVP is ProdVP, (
            "pan_cancer_xnode.ViabilityPath is a duplicate, not core.data_structures.ViabilityPath"
        )

    def test_hitting_set_identity(self):
        from core.data_structures import HittingSet as CoreHS
        from pan_cancer_xnode import HittingSet as ProdHS
        assert CoreHS is ProdHS, (
            "pan_cancer_xnode.HittingSet is a duplicate, not core.data_structures.HittingSet"
        )

    def test_target_node_identity(self):
        from core.data_structures import TargetNode as CoreTN
        from pan_cancer_xnode import TargetNode as ProdTN
        assert CoreTN is ProdTN, (
            "pan_cancer_xnode.TargetNode is a duplicate, not core.data_structures.TargetNode"
        )

    def test_node_cost_identity(self):
        from core.data_structures import NodeCost as CoreNC
        from pan_cancer_xnode import NodeCost as ProdNC
        assert CoreNC is ProdNC, (
            "pan_cancer_xnode.NodeCost is a duplicate, not core.data_structures.NodeCost"
        )

    def test_drug_target_identity(self):
        from core.data_structures import DrugTarget as CoreDT
        from pan_cancer_xnode import DrugTarget as ProdDT
        assert CoreDT is ProdDT, (
            "pan_cancer_xnode.DrugTarget is a duplicate, not core.data_structures.DrugTarget"
        )

    def test_triple_combination_identity(self):
        from core.data_structures import TripleCombination as CoreTC
        from pan_cancer_xnode import TripleCombination as ProdTC
        assert CoreTC is ProdTC, (
            "pan_cancer_xnode.TripleCombination is a duplicate, not core.data_structures.TripleCombination"
        )

    def test_cancer_type_analysis_identity(self):
        from core.data_structures import CancerTypeAnalysis as CoreCTA
        from pan_cancer_xnode import CancerTypeAnalysis as ProdCTA
        assert CoreCTA is ProdCTA, (
            "pan_cancer_xnode.CancerTypeAnalysis is a duplicate, not core.data_structures.CancerTypeAnalysis"
        )


# ============================================================================
# API completeness — methods tests rely on must exist on the canonical class
# ============================================================================

class TestAPICompleteness:
    """Verify the single-source class has all methods that tests and prod use."""

    def test_viability_path_has_contains(self):
        from core.data_structures import ViabilityPath
        vp = ViabilityPath(path_id="p1", nodes=frozenset({"A", "B"}), context="test")
        assert "A" in vp
        assert "C" not in vp

    def test_viability_path_has_intersects(self):
        from core.data_structures import ViabilityPath
        vp = ViabilityPath(path_id="p1", nodes=frozenset({"A", "B"}), context="test")
        assert vp.intersects({"A", "Z"})
        assert not vp.intersects({"X", "Y"})

    def test_hitting_set_has_contains(self):
        from core.data_structures import HittingSet
        hs = HittingSet(targets=frozenset({"X", "Y"}), total_cost=1.0,
                        coverage=1.0, paths_covered={"p1"})
        assert "X" in hs
        assert "Z" not in hs

    def test_hitting_set_has_hits_path(self):
        from core.data_structures import ViabilityPath, HittingSet
        vp = ViabilityPath(path_id="p1", nodes=frozenset({"A", "B"}), context="test")
        hs = HittingSet(targets=frozenset({"B", "C"}), total_cost=1.0,
                        coverage=1.0, paths_covered={"p1"})
        assert hs.hits_path(vp)

    def test_triple_combination_has_contains(self):
        from core.data_structures import TripleCombination
        tc = TripleCombination(
            targets=("A", "B", "C"), total_cost=1.0, synergy_score=0.5,
            resistance_score=0.3, pathway_coverage={}, coverage=0.9,
            druggable_count=2, combined_score=1.5
        )
        assert "A" in tc
        assert "D" not in tc

    def test_triple_combination_has_drugs_property(self):
        from core.data_structures import TripleCombination, DrugTarget
        dt = DrugTarget(gene="A", available_drugs=["DrugA"], clinical_stage="approved",
                        known_toxicities=[])
        tc = TripleCombination(
            targets=("A", "B", "C"), total_cost=1.0, synergy_score=0.5,
            resistance_score=0.3, pathway_coverage={}, coverage=0.9,
            druggable_count=1, combined_score=1.5,
            drug_info={"A": dt, "B": None, "C": None}
        )
        drugs = tc.drugs
        assert len(drugs) == 3
        assert drugs[0] == "DrugA"

    def test_triple_combination_has_combo_tox_fields(self):
        from core.data_structures import TripleCombination
        tc = TripleCombination(
            targets=("A", "B", "C"), total_cost=1.0, synergy_score=0.5,
            resistance_score=0.3, pathway_coverage={}, coverage=0.9,
            druggable_count=2, combined_score=1.5,
            combo_tox_score=0.42, combo_tox_details={"ddi_pairs": 1}
        )
        assert tc.combo_tox_score == 0.42
        assert tc.combo_tox_details["ddi_pairs"] == 1

    def test_triple_combination_has_confidence_interval(self):
        from core.data_structures import TripleCombination
        tc = TripleCombination(
            targets=("A", "B", "C"), total_cost=1.0, synergy_score=0.5,
            resistance_score=0.3, pathway_coverage={}, coverage=0.9,
            druggable_count=2, combined_score=1.5,
            confidence_interval=(1.2, 1.8)
        )
        assert tc.confidence_interval == (1.2, 1.8)

    def test_drug_target_has_is_approved(self):
        from core.data_structures import DrugTarget
        dt = DrugTarget(gene="X", available_drugs=["Drug1"], clinical_stage="approved",
                        known_toxicities=[])
        assert dt.is_approved is True
        dt2 = DrugTarget(gene="Y", available_drugs=[], clinical_stage="phase2",
                         known_toxicities=[])
        assert dt2.is_approved is False

    def test_drug_target_has_is_clinical(self):
        from core.data_structures import DrugTarget
        dt = DrugTarget(gene="X", available_drugs=[], clinical_stage="phase1",
                        known_toxicities=[])
        assert dt.is_clinical is True
        dt2 = DrugTarget(gene="Y", available_drugs=[], clinical_stage="preclinical",
                         known_toxicities=[])
        assert dt2.is_clinical is False

    def test_cancer_type_analysis_has_pharmacological_validation(self):
        from core.data_structures import CancerTypeAnalysis
        cta = CancerTypeAnalysis(
            cancer_type="test", lineage="test", n_cell_lines=1,
            cell_line_ids=[], driver_mutations={}, essential_genes={},
            viability_paths=[], minimal_hitting_sets=[],
            top_x_node_sets=[], recommended_combination=None,
        )
        assert cta.pharmacological_validation is None
        cta.pharmacological_validation = {"tier": 3}
        assert cta.pharmacological_validation["tier"] == 3

    def test_cancer_type_analysis_has_predictions(self):
        from core.data_structures import CancerTypeAnalysis
        cta = CancerTypeAnalysis(
            cancer_type="test", lineage="test", n_cell_lines=1,
            cell_line_ids=[], driver_mutations={}, essential_genes={},
            viability_paths=[], minimal_hitting_sets=[],
            top_x_node_sets=[], recommended_combination=None,
        )
        assert cta.has_predictions is False
        cta.recommended_combination = ["A", "B"]
        assert cta.has_predictions is True
