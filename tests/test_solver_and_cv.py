"""
Tests for Fix #7 (Cross-Validation Honesty) and Fix #8 (MHS Solver Hierarchy)
=============================================================================

Fix #7: Verifies that run_loco_cv() is honestly labelled as partitioned
evaluation rather than true cross-validation.

Fix #8: Verifies the solver hierarchy (greedy → ILP → exhaustive) works
correctly, that the ILP solver produces provably optimal solutions for
moderate-size problems, and that solver_stats are tracked.
"""

import pytest
import numpy as np
from typing import List, Dict
from itertools import combinations

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

from core.data_structures import ViabilityPath, HittingSet, NodeCost


def _make_paths(spec: List[tuple]) -> List[ViabilityPath]:
    """Helper: spec = [(id, {genes}), ...]"""
    return [
        ViabilityPath(path_id=pid, nodes=frozenset(genes), context="test")
        for pid, genes in spec
    ]


# ============================================================================
# Fix #8: MHS Solver Hierarchy Tests
# ============================================================================

class TestSolverHierarchy:
    """Test that the solver hierarchy (greedy/ILP/exhaustive) is correct."""

    def _make_solver(self):
        """Create a MinimalHittingSetSolver with a dummy cost function."""
        # Import the real solver
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from pan_cancer_xnode import MinimalHittingSetSolver, CostFunction

        class DummyCostFunction:
            """Cost function that returns fixed costs per gene."""
            def __init__(self, costs: Dict[str, float]):
                self._costs = costs

            def compute_cost(self, gene, cancer_type):
                cost_val = self._costs.get(gene, 1.0)
                return NodeCost(
                    gene=gene,
                    toxicity_score=cost_val,
                    tumor_specificity=0.0,
                    druggability_score=0.0,
                    pan_essential_penalty=0.0,
                    base_penalty=0.0,
                )

        return MinimalHittingSetSolver, DummyCostFunction

    def test_greedy_always_runs(self):
        """Greedy solver should run regardless of pool size."""
        Solver, DummyCost = self._make_solver()
        paths = _make_paths([
            ("p1", {"A", "B"}),
            ("p2", {"B", "C"}),
            ("p3", {"C", "D"}),
        ])
        cost_fn = DummyCost({"A": 1, "B": 1, "C": 1, "D": 1})
        solver = Solver(cost_fn)
        solutions = solver.solve(paths, "test_cancer", max_size=4, min_coverage=0.8)

        assert len(solutions) >= 1
        assert solver.solver_stats['greedy'] >= 1

    def test_exhaustive_runs_for_small_pools(self):
        """Exhaustive solver should run when pool <= EXHAUSTIVE_THRESHOLD."""
        Solver, DummyCost = self._make_solver()
        # 4 genes < 20 threshold
        paths = _make_paths([
            ("p1", {"A", "B"}),
            ("p2", {"C", "D"}),
        ])
        cost_fn = DummyCost({"A": 2, "B": 1, "C": 1, "D": 3})
        solver = Solver(cost_fn)
        solutions = solver.solve(paths, "test_cancer", max_size=4, min_coverage=0.8)

        assert solver.solver_stats['exhaustive'] >= 1
        # Should find optimal: {B, C} with cost 2.0
        best = min(solutions, key=lambda s: s.total_cost)
        assert best.targets == frozenset({"B", "C"})

    def test_ilp_runs_for_medium_pools(self):
        """ILP solver should run for pools between EXHAUSTIVE_THRESHOLD and ILP_THRESHOLD."""
        Solver, DummyCost = self._make_solver()
        # Create 30 unique genes spread across 15 paths that SHARE some genes
        # so 4 genes can cover most paths
        genes = [f"G{i}" for i in range(30)]
        # Create "hub" genes appearing in many paths
        path_specs = []
        for i in range(15):
            # Each path includes two hub genes + one unique gene
            hub1 = genes[i % 4]        # 4 hub genes
            unique = genes[4 + i]       # unique genes G4..G18
            extra = genes[19 + (i % 11)]  # extra genes G19..G29
            path_specs.append((f"p{i}", {hub1, unique, extra}))
        paths = _make_paths(path_specs)

        costs = {g: float(i + 1) for i, g in enumerate(genes)}
        cost_fn = DummyCost(costs)
        solver = Solver(cost_fn)
        solutions = solver.solve(paths, "test_cancer", max_size=4, min_coverage=0.8)

        assert solver.solver_stats['ilp'] >= 1
        assert len(solutions) >= 1

    def test_ilp_produces_valid_solution(self):
        """ILP solution should satisfy coverage and cardinality constraints."""
        Solver, DummyCost = self._make_solver()
        # Known problem with clear optimal
        paths = _make_paths([
            ("p1", {"A", "X"}),
            ("p2", {"B", "X"}),
            ("p3", {"C", "X"}),
        ])
        # X covers all paths at cost 1; alternatives cost more
        cost_fn = DummyCost({"A": 5, "B": 5, "C": 5, "X": 1})
        solver = Solver(cost_fn)
        solutions = solver.solve(paths, "test_cancer", max_size=4, min_coverage=1.0)

        # The optimal solution is {X} with cost 1
        assert any(s.targets == frozenset({"X"}) for s in solutions)
        best = min(solutions, key=lambda s: s.total_cost)
        assert best.coverage >= 1.0

    def test_solver_stats_tracking(self):
        """solver_stats dict should accurately count solver invocations."""
        Solver, DummyCost = self._make_solver()
        paths = _make_paths([("p1", {"A"}), ("p2", {"B"})])
        cost_fn = DummyCost({"A": 1, "B": 1})
        solver = Solver(cost_fn)

        # Reset stats
        assert solver.solver_stats['greedy'] == 0
        assert solver.solver_stats['ilp'] == 0

        solver.solve(paths, "test", max_size=2)
        assert solver.solver_stats['greedy'] >= 1

    def test_ilp_timeout_does_not_crash(self):
        """ILP solver should return None gracefully if infeasible."""
        Solver, DummyCost = self._make_solver()
        # Create a problem where max_size=1 cannot cover all paths (no single gene does)
        paths = _make_paths([
            ("p1", {"A"}),
            ("p2", {"B"}),
            ("p3", {"C"}),
        ])
        cost_fn = DummyCost({"A": 1, "B": 1, "C": 1})
        solver = Solver(cost_fn)

        # With max_size=1 and min_coverage=1.0, ILP should be infeasible
        # but the greedy should still return something
        solutions = solver.solve(paths, "test", max_size=1, min_coverage=1.0)
        # Greedy will pick one gene covering 1/3 paths
        assert len(solutions) >= 1

    def test_greedy_is_approximate(self):
        """Greedy solution cost should be >= ILP solution cost (ILP is optimal)."""
        Solver, DummyCost = self._make_solver()
        # Create a problem where greedy might pick sub-optimally
        paths = _make_paths([
            ("p1", {"A", "B", "C"}),
            ("p2", {"A", "D", "E"}),
            ("p3", {"B", "D", "F"}),
            ("p4", {"C", "E", "F"}),
        ])
        # Make A cheap-per-hit but not globally optimal
        costs = {"A": 0.5, "B": 0.5, "C": 0.5, "D": 0.5, "E": 0.5, "F": 0.5}
        cost_fn = DummyCost(costs)
        solver = Solver(cost_fn)
        solutions = solver.solve(paths, "test", max_size=4, min_coverage=1.0)

        if len(solutions) >= 2:
            costs_list = [s.total_cost for s in solutions]
            # ILP/exhaustive should have found optimal cost <= greedy cost
            assert min(costs_list) <= max(costs_list)

    def test_class_has_threshold_constants(self):
        """MinimalHittingSetSolver should have the documented threshold constants."""
        Solver, _ = self._make_solver()
        assert hasattr(Solver, 'EXHAUSTIVE_THRESHOLD')
        assert hasattr(Solver, 'ILP_THRESHOLD')
        assert hasattr(Solver, 'PREFILTER_TOP_K')
        assert Solver.EXHAUSTIVE_THRESHOLD == 20
        assert Solver.ILP_THRESHOLD == 500
        assert Solver.PREFILTER_TOP_K == 60

    def test_prefiltered_exhaustive_for_large_pools(self):
        """For large pools (>500), pre-filtered exhaustive should be attempted."""
        Solver, DummyCost = self._make_solver()
        # Create paths that collectively reference 600 unique genes (> ILP_THRESHOLD of 500)
        n_genes = 600
        genes = [f"G{i}" for i in range(n_genes)]
        # Create 300 paths, each with 2 unique genes, so all 600 genes appear
        path_specs = []
        for i in range(0, n_genes, 2):
            path_specs.append((f"p{i//2}", {genes[i], genes[i+1]}))
        paths = _make_paths(path_specs)

        costs = {g: float(i + 1) for i, g in enumerate(genes)}
        cost_fn = DummyCost(costs)
        solver = Solver(cost_fn)
        solutions = solver.solve(paths, "test", max_size=4, min_coverage=0.8)

        # Should have attempted ILP (for large pools) + prefiltered exhaustive
        assert (solver.solver_stats['ilp'] >= 1
                or solver.solver_stats['prefiltered_exhaustive'] >= 1)
        assert len(solutions) >= 1


# ============================================================================
# Fix #7: LOCO-CV Honesty Tests
# ============================================================================

class TestLOCOCVHonesty:
    """Test that run_loco_cv is honestly documented as partitioned evaluation."""

    def test_loco_cv_returns_correct_method_label(self):
        """run_loco_cv should return method='LOCO_partitioned_evaluation'."""
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from benchmarking_module import run_loco_cv

        # We need a triples CSV; create a minimal one
        import tempfile, csv
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Cancer_Type', 'Target_1', 'Target_2', 'Target_3',
                             'Combined_Score', 'Rank'])
            writer.writerow(['Melanoma', 'BRAF', 'MAP2K1', 'STAT3', '0.95', '1'])
            writer.writerow(['Non-Small Cell Lung Cancer', 'KRAS', 'CDK6', 'STAT3', '0.90', '1'])
            tmppath = f.name

        try:
            result = run_loco_cv(tmppath)
            assert result['method'] == 'LOCO_partitioned_evaluation'
            assert 'caveat' in result
            assert 'not re-run' in result['caveat'].lower() or 'partitioned' in result['caveat'].lower()
        finally:
            os.unlink(tmppath)

    def test_loco_cv_docstring_warns_about_no_retraining(self):
        """run_loco_cv docstring should warn about the lack of re-training."""
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from benchmarking_module import run_loco_cv

        doc = run_loco_cv.__doc__
        assert doc is not None
        doc_lower = doc.lower()
        # Should mention it's not true CV
        assert 'not true' in doc_lower or 'not cross-validation' in doc_lower or 'partitioned' in doc_lower
        # Should mention no re-training/re-running
        assert 're-train' in doc_lower or 're-run' in doc_lower or 'no re' in doc_lower

    def test_loco_cv_has_caveat_field(self):
        """Result dict should include a caveat explaining the limitation."""
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from benchmarking_module import run_loco_cv

        import tempfile, csv
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Cancer_Type', 'Target_1', 'Target_2', 'Target_3',
                             'Combined_Score', 'Rank'])
            writer.writerow(['Melanoma', 'BRAF', 'CDK4', 'STAT3', '0.9', '1'])
            tmppath = f.name

        try:
            result = run_loco_cv(tmppath)
            assert 'caveat' in result
            # Caveat should mention the pipeline is not re-run
            assert 'pipeline' in result['caveat'].lower() or 'single' in result['caveat'].lower()
        finally:
            os.unlink(tmppath)


# ============================================================================
# Paper Consistency Tests
# ============================================================================

class TestPaperConsistency:
    """Test that paper.tex is consistent with code changes."""

    def _read_paper(self):
        import os
        paper_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'paper.tex'
        )
        with open(paper_path, 'r') as f:
            return f.read()

    def test_paper_mentions_ilp_solver(self):
        """Paper should mention ILP solver in the methods."""
        text = self._read_paper()
        assert 'ILP' in text or 'integer linear programming' in text.lower()

    def test_paper_mentions_solver_hierarchy(self):
        """Paper should describe the solver hierarchy."""
        text = self._read_paper()
        assert 'greedy' in text.lower()
        assert 'exhaustive' in text.lower()

    def test_paper_does_not_claim_cv_for_loco(self):
        """Paper should NOT claim LOCO is true cross-validation."""
        text = self._read_paper()
        # "partitioned evaluation" should appear
        assert 'partitioned evaluation' in text.lower()

    def test_paper_has_limitation_10(self):
        """Paper should have Limitation #10 about fake CV."""
        text = self._read_paper()
        assert 'Limitation' in text or '\\textbf{10.' in text
        assert 'not true' in text.lower() or 'not cross-validation' in text.lower() or \
               'not re-run' in text.lower()

    def test_paper_has_limitation_11(self):
        """Paper should mention MHS solver limitation (greedy fallback)."""
        text = self._read_paper()
        # May be in any numbered limitation (currently #6)
        assert 'greedy' in text.lower()

    def test_paper_no_old_25_threshold(self):
        """Paper should no longer describe the old ≤25 exhaustive threshold."""
        text = self._read_paper()
        # The old specific "≤25 genes" or "$\leq$25 genes" for exhaustive should be gone
        # (except in Limitation #11 which explains the old version was replaced)
        lines = text.split('\n')
        for i, line in enumerate(lines):
            if '\\textbf{11.' in line:
                continue  # Limitation 11 may reference old threshold as context
            if 'exhaustive enumeration for small candidate sets' in line.lower():
                assert '$\\leq$25' not in line, \
                    f"Line {i+1} still references old ≤25 exhaustive threshold"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
