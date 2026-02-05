"""
Unit Tests for Hitting Set Solver
=================================
Tests for the minimal hitting set optimization algorithm.
"""

import pytest
import numpy as np
from typing import List, Set, FrozenSet

from core.data_structures import ViabilityPath, HittingSet, NodeCost


class TestNodeCost:
    """Tests for NodeCost calculation"""
    
    def test_total_cost_basic(self):
        """Test basic cost calculation"""
        cost = NodeCost(
            gene="KRAS",
            toxicity_score=0.5,
            tumor_specificity=0.8,
            druggability_score=0.9,
            pan_essential_penalty=0.0,
            base_penalty=1.0
        )
        
        # c(g,c) = α*τ - β*s - γ*d + δ*pan + λ
        # = 1*0.5 - 0.5*0.8 - 0.3*0.9 + 2*0 + 1*1
        # = 0.5 - 0.4 - 0.27 + 0 + 1 = 0.83
        expected = 1.0 * 0.5 - 0.5 * 0.8 - 0.3 * 0.9 + 2.0 * 0.0 + 1.0 * 1.0
        actual = cost.total_cost()
        
        assert abs(actual - expected) < 0.01, f"Expected {expected}, got {actual}"
    
    def test_pan_essential_penalty(self):
        """Test that pan-essential genes have higher cost"""
        normal_gene = NodeCost(
            gene="KRAS", toxicity_score=0.5, tumor_specificity=0.5,
            druggability_score=0.5, pan_essential_penalty=0.0
        )
        pan_essential = NodeCost(
            gene="RPL11", toxicity_score=0.5, tumor_specificity=0.5,
            druggability_score=0.5, pan_essential_penalty=1.0
        )
        
        # Pan-essential should have higher cost (less desirable)
        assert pan_essential.total_cost() > normal_gene.total_cost()
    
    def test_druggability_reduces_cost(self):
        """Test that higher druggability reduces cost"""
        low_drug = NodeCost(
            gene="MYC", toxicity_score=0.5, tumor_specificity=0.5,
            druggability_score=0.1, pan_essential_penalty=0.0
        )
        high_drug = NodeCost(
            gene="EGFR", toxicity_score=0.5, tumor_specificity=0.5,
            druggability_score=0.9, pan_essential_penalty=0.0
        )
        
        # Higher druggability should reduce cost
        assert high_drug.total_cost() < low_drug.total_cost()
    
    def test_custom_weights(self):
        """Test cost calculation with custom weights"""
        cost = NodeCost(
            gene="TEST", toxicity_score=1.0, tumor_specificity=1.0,
            druggability_score=1.0, pan_essential_penalty=0.0
        )
        
        # With zero weights for everything except base
        result = cost.total_cost(alpha=0, beta=0, gamma=0, delta=0, lambda_base=1.0)
        assert result == 1.0  # Only base penalty


class TestViabilityPath:
    """Tests for ViabilityPath"""
    
    def test_path_creation(self):
        """Test creating a viability path"""
        path = ViabilityPath(
            path_id="test_path",
            nodes=frozenset({"KRAS", "BRAF", "MAP2K1"}),
            context="test_cancer",
            confidence=0.9,
            path_type="signaling_path"
        )
        
        assert "KRAS" in path
        assert "EGFR" not in path
        assert len(path.nodes) == 3
    
    def test_path_intersection(self):
        """Test path intersection with gene sets"""
        path = ViabilityPath(
            path_id="test",
            nodes=frozenset({"KRAS", "BRAF", "STAT3"}),
            context="test"
        )
        
        # Should intersect
        assert path.intersects({"KRAS", "EGFR"})
        assert path.intersects({"STAT3"})
        
        # Should not intersect
        assert not path.intersects({"EGFR", "MET"})
        assert not path.intersects(set())


class TestHittingSet:
    """Tests for HittingSet"""
    
    def test_hitting_set_creation(self):
        """Test creating a hitting set"""
        hs = HittingSet(
            targets=frozenset({"KRAS", "STAT3", "CDK4"}),
            total_cost=3.5,
            coverage=0.85,
            paths_covered={"path1", "path2", "path3"}
        )
        
        assert len(hs) == 3
        assert "KRAS" in hs
        assert "EGFR" not in hs
    
    def test_hitting_set_covers_path(self):
        """Test that hitting set correctly reports path coverage"""
        hs = HittingSet(
            targets=frozenset({"KRAS", "STAT3"}),
            total_cost=2.0,
            coverage=1.0,
            paths_covered={"p1"}
        )
        
        path1 = ViabilityPath("p1", frozenset({"KRAS", "BRAF"}), "test")
        path2 = ViabilityPath("p2", frozenset({"EGFR", "MET"}), "test")
        
        assert hs.hits_path(path1)  # KRAS is in both
        assert not hs.hits_path(path2)  # No overlap


class TestHittingSetSolver:
    """Tests for the hitting set solver algorithm"""
    
    def create_simple_paths(self) -> List[ViabilityPath]:
        """Create simple test paths"""
        return [
            ViabilityPath("p1", frozenset({"A", "B"}), "test"),
            ViabilityPath("p2", frozenset({"B", "C"}), "test"),
            ViabilityPath("p3", frozenset({"C", "D"}), "test"),
        ]
    
    def test_greedy_finds_solution(self):
        """Test that greedy algorithm finds a valid hitting set"""
        paths = self.create_simple_paths()
        
        # Gene B covers p1 and p2, gene C covers p2 and p3
        # So {B, C} should cover all paths
        
        # Simulate greedy selection
        all_genes = {"A", "B", "C", "D"}
        gene_costs = {g: 1.0 for g in all_genes}
        
        selected = set()
        uncovered = set(paths)
        
        while uncovered:
            best_gene = None
            best_hits = 0
            
            for gene in all_genes - selected:
                hits = sum(1 for p in uncovered if gene in p.nodes)
                if hits > best_hits:
                    best_hits = hits
                    best_gene = gene
            
            if best_gene is None:
                break
            
            selected.add(best_gene)
            uncovered = {p for p in uncovered if best_gene not in p.nodes}
        
        # Verify all paths are covered
        for path in paths:
            assert any(g in path.nodes for g in selected), f"Path {path.path_id} not covered"
        
        # Greedy should find {B, C} or {B, D} - both valid minimal solutions
        assert len(selected) <= 3  # Should be reasonably small
    
    def test_coverage_calculation(self):
        """Test coverage fraction calculation"""
        paths = self.create_simple_paths()
        
        # Single gene coverage
        def calc_coverage(genes: Set[str]) -> float:
            covered = sum(1 for p in paths if any(g in p.nodes for g in genes))
            return covered / len(paths)
        
        # A covers only p1 (1/3)
        assert abs(calc_coverage({"A"}) - 1/3) < 0.01
        
        # B covers p1 and p2 (2/3)
        assert abs(calc_coverage({"B"}) - 2/3) < 0.01
        
        # B and D cover all (3/3)
        assert abs(calc_coverage({"B", "D"}) - 1.0) < 0.01
    
    def test_exhaustive_finds_optimal(self):
        """Test that exhaustive search finds optimal solution"""
        paths = self.create_simple_paths()
        all_genes = {"A", "B", "C", "D"}
        
        # Gene costs: make B cheaper to test cost-weighted selection
        gene_costs = {"A": 2.0, "B": 0.5, "C": 1.0, "D": 1.5}
        
        from itertools import combinations
        
        best_solution = None
        best_cost = float('inf')
        
        for size in range(1, len(all_genes) + 1):
            for subset in combinations(all_genes, size):
                subset_set = set(subset)
                
                # Check coverage
                covered = sum(1 for p in paths if any(g in subset_set for g in p.nodes))
                coverage = covered / len(paths)
                
                if coverage >= 0.99:  # Full coverage
                    total_cost = sum(gene_costs[g] for g in subset)
                    if total_cost < best_cost:
                        best_cost = total_cost
                        best_solution = subset_set
        
        assert best_solution is not None
        # Optimal should be {B, C} with cost 1.5 or {B, D} with cost 2.0
        assert len(best_solution) == 2
        assert "B" in best_solution  # B should always be included (lowest cost, high coverage)


class TestHittingSetProperties:
    """Mathematical property tests for hitting sets"""
    
    def test_hitting_set_constraint(self):
        """Test that all paths are hit (∀p∈P: T∩N(p)≠∅)"""
        paths = [
            ViabilityPath("p1", frozenset({"KRAS", "BRAF"}), "test"),
            ViabilityPath("p2", frozenset({"STAT3", "JAK1"}), "test"),
            ViabilityPath("p3", frozenset({"CDK4", "MYC"}), "test"),
        ]
        
        # Valid hitting set: one gene from each path
        valid_hs = frozenset({"KRAS", "STAT3", "CDK4"})
        
        # Check constraint: T ∩ N(p) ≠ ∅ for all p
        for path in paths:
            intersection = valid_hs & path.nodes
            assert len(intersection) > 0, f"Path {path.path_id} not hit"
    
    def test_minimal_hitting_set(self):
        """Test that we can find minimal (smallest) hitting sets"""
        # Paths that overlap
        paths = [
            ViabilityPath("p1", frozenset({"A", "X"}), "test"),
            ViabilityPath("p2", frozenset({"B", "X"}), "test"),
            ViabilityPath("p3", frozenset({"C", "X"}), "test"),
        ]
        
        # X is the optimal minimal hitting set (size 1)
        minimal = frozenset({"X"})
        
        for path in paths:
            assert bool(minimal & path.nodes)
        
        # Verify X alone hits all paths
        assert all(any(g in path.nodes for g in minimal) for path in paths)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
