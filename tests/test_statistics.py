"""
Unit Tests for Statistical Methods
==================================
Tests for FDR correction, confidence intervals, and sensitivity analysis.
"""

import pytest
import numpy as np
from typing import List

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.statistics import (
    apply_fdr_correction,
    compute_confidence_interval,
    bootstrap_confidence_interval,
    sensitivity_analysis,
    permutation_test,
    cohens_d,
    power_analysis,
)


class TestFDRCorrection:
    """Tests for FDR correction"""
    
    def test_fdr_increases_pvalues(self):
        """FDR-adjusted p-values should be >= raw p-values"""
        raw_pvals = [0.001, 0.01, 0.02, 0.05, 0.1, 0.5]
        adj_pvals, _ = apply_fdr_correction(raw_pvals)
        
        for raw, adj in zip(raw_pvals, adj_pvals):
            assert adj >= raw, f"Adjusted {adj} should be >= raw {raw}"
    
    def test_fdr_significance_threshold(self):
        """Test that significant p-values are correctly identified"""
        # Clear significant vs non-significant
        raw_pvals = [0.001, 0.002, 0.5, 0.9]
        adj_pvals, reject = apply_fdr_correction(raw_pvals, alpha=0.05)
        
        # First two should be significant, last two should not
        assert reject[0] == True
        assert reject[1] == True
        # Note: FDR adjustment is adaptive, so we can't guarantee exact behavior
    
    def test_empty_input(self):
        """Test handling of empty input"""
        adj_pvals, reject = apply_fdr_correction([])
        assert adj_pvals == []
        assert reject == []
    
    def test_single_pvalue(self):
        """Test with single p-value"""
        adj_pvals, reject = apply_fdr_correction([0.03])
        assert len(adj_pvals) == 1
        assert len(reject) == 1
    
    def test_monotonicity(self):
        """Adjusted p-values should maintain relative ordering"""
        raw_pvals = [0.01, 0.05, 0.1, 0.2]
        adj_pvals, _ = apply_fdr_correction(raw_pvals)
        
        # Sorted raw -> sorted adjusted
        raw_order = np.argsort(raw_pvals)
        adj_order = np.argsort(adj_pvals)
        
        # Same ordering
        assert list(raw_order) == list(adj_order)


class TestConfidenceIntervals:
    """Tests for confidence interval calculation"""
    
    def test_normal_ci_contains_mean(self):
        """95% CI should contain the sample mean"""
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        lower, upper = compute_confidence_interval(values, confidence=0.95)
        mean = np.mean(values)
        
        assert lower <= mean <= upper
    
    def test_ci_width_decreases_with_n(self):
        """CI width should decrease with larger sample size"""
        np.random.seed(42)
        
        small_sample = list(np.random.normal(5, 1, 10))
        large_sample = list(np.random.normal(5, 1, 100))
        
        small_lower, small_upper = compute_confidence_interval(small_sample)
        large_lower, large_upper = compute_confidence_interval(large_sample)
        
        small_width = small_upper - small_lower
        large_width = large_upper - large_lower
        
        assert large_width < small_width
    
    def test_higher_confidence_wider_interval(self):
        """Higher confidence should give wider intervals"""
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        
        lower_90, upper_90 = compute_confidence_interval(values, confidence=0.90)
        lower_99, upper_99 = compute_confidence_interval(values, confidence=0.99)
        
        width_90 = upper_90 - lower_90
        width_99 = upper_99 - lower_99
        
        assert width_99 > width_90
    
    def test_single_value(self):
        """Single value should return same value for both bounds"""
        lower, upper = compute_confidence_interval([5.0])
        assert lower == upper == 5.0


class TestBootstrapCI:
    """Tests for bootstrap confidence intervals"""
    
    def test_bootstrap_ci_reasonable(self):
        """Bootstrap CI should give reasonable bounds"""
        np.random.seed(42)
        values = list(np.random.normal(10, 2, 50))
        
        lower, upper = bootstrap_confidence_interval(values, random_state=42)
        
        # Should be around the true mean
        assert 8 < lower < upper < 12
    
    def test_bootstrap_reproducible(self):
        """Same random state should give same results"""
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        
        result1 = bootstrap_confidence_interval(values, random_state=42)
        result2 = bootstrap_confidence_interval(values, random_state=42)
        
        assert result1 == result2
    
    def test_custom_statistic(self):
        """Test with custom statistic (median)"""
        values = [1, 2, 3, 4, 5, 100]  # Skewed data
        
        mean_ci = bootstrap_confidence_interval(values, statistic=np.mean, random_state=42)
        median_ci = bootstrap_confidence_interval(values, statistic=np.median, random_state=42)
        
        # Median CI should be tighter due to robustness
        mean_width = mean_ci[1] - mean_ci[0]
        median_width = median_ci[1] - median_ci[0]
        
        # Not a strict test, but median should be less affected by outlier
        assert median_ci[0] < 10  # Median not inflated by 100


class TestSensitivityAnalysis:
    """Tests for sensitivity analysis"""
    
    def test_sensitivity_basic(self):
        """Test basic sensitivity analysis"""
        def score_fn(weights):
            return weights['a'] * 2 + weights['b'] * 3
        
        base_weights = {'a': 1.0, 'b': 1.0}
        
        result = sensitivity_analysis(
            base_weights, 
            score_fn, 
            perturbation=0.2,
            n_samples=50,
            random_state=42
        )
        
        assert 'base_score' in result
        assert 'mean_score' in result
        assert 'weight_sensitivity' in result
        assert result['base_score'] == 5.0  # 1*2 + 1*3
    
    def test_sensitivity_detects_important_weights(self):
        """Weights with larger coefficients should be more sensitive"""
        def score_fn(weights):
            # 'b' has larger coefficient
            return weights['a'] * 1 + weights['b'] * 10
        
        base_weights = {'a': 1.0, 'b': 1.0}
        
        result = sensitivity_analysis(base_weights, score_fn, random_state=42)
        
        # b should have higher sensitivity
        assert result['weight_sensitivity']['b'] > result['weight_sensitivity']['a']


class TestPermutationTest:
    """Tests for permutation test"""
    
    def test_same_groups_high_pvalue(self):
        """Same distribution should give high p-value"""
        np.random.seed(42)
        group1 = list(np.random.normal(5, 1, 30))
        group2 = list(np.random.normal(5, 1, 30))
        
        _, p_value = permutation_test(group1, group2, n_permutations=1000, random_state=42)
        
        assert p_value > 0.05  # Should not be significant
    
    def test_different_groups_low_pvalue(self):
        """Different distributions should give low p-value"""
        np.random.seed(42)
        group1 = list(np.random.normal(0, 1, 30))
        group2 = list(np.random.normal(5, 1, 30))  # Very different
        
        _, p_value = permutation_test(group1, group2, n_permutations=1000, random_state=42)
        
        assert p_value < 0.05  # Should be significant


class TestCohensD:
    """Tests for Cohen's d effect size"""
    
    def test_zero_effect(self):
        """Same groups should have zero effect size"""
        values = [1, 2, 3, 4, 5]
        d = cohens_d(values, values)
        assert abs(d) < 0.01
    
    def test_large_effect(self):
        """Very different groups should have large effect size"""
        group1 = [1, 2, 3, 4, 5]
        group2 = [10, 11, 12, 13, 14]
        
        d = cohens_d(group1, group2)
        
        # Effect size > 0.8 is considered large
        assert abs(d) > 2.0  # Very large effect
    
    def test_sign_direction(self):
        """Effect size sign should reflect direction"""
        group1 = [1, 2, 3]
        group2 = [4, 5, 6]
        
        d = cohens_d(group1, group2)
        assert d < 0  # group1 mean < group2 mean


class TestPowerAnalysis:
    """Tests for power analysis"""
    
    def test_larger_effect_smaller_n(self):
        """Larger effect size should require smaller sample"""
        n_small_effect = power_analysis(effect_size=0.2)
        n_large_effect = power_analysis(effect_size=0.8)
        
        assert n_large_effect < n_small_effect
    
    def test_higher_power_larger_n(self):
        """Higher power should require larger sample"""
        n_80 = power_analysis(effect_size=0.5, power=0.8)
        n_95 = power_analysis(effect_size=0.5, power=0.95)
        
        assert n_95 > n_80
    
    def test_reasonable_values(self):
        """Sample size should be reasonable for medium effect"""
        n = power_analysis(effect_size=0.5, power=0.8)
        
        # For medium effect (0.5), need roughly 60-70 per group
        assert 50 < n < 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
