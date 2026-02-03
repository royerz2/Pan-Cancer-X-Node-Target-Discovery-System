"""
Statistical Methods for ALIN Framework
======================================
FDR correction, confidence intervals, sensitivity analysis, and power analysis.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Callable, Any
from scipy import stats
import warnings


def apply_fdr_correction(p_values: List[float], 
                         method: str = 'fdr_bh',
                         alpha: float = 0.05) -> Tuple[List[float], List[bool]]:
    """
    Apply False Discovery Rate (FDR) correction for multiple hypothesis testing.
    
    This is CRITICAL for -omics studies where thousands of genes are tested.
    
    Args:
        p_values: List of raw p-values from statistical tests
        method: Correction method ('fdr_bh' for Benjamini-Hochberg, 
                'bonferroni', 'holm', 'fdr_by')
        alpha: Significance level (default 0.05)
        
    Returns:
        Tuple of (adjusted_pvalues, reject_null_hypothesis)
        
    Example:
        >>> p_vals = [0.001, 0.01, 0.04, 0.05, 0.1]
        >>> adj_p, significant = apply_fdr_correction(p_vals)
        >>> # adj_p will be higher than raw p-values
        >>> # significant will be [True, True, False, False, False] typically
    """
    try:
        from statsmodels.stats.multitest import multipletests
        reject, pvals_corrected, _, _ = multipletests(
            p_values, alpha=alpha, method=method
        )
        return list(pvals_corrected), list(reject)
    except ImportError:
        # Fallback: Benjamini-Hochberg manual implementation
        n = len(p_values)
        if n == 0:
            return [], []
        
        # Sort p-values and track original indices
        sorted_indices = np.argsort(p_values)
        sorted_pvals = np.array(p_values)[sorted_indices]
        
        # BH adjustment: p_adj[i] = min(p[i] * n / (i+1), 1.0)
        adjusted = np.zeros(n)
        for i in range(n):
            adjusted[i] = min(sorted_pvals[i] * n / (i + 1), 1.0)
        
        # Ensure monotonicity (larger indices can't have smaller adjusted p-values)
        for i in range(n - 2, -1, -1):
            adjusted[i] = min(adjusted[i], adjusted[i + 1])
        
        # Restore original order
        final_adjusted = np.zeros(n)
        final_adjusted[sorted_indices] = adjusted
        
        reject = final_adjusted < alpha
        return list(final_adjusted), list(reject)


def compute_confidence_interval(values: List[float], 
                                confidence: float = 0.95) -> Tuple[float, float]:
    """
    Compute confidence interval using normal approximation.
    
    Args:
        values: Sample values
        confidence: Confidence level (default 0.95 for 95% CI)
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if len(values) < 2:
        mean = np.mean(values) if values else 0.0
        return (mean, mean)
    
    n = len(values)
    mean = np.mean(values)
    se = stats.sem(values)  # Standard error of mean
    
    # t-distribution for small samples
    alpha = 1 - confidence
    t_crit = stats.t.ppf(1 - alpha/2, df=n-1)
    
    margin = t_crit * se
    return (mean - margin, mean + margin)


def bootstrap_confidence_interval(values: List[float],
                                  statistic: Callable = np.mean,
                                  n_bootstrap: int = 1000,
                                  confidence: float = 0.95,
                                  random_state: Optional[int] = None) -> Tuple[float, float]:
    """
    Compute confidence interval using bootstrap resampling.
    
    More robust than normal approximation for non-normal distributions.
    
    Args:
        values: Sample values
        statistic: Function to compute statistic (default: mean)
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if len(values) < 2:
        stat = statistic(values) if values else 0.0
        return (stat, stat)
    
    rng = np.random.RandomState(random_state)
    values_array = np.array(values)
    n = len(values_array)
    
    # Generate bootstrap samples
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample = rng.choice(values_array, size=n, replace=True)
        bootstrap_stats.append(statistic(sample))
    
    # Compute percentile CI
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_stats, alpha/2 * 100)
    upper = np.percentile(bootstrap_stats, (1 - alpha/2) * 100)
    
    return (lower, upper)


def sensitivity_analysis(base_weights: Dict[str, float],
                        score_function: Callable[[Dict[str, float]], float],
                        perturbation: float = 0.2,
                        n_samples: int = 100,
                        random_state: Optional[int] = None) -> Dict[str, Any]:
    """
    Perform sensitivity analysis on scoring weights.
    
    Tests how stable rankings are when weights are perturbed ±perturbation%.
    
    Args:
        base_weights: Dict of weight names to values (e.g., {'cost': 0.3, 'synergy': 0.25})
        score_function: Function that takes weights dict and returns score
        perturbation: Fraction to perturb weights (default 0.2 = ±20%)
        n_samples: Number of random perturbations to test
        random_state: Random seed
        
    Returns:
        Dict with:
            - 'base_score': Score with original weights
            - 'mean_score': Mean score across perturbations
            - 'std_score': Standard deviation
            - 'ci_95': 95% confidence interval
            - 'rank_stability': Fraction of runs with same rank order
            - 'weight_sensitivity': Dict of weight name to sensitivity score
    """
    rng = np.random.RandomState(random_state)
    
    # Base score
    base_score = score_function(base_weights)
    
    # Perturbed scores
    perturbed_scores = []
    weight_names = list(base_weights.keys())
    
    for _ in range(n_samples):
        perturbed = {}
        for name, value in base_weights.items():
            # Perturb by ±perturbation
            factor = 1 + rng.uniform(-perturbation, perturbation)
            perturbed[name] = value * factor
        
        try:
            score = score_function(perturbed)
            perturbed_scores.append(score)
        except Exception:
            continue
    
    if not perturbed_scores:
        return {
            'base_score': base_score,
            'mean_score': base_score,
            'std_score': 0.0,
            'ci_95': (base_score, base_score),
            'rank_stability': 1.0,
            'weight_sensitivity': {k: 0.0 for k in weight_names}
        }
    
    # Compute individual weight sensitivities
    weight_sensitivity = {}
    for name in weight_names:
        # Perturb only this weight
        plus_weights = base_weights.copy()
        plus_weights[name] *= (1 + perturbation)
        minus_weights = base_weights.copy()
        minus_weights[name] *= (1 - perturbation)
        
        try:
            plus_score = score_function(plus_weights)
            minus_score = score_function(minus_weights)
            sensitivity = abs(plus_score - minus_score) / (2 * perturbation * base_weights[name])
        except Exception:
            sensitivity = 0.0
        
        weight_sensitivity[name] = sensitivity
    
    return {
        'base_score': base_score,
        'mean_score': np.mean(perturbed_scores),
        'std_score': np.std(perturbed_scores),
        'ci_95': (np.percentile(perturbed_scores, 2.5), np.percentile(perturbed_scores, 97.5)),
        'rank_stability': np.mean([1 if abs(s - base_score) < 0.1 else 0 for s in perturbed_scores]),
        'weight_sensitivity': weight_sensitivity
    }


def permutation_test(group1: List[float], 
                     group2: List[float],
                     statistic: Callable = lambda x, y: np.mean(x) - np.mean(y),
                     n_permutations: int = 10000,
                     alternative: str = 'two-sided',
                     random_state: Optional[int] = None) -> Tuple[float, float]:
    """
    Permutation test for comparing two groups.
    
    Non-parametric alternative to t-test.
    
    Args:
        group1: Values for first group
        group2: Values for second group
        statistic: Function computing test statistic
        n_permutations: Number of permutations
        alternative: 'two-sided', 'greater', or 'less'
        random_state: Random seed
        
    Returns:
        Tuple of (observed_statistic, p_value)
    """
    rng = np.random.RandomState(random_state)
    
    combined = np.concatenate([group1, group2])
    n1 = len(group1)
    
    observed = statistic(np.array(group1), np.array(group2))
    
    count = 0
    for _ in range(n_permutations):
        rng.shuffle(combined)
        perm_stat = statistic(combined[:n1], combined[n1:])
        
        if alternative == 'two-sided':
            if abs(perm_stat) >= abs(observed):
                count += 1
        elif alternative == 'greater':
            if perm_stat >= observed:
                count += 1
        else:  # 'less'
            if perm_stat <= observed:
                count += 1
    
    p_value = (count + 1) / (n_permutations + 1)
    return observed, p_value


def power_analysis(effect_size: float,
                   alpha: float = 0.05,
                   power: float = 0.8,
                   test_type: str = 'two-sample') -> int:
    """
    Compute required sample size for desired statistical power.
    
    Args:
        effect_size: Expected effect size (Cohen's d for t-test)
        alpha: Significance level
        power: Desired power (1 - Type II error rate)
        test_type: 'two-sample' or 'one-sample'
        
    Returns:
        Required sample size per group
    """
    try:
        from statsmodels.stats.power import TTestIndPower, TTestPower
        
        if test_type == 'two-sample':
            analysis = TTestIndPower()
        else:
            analysis = TTestPower()
        
        n = analysis.solve_power(
            effect_size=effect_size,
            alpha=alpha,
            power=power,
            alternative='two-sided'
        )
        return int(np.ceil(n))
    except ImportError:
        # Rough approximation using normal distribution
        z_alpha = stats.norm.ppf(1 - alpha/2)
        z_beta = stats.norm.ppf(power)
        n = 2 * ((z_alpha + z_beta) / effect_size) ** 2
        return int(np.ceil(n))


def cohens_d(group1: List[float], group2: List[float]) -> float:
    """
    Compute Cohen's d effect size for two groups.
    
    Args:
        group1: Values for first group
        group2: Values for second group
        
    Returns:
        Cohen's d effect size
    """
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    
    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0.0
    
    return (mean1 - mean2) / pooled_std
