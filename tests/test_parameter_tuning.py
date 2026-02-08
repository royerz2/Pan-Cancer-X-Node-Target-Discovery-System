"""
Tests for parameter_tuning module.

Tests cover:
- Parameter grid definitions and constraints
- Weight preset normalization
- TuningTrial objective computation
- Manual ARI implementation
- Grid construction logic
"""

import pytest
import numpy as np


# ---------------------------------------------------------------------------
# Test parameter space definitions
# ---------------------------------------------------------------------------

class TestParameterGrid:
    """Verify parameter grid is well-defined."""

    def test_param_grid_keys(self):
        from parameter_tuning import PARAM_GRID
        expected = {'dependency_threshold', 'min_selectivity_fraction',
                    'pan_essential_threshold', 'n_cluster_divisor',
                    'max_path_length'}
        assert set(PARAM_GRID.keys()) == expected

    def test_param_grid_values_are_lists(self):
        from parameter_tuning import PARAM_GRID
        for key, values in PARAM_GRID.items():
            assert isinstance(values, list), f"{key} should be a list"
            assert len(values) >= 2, f"{key} needs at least 2 values"

    def test_dependency_thresholds_are_negative(self):
        from parameter_tuning import PARAM_GRID
        for t in PARAM_GRID['dependency_threshold']:
            assert t < 0, f"Dependency threshold {t} should be negative"

    def test_selectivity_fractions_in_01(self):
        from parameter_tuning import PARAM_GRID
        for f in PARAM_GRID['min_selectivity_fraction']:
            assert 0 < f < 1, f"Selectivity fraction {f} should be in (0, 1)"

    def test_pan_essential_thresholds_in_01(self):
        from parameter_tuning import PARAM_GRID
        for t in PARAM_GRID['pan_essential_threshold']:
            assert 0 < t < 1, f"Pan-essential threshold {t} should be in (0, 1)"

    def test_path_lengths_are_positive(self):
        from parameter_tuning import PARAM_GRID
        for l in PARAM_GRID['max_path_length']:
            assert l > 0, f"Path length {l} should be positive"


# ---------------------------------------------------------------------------
# Test weight presets
# ---------------------------------------------------------------------------

class TestWeightPresets:
    """Verify weight presets are internally consistent."""

    def test_all_presets_have_same_keys(self):
        from parameter_tuning import WEIGHT_PRESETS
        keys_set = None
        for name, weights in WEIGHT_PRESETS.items():
            if keys_set is None:
                keys_set = set(weights.keys())
            else:
                assert set(weights.keys()) == keys_set, \
                    f"Preset '{name}' has different keys"

    def test_weights_sum_approximately_to_one(self):
        from parameter_tuning import WEIGHT_PRESETS
        for name, weights in WEIGHT_PRESETS.items():
            total = sum(weights.values())
            assert abs(total - 1.0) < 0.01, \
                f"Preset '{name}' weights sum to {total}, expected ~1.0"

    def test_all_weights_non_negative(self):
        from parameter_tuning import WEIGHT_PRESETS
        for name, weights in WEIGHT_PRESETS.items():
            for k, v in weights.items():
                assert v >= 0, f"Preset '{name}', weight '{k}' = {v} is negative"

    def test_original_preset_matches_defaults(self):
        from parameter_tuning import WEIGHT_PRESETS
        orig = WEIGHT_PRESETS['original']
        assert orig['cost'] == 0.22
        assert orig['synergy'] == 0.18
        assert orig['resistance'] == 0.18
        assert orig['coverage'] == 0.14
        assert orig['druggability'] == 0.10

    def test_at_least_five_presets(self):
        from parameter_tuning import WEIGHT_PRESETS
        assert len(WEIGHT_PRESETS) >= 5


# ---------------------------------------------------------------------------
# Test TuningTrial
# ---------------------------------------------------------------------------

class TestTuningTrial:
    """Test the TuningTrial dataclass."""

    def test_objective_computation(self):
        from parameter_tuning import TuningTrial
        trial = TuningTrial(
            trial_id=1,
            params={'dependency_threshold': -0.5},
            pairwise_recall=0.5,
            exact_recall=0.2,
            loco_pairwise_std=0.1,
        )
        obj = trial.compute_objective()
        expected = 0.5 + 0.10 * 0.2 - 0.05 * 0.1
        assert abs(obj - expected) < 1e-10

    def test_objective_zero_when_no_recall(self):
        from parameter_tuning import TuningTrial
        trial = TuningTrial(
            trial_id=2,
            params={},
            pairwise_recall=0.0,
            exact_recall=0.0,
            loco_pairwise_std=0.0,
        )
        assert trial.compute_objective() == 0.0

    def test_higher_recall_higher_objective(self):
        from parameter_tuning import TuningTrial
        t1 = TuningTrial(trial_id=1, params={}, pairwise_recall=0.3)
        t2 = TuningTrial(trial_id=2, params={}, pairwise_recall=0.7)
        assert t2.compute_objective() > t1.compute_objective()


# ---------------------------------------------------------------------------
# Test manual ARI
# ---------------------------------------------------------------------------

class TestManualARI:
    """Test the manual Adjusted Rand Index implementation."""

    def test_perfect_agreement(self):
        from parameter_tuning import _manual_ari
        labels = ['A', 'A', 'B', 'B', 'C', 'C']
        ari = _manual_ari(labels, labels)
        assert abs(ari - 1.0) < 1e-10

    def test_self_agreement(self):
        from parameter_tuning import _manual_ari
        labels = [1, 1, 2, 2, 3, 3, 3]
        ari = _manual_ari(labels, labels)
        assert abs(ari - 1.0) < 1e-10

    def test_random_labels_near_zero(self):
        from parameter_tuning import _manual_ari
        np.random.seed(42)
        n = 100
        true_labels = np.random.choice(5, n).tolist()
        pred_labels = np.random.choice(5, n).tolist()
        ari = _manual_ari(true_labels, pred_labels)
        assert -0.5 < ari < 0.5  # should be near zero for random

    def test_empty_returns_zero(self):
        from parameter_tuning import _manual_ari
        assert _manual_ari([], []) == 0.0

    def test_single_element(self):
        from parameter_tuning import _manual_ari
        assert _manual_ari(['A'], ['B']) == 0.0


# ---------------------------------------------------------------------------
# Test grid construction
# ---------------------------------------------------------------------------

class TestGridConstruction:
    """Test GridSearchTuner grid building logic."""

    def test_stage1_grid_size(self):
        """Full grid should be product of all param values."""
        from parameter_tuning import PARAM_GRID, GridSearchTuner, PipelineEvaluator
        import unittest.mock as mock

        # Don't actually instantiate the evaluator
        with mock.patch.object(PipelineEvaluator, '__init__', lambda self, **kw: None):
            evaluator = PipelineEvaluator.__new__(PipelineEvaluator)
            tuner = GridSearchTuner(evaluator, output_dir="/tmp/test_tuning")

        configs = tuner._build_stage1_grid()
        expected_size = 1
        for values in PARAM_GRID.values():
            expected_size *= len(values)
        assert len(configs) == expected_size

    def test_stage1_sampling(self):
        from parameter_tuning import GridSearchTuner, PipelineEvaluator
        import unittest.mock as mock

        with mock.patch.object(PipelineEvaluator, '__init__', lambda self, **kw: None):
            evaluator = PipelineEvaluator.__new__(PipelineEvaluator)
            tuner = GridSearchTuner(evaluator, output_dir="/tmp/test_tuning")

        configs = tuner._build_stage1_grid(sample_n=50)
        assert len(configs) == 50

    def test_stage2_grid_has_all_presets(self):
        from parameter_tuning import WEIGHT_PRESETS, GridSearchTuner, PipelineEvaluator
        import unittest.mock as mock

        with mock.patch.object(PipelineEvaluator, '__init__', lambda self, **kw: None):
            evaluator = PipelineEvaluator.__new__(PipelineEvaluator)
            tuner = GridSearchTuner(evaluator, output_dir="/tmp/test_tuning")

        best_upstream = {'dependency_threshold': -0.5, 'weight_preset': 'original'}
        configs = tuner._build_stage2_grid(best_upstream)
        assert len(configs) == len(WEIGHT_PRESETS)
        preset_names = {c['weight_preset'] for c in configs}
        assert preset_names == set(WEIGHT_PRESETS.keys())

    def test_all_configs_have_weight_preset(self):
        from parameter_tuning import GridSearchTuner, PipelineEvaluator
        import unittest.mock as mock

        with mock.patch.object(PipelineEvaluator, '__init__', lambda self, **kw: None):
            evaluator = PipelineEvaluator.__new__(PipelineEvaluator)
            tuner = GridSearchTuner(evaluator, output_dir="/tmp/test_tuning")

        configs = tuner._build_stage1_grid(sample_n=10)
        for c in configs:
            assert 'weight_preset' in c


# ---------------------------------------------------------------------------
# Test default params
# ---------------------------------------------------------------------------

class TestDefaultParams:
    """Ensure defaults are sensible."""

    def test_defaults_are_within_grid(self):
        from parameter_tuning import DEFAULT_PARAMS, PARAM_GRID
        for key in PARAM_GRID:
            assert DEFAULT_PARAMS[key] in PARAM_GRID[key], \
                f"Default {key}={DEFAULT_PARAMS[key]} not in grid {PARAM_GRID[key]}"

    def test_default_weight_preset_exists(self):
        from parameter_tuning import DEFAULT_PARAMS, WEIGHT_PRESETS
        assert DEFAULT_PARAMS['weight_preset'] in WEIGHT_PRESETS


# ---------------------------------------------------------------------------
# Tests for viability path enrichment validation (ORA)
# ---------------------------------------------------------------------------

class TestPathwayEnrichmentORA:
    """Tests for the hypergeometric ORA module."""

    def test_ora_pathway_sets_defined(self):
        from parameter_tuning import _ORA_PATHWAY_SETS
        assert isinstance(_ORA_PATHWAY_SETS, dict)
        assert len(_ORA_PATHWAY_SETS) >= 10, "Should have at least 10 pathway sets"
        for name, genes in _ORA_PATHWAY_SETS.items():
            assert isinstance(genes, set), f"{name} should be a set"
            assert len(genes) >= 5, f"{name} has too few genes ({len(genes)})"

    def test_ora_pathway_sets_no_overlap_major(self):
        """Major pathways should be mostly distinct."""
        from parameter_tuning import _ORA_PATHWAY_SETS
        mapk = _ORA_PATHWAY_SETS.get('MAPK_cascade', set())
        cell_cycle = _ORA_PATHWAY_SETS.get('Cell_cycle', set())
        # These should be largely distinct
        overlap = mapk & cell_cycle
        assert len(overlap) <= 2, f"MAPK and Cell_cycle share too many genes: {overlap}"

    def test_hypergeom_pvalue_perfect_overlap(self):
        from parameter_tuning import _hypergeom_pvalue
        # All pathway genes in query: should be very significant
        pval = _hypergeom_pvalue(k=10, M=18000, n=10, N=10)
        assert pval < 1e-10

    def test_hypergeom_pvalue_no_overlap(self):
        from parameter_tuning import _hypergeom_pvalue
        # k=0 overlap should give p ~= 1
        # sf(k-1, M, n, N) = sf(-1, ...) = 1.0
        pval = _hypergeom_pvalue(k=0, M=18000, n=100, N=100)
        assert pval > 0.99

    def test_hypergeom_pvalue_moderate(self):
        from parameter_tuning import _hypergeom_pvalue
        # 5 hits out of 50-gene query in 100-gene pathway from 18000 background
        pval = _hypergeom_pvalue(k=5, M=18000, n=100, N=50)
        assert 0.0 < pval < 1.0

    def test_validate_function_importable(self):
        """The enrichment validation function should be importable."""
        from parameter_tuning import validate_viability_paths_enrichment
        assert callable(validate_viability_paths_enrichment)

    def test_ora_total_gene_count(self):
        """All pathway gene sets combined should cover >100 unique genes."""
        from parameter_tuning import _ORA_PATHWAY_SETS
        all_genes = set()
        for genes in _ORA_PATHWAY_SETS.values():
            all_genes |= genes
        assert len(all_genes) >= 100, f"Only {len(all_genes)} unique genes across all pathways"
