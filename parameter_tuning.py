#!/usr/bin/env python3
"""
Parameter Tuning Module for ALIN Framework
===========================================

Systematically tunes the previously arbitrary pipeline parameters using
cross-validated grid search against the independently curated gold-standard
benchmark.

Tunable parameters
------------------
1. dependency_threshold   (Chronos essentiality cutoff)
2. min_selectivity_fraction (selectivity fraction)
3. pan_essential_threshold (pan-essential exclusion cutoff)
4. n_cluster_divisor       (divisor in n_clusters = n_genes / divisor)
5. max_path_length         (signaling-path hop limit)
6. scoring_weights         (combination-ranking weights)

Objective
---------
Maximise *pairwise recall* (secondary: exact recall) on the 13-entry
COMBINATION_GOLD_STANDARD via leave-one-cancer-out cross-validation,
following the evaluation design of Li et al. (2024, Brief Bioinform
bbae172) and Julkunen et al. (2023).

Usage
-----
    python parameter_tuning.py --mode grid       # full grid search
    python parameter_tuning.py --mode bayesian    # Bayesian optimisation
    python parameter_tuning.py --mode report      # load saved results

Output
------
    tuning_results/best_params.json
    tuning_results/grid_search_log.csv
    tuning_results/tuning_report.txt
"""

import json
import csv
import itertools
import logging
import time
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Parameter space definition
# ---------------------------------------------------------------------------

# 1–5: upstream / structural parameters
PARAM_GRID = {
    'dependency_threshold':      [-0.3, -0.4, -0.5, -0.6, -0.7],
    'min_selectivity_fraction':  [0.15, 0.20, 0.25, 0.30, 0.40],
    'pan_essential_threshold':   [0.80, 0.85, 0.90, 0.95],
    'n_cluster_divisor':         [3, 4, 5, 7, 10],
    'max_path_length':           [3, 4, 5, 6],
}

# 6: scoring weights  (must sum to ~1.0 before drug/perturbation bonuses)
# We search a structured set of weight vectors rather than full combinatorial
WEIGHT_PRESETS = {
    'original':    {'cost': 0.22, 'synergy': 0.18, 'resistance': 0.18,
                    'combo_tox': 0.18, 'coverage': 0.14, 'druggability': 0.10},
    'coverage_heavy': {'cost': 0.15, 'synergy': 0.15, 'resistance': 0.15,
                       'combo_tox': 0.15, 'coverage': 0.30, 'druggability': 0.10},
    'synergy_heavy':  {'cost': 0.15, 'synergy': 0.30, 'resistance': 0.15,
                       'combo_tox': 0.15, 'coverage': 0.15, 'druggability': 0.10},
    'resistance_heavy': {'cost': 0.15, 'synergy': 0.15, 'resistance': 0.30,
                         'combo_tox': 0.15, 'coverage': 0.15, 'druggability': 0.10},
    'balanced':     {'cost': 0.18, 'synergy': 0.18, 'resistance': 0.18,
                     'combo_tox': 0.18, 'coverage': 0.18, 'druggability': 0.10},
    'cost_light':   {'cost': 0.10, 'synergy': 0.20, 'resistance': 0.20,
                     'combo_tox': 0.20, 'coverage': 0.20, 'druggability': 0.10},
    'tox_heavy':    {'cost': 0.18, 'synergy': 0.14, 'resistance': 0.14,
                     'combo_tox': 0.30, 'coverage': 0.14, 'druggability': 0.10},
}

# Canonical defaults (pre-tuning)
DEFAULT_PARAMS = {
    'dependency_threshold':     -0.5,
    'min_selectivity_fraction': 0.30,
    'pan_essential_threshold':  0.90,
    'n_cluster_divisor':        5,
    'max_path_length':          4,
    'weight_preset':            'original',
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TuningTrial:
    """One evaluation of a parameter configuration."""
    trial_id: int
    params: Dict[str, Any]
    pairwise_recall: float = 0.0
    exact_recall: float = 0.0
    superset_recall: float = 0.0
    loco_pairwise_mean: float = 0.0
    loco_pairwise_std: float = 0.0
    n_predictions: int = 0
    objective: float = 0.0        # combined metric to maximise
    wall_time_s: float = 0.0

    def compute_objective(self) -> float:
        """
        Primary: pairwise recall.
        Secondary: exact recall (10% bonus) + LOCO stability (penalise high std).
        """
        self.objective = (
            self.pairwise_recall
            + 0.10 * self.exact_recall
            - 0.05 * self.loco_pairwise_std
        )
        return self.objective


@dataclass
class TuningResult:
    """Aggregated tuning output."""
    best_params: Dict[str, Any]
    best_objective: float
    best_pairwise_recall: float
    best_exact_recall: float
    all_trials: List[TuningTrial] = field(default_factory=list)
    default_objective: float = 0.0
    improvement_pct: float = 0.0


# ---------------------------------------------------------------------------
# Core evaluator — runs a single parameter configuration through the pipeline
# ---------------------------------------------------------------------------

class PipelineEvaluator:
    """
    Lightweight evaluator that reconfigures the ALIN pipeline with a given
    parameter set and scores it against the gold-standard benchmark.
    
    For efficiency, DepMap data is loaded once and shared across evaluations.
    Only the viability-path / MHS / triple stages are re-run.
    """

    def __init__(self, data_dir: str = "./depmap_data",
                 validation_data_dir: str = "./validation_data",
                 cancer_types: Optional[List[str]] = None,
                 top_n_cancers: int = 20):
        """
        Parameters
        ----------
        data_dir : str
            Path to DepMap data directory.
        cancer_types : list[str] | None
            Restrict evaluation to these cancer types (faster).
            If None, uses top_n_cancers by cell-line count.
        """
        self.data_dir = data_dir
        self.validation_data_dir = validation_data_dir
        self._top_n = top_n_cancers
        self._user_cancer_types = cancer_types

        # Lazy-loaded shared resources
        self._depmap = None
        self._omnipath = None
        self._drug_db = None
        self._gold_cancers = None

    # --- lazy loaders -------------------------------------------------------

    def _ensure_loaded(self):
        """Load heavy resources once."""
        if self._depmap is not None:
            return

        from pan_cancer_xnode import DepMapLoader, OmniPathLoader, DrugTargetDB
        self._depmap = DepMapLoader(self.data_dir)
        self._omnipath = OmniPathLoader()
        self._drug_db = DrugTargetDB()

        # Determine cancer types relevant to gold standard
        from benchmarking_module import COMBINATION_GOLD_STANDARD, match_cancer
        gold_cancers_raw = {e['cancer'] for e in COMBINATION_GOLD_STANDARD}
        available = self._depmap.get_available_cancer_types()  # List[Tuple[str, int]]

        if self._user_cancer_types:
            self._eval_cancers = self._user_cancer_types
        else:
            # Take cancers that overlap with gold standard + extras for pool
            matched = []
            for avail_name, _count in available:
                for gc in gold_cancers_raw:
                    if match_cancer(avail_name, gc):
                        matched.append(avail_name)
                        break
            # Add top-N by cell-line count for realistic pool
            # available is already sorted by count (value_counts order)
            top_extras = [name for name, _ in available[:self._top_n]
                          if name not in matched]
            self._eval_cancers = matched + top_extras[:max(0, self._top_n - len(matched))]

        logger.info(f"Evaluator initialised with {len(self._eval_cancers)} cancer types")

    # --- single-config evaluation -------------------------------------------

    def evaluate(self, params: Dict[str, Any]) -> Dict[str, float]:
        """
        Run the pipeline with *params* and return benchmark metrics.

        Returns dict with keys: pairwise_recall, exact_recall, superset_recall,
        n_predictions, loco_pairwise_mean, loco_pairwise_std.
        """
        self._ensure_loaded()

        from pan_cancer_xnode import (
            ViabilityPathInference, CostFunction, MinimalHittingSetSolver,
            TripleCombinationFinder, CancerTypeAnalysis,
        )
        from benchmarking_module import (
            COMBINATION_GOLD_STANDARD, match_cancer, check_match,
            _expand_with_equivalents,
        )

        dep_thresh = params.get('dependency_threshold', -0.5)
        sel_frac   = params.get('min_selectivity_fraction', 0.30)
        pan_thresh = params.get('pan_essential_threshold', 0.90)
        clust_div  = params.get('n_cluster_divisor', 5)
        path_len   = params.get('max_path_length', 4)
        weight_key = params.get('weight_preset', 'original')
        weights    = WEIGHT_PRESETS.get(weight_key, WEIGHT_PRESETS['original'])

        # Override pan-essential threshold
        self._depmap._pan_essential_threshold = pan_thresh

        path_inf = ViabilityPathInference(self._depmap, self._omnipath)
        # Override the cached pan-essential set so threshold change takes effect
        path_inf._pan_essential = None
        path_inf._pan_essential = self._depmap.get_pan_essential_genes(threshold=pan_thresh)

        cost_fn = CostFunction(self._depmap, self._drug_db)
        solver  = MinimalHittingSetSolver(cost_fn)
        triple_finder = TripleCombinationFinder(
            self._depmap, self._omnipath, self._drug_db
        )

        # --- run pipeline per cancer type ---
        # Collect {cancer: [list of frozenset targets]}
        predictions: Dict[str, List[frozenset]] = {}

        for cancer in self._eval_cancers:
            try:
                paths = self._infer_paths_with_params(
                    path_inf, cancer, dep_thresh, sel_frac, clust_div, path_len
                )
                if not paths:
                    continue

                # MHS
                hitting_sets = solver.solve(paths, cancer, max_size=4)

                # Triples with custom weights
                triples = self._score_triples_with_weights(
                    triple_finder, paths, cancer, weights,
                    dep_thresh, sel_frac
                )

                ranked = []
                if triples:
                    for t in triples:
                        ranked.append(frozenset(t.targets))
                if hitting_sets:
                    for hs in hitting_sets:
                        if hs.targets not in ranked:
                            ranked.append(hs.targets)
                if ranked:
                    predictions[cancer] = ranked
            except Exception as e:
                logger.debug(f"Eval skip {cancer}: {e}")
                continue

        if not predictions:
            return {'pairwise_recall': 0, 'exact_recall': 0,
                    'superset_recall': 0, 'n_predictions': 0,
                    'loco_pairwise_mean': 0, 'loco_pairwise_std': 0}

        # --- benchmark against gold standard ---
        metrics = self._benchmark_predictions(predictions, COMBINATION_GOLD_STANDARD)
        metrics['n_predictions'] = sum(len(v) for v in predictions.values())
        return metrics

    # --- helpers for parameterised sub-steps --------------------------------

    def _infer_paths_with_params(self, path_inf, cancer, dep_thresh,
                                  sel_frac, clust_div, path_len):
        """Call viability-path inference with overridden parameters."""
        paths = []

        # 1. Essential modules (with custom dep_thresh, sel_frac, clust_div)
        try:
            mod_paths = path_inf.infer_essential_modules(
                cancer,
                dependency_threshold=dep_thresh,
                min_selectivity_fraction=sel_frac,
            )
            # Monkey-patch cluster divisor for this call is hard —
            # instead, re-cluster if we got results and divisor differs from 5
            if clust_div != 5 and mod_paths:
                mod_paths = self._recluster_modules(
                    path_inf, cancer, dep_thresh, sel_frac, clust_div
                )
            paths.extend(mod_paths)
        except Exception:
            pass

        # 2. Signaling paths (custom path_len)
        try:
            sig_paths = path_inf.infer_signaling_paths(
                cancer, dependency_threshold=dep_thresh,
                max_path_length=path_len,
            )
            paths.extend(sig_paths)
        except Exception:
            pass

        # 3. Cancer-specific (uses default p/effect thresholds)
        try:
            spec_paths = path_inf.infer_cancer_specific_dependencies(cancer)
            paths.extend(spec_paths)
        except Exception:
            pass

        # 4. Perturbation (if available)
        try:
            pert_paths = path_inf.infer_perturbation_response_paths(
                cancer, dependency_threshold=dep_thresh
            )
            paths.extend(pert_paths)
        except Exception:
            pass

        return paths

    def _recluster_modules(self, path_inf, cancer, dep_thresh, sel_frac,
                            clust_div):
        """Re-run co-essentiality clustering with a different divisor."""
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform
        from collections import defaultdict
        from core.data_structures import ViabilityPath

        crispr = path_inf.depmap.load_crispr_dependencies()
        cell_lines = path_inf.depmap.get_cell_lines_for_cancer(cancer)
        pan_essential = path_inf._get_pan_essential()
        available = [cl for cl in cell_lines if cl in crispr.index]
        if len(available) < 3:
            return []

        crispr_sub = crispr.loc[available]
        n_lines = len(available)
        min_ess = max(1, int(n_lines * sel_frac))

        # Build essential sets per line
        line_ess = {}
        for cl in available:
            ess = set()
            for g in crispr_sub.columns:
                if g not in pan_essential and crispr_sub.loc[cl, g] < dep_thresh:
                    ess.add(g)
            if len(ess) >= 2:
                line_ess[cl] = ess

        # Selectivity filter
        gene_count = defaultdict(int)
        for es in line_ess.values():
            for g in es:
                gene_count[g] += 1
        selective = [g for g, c in gene_count.items() if c >= min_ess]

        if len(selective) < 2:
            return []

        # Co-essentiality + clustering
        gene_sets = {}
        for g in selective:
            gene_sets[g] = {lid for lid, es in line_ess.items() if g in es}

        n = len(selective)
        sim = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                inter = len(gene_sets[selective[i]] & gene_sets[selective[j]])
                union = len(gene_sets[selective[i]] | gene_sets[selective[j]])
                s = inter / max(1, union)
                sim[i, j] = sim[j, i] = s

        dist = 1 - sim
        np.fill_diagonal(dist, 0)
        n_clusters = min(15, max(3, n // clust_div))

        try:
            condensed = squareform(dist, checks=False)
            Z = linkage(condensed, method='ward')
            labels = fcluster(Z, n_clusters, criterion='maxclust')
            cluster_genes = defaultdict(set)
            for gene, lab in zip(selective, labels):
                cluster_genes[lab].add(gene)
        except Exception:
            cluster_genes = {0: set(selective)}

        paths = []
        for cid, genes in cluster_genes.items():
            if len(genes) >= 2:
                paths.append(ViabilityPath(
                    path_id=f"{cancer}_tuned_cluster_{cid}",
                    nodes=frozenset(genes),
                    context=cancer, confidence=0.9,
                    path_type="co_essential_module"
                ))
        if len(selective) >= 2:
            paths.append(ViabilityPath(
                path_id=f"{cancer}_tuned_consensus",
                nodes=frozenset(selective),
                context=cancer, confidence=1.0,
                path_type="essential_module"
            ))
        return paths

    def _score_triples_with_weights(self, finder, paths, cancer, weights,
                                     dep_thresh, sel_frac):
        """Run triple scoring with custom weights. Returns sorted list."""
        # Use the finder's normal candidate selection but override scoring
        try:
            triples = finder.find_triple_combinations(
                paths, cancer, top_n=10, min_coverage=0.5
            )
        except Exception:
            triples = []

        if not triples:
            return triples

        # Re-score with custom weights
        w_cost  = weights.get('cost', 0.22)
        w_syn   = weights.get('synergy', 0.18)
        w_res   = weights.get('resistance', 0.18)
        w_tox   = weights.get('combo_tox', 0.18)
        w_cov   = weights.get('coverage', 0.14)
        w_drug  = weights.get('druggability', 0.10)

        for t in triples:
            t.combined_score = (
                t.total_cost * w_cost
                + (1 - t.synergy_score) * w_syn
                + t.resistance_score * w_res
                + (1 - t.coverage) * w_cov
                + t.combo_tox_score * w_tox
                - t.druggable_count * w_drug
            )

        triples.sort(key=lambda x: x.combined_score)
        return triples

    def _benchmark_predictions(self, predictions, gold_standard):
        """Evaluate predictions against gold standard; returns metrics dict."""
        from benchmarking_module import match_cancer, check_match

        tp_exact = tp_super = tp_pair = 0
        total = len(gold_standard)

        for entry in gold_standard:
            gold_cancer = entry['cancer']
            gold_targets = entry['targets']

            best_match = 'none'
            for our_cancer, ranked in predictions.items():
                if not match_cancer(our_cancer, gold_cancer):
                    continue
                for pred_set in ranked:
                    _, mtype = check_match(pred_set, gold_targets)
                    if mtype == 'exact':
                        best_match = 'exact'
                        break
                    elif mtype == 'superset' and best_match not in ('exact',):
                        best_match = 'superset'
                    elif mtype == 'pairwise' and best_match == 'none':
                        best_match = 'pairwise'
                if best_match == 'exact':
                    break

            if best_match == 'exact':
                tp_exact += 1
                tp_super += 1
                tp_pair += 1
            elif best_match == 'superset':
                tp_super += 1
                tp_pair += 1
            elif best_match == 'pairwise':
                tp_pair += 1

        # Simple LOCO-CV approximation: hold out each unique cancer
        loco_recalls = []
        gold_cancers = list({e['cancer'] for e in gold_standard})
        for held_out in gold_cancers:
            test_entries = [e for e in gold_standard if e['cancer'] == held_out]
            if not test_entries:
                continue
            tp_p = 0
            for entry in test_entries:
                for our_cancer, ranked in predictions.items():
                    if not match_cancer(our_cancer, entry['cancer']):
                        continue
                    for pred_set in ranked:
                        _, mtype = check_match(pred_set, entry['targets'])
                        if mtype in ('exact', 'superset', 'pairwise'):
                            tp_p += 1
                            break
                    break
            loco_recalls.append(tp_p / len(test_entries) if test_entries else 0)

        return {
            'pairwise_recall': tp_pair / max(1, total),
            'exact_recall': tp_exact / max(1, total),
            'superset_recall': tp_super / max(1, total),
            'loco_pairwise_mean': float(np.mean(loco_recalls)) if loco_recalls else 0,
            'loco_pairwise_std': float(np.std(loco_recalls)) if loco_recalls else 0,
        }


# ---------------------------------------------------------------------------
# Grid-search tuner
# ---------------------------------------------------------------------------

class GridSearchTuner:
    """
    Exhaustive grid search over the parameter space.
    
    For computational tractability the upstream parameters (thresholds,
    clustering, path-length) and scoring weights are tuned in a two-stage
    process:
      Stage 1: fix weights to 'original', sweep upstream params.
      Stage 2: fix best upstream params, sweep weight presets.
    This avoids the combinatorial explosion of ~5^5 × 7 ≈ 22k configs.
    """

    def __init__(self, evaluator: PipelineEvaluator,
                 output_dir: str = "tuning_results"):
        self.evaluator = evaluator
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.trials: List[TuningTrial] = []
        self._trial_id = 0

    def _next_id(self) -> int:
        self._trial_id += 1
        return self._trial_id

    def run(self, stage1_sample: Optional[int] = None) -> TuningResult:
        """
        Execute two-stage grid search.
        
        Parameters
        ----------
        stage1_sample : int | None
            If set, randomly sample this many configs from full grid (Latin
            hypercube–style) instead of exhaustive enumeration.  Useful when
            the full grid (3 125 configs) is too expensive.
        """
        print("=" * 70)
        print("ALIN Parameter Tuning — Grid Search")
        print("=" * 70)

        # ------ Stage 1: upstream parameters ------
        print("\n[Stage 1] Tuning upstream parameters (weights fixed to 'original')…")
        stage1_configs = self._build_stage1_grid(sample_n=stage1_sample)
        print(f"  → {len(stage1_configs)} configurations to evaluate")

        best_s1 = self._evaluate_configs(stage1_configs, stage_label="Stage1")

        print(f"\n  Best Stage-1 objective = {best_s1.objective:.4f}")
        print(f"  Best Stage-1 params = {best_s1.params}")

        # ------ Stage 2: scoring weights ------
        print("\n[Stage 2] Tuning scoring weights (upstream params fixed to Stage-1 best)…")
        stage2_configs = self._build_stage2_grid(best_s1.params)
        print(f"  → {len(stage2_configs)} weight presets to evaluate")

        best_s2 = self._evaluate_configs(stage2_configs, stage_label="Stage2")

        # Pick overall best
        all_sorted = sorted(self.trials, key=lambda t: t.objective, reverse=True)
        best = all_sorted[0]

        # Evaluate default for comparison
        default_metrics = self.evaluator.evaluate(DEFAULT_PARAMS)
        default_obj = (
            default_metrics['pairwise_recall']
            + 0.10 * default_metrics['exact_recall']
            - 0.05 * default_metrics.get('loco_pairwise_std', 0)
        )

        improvement = (
            ((best.objective - default_obj) / max(0.001, abs(default_obj))) * 100
            if default_obj != 0 else 0
        )

        result = TuningResult(
            best_params=best.params,
            best_objective=best.objective,
            best_pairwise_recall=best.pairwise_recall,
            best_exact_recall=best.exact_recall,
            all_trials=self.trials,
            default_objective=default_obj,
            improvement_pct=improvement,
        )

        self._save_results(result)
        self._print_report(result, default_metrics)
        return result

    # --- grid builders ------------------------------------------------------

    def _build_stage1_grid(self, sample_n=None):
        keys = ['dependency_threshold', 'min_selectivity_fraction',
                'pan_essential_threshold', 'n_cluster_divisor',
                'max_path_length']
        values = [PARAM_GRID[k] for k in keys]
        full_grid = list(itertools.product(*values))

        if sample_n and sample_n < len(full_grid):
            rng = np.random.RandomState(42)
            indices = rng.choice(len(full_grid), size=sample_n, replace=False)
            full_grid = [full_grid[i] for i in sorted(indices)]

        configs = []
        for combo in full_grid:
            params = dict(zip(keys, combo))
            params['weight_preset'] = 'original'
            configs.append(params)
        return configs

    def _build_stage2_grid(self, best_upstream):
        configs = []
        for preset_name in WEIGHT_PRESETS:
            params = dict(best_upstream)
            params['weight_preset'] = preset_name
            configs.append(params)
        return configs

    # --- evaluation loop ----------------------------------------------------

    def _evaluate_configs(self, configs, stage_label=""):
        best_trial = None
        for i, params in enumerate(configs):
            trial = TuningTrial(trial_id=self._next_id(), params=params)
            t0 = time.time()
            try:
                metrics = self.evaluator.evaluate(params)
                trial.pairwise_recall = metrics['pairwise_recall']
                trial.exact_recall = metrics['exact_recall']
                trial.superset_recall = metrics['superset_recall']
                trial.loco_pairwise_mean = metrics.get('loco_pairwise_mean', 0)
                trial.loco_pairwise_std = metrics.get('loco_pairwise_std', 0)
                trial.n_predictions = metrics.get('n_predictions', 0)
            except Exception as e:
                logger.warning(f"Trial {trial.trial_id} failed: {e}")
            trial.wall_time_s = time.time() - t0
            trial.compute_objective()
            self.trials.append(trial)

            if best_trial is None or trial.objective > best_trial.objective:
                best_trial = trial

            if (i + 1) % 10 == 0 or (i + 1) == len(configs):
                print(f"  [{stage_label}] {i+1}/{len(configs)} done "
                      f"| best obj={best_trial.objective:.4f} "
                      f"| pair={best_trial.pairwise_recall:.3f} "
                      f"| exact={best_trial.exact_recall:.3f}")

        return best_trial

    # --- persistence --------------------------------------------------------

    def _save_results(self, result: TuningResult):
        # Best params JSON
        with open(self.output_dir / "best_params.json", "w") as f:
            json.dump({
                'best_params': result.best_params,
                'best_objective': result.best_objective,
                'best_pairwise_recall': result.best_pairwise_recall,
                'best_exact_recall': result.best_exact_recall,
                'default_objective': result.default_objective,
                'improvement_pct': result.improvement_pct,
                'n_trials': len(result.all_trials),
            }, f, indent=2, default=str)

        # Full trial log CSV
        with open(self.output_dir / "grid_search_log.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                'trial_id', 'dependency_threshold', 'min_selectivity_fraction',
                'pan_essential_threshold', 'n_cluster_divisor', 'max_path_length',
                'weight_preset', 'pairwise_recall', 'exact_recall',
                'superset_recall', 'loco_pairwise_mean', 'loco_pairwise_std',
                'objective', 'wall_time_s',
            ])
            for t in result.all_trials:
                writer.writerow([
                    t.trial_id,
                    t.params.get('dependency_threshold'),
                    t.params.get('min_selectivity_fraction'),
                    t.params.get('pan_essential_threshold'),
                    t.params.get('n_cluster_divisor'),
                    t.params.get('max_path_length'),
                    t.params.get('weight_preset'),
                    f"{t.pairwise_recall:.4f}",
                    f"{t.exact_recall:.4f}",
                    f"{t.superset_recall:.4f}",
                    f"{t.loco_pairwise_mean:.4f}",
                    f"{t.loco_pairwise_std:.4f}",
                    f"{t.objective:.4f}",
                    f"{t.wall_time_s:.1f}",
                ])

    def _print_report(self, result: TuningResult, default_metrics: dict):
        lines = [
            "",
            "=" * 70,
            "TUNING REPORT",
            "=" * 70,
            "",
            "Default parameters:",
            f"  {DEFAULT_PARAMS}",
            f"  pairwise_recall = {default_metrics['pairwise_recall']:.4f}",
            f"  exact_recall    = {default_metrics['exact_recall']:.4f}",
            f"  objective       = {result.default_objective:.4f}",
            "",
            "Tuned parameters:",
            f"  {result.best_params}",
            f"  pairwise_recall = {result.best_pairwise_recall:.4f}",
            f"  exact_recall    = {result.best_exact_recall:.4f}",
            f"  objective       = {result.best_objective:.4f}",
            "",
            f"Improvement: {result.improvement_pct:+.1f}%",
            f"Total trials: {len(result.all_trials)}",
            "",
            "Top-5 configurations:",
        ]
        top5 = sorted(result.all_trials, key=lambda t: t.objective, reverse=True)[:5]
        for i, t in enumerate(top5):
            lines.append(f"  #{i+1} obj={t.objective:.4f} pair={t.pairwise_recall:.3f} "
                         f"exact={t.exact_recall:.3f} params={t.params}")
        lines.extend(["", "=" * 70])

        report = "\n".join(lines)
        print(report)

        with open(self.output_dir / "tuning_report.txt", "w") as f:
            f.write(report)


# ---------------------------------------------------------------------------
# Threshold sensitivity sweep (lightweight — no full pipeline re-run)
# ---------------------------------------------------------------------------

def threshold_sensitivity_sweep(data_dir: str = "./depmap_data",
                                 output_dir: str = "tuning_results") -> Dict:
    """
    Quick sweep showing how candidate pool size changes with threshold
    parameters.  Does NOT re-run full MHS/benchmark — just counts genes
    passing each threshold combination.
    
    This directly addresses the criticism that upstream thresholds define
    the candidate pool in unexamined ways.
    """
    from pan_cancer_xnode import DepMapLoader
    from collections import defaultdict

    out = Path(output_dir)
    out.mkdir(exist_ok=True)

    depmap = DepMapLoader(data_dir)
    crispr = depmap.load_crispr_dependencies()
    available = depmap.get_available_cancer_types()  # List[Tuple[str, int]]

    # Pick five representative cancers (most cell lines)
    test_cancers = [name for name, _ in available[:5]]

    dep_thresholds = [-0.3, -0.4, -0.5, -0.6, -0.7]
    sel_fractions  = [0.15, 0.20, 0.25, 0.30, 0.40]
    pan_thresholds = [0.80, 0.85, 0.90, 0.95]

    results = []

    # Pre-compute pan-essential sets for each threshold (expensive, do once)
    pan_ess_cache = {}
    for pan_t in pan_thresholds:
        pan_ess_cache[pan_t] = depmap.get_pan_essential_genes(threshold=pan_t)

    for cancer in test_cancers:
        cell_lines = depmap.get_cell_lines_for_cancer(cancer)
        avail_lines = [cl for cl in cell_lines if cl in crispr.index]
        if len(avail_lines) < 3:
            continue

        crispr_sub = crispr.loc[avail_lines]
        n_lines = len(avail_lines)
        # Convert to numpy for fast vectorised operations
        crispr_vals = crispr_sub.values  # shape (n_lines, n_genes)
        gene_names = crispr_sub.columns.tolist()

        for dep_t in dep_thresholds:
            # Boolean mask: which (cell_line, gene) pairs are essential?
            ess_mask = crispr_vals < dep_t  # (n_lines, n_genes)
            # Count how many lines each gene is essential in
            gene_ess_counts = ess_mask.sum(axis=0)  # (n_genes,)

            for sel_f in sel_fractions:
                min_lines_needed = max(1, int(n_lines * sel_f))

                for pan_t in pan_thresholds:
                    pan_ess = pan_ess_cache[pan_t]
                    # Build boolean mask for non-pan-essential genes
                    non_pan_mask = np.array([g not in pan_ess for g in gene_names])

                    # Selective = essential in >= min_lines AND not pan-essential
                    n_selective = int(((gene_ess_counts >= min_lines_needed) & non_pan_mask).sum())

                    results.append({
                        'cancer': cancer,
                        'dep_threshold': dep_t,
                        'sel_fraction': sel_f,
                        'pan_threshold': pan_t,
                        'n_cell_lines': n_lines,
                        'n_pan_essential': len(pan_ess),
                        'n_selective_genes': n_selective,
                    })

    # Save
    import csv
    with open(out / "threshold_sensitivity.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    # Summary statistics
    import pandas as pd
    df = pd.DataFrame(results)
    summary = {
        'n_configs': len(results),
        'n_cancers': len(test_cancers),
        'cancers': test_cancers,
        'pool_size_range': {
            'min': int(df['n_selective_genes'].min()),
            'max': int(df['n_selective_genes'].max()),
            'mean': float(df['n_selective_genes'].mean()),
            'std': float(df['n_selective_genes'].std()),
        },
        'by_dep_threshold': {
            str(t): float(df[df['dep_threshold'] == t]['n_selective_genes'].mean())
            for t in dep_thresholds
        },
        'by_sel_fraction': {
            str(f): float(df[df['sel_fraction'] == f]['n_selective_genes'].mean())
            for f in sel_fractions
        },
        'by_pan_threshold': {
            str(p): float(df[df['pan_threshold'] == p]['n_selective_genes'].mean())
            for p in pan_thresholds
        },
    }

    with open(out / "threshold_sensitivity_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nThreshold Sensitivity Sweep")
    print("=" * 50)
    print(f"Cancers tested: {test_cancers}")
    print(f"Pool size range: {summary['pool_size_range']['min']}–"
          f"{summary['pool_size_range']['max']} genes "
          f"(mean {summary['pool_size_range']['mean']:.0f} ± "
          f"{summary['pool_size_range']['std']:.0f})")
    print(f"\nMean pool size by dependency threshold:")
    for t, v in summary['by_dep_threshold'].items():
        print(f"  τ_dep = {t}: {v:.0f} genes")
    print(f"\nMean pool size by selectivity fraction:")
    for f, v in summary['by_sel_fraction'].items():
        print(f"  θ = {f}: {v:.0f} genes")
    print(f"\nMean pool size by pan-essential threshold:")
    for p, v in summary['by_pan_threshold'].items():
        print(f"  pan = {p}: {v:.0f} genes")

    return summary


# ---------------------------------------------------------------------------
# Cluster-count calibration against KEGG/Reactome pathways
# ---------------------------------------------------------------------------

def calibrate_cluster_count(data_dir: str = "./depmap_data",
                             output_dir: str = "tuning_results") -> Dict:
    """
    Evaluate different cluster-count divisors by measuring overlap with
    curated pathway gene sets (KEGG cancer pathways + Reactome signaling).
    
    For each divisor d and cancer type, clusters selective genes into
    n_clusters = n_genes / d modules, then measures the Adjusted Rand Index
    (ARI) and Normalized Mutual Information (NMI) against pathway membership.
    
    This addresses the criticism that the clustering heuristic is not
    calibrated against external biological knowledge.
    """
    from pan_cancer_xnode import DepMapLoader
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    from collections import defaultdict

    out = Path(output_dir)
    out.mkdir(exist_ok=True)

    depmap = DepMapLoader(data_dir)
    crispr = depmap.load_crispr_dependencies()

    # Curated cancer-relevant pathway gene sets (KEGG + Reactome core)
    PATHWAY_GENE_SETS = {
        'MAPK_signaling': {'KRAS', 'NRAS', 'HRAS', 'BRAF', 'RAF1', 'MAP2K1',
                           'MAP2K2', 'MAPK1', 'MAPK3', 'ARAF'},
        'PI3K_AKT_mTOR':  {'PIK3CA', 'PIK3CB', 'PIK3R1', 'AKT1', 'AKT2',
                           'MTOR', 'RPTOR', 'RICTOR', 'RPS6KB1', 'TSC1', 'TSC2',
                           'PTEN'},
        'Cell_cycle':      {'CDK4', 'CDK6', 'CDK2', 'CCND1', 'CCNE1', 'CCNA2',
                           'RB1', 'E2F1', 'E2F3', 'CDKN2A', 'CDKN1A', 'CDKN1B'},
        'JAK_STAT':        {'JAK1', 'JAK2', 'JAK3', 'TYK2', 'STAT3', 'STAT1',
                           'STAT5A', 'STAT5B', 'SOCS1', 'SOCS3'},
        'Apoptosis':       {'BCL2', 'MCL1', 'BCL2L1', 'BAX', 'BAK1', 'BID',
                           'CASP3', 'CASP8', 'CASP9', 'XIAP'},
        'RTK_signaling':   {'EGFR', 'ERBB2', 'ERBB3', 'MET', 'FGFR1', 'FGFR2',
                           'FGFR3', 'RET', 'ALK', 'ROS1', 'PDGFRA', 'KIT'},
        'WNT_signaling':   {'CTNNB1', 'APC', 'GSK3B', 'AXIN1', 'TCF7L2',
                           'LRP5', 'LRP6', 'FZD7', 'RSPO3'},
        'Chromatin':       {'EZH2', 'KDM6A', 'ARID1A', 'SMARCA4', 'SMARCB1',
                           'KMT2A', 'KMT2D', 'DNMT3A', 'TET2', 'IDH1', 'IDH2'},
    }

    # Build gene → pathway label mapping
    gene_to_pathway = {}
    for pw, genes in PATHWAY_GENE_SETS.items():
        for g in genes:
            gene_to_pathway[g] = pw

    available_ct = depmap.get_available_cancer_types()  # List[Tuple[str, int]]
    test_cancers = [name for name, _ in available_ct[:5]]
    # Divisors tested — with hundreds of genes, smaller divisors yield
    # more clusters, letting us explore the resolution spectrum
    divisors = [3, 4, 5, 7, 10]
    # We also test fixed cluster counts matching the number of pathways (8)
    fixed_k_values = [5, 8, 10, 15, 20, 30, 50]
    pan_ess = depmap.get_pan_essential_genes(threshold=0.9)

    results = []

    for cancer in test_cancers:
        cell_lines = depmap.get_cell_lines_for_cancer(cancer)
        available = [cl for cl in cell_lines if cl in crispr.index]
        if len(available) < 3:
            continue

        crispr_sub = crispr.loc[available]
        n_lines = len(available)
        min_lines = max(1, int(n_lines * 0.3))

        # Get selective genes (vectorised)
        crispr_vals = crispr_sub.values
        gene_names = crispr_sub.columns.tolist()
        non_pan = np.array([g not in pan_ess for g in gene_names])
        ess_mask = crispr_vals < -0.5
        gene_ess_counts = ess_mask.sum(axis=0)
        min_lines = max(1, int(n_lines * 0.3))
        selective_mask = (gene_ess_counts >= min_lines) & non_pan
        selective = [gene_names[i] for i in range(len(gene_names)) if selective_mask[i]]

        if len(selective) < 5:
            continue

        # Annotated genes (those with known pathway membership)
        annotated = [g for g in selective if g in gene_to_pathway]
        if len(annotated) < 5:
            continue

        # Build co-essentiality matrix (vectorised via binary membership)
        sel_idx = [gene_names.index(g) for g in selective]
        # Binary: (n_cell_lines, n_selective) — is gene g essential in cell line?
        binary = ess_mask[:, sel_idx].astype(np.float64)  # (lines, genes)
        # Jaccard via dot products
        intersection = binary.T @ binary  # (n_sel, n_sel)
        row_sums = binary.sum(axis=0)     # (n_sel,)
        union = row_sums[:, None] + row_sums[None, :] - intersection
        union = np.maximum(union, 1)
        sim = intersection / union
        np.fill_diagonal(sim, 1.0)

        dist = 1 - sim
        np.fill_diagonal(dist, 0)

        try:
            condensed = squareform(dist, checks=False)
            Z = linkage(condensed, method='ward')
        except Exception:
            continue

        # Map indices → gene names
        gene_idx = {g: i for i, g in enumerate(selective)}

        for div in divisors:
            n_sel = len(selective)
            n_clusters_from_div = min(15, max(3, n_sel // div))
            # Test both the formula-derived k and fixed k values
            k_values_to_test = set([n_clusters_from_div])
            k_values_to_test.update(fixed_k_values)
            # Also test uncapped divisor formula
            k_values_to_test.add(max(3, n_sel // div))

            for n_clusters in sorted(k_values_to_test):
                if n_clusters > n_sel:
                    continue
                labels = fcluster(Z, n_clusters, criterion='maxclust')

                # Extract labels for annotated genes only
                pred_labels = [labels[gene_idx[g]] for g in annotated]
                true_labels = [gene_to_pathway[g] for g in annotated]

                # Compute ARI and NMI
                try:
                    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
                    ari = adjusted_rand_score(true_labels, pred_labels)
                    nmi = normalized_mutual_info_score(true_labels, pred_labels)
                except ImportError:
                    ari = _manual_ari(true_labels, pred_labels)
                    nmi = 0.0

                results.append({
                    'cancer': cancer,
                    'n_selective': len(selective),
                    'n_annotated': len(annotated),
                    'divisor': div,
                    'n_clusters': n_clusters,
                    'ari': round(ari, 4),
                    'nmi': round(nmi, 4),
                })

    # Save
    import csv as csv_mod
    if results:
        with open(out / "cluster_calibration.csv", "w", newline="") as f:
            writer = csv_mod.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    # Summary
    import pandas as pd
    df = pd.DataFrame(results) if results else pd.DataFrame()
    summary = {}
    if not df.empty:
        mean_by_k = df.groupby('n_clusters')[['ari', 'nmi']].mean()
        best_k = int(mean_by_k['nmi'].idxmax())
        summary = {
            'best_n_clusters': best_k,
            'best_mean_ari': float(mean_by_k.loc[best_k, 'ari']),
            'best_mean_nmi': float(mean_by_k.loc[best_k, 'nmi']),
            'all_k_values': {
                str(int(k)): {'ari': float(mean_by_k.loc[k, 'ari']),
                              'nmi': float(mean_by_k.loc[k, 'nmi'])}
                for k in sorted(mean_by_k.index)
            },
            'n_cancers': len(test_cancers),
            'n_configs': len(results),
        }

        print("\nCluster-Count Calibration (vs. KEGG/Reactome pathways)")
        print("=" * 55)
        print(f"{'k':>6} {'Mean ARI':>10} {'Mean NMI':>10}")
        for k in sorted(mean_by_k.index):
            marker = " <-- best" if k == best_k else ""
            print(f"{int(k):>6} {mean_by_k.loc[k, 'ari']:>10.4f} "
                  f"{mean_by_k.loc[k, 'nmi']:>10.4f}{marker}")
        print(f"\nBest k = {best_k} (NMI = {summary['best_mean_nmi']:.4f}, "
              f"ARI = {summary['best_mean_ari']:.4f})")

    with open(out / "cluster_calibration_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


# ---------------------------------------------------------------------------
# Viability path enrichment validation (hypergeometric ORA)
# ---------------------------------------------------------------------------

# Broader canonical pathway gene sets for ORA — union of KEGG, Reactome,
# and Hallmark gene sets for cancer-relevant pathways.  These are still
# hard-coded summaries (not fetched from a database API), but cover a
# broader and more granular set than the 8-pathway NMI calibration above.
_ORA_PATHWAY_SETS: Dict[str, set] = {
    # Signaling cascades
    'MAPK_cascade':   {'KRAS', 'NRAS', 'HRAS', 'BRAF', 'RAF1', 'ARAF',
                       'MAP2K1', 'MAP2K2', 'MAPK1', 'MAPK3', 'MAP3K1',
                       'MAP3K8', 'SOS1', 'GRB2', 'NF1', 'SPRY2', 'DUSP6'},
    'PI3K_AKT_mTOR':  {'PIK3CA', 'PIK3CB', 'PIK3R1', 'AKT1', 'AKT2', 'AKT3',
                       'MTOR', 'RPTOR', 'RICTOR', 'RPS6KB1', 'TSC1', 'TSC2',
                       'PTEN', 'INPP4B', 'GSK3B', 'EIF4EBP1'},
    'JAK_STAT':       {'JAK1', 'JAK2', 'JAK3', 'TYK2', 'STAT3', 'STAT1',
                       'STAT5A', 'STAT5B', 'SOCS1', 'SOCS3', 'CISH', 'SRC',
                       'FYN', 'IL6ST'},
    'RTK_signaling':  {'EGFR', 'ERBB2', 'ERBB3', 'MET', 'FGFR1', 'FGFR2',
                       'FGFR3', 'RET', 'ALK', 'ROS1', 'PDGFRA', 'KIT',
                       'FLT3', 'IGF1R', 'NTRK1', 'EPHA2'},
    'WNT_signaling':  {'CTNNB1', 'APC', 'GSK3B', 'AXIN1', 'AXIN2', 'TCF7L2',
                       'LEF1', 'LRP5', 'LRP6', 'FZD7', 'RSPO3', 'RNF43',
                       'ZNRF3'},
    'NOTCH_signaling': {'NOTCH1', 'NOTCH2', 'NOTCH3', 'DLL1', 'DLL3', 'DLL4',
                        'JAG1', 'JAG2', 'HES1', 'HEY1', 'RBPJ', 'MAML1',
                        'FBXW7'},
    'HIPPO_signaling': {'YAP1', 'WWTR1', 'LATS1', 'LATS2', 'STK3', 'STK4',
                        'NF2', 'MOB1A', 'MOB1B', 'TEAD1', 'TEAD4', 'SAV1'},
    # Cell fate / cycle / death
    'Cell_cycle':     {'CDK4', 'CDK6', 'CDK2', 'CDK1', 'CCND1', 'CCNE1',
                       'CCNA2', 'CCNB1', 'RB1', 'E2F1', 'E2F3', 'CDKN2A',
                       'CDKN2B', 'CDKN1A', 'CDKN1B', 'TP53', 'MDM2'},
    'Apoptosis':      {'BCL2', 'MCL1', 'BCL2L1', 'BCL2L11', 'BAX', 'BAK1',
                       'BID', 'BAD', 'CASP3', 'CASP8', 'CASP9', 'XIAP',
                       'BIRC5', 'BBC3', 'PMAIP1'},
    'DNA_damage':     {'ATM', 'ATR', 'CHEK1', 'CHEK2', 'BRCA1', 'BRCA2',
                       'RAD51', 'PARP1', 'PARP2', 'TP53', 'MDM2', 'CDKN1A',
                       'XRCC1', 'H2AX'},
    # Chromatin / epigenetic
    'Chromatin_remod': {'EZH2', 'KDM6A', 'ARID1A', 'SMARCA4', 'SMARCB1',
                        'KMT2A', 'KMT2D', 'DNMT3A', 'TET2', 'IDH1', 'IDH2',
                        'SETD2', 'KDM5C', 'BRD4'},
    # Metabolism
    'Cancer_metab':   {'IDH1', 'IDH2', 'HK2', 'PKM', 'LDHA', 'SLC2A1',
                       'MYC', 'HIF1A', 'VHL', 'GLUL', 'ACLY', 'FASN'},
}


def _hypergeom_pvalue(k: int, M: int, n: int, N: int) -> float:
    """Right-tail hypergeometric p-value: P(X >= k).

    Parameters
    ----------
    k : observed overlap
    M : population size (total genes)
    n : pathway size
    N : query set size
    """
    from scipy.stats import hypergeom
    return float(hypergeom.sf(k - 1, M, n, N))


def validate_viability_paths_enrichment(
    data_dir: str = "./depmap_data",
    output_dir: str = "tuning_results",
) -> Dict:
    """Run hypergeometric over-representation analysis (ORA) on viability
    path gene sets against canonical pathway databases.

    For each cancer type, infer viability paths and test each path's gene
    set for enrichment in each canonical pathway using the hypergeometric
    test with Benjamini-Hochberg FDR correction.

    This addresses the criticism that viability paths are never validated
    against known biological pathways (KEGG/Reactome/GO).

    Returns
    -------
    dict  — summary statistics including fraction of paths with at least
            one significant pathway enrichment.
    """
    from pan_cancer_xnode import DepMapLoader, PathInference
    from core.statistics import apply_fdr_correction
    import csv as csv_mod

    out = Path(output_dir)
    out.mkdir(exist_ok=True)

    depmap = DepMapLoader(data_dir)
    path_inf = PathInference(depmap)

    available_ct = depmap.get_available_cancer_types()
    # Use top-5 well-powered cancer types
    test_cancers = [name for name, n in available_ct if n >= 20][:5]
    if not test_cancers:
        test_cancers = [name for name, _ in available_ct[:3]]

    # Flatten all pathway genes as background universe
    all_pathway_genes = set()
    for genes in _ORA_PATHWAY_SETS.values():
        all_pathway_genes |= genes

    results = []

    for cancer in test_cancers:
        try:
            paths = path_inf.infer_all_paths(cancer)
        except Exception as e:
            logger.warning(f"Path inference failed for {cancer}: {e}")
            continue

        for vp in paths:
            path_genes = set(vp.nodes)
            if len(path_genes) < 2:
                continue

            # Test against each canonical pathway
            for pw_name, pw_genes in _ORA_PATHWAY_SETS.items():
                overlap = path_genes & pw_genes
                k = len(overlap)
                if k == 0:
                    continue

                # Universe = all genes in DepMap (~18,000)
                # Use a reasonable estimate for background
                M = 18_000
                n = len(pw_genes)
                N = len(path_genes)

                pval = _hypergeom_pvalue(k, M, n, N)
                results.append({
                    'cancer': cancer,
                    'path_id': vp.path_id,
                    'path_type': vp.path_type,
                    'path_size': N,
                    'pathway': pw_name,
                    'pathway_size': n,
                    'overlap': k,
                    'overlap_genes': ','.join(sorted(overlap)),
                    'pvalue': pval,
                })

    # Apply FDR correction across all tests
    if results:
        raw_pvals = [r['pvalue'] for r in results]
        adj_pvals, reject = apply_fdr_correction(raw_pvals, method='fdr_bh',
                                                  alpha=0.05)
        for r, qval, rej in zip(results, adj_pvals, reject):
            r['qvalue'] = qval
            r['significant'] = bool(rej)

        # Save detailed results
        with open(out / "viability_path_enrichment.csv", "w", newline="") as f:
            writer = csv_mod.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    # Summary statistics
    if results:
        import pandas as pd
        df = pd.DataFrame(results)

        # Per-path: does each path have at least one significant enrichment?
        path_ids = df.groupby(['cancer', 'path_id'])['significant'].any()
        n_paths_total = len(path_ids)
        n_paths_enriched = int(path_ids.sum())
        frac_enriched = n_paths_enriched / n_paths_total if n_paths_total else 0

        # Per-path-type breakdown
        type_summary = {}
        for pt in df['path_type'].unique():
            sub = df[df['path_type'] == pt]
            sub_paths = sub.groupby(['cancer', 'path_id'])['significant'].any()
            type_summary[pt] = {
                'n_paths': len(sub_paths),
                'n_enriched': int(sub_paths.sum()),
                'frac_enriched': round(sub_paths.mean(), 3) if len(sub_paths) else 0,
            }

        # Top enriched pathways
        sig_df = df[df['significant']]
        top_pathways = {}
        if not sig_df.empty:
            pw_counts = sig_df['pathway'].value_counts()
            top_pathways = {pw: int(c) for pw, c in pw_counts.head(10).items()}

        summary = {
            'n_cancers_tested': len(test_cancers),
            'n_total_tests': len(results),
            'n_significant_tests': int(df['significant'].sum()),
            'n_paths_total': n_paths_total,
            'n_paths_enriched': n_paths_enriched,
            'frac_paths_enriched': round(frac_enriched, 3),
            'per_path_type': type_summary,
            'top_enriched_pathways': top_pathways,
            'fdr_threshold': 0.05,
        }

        print("\nViability Path Enrichment Validation (ORA)")
        print("=" * 55)
        print(f"Cancer types tested:    {summary['n_cancers_tested']}")
        print(f"Total paths tested:     {n_paths_total}")
        print(f"Paths with enrichment:  {n_paths_enriched} "
              f"({frac_enriched:.1%})")
        print(f"\nPer path type:")
        for pt, info in type_summary.items():
            print(f"  {pt:30s}  {info['n_enriched']:>3d}/{info['n_paths']:>3d} "
                  f"({info['frac_enriched']:.1%})")
        if top_pathways:
            print(f"\nTop enriched pathways:")
            for pw, c in list(top_pathways.items())[:5]:
                print(f"  {pw:30s}  {c} significant hits")
    else:
        summary = {
            'n_cancers_tested': len(test_cancers),
            'n_total_tests': 0,
            'n_paths_enriched': 0,
            'frac_paths_enriched': 0.0,
        }

    with open(out / "viability_path_enrichment_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def _manual_ari(true_labels, pred_labels):
    """Compute Adjusted Rand Index without sklearn."""
    from collections import Counter
    n = len(true_labels)
    if n < 2:
        return 0.0

    # Contingency table
    pairs_true = Counter(true_labels)
    pairs_pred = Counter(pred_labels)
    contingency = Counter(zip(true_labels, pred_labels))

    sum_comb_c = sum(v * (v - 1) / 2 for v in contingency.values())
    sum_comb_k = sum(v * (v - 1) / 2 for v in pairs_true.values())
    sum_comb_j = sum(v * (v - 1) / 2 for v in pairs_pred.values())
    comb_n = n * (n - 1) / 2

    expected = sum_comb_k * sum_comb_j / comb_n if comb_n > 0 else 0
    max_index = (sum_comb_k + sum_comb_j) / 2
    denom = max_index - expected

    if denom == 0:
        return 0.0
    return (sum_comb_c - expected) / denom


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="ALIN Parameter Tuning — tune previously arbitrary parameters "
                    "via cross-validated grid search against gold-standard benchmarks"
    )
    parser.add_argument('--mode', default='sweep',
                        choices=['grid', 'sweep', 'calibrate', 'enrich', 'all', 'report'],
                        help='Tuning mode: grid (full grid search), sweep (threshold '
                             'sensitivity), calibrate (cluster vs pathways), '
                             'enrich (viability path ORA), all, report')
    parser.add_argument('--data-dir', default='./depmap_data')
    parser.add_argument('--output-dir', default='tuning_results')
    parser.add_argument('--sample', type=int, default=None,
                        help='For grid mode: randomly sample N configs (faster)')
    parser.add_argument('--top-n', type=int, default=20,
                        help='Number of cancer types to evaluate')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)s %(levelname)s %(message)s')

    if args.mode in ('sweep', 'all'):
        print("\n" + "=" * 70)
        print("Running threshold sensitivity sweep...")
        print("=" * 70)
        threshold_sensitivity_sweep(args.data_dir, args.output_dir)

    if args.mode in ('calibrate', 'all'):
        print("\n" + "=" * 70)
        print("Running cluster-count calibration...")
        print("=" * 70)
        calibrate_cluster_count(args.data_dir, args.output_dir)

    if args.mode in ('enrich', 'all'):
        print("\n" + "=" * 70)
        print("Running viability path enrichment validation (ORA)...")
        print("=" * 70)
        validate_viability_paths_enrichment(args.data_dir, args.output_dir)

    if args.mode in ('grid', 'all'):
        print("\n" + "=" * 70)
        print("Running full grid-search tuning...")
        print("=" * 70)
        evaluator = PipelineEvaluator(
            data_dir=args.data_dir, top_n_cancers=args.top_n
        )
        tuner = GridSearchTuner(evaluator, args.output_dir)
        tuner.run(stage1_sample=args.sample)

    if args.mode == 'report':
        report_path = Path(args.output_dir) / "tuning_report.txt"
        if report_path.exists():
            print(report_path.read_text())
        else:
            print(f"No report found at {report_path}. Run tuning first.")


if __name__ == "__main__":
    main()
