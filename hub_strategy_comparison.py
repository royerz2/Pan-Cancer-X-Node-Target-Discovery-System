#!/usr/bin/env python3
"""
Hub-Penalty Strategy Comparison
================================

Compares multiple hub-handling strategies:
1. Current hub penalty (excess * 1.5 above median)
2. Degree-normalized path scoring (PageRank-like)
3. Hard cap on hub reuse (max frequency per gene)
4. No hub penalty (baseline)

Also addresses literature features: since Δ=0.0 ablation showed
literature features contribute nothing, we test a "lean" scoring
formula that removes decorative components and redistributes weight.

Metrics:
  - Hub dominance: max gene frequency, Shannon entropy, Gini coefficient
  - Benchmark concordance: any-overlap, pair-overlap, exact vs 43-entry gold standard
"""

import sys
import os
import json
import math
import copy
import logging
import warnings
import tempfile
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional, Set
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.WARNING)

# ---------------------------------------------------------------------------
# Gold standard benchmark (43-entry)
# ---------------------------------------------------------------------------
from gold_standard import GOLD_STANDARD, run_benchmark, CANCER_ALIASES

# ---------------------------------------------------------------------------
# Pipeline imports
# ---------------------------------------------------------------------------
from pan_cancer_xnode import (
    TripleCombinationFinder,
    XNodeNetworkAnalyzer,
    MinimalHittingSetSolver,
    CostFunction,
    DrugTargetDB,
    SynergyScorer,
    ResistanceProbabilityEstimator,
    DepMapLoader,
    OmniPathLoader,
    PanCancerXNodeAnalyzer,
)
from core.data_structures import TripleCombination

# ---------------------------------------------------------------------------
# Hub dominance metrics
# ---------------------------------------------------------------------------

def compute_hub_metrics(df: pd.DataFrame) -> Dict:
    """Compute hub-dominance metrics from a triple predictions CSV."""
    n = len(df)
    if n == 0:
        return {'max_freq': 0, 'entropy': 0, 'gini': 0, 'top_gene': 'N/A',
                'n_unique_genes': 0, 'stat3_freq': 0}
    
    genes = Counter()
    for _, r in df.iterrows():
        for col in ['Target 1', 'Target 2', 'Target 3']:
            if col in df.columns:
                g = r[col]
                if pd.notna(g) and str(g) != '':
                    genes[str(g)] += 1
    
    total_slots = sum(genes.values())
    freqs = np.array(list(genes.values())) / n  # per-cancer frequency
    probs = np.array(list(genes.values())) / total_slots  # probability mass
    
    # Shannon entropy (higher = more diverse)
    entropy = -sum(p * math.log2(p) for p in probs if p > 0)
    
    # Gini coefficient (lower = more equal)
    sorted_vals = np.sort(np.array(list(genes.values()), dtype=float))
    n_genes = len(sorted_vals)
    if n_genes > 0 and sorted_vals.sum() > 0:
        index = np.arange(1, n_genes + 1)
        gini = (2 * np.sum(index * sorted_vals) - (n_genes + 1) * sorted_vals.sum()) / (n_genes * sorted_vals.sum())
    else:
        gini = 0.0
    
    top_gene, top_count = genes.most_common(1)[0]
    stat3_count = genes.get('STAT3', 0)
    
    return {
        'max_freq': top_count / n,
        'max_freq_pct': f"{top_count / n * 100:.1f}%",
        'top_gene': top_gene,
        'entropy': round(entropy, 3),
        'gini': round(gini, 3),
        'n_unique_genes': len(genes),
        'stat3_freq': stat3_count / n,
        'stat3_freq_pct': f"{stat3_count / n * 100:.1f}%",
        'gene_counts': dict(genes.most_common(10)),
    }


def run_gold_benchmark(csv_path: str) -> Dict:
    """Run against 43-entry gold standard, return recall dict."""
    result = run_benchmark(csv_path, verbose=False)
    return result['recall']


# ===========================================================================
# STRATEGY 1: Current hub penalty (excess * 1.5)
# ===========================================================================
# Already have results in results/triple_combinations.csv

# ===========================================================================
# STRATEGY 2: Degree-normalized path scoring (PageRank-like)
# ===========================================================================
# Instead of a penalty on the final score, normalize gene weight
# in the path-coverage computation by 1/log(degree+1).

# ===========================================================================
# STRATEGY 3: Hard cap on hub reuse
# ===========================================================================
# After ranking, post-process: if a gene appears in >K cancer types,
# replace with next-best triple that doesn't use that gene.

# ===========================================================================
# STRATEGY 4: Lean scoring (remove decorative literature features)
# ===========================================================================
# Remove literature features (synergy dict, resistance dict, DDI, 
# perturbation) and redistribute weight to data-driven components.

# ===========================================================================
# Implementation: Monkey-patch TripleCombinationFinder to test strategies
# ===========================================================================

def run_strategy(strategy_name: str, analyzer: PanCancerXNodeAnalyzer, 
                 cancer_types: List[str], base_df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """
    Re-run the ranking stage with a specific hub strategy.
    Returns (DataFrame of results, path to CSV).
    """
    print(f"\n{'='*70}")
    print(f"Running strategy: {strategy_name}")
    print(f"{'='*70}")
    
    finder = analyzer.triple_finder
    drug_db = analyzer.drug_db
    
    results_rows = []
    
    for cancer_type in sorted(cancer_types):
        print(f"  Processing: {cancer_type}...", end='', flush=True)
        
        try:
            # Get paths for this cancer
            paths = analyzer.path_inference.infer_all_paths(cancer_type)
            if not paths or len(paths) < 3:
                print(" (no paths, skipping)")
                continue
            
            if strategy_name == 'no_hub_penalty':
                finder.disable_hub_penalty = True
                triples = finder.find_triple_combinations(
                    cancer_type=cancer_type,
                    paths=paths,
                    min_coverage=0.3,
                    top_n=5,
                )
                finder.disable_hub_penalty = False
                
            elif strategy_name == 'current_hub_penalty':
                finder.disable_hub_penalty = False
                triples = finder.find_triple_combinations(
                    cancer_type=cancer_type,
                    paths=paths,
                    min_coverage=0.3,
                    top_n=5,
                )
                
            elif strategy_name == 'pagerank_normalized':
                triples = _run_pagerank_strategy(finder, cancer_type, paths, drug_db)
                
            elif strategy_name == 'lean_scoring':
                triples = _run_lean_strategy(finder, cancer_type, paths, drug_db)
                
            elif strategy_name == 'pagerank_lean':
                triples = _run_pagerank_lean_strategy(finder, cancer_type, paths, drug_db)
                
            elif strategy_name == 'hard_cap':
                # Run unconstrained first, post-process later
                finder.disable_hub_penalty = True
                triples = finder.find_triple_combinations(
                    cancer_type=cancer_type,
                    paths=paths,
                    min_coverage=0.3,
                    top_n=50,
                )
                finder.disable_hub_penalty = False
                
            else:
                raise ValueError(f"Unknown strategy: {strategy_name}")
            
            if triples:
                best = triples[0]
                targets = list(best.targets)
                drugs = []
                for t in targets:
                    info = drug_db.get_drug_info(t)
                    if info and info.available_drugs:
                        drugs.append(info.available_drugs[0])
                    else:
                        drugs.append('investigational')
                while len(drugs) < 3:
                    drugs.append('')
                while len(targets) < 3:
                    targets.append('')
                
                cell_lines_val = base_df[base_df['Cancer Type'] == cancer_type]['Cell Lines'].values
                row = {
                    'Cancer Type': cancer_type,
                    'Cell Lines': cell_lines_val[0] if len(cell_lines_val) > 0 else 0,
                    'Target 1': targets[0],
                    'Target 2': targets[1],
                    'Target 3': targets[2],
                    'Drug 1': drugs[0],
                    'Drug 2': drugs[1],
                    'Drug 3': drugs[2],
                    'Synergy': round(best.synergy_score, 2),
                    'Resistance': round(best.resistance_score, 2),
                    'Coverage': f"{best.coverage * 100:.1f}%",
                    'Druggable': f"{sum(1 for t in targets if t and drug_db.get_druggability_score(t) >= 0.6)}/{len([t for t in targets if t])}",
                    'Best_Combo_Size': len([t for t in targets if t]),
                    'Best_Combo_1': targets[0],
                    'Best_Combo_2': targets[1],
                    'Best_Combo_3': targets[2],
                    'Best_Combo_Score': round(best.combined_score, 3),
                }
                results_rows.append(row)
                print(f" -> {tuple(t for t in targets if t)}")
            else:
                print(" (no viable triples)")
                
        except Exception as e:
            print(f" ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not results_rows:
        print("  WARNING: No results generated!")
        return pd.DataFrame(), ''
    
    df = pd.DataFrame(results_rows)
    
    # Hard cap post-processing: replace over-represented gene's excess cancers
    if strategy_name == 'hard_cap':
        df = _apply_hard_cap_postprocess(df, max_freq=7)
    
    # Save
    out_dir = Path('hub_strategy_results')
    out_dir.mkdir(exist_ok=True)
    csv_path = str(out_dir / f'triples_{strategy_name}.csv')
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path} ({len(df)} rows)")
    
    return df, csv_path


def _run_pagerank_strategy(finder, cancer_type, paths, drug_db):
    """
    PageRank-like degree normalization:
    Instead of raw path coverage, weight each path hit by 1/log2(degree+2)
    where degree = number of paths the gene appears in.
    This down-weights hub contribution in coverage, without a separate penalty term.
    """
    import itertools
    
    # Get candidate genes (reuse finder's logic)
    all_genes = set()
    for p in paths:
        all_genes.update(p.nodes)
    
    # Build candidate pool (druggable + priority)
    candidate_genes = []
    for g in all_genes:
        score = drug_db.get_druggability_score(g)
        candidate_genes.append((g, score))
    candidate_genes.sort(key=lambda x: -x[1])
    candidate_genes = [g for g, s in candidate_genes[:50]]
    
    if len(candidate_genes) < 3:
        return []
    
    # Compute gene-path degree
    gene_degree = {}
    for g in candidate_genes:
        gene_degree[g] = sum(1 for p in paths if g in p.nodes)
    
    # PageRank-like weight: 1 / log2(degree + 2)
    gene_weight = {}
    for g in candidate_genes:
        gene_weight[g] = 1.0 / math.log2(gene_degree.get(g, 0) + 2)
    
    # Enumerate triples, compute PageRank-weighted coverage
    gene_costs = {}
    for g in candidate_genes:
        cost_obj = finder.cost_fn.compute_cost(g, cancer_type)
        gene_costs[g] = cost_obj.total_cost()
    
    scored = []
    for combo in itertools.combinations(candidate_genes, 3):
        combo_set = set(combo)
        
        # Weighted coverage: each path's "covered" value = max gene_weight in path
        weighted_coverage = 0
        raw_covered = 0
        for p in paths:
            genes_in_path = combo_set & set(p.nodes)
            if genes_in_path:
                raw_covered += 1
                # Contribution weighted by the MINIMUM-degree gene (most specific)
                weighted_coverage += max(gene_weight[g] for g in genes_in_path)
        
        raw_cov = raw_covered / len(paths)
        if raw_cov < 0.3:
            continue
        
        # Normalize weighted coverage
        max_possible_wt_cov = sum(1.0 / math.log2(2) for _ in paths)  # if all covered by degree-0 gene
        norm_weighted_cov = weighted_coverage / max_possible_wt_cov if max_possible_wt_cov > 0 else 0
        
        total_cost = sum(gene_costs.get(g, 1.0) for g in combo)
        
        # Data-driven synergy from co-essentiality
        synergy = finder.synergy_scorer.compute_synergy_score(combo_set)
        try:
            from pharmacological_validation import CoEssentialityInteractionEstimator
            dd = CoEssentialityInteractionEstimator.score_combination(
                targets=tuple(sorted(combo)),
                depmap_df=finder.depmap._crispr_df if hasattr(finder.depmap, '_crispr_df') and finder.depmap._crispr_df is not None else None,
                cell_lines=[],
                original_synergy=synergy,
                original_pathway_diversity=len(set(
                    finder.synergy_scorer.PATHWAY_ASSIGNMENT.get(g, g) for g in combo
                )) / max(len(combo), 1),
            )
            synergy = dd.data_driven_synergy
        except Exception:
            pass
        
        resistance = finder.resistance_estimator.estimate_resistance_probability(combo_set, cancer_type)
        druggable_count = sum(1 for g in combo if drug_db.get_druggability_score(g) >= 0.6)
        
        # PageRank-normalized score (no separate hub penalty needed)
        # Use weighted coverage instead of raw coverage
        combined_score = (
            total_cost * 0.25 +
            (1 - synergy) * 0.20 +
            resistance * 0.20 +
            (1 - norm_weighted_cov) * 0.25 +    # PageRank-normalized coverage (upweighted)
            - druggable_count * 0.10
        )
        tc = TripleCombination(
            targets=tuple(sorted(combo)),
            total_cost=total_cost,
            synergy_score=synergy,
            resistance_score=resistance,
            pathway_coverage={},
            coverage=raw_cov,
            druggable_count=druggable_count,
            combined_score=combined_score,
            drug_info={g: drug_db.get_drug_info(g) for g in combo},
        )
        scored.append(tc)
    
    scored.sort(key=lambda x: x.combined_score)
    return scored[:5]


def _run_lean_strategy(finder, cancer_type, paths, drug_db):
    """
    Lean scoring: remove ALL literature features (synergy dict, resistance dict,
    DDI/toxicity, perturbation), keep only:
    - DepMap co-essentiality synergy
    - path coverage
    - node cost (essentiality + specificity + druggability)
    - current hub penalty
    
    Redistributes weights to data-driven components only.
    """
    import itertools
    
    all_genes = set()
    for p in paths:
        all_genes.update(p.nodes)
    
    candidate_genes = []
    for g in all_genes:
        score = drug_db.get_druggability_score(g)
        candidate_genes.append((g, score))
    candidate_genes.sort(key=lambda x: -x[1])
    candidate_genes = [g for g, s in candidate_genes[:50]]
    
    if len(candidate_genes) < 3:
        return []
    
    gene_costs = {}
    for g in candidate_genes:
        cost_obj = finder.cost_fn.compute_cost(g, cancer_type)
        gene_costs[g] = cost_obj.total_cost()
    
    # Gene path frequencies for hub penalty
    gene_path_freqs = {}
    for g in candidate_genes:
        gene_path_freqs[g] = sum(1 for p in paths if g in p.nodes) / max(len(paths), 1)
    freq_values = sorted(gene_path_freqs.values())
    median_freq = freq_values[len(freq_values) // 2] if freq_values else 0.3
    
    scored = []
    for combo in itertools.combinations(candidate_genes, 3):
        combo_set = set(combo)
        covered = sum(1 for p in paths if any(g in combo_set for g in p.nodes))
        coverage = covered / len(paths)
        if coverage < 0.3:
            continue
        
        total_cost = sum(gene_costs.get(g, 1.0) for g in combo)
        druggable_count = sum(1 for g in combo if drug_db.get_druggability_score(g) >= 0.6)
        
        # Data-driven synergy ONLY (no KNOWN_SYNERGIES dict)
        pathway_diversity = len(set(
            finder.synergy_scorer.PATHWAY_ASSIGNMENT.get(g, g) for g in combo
        )) / max(len(combo), 1)
        synergy = pathway_diversity * 0.6  # base pathway diversity only
        try:
            from pharmacological_validation import CoEssentialityInteractionEstimator
            dd = CoEssentialityInteractionEstimator.score_combination(
                targets=tuple(sorted(combo)),
                depmap_df=finder.depmap._crispr_df if hasattr(finder.depmap, '_crispr_df') and finder.depmap._crispr_df is not None else None,
                cell_lines=[],
                original_synergy=0.5,
                original_pathway_diversity=pathway_diversity,
            )
            synergy = dd.data_driven_synergy
        except Exception:
            pass
        
        # Hub penalty (same as current)
        hub_penalty = 0.0
        for g in combo:
            excess = gene_path_freqs.get(g, 0) - median_freq
            if excess > 0:
                hub_penalty += excess * 1.5
        
        # Lean score: only data-driven components
        # cost + coverage + co-essentiality synergy + hub penalty + druggability
        combined_score = (
            total_cost * 0.30 +            # Upweighted: cost encodes essentiality+specificity
            (1 - synergy) * 0.25 +         # Co-essentiality synergy
            (1 - coverage) * 0.25 +        # Path coverage
            hub_penalty -                   # Hub penalty
            druggable_count * 0.10          # Druggability bonus
        )
        # Note: resistance (literature) and combo_tox (literature) removed
        # Their 0.36 combined weight redistributed to cost (+0.08) and synergy/coverage (+0.11 each)
        
        tc = TripleCombination(
            targets=tuple(sorted(combo)),
            total_cost=total_cost,
            synergy_score=synergy,
            resistance_score=0.0,
            pathway_coverage={},
            coverage=coverage,
            druggable_count=druggable_count,
            combined_score=combined_score,
            drug_info={g: drug_db.get_drug_info(g) for g in combo},
        )
        scored.append(tc)
    
    scored.sort(key=lambda x: x.combined_score)
    return scored[:5]


def _run_pagerank_lean_strategy(finder, cancer_type, paths, drug_db):
    """
    Best-of-both: PageRank-normalized coverage + lean (no-literature) scoring.
    This is the principled replacement candidate.
    """
    import itertools
    
    all_genes = set()
    for p in paths:
        all_genes.update(p.nodes)
    
    candidate_genes = []
    for g in all_genes:
        score = drug_db.get_druggability_score(g)
        candidate_genes.append((g, score))
    candidate_genes.sort(key=lambda x: -x[1])
    candidate_genes = [g for g, s in candidate_genes[:50]]
    
    if len(candidate_genes) < 3:
        return []
    
    gene_costs = {}
    for g in candidate_genes:
        cost_obj = finder.cost_fn.compute_cost(g, cancer_type)
        gene_costs[g] = cost_obj.total_cost()
    
    # Gene-path degree for PageRank normalization
    gene_degree = {}
    for g in candidate_genes:
        gene_degree[g] = sum(1 for p in paths if g in p.nodes)
    gene_weight = {g: 1.0 / math.log2(gene_degree.get(g, 0) + 2) for g in candidate_genes}
    
    scored = []
    for combo in itertools.combinations(candidate_genes, 3):
        combo_set = set(combo)
        
        # PageRank-weighted coverage
        weighted_cov = 0
        raw_covered = 0
        for p in paths:
            genes_in_path = combo_set & set(p.nodes)
            if genes_in_path:
                raw_covered += 1
                weighted_cov += max(gene_weight[g] for g in genes_in_path)
        
        raw_cov = raw_covered / len(paths)
        if raw_cov < 0.3:
            continue
        
        max_w = len(paths) * (1.0 / math.log2(2))
        norm_wcov = weighted_cov / max_w if max_w > 0 else 0
        
        total_cost = sum(gene_costs.get(g, 1.0) for g in combo)
        druggable_count = sum(1 for g in combo if drug_db.get_druggability_score(g) >= 0.6)
        
        # Co-essentiality synergy only
        pathway_diversity = len(set(
            finder.synergy_scorer.PATHWAY_ASSIGNMENT.get(g, g) for g in combo
        )) / max(len(combo), 1)
        synergy = pathway_diversity * 0.6
        try:
            from pharmacological_validation import CoEssentialityInteractionEstimator
            dd = CoEssentialityInteractionEstimator.score_combination(
                targets=tuple(sorted(combo)),
                depmap_df=finder.depmap._crispr_df if hasattr(finder.depmap, '_crispr_df') and finder.depmap._crispr_df is not None else None,
                cell_lines=[],
                original_synergy=0.5,
                original_pathway_diversity=pathway_diversity,
            )
            synergy = dd.data_driven_synergy
        except Exception:
            pass
        
        # No separate hub penalty — PageRank normalization handles it
        combined_score = (
            total_cost * 0.30 +
            (1 - synergy) * 0.25 +
            (1 - norm_wcov) * 0.30 +    # PageRank-weighted (replaces hub penalty)
            - druggable_count * 0.10
        )
        
        tc = TripleCombination(
            targets=tuple(sorted(combo)),
            total_cost=total_cost,
            synergy_score=synergy,
            resistance_score=0.0,
            pathway_coverage={},
            coverage=raw_cov,
            druggable_count=druggable_count,
            combined_score=combined_score,
            drug_info={g: drug_db.get_drug_info(g) for g in combo},
        )
        scored.append(tc)
    
    scored.sort(key=lambda x: x.combined_score)
    return scored[:5]


def _apply_hard_cap_postprocess(df, max_freq=7):
    """
    Hard-cap post-processing: if a gene appears in > max_freq cancer types,
    mark it. This evaluates how much the hard cap would affect predictions.
    In practice, for a full implementation you'd re-rank with the gene excluded.
    """
    gene_freq = Counter()
    for _, r in df.iterrows():
        for col in ['Target 1', 'Target 2', 'Target 3']:
            g = r[col]
            if pd.notna(g) and str(g) != '':
                gene_freq[str(g)] += 1
    
    over_cap = {g for g, c in gene_freq.items() if c > max_freq}
    if over_cap:
        print(f"  Hard cap ({max_freq}): genes over limit: {over_cap}")
    return df


# ===========================================================================
# MAIN: Run all strategies and compare
# ===========================================================================

def main():
    print("=" * 70)
    print("HUB-PENALTY STRATEGY COMPARISON")
    print("=" * 70)
    
    # Initialize pipeline once (expensive)
    print("\nInitializing pipeline...")
    analyzer = PanCancerXNodeAnalyzer()
    
    # Load base results for cancer type list
    full_csv = 'ablation_results/triple_combinations_full_pipeline.csv'
    if not os.path.exists(full_csv):
        full_csv = 'results/triple_combinations.csv'
    base_df = pd.read_csv(full_csv)
    cancer_types = list(base_df['Cancer Type'].unique())
    print(f"Cancer types to evaluate: {len(cancer_types)}")
    
    strategies = [
        'no_hub_penalty',       # Baseline: no hub handling
        'current_hub_penalty',  # Strategy 1: current excess*1.5
        'pagerank_normalized',  # Strategy 2: degree-normalized coverage
        'lean_scoring',         # Strategy 3: remove lit features + current hub
        'pagerank_lean',        # Strategy 4: PageRank + lean (principled combo)
    ]
    
    results = {}
    
    for strat in strategies:
        try:
            df, csv_path = run_strategy(strat, analyzer, cancer_types, base_df)
            if csv_path and len(df) > 0:
                hub = compute_hub_metrics(df)
                bench = run_gold_benchmark(csv_path)
                results[strat] = {
                    'hub_metrics': hub,
                    'benchmark': bench,
                    'n_cancers': len(df),
                    'csv_path': csv_path,
                }
                print(f"\n  Hub metrics: max_freq={hub['max_freq_pct']} ({hub['top_gene']}), "
                      f"STAT3={hub['stat3_freq_pct']}, entropy={hub['entropy']}, gini={hub['gini']}")
                print(f"  Benchmark:   any_overlap={bench['any_overlap']:.1%} ({bench['n_any_overlap']}/{bench['n_entries']}), "
                      f"pair_overlap={bench['pair_overlap']:.1%} ({bench['n_pair_overlap']}/{bench['n_entries']}), "
                      f"exact={bench['exact']:.1%} ({bench['n_exact']}/{bench['n_entries']})")
            else:
                print(f"  SKIPPED: No results for {strat}")
        except Exception as e:
            print(f"  ERROR in {strat}: {e}")
            import traceback
            traceback.print_exc()
    
    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print("\n\n" + "=" * 90)
    print("COMPARISON SUMMARY")
    print("=" * 90)
    
    header = f"{'Strategy':<25} {'Cancers':>7} {'AnyOvlp':>8} {'PairOvlp':>8} {'Exact':>6} {'MaxFreq':>8} {'STAT3':>7} {'Entropy':>8} {'Gini':>6} {'UniqueG':>7}"
    print(header)
    print("-" * len(header))
    
    for strat in strategies:
        if strat in results:
            r = results[strat]
            h = r['hub_metrics']
            b = r['benchmark']
            print(f"{strat:<25} {r['n_cancers']:>7} "
                  f"{b['any_overlap']*100:>7.1f}% "
                  f"{b['pair_overlap']*100:>7.1f}% "
                  f"{b['exact']*100:>5.1f}% "
                  f"{h['max_freq']*100:>7.1f}% "
                  f"{h['stat3_freq']*100:>6.1f}% "
                  f"{h['entropy']:>8.3f} "
                  f"{h['gini']:>6.3f} "
                  f"{h['n_unique_genes']:>7}")
    
    # -----------------------------------------------------------------------
    # Save full results
    # -----------------------------------------------------------------------
    out_dir = Path('hub_strategy_results')
    
    # Save JSON
    serializable = {}
    for k, v in results.items():
        s = copy.deepcopy(v)
        s['hub_metrics'].pop('gene_counts', None)
        serializable[k] = s
    
    with open(out_dir / 'strategy_comparison.json', 'w') as f:
        json.dump(serializable, f, indent=2, default=str)
    
    print(f"\nResults saved to {out_dir}/")
    
    # -----------------------------------------------------------------------
    # Generate figure
    # -----------------------------------------------------------------------
    try:
        _generate_figure(results, out_dir)
    except Exception as e:
        print(f"Figure generation failed: {e}")
    
    return results


def _generate_figure(results, out_dir):
    """Generate comparison figure."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    strategies = list(results.keys())
    labels = {
        'no_hub_penalty': 'No hub\nhandling',
        'current_hub_penalty': 'Current\n(excess×1.5)',
        'pagerank_normalized': 'PageRank\nnormalized',
        'lean_scoring': 'Lean\nscoring',
        'pagerank_lean': 'PageRank\n+ lean',
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    
    x = np.arange(len(strategies))
    width = 0.55
    
    # Panel A: Benchmark concordance
    ax = axes[0]
    any_ovl = [results[s]['benchmark']['any_overlap'] * 100 for s in strategies]
    pair_ovl = [results[s]['benchmark']['pair_overlap'] * 100 for s in strategies]
    exact = [results[s]['benchmark']['exact'] * 100 for s in strategies]
    
    ax.bar(x - 0.2, any_ovl, 0.2, label='Any-overlap', color='#2196F3', alpha=0.8)
    ax.bar(x, pair_ovl, 0.2, label='Pair-overlap', color='#FF9800', alpha=0.8)
    ax.bar(x + 0.2, exact, 0.2, label='Exact', color='#4CAF50', alpha=0.8)
    ax.set_ylabel('Concordance (%)')
    ax.set_title('A. Benchmark Performance')
    ax.set_xticks(x)
    ax.set_xticklabels([labels.get(s, s) for s in strategies], fontsize=7)
    ax.legend(fontsize=7, loc='upper right')
    ax.set_ylim(0, max(any_ovl) * 1.3)
    
    # Panel B: Hub dominance
    ax = axes[1]
    max_freq = [results[s]['hub_metrics']['max_freq'] * 100 for s in strategies]
    stat3_freq = [results[s]['hub_metrics']['stat3_freq'] * 100 for s in strategies]
    
    ax.bar(x - 0.15, max_freq, 0.3, label='Max gene freq', color='#F44336', alpha=0.8)
    ax.bar(x + 0.15, stat3_freq, 0.3, label='STAT3 freq', color='#9C27B0', alpha=0.8)
    ax.set_ylabel('Frequency (%)')
    ax.set_title('B. Hub Dominance')
    ax.set_xticks(x)
    ax.set_xticklabels([labels.get(s, s) for s in strategies], fontsize=7)
    ax.legend(fontsize=7)
    ax.set_ylim(0, 110)
    
    # Panel C: Diversity metrics
    ax = axes[2]
    entropy = [results[s]['hub_metrics']['entropy'] for s in strategies]
    n_unique = [results[s]['hub_metrics']['n_unique_genes'] for s in strategies]
    
    ax2 = ax.twinx()
    bars1 = ax.bar(x - 0.15, entropy, 0.3, label='Shannon entropy', color='#00BCD4', alpha=0.8)
    bars2 = ax2.bar(x + 0.15, n_unique, 0.3, label='Unique genes', color='#795548', alpha=0.8)
    ax.set_ylabel('Shannon Entropy (bits)')
    ax2.set_ylabel('Unique Genes')
    ax.set_title('C. Target Diversity')
    ax.set_xticks(x)
    ax.set_xticklabels([labels.get(s, s) for s in strategies], fontsize=7)
    lines1, lab1 = ax.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lab1 + lab2, fontsize=7, loc='upper left')
    
    plt.tight_layout()
    
    for fmt in ['png', 'pdf']:
        fig.savefig(str(out_dir / f'hub_strategy_comparison.{fmt}'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Figure saved to {out_dir}/hub_strategy_comparison.{{png,pdf}}")


if __name__ == '__main__':
    main()
