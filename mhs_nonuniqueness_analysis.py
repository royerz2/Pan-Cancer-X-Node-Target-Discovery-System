#!/usr/bin/env python3
"""
MHS Non-Uniqueness and Biological Plausibility Analysis
========================================================

1. Enumerate top-N near-optimal MHS solutions per cancer type
2. Measure solution diversity (Jaccard distance, gene overlap)
3. Apply biological-plausibility constraints:
   a. Druggability filter (>= 1 druggable target required)
   b. Pan-essential exclusion (hard filter, not just cost penalty)
   c. Toxicity ceiling (exclude high-toxicity combos)
   d. Same-pathway redundancy penalty
4. Report how often the recommended solution changes under constraints
"""

import sys, os, json, logging, warnings
from itertools import combinations
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ── Imports from pipeline ───────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pan_cancer_xnode import (
    DepMapLoader, OmniPathLoader, DrugTargetDB,
    CostFunction, MinimalHittingSetSolver,
    ViabilityPathInference, PanCancerXNodeAnalyzer,
)
from core.data_structures import HittingSet, ViabilityPath

# ── Configuration ───────────────────────────────────────────────────────
CANCER_TYPES = [
    "Acute Myeloid Leukemia", "Anaplastic Thyroid Cancer",
    "Bladder Urothelial Carcinoma", "Colorectal Adenocarcinoma",
    "Diffuse Glioma", "Endometrial Carcinoma",
    "Esophagogastric Adenocarcinoma",
    "Head and Neck Squamous Cell Carcinoma",
    "Hepatocellular Carcinoma", "Invasive Breast Carcinoma",
    "Liposarcoma", "Melanoma", "Non-Small Cell Lung Cancer",
    "Ovarian Epithelial Tumor", "Pancreatic Adenocarcinoma",
    "Prostate Adenocarcinoma", "Renal Cell Carcinoma",
]
TOP_N = 20          # max solutions to enumerate
MAX_SIZE = 4
MIN_COVERAGE = 0.8
OUTPUT_DIR = "mhs_nonuniqueness_results"

# ── Constraint definitions ──────────────────────────────────────────────
# Druggability: at least 1 target must have druggability >= 0.6
MIN_DRUGGABLE_TARGETS = 1
DRUGGABILITY_THRESHOLD = 0.6

# Toxicity ceiling: mean combo toxicity must be < 0.8
TOXICITY_CEILING = 0.8

# Same-pathway redundancy: penalize combos where >50% of targets
# come from the same viability-path cluster
MAX_SAME_PATHWAY_FRAC = 0.5


def jaccard_distance(s1: frozenset, s2: frozenset) -> float:
    """Jaccard distance between two sets (1 - Jaccard index)."""
    if len(s1 | s2) == 0:
        return 0.0
    return 1.0 - len(s1 & s2) / len(s1 | s2)


def pairwise_jaccard_matrix(solutions: List[HittingSet]) -> np.ndarray:
    """NxN Jaccard distance matrix for N solutions."""
    n = len(solutions)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = jaccard_distance(solutions[i].targets, solutions[j].targets)
            mat[i, j] = mat[j, i] = d
    return mat


def diversity_metrics(solutions: List[HittingSet]) -> Dict:
    """Compute diversity metrics for a set of MHS solutions."""
    if len(solutions) <= 1:
        return {
            'n_solutions': len(solutions),
            'mean_jaccard': 0.0,
            'min_jaccard': 0.0,
            'max_jaccard': 0.0,
            'n_unique_genes': len(solutions[0].targets) if solutions else 0,
            'gene_entropy': 0.0,
        }

    jmat = pairwise_jaccard_matrix(solutions)
    upper = jmat[np.triu_indices_from(jmat, k=1)]

    all_genes = Counter()
    for s in solutions:
        all_genes.update(s.targets)
    total = sum(all_genes.values())
    freqs = np.array([c / total for c in all_genes.values()])
    entropy = -np.sum(freqs * np.log2(freqs + 1e-12))

    return {
        'n_solutions': len(solutions),
        'mean_jaccard': float(np.mean(upper)) if len(upper) else 0.0,
        'min_jaccard': float(np.min(upper)) if len(upper) else 0.0,
        'max_jaccard': float(np.max(upper)) if len(upper) else 0.0,
        'n_unique_genes': len(all_genes),
        'gene_entropy': float(entropy),
    }


def apply_constraints(
    solutions: List[HittingSet],
    drug_db: DrugTargetDB,
    cost_fn: CostFunction,
    cancer_type: str,
    paths: List[ViabilityPath],
) -> Tuple[List[HittingSet], Dict]:
    """Apply biological-plausibility constraints. Return filtered list + stats."""
    stats = {
        'before': len(solutions),
        'removed_no_druggable': 0,
        'removed_pan_essential': 0,
        'removed_high_toxicity': 0,
        'removed_pathway_redundant': 0,
    }

    # Precompute per-gene properties
    pan_essential = cost_fn._get_pan_essential()
    gene_druggability = {}
    gene_toxicity = {}
    for sol in solutions:
        for g in sol.targets:
            if g not in gene_druggability:
                gene_druggability[g] = drug_db.get_druggability_score(g)
                gene_toxicity[g] = cost_fn._get_toxicity_score(g)

    # Build gene → pathway-cluster mapping for redundancy check
    gene_to_paths = defaultdict(set)
    for p in paths:
        for g in p.nodes:
            gene_to_paths[g].add(p.path_id)

    filtered = []
    for sol in solutions:
        targets = list(sol.targets)

        # 1. Druggability filter
        n_druggable = sum(1 for g in targets if gene_druggability.get(g, 0) >= DRUGGABILITY_THRESHOLD)
        if n_druggable < MIN_DRUGGABLE_TARGETS:
            stats['removed_no_druggable'] += 1
            continue

        # 2. Pan-essential exclusion (hard filter)
        n_pan = sum(1 for g in targets if g in pan_essential)
        if n_pan > 0:
            stats['removed_pan_essential'] += 1
            continue

        # 3. Toxicity ceiling
        mean_tox = np.mean([gene_toxicity.get(g, 0.5) for g in targets])
        if mean_tox > TOXICITY_CEILING:
            stats['removed_high_toxicity'] += 1
            continue

        # 4. Same-pathway redundancy
        if len(targets) >= 2:
            # For each pair, check if they share >50% of their pathway memberships
            path_sets = [gene_to_paths[g] for g in targets]
            # Find max overlap fraction between any pair
            max_overlap = 0.0
            for i in range(len(targets)):
                for j in range(i + 1, len(targets)):
                    si, sj = path_sets[i], path_sets[j]
                    if len(si | sj) > 0:
                        overlap = len(si & sj) / len(si | sj)
                        max_overlap = max(max_overlap, overlap)
            if max_overlap > MAX_SAME_PATHWAY_FRAC:
                stats['removed_pathway_redundant'] += 1
                continue

        filtered.append(sol)

    stats['after'] = len(filtered)
    return filtered, stats


def enumerate_near_optimal(
    solver: MinimalHittingSetSolver,
    paths: List[ViabilityPath],
    gene_costs: Dict[str, float],
    max_size: int,
    min_coverage: float,
    cost_tolerance: float = 1.5,
    k: int = TOP_N,
) -> List[HittingSet]:
    """
    Enumerate k near-optimal MHS solutions using iterative ILP
    with solution-exclusion constraints (k-best approach).

    1. Solve ILP optimally → solution S*
    2. Add constraint: exclude S* (sum of its vars ≤ |S*| - 1)
    3. Repeat up to k times or until cost > tolerance * optimal
    """
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        from scipy.sparse import csc_matrix, hstack, vstack, eye
        import math
    except ImportError:
        print("  scipy not available; falling back to greedy only")
        greedy = solver._solve_greedy(paths, gene_costs, max_size)
        return [greedy] if greedy else []

    all_genes = set()
    for p in paths:
        all_genes.update(p.nodes)

    genes = sorted(gene_costs.keys())
    gene_idx = {g: i for i, g in enumerate(genes)}
    n_genes = len(genes)
    n_paths = len(paths)

    if n_genes == 0 or n_paths == 0:
        return []

    # Total variables: n_genes (x_g) + n_paths (y_p)
    n_vars = n_genes + n_paths

    # Objective: minimize total cost of selected genes
    c = np.zeros(n_vars, dtype=float)
    c[:n_genes] = [gene_costs[g] for g in genes]

    # Build path-gene incidence matrix
    rows, cols = [], []
    for p_idx, path in enumerate(paths):
        for node in path.nodes:
            if node in gene_idx:
                rows.append(p_idx)
                cols.append(gene_idx[node])

    if not rows:
        return []

    A_path_gene = csc_matrix(
        (np.ones(len(rows)), (rows, cols)),
        shape=(n_paths, n_genes)
    )

    # Base constraints
    neg_I = -eye(n_paths, format='csc')
    cov_matrix = hstack([A_path_gene, neg_I], format='csc')
    cov_constraint = LinearConstraint(cov_matrix, lb=np.zeros(n_paths))

    min_covered = math.ceil(min_coverage * n_paths)
    sum_y = csc_matrix(
        np.concatenate([np.zeros(n_genes), np.ones(n_paths)]).reshape(1, -1)
    )
    coverage_constraint = LinearConstraint(sum_y, lb=np.array([min_covered]))

    sum_x = csc_matrix(
        np.concatenate([np.ones(n_genes), np.zeros(n_paths)]).reshape(1, -1)
    )
    card_constraint = LinearConstraint(sum_x, ub=np.array([max_size]))

    bounds = Bounds(lb=np.zeros(n_vars), ub=np.ones(n_vars))
    integrality = np.ones(n_vars)

    solutions = []
    exclusion_constraints = []
    optimal_cost = None

    for iteration in range(k):
        all_constraints = [cov_constraint, coverage_constraint, card_constraint] + exclusion_constraints

        try:
            result = milp(
                c=c,
                constraints=all_constraints,
                integrality=integrality,
                bounds=bounds,
                options={'time_limit': 15}
            )
        except Exception:
            break

        if not result.success:
            break

        x_vals = result.x[:n_genes]
        selected = {genes[i] for i in range(n_genes) if x_vals[i] > 0.5}
        if not selected:
            break

        total_cost = sum(gene_costs[g] for g in selected)

        # Check cost tolerance
        if optimal_cost is None:
            optimal_cost = total_cost
        elif total_cost > optimal_cost * cost_tolerance:
            break

        covered_count = sum(
            1 for p in paths if any(g in selected for g in p.nodes)
        )
        coverage = covered_count / n_paths
        paths_covered = {
            p.path_id for p in paths if any(g in selected for g in p.nodes)
        }

        hs = HittingSet(
            targets=frozenset(selected),
            total_cost=total_cost,
            coverage=coverage,
            paths_covered=paths_covered,
        )
        solutions.append(hs)

        # Add exclusion constraint: sum of x_g for genes in this solution <= |solution| - 1
        excl_vec = np.zeros((1, n_vars))
        for g in selected:
            excl_vec[0, gene_idx[g]] = 1.0
        excl_matrix = csc_matrix(excl_vec)
        excl_constraint = LinearConstraint(excl_matrix, ub=np.array([len(selected) - 1]))
        exclusion_constraints.append(excl_constraint)

    # Also try greedy (may differ from ILP)
    greedy = solver._solve_greedy(paths, gene_costs, max_size)
    if greedy:
        solutions.append(greedy)

    # Deduplicate and sort
    seen = set()
    unique = []
    for s in solutions:
        if s.targets not in seen:
            seen.add(s.targets)
            unique.append(s)
    unique.sort(key=lambda x: (len(x.targets), x.total_cost))
    return unique[:k]


def run_analysis():
    """Main analysis loop."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 72)
    print("MHS NON-UNIQUENESS AND BIOLOGICAL PLAUSIBILITY ANALYSIS")
    print("=" * 72)
    print(f"Cancer types: {len(CANCER_TYPES)}")
    print(f"Top-N solutions: {TOP_N}")
    print(f"Cost tolerance: 1.5x optimal")
    print(f"Constraints: druggability>={DRUGGABILITY_THRESHOLD}, "
          f"no pan-essential, toxicity<{TOXICITY_CEILING}, "
          f"pathway-overlap<{MAX_SAME_PATHWAY_FRAC}")
    print()

    # Initialize pipeline
    print("Initializing pipeline components...")
    analyzer = PanCancerXNodeAnalyzer()
    depmap = analyzer.depmap
    omnipath = analyzer.omnipath
    drug_db = analyzer.drug_db
    cost_fn = analyzer.cost_fn
    solver = analyzer.solver

    all_results = []
    summary_rows = []

    for ct_idx, cancer_type in enumerate(CANCER_TYPES, 1):
        print(f"\n[{ct_idx}/{len(CANCER_TYPES)}] {cancer_type}")
        print("-" * 60)

        try:
            # Get viability paths
            print("  Inferring viability paths...")
            paths = analyzer.path_inference.infer_all_paths(cancer_type)
            if not paths:
                print("  ⚠ No viability paths — skipping")
                continue

            all_genes = set()
            for p in paths:
                all_genes.update(p.nodes)

            # Compute gene costs
            gene_costs = {}
            for g in all_genes:
                cost_obj = cost_fn.compute_cost(g, cancer_type)
                gene_costs[g] = cost_obj.total_cost()

            # 1. Enumerate near-optimal solutions
            print(f"  Enumerating near-optimal MHS (pool={len(all_genes)} genes, "
                  f"{len(paths)} paths)...")
            solutions = enumerate_near_optimal(
                solver, paths, gene_costs,
                MAX_SIZE, MIN_COVERAGE, cost_tolerance=1.5
            )
            print(f"  Found {len(solutions)} near-optimal solutions")

            if not solutions:
                print("  ⚠ No feasible solutions — skipping")
                continue

            # 2. Diversity metrics (unconstrained)
            div_unconstrained = diversity_metrics(solutions)
            print(f"  Unconstrained diversity: {div_unconstrained['n_unique_genes']} unique genes, "
                  f"mean Jaccard={div_unconstrained['mean_jaccard']:.3f}")

            # Record top-1 unconstrained
            top1_unconstrained = solutions[0].targets

            # 3. Apply biological-plausibility constraints
            print("  Applying constraints...")
            constrained, constraint_stats = apply_constraints(
                solutions, drug_db, cost_fn, cancer_type, paths
            )
            print(f"  After constraints: {len(constrained)} solutions surviving "
                  f"(removed: {constraint_stats['removed_no_druggable']} druggability, "
                  f"{constraint_stats['removed_pan_essential']} pan-essential, "
                  f"{constraint_stats['removed_high_toxicity']} toxicity, "
                  f"{constraint_stats['removed_pathway_redundant']} pathway-redundant)")

            # 4. Diversity metrics (constrained)
            div_constrained = diversity_metrics(constrained) if constrained else {
                'n_solutions': 0, 'mean_jaccard': 0, 'n_unique_genes': 0, 'gene_entropy': 0
            }

            # 5. Check if top-1 changed
            top1_constrained = constrained[0].targets if constrained else frozenset()
            top1_changed = (top1_constrained != top1_unconstrained)
            top1_jaccard = jaccard_distance(top1_unconstrained, top1_constrained) if constrained else 1.0

            if top1_changed and constrained:
                print(f"  ★ Top-1 CHANGED: {set(top1_unconstrained)} → {set(top1_constrained)} "
                      f"(Jaccard dist={top1_jaccard:.3f})")
            else:
                print(f"  Top-1 unchanged: {set(top1_unconstrained)}")

            # 6. Gene frequency across solutions
            gene_freq_unconstrained = Counter()
            for s in solutions:
                gene_freq_unconstrained.update(s.targets)
            gene_freq_constrained = Counter()
            for s in constrained:
                gene_freq_constrained.update(s.targets)

            # Build result row
            row = {
                'cancer_type': cancer_type,
                'n_paths': len(paths),
                'pool_size': len(all_genes),
                # Unconstrained
                'n_unconstrained': len(solutions),
                'unc_unique_genes': div_unconstrained['n_unique_genes'],
                'unc_mean_jaccard': div_unconstrained['mean_jaccard'],
                'unc_gene_entropy': div_unconstrained['gene_entropy'],
                'unc_top1': '+'.join(sorted(top1_unconstrained)),
                'unc_top1_cost': solutions[0].total_cost,
                'unc_top1_coverage': solutions[0].coverage,
                'unc_top1_size': len(solutions[0].targets),
                # Constrained
                'n_constrained': len(constrained),
                'con_unique_genes': div_constrained.get('n_unique_genes', 0),
                'con_mean_jaccard': div_constrained.get('mean_jaccard', 0),
                'con_gene_entropy': div_constrained.get('gene_entropy', 0),
                'con_top1': '+'.join(sorted(top1_constrained)) if constrained else 'NONE',
                'con_top1_cost': constrained[0].total_cost if constrained else float('nan'),
                'con_top1_coverage': constrained[0].coverage if constrained else 0,
                'con_top1_size': len(constrained[0].targets) if constrained else 0,
                # Changeover
                'top1_changed': top1_changed,
                'top1_jaccard_dist': top1_jaccard,
                # Constraint removal counts
                'removed_druggability': constraint_stats['removed_no_druggable'],
                'removed_pan_essential': constraint_stats['removed_pan_essential'],
                'removed_toxicity': constraint_stats['removed_high_toxicity'],
                'removed_pathway_redundant': constraint_stats['removed_pathway_redundant'],
                # Top gene frequencies
                'unc_top_genes': ', '.join(f"{g}({c})" for g, c in gene_freq_unconstrained.most_common(5)),
                'con_top_genes': ', '.join(f"{g}({c})" for g, c in gene_freq_constrained.most_common(5)),
            }
            summary_rows.append(row)

            # Detailed per-cancer JSON
            all_results.append({
                'cancer_type': cancer_type,
                'unconstrained_solutions': [
                    {'targets': sorted(s.targets), 'cost': s.total_cost,
                     'coverage': s.coverage, 'size': len(s.targets)}
                    for s in solutions
                ],
                'constrained_solutions': [
                    {'targets': sorted(s.targets), 'cost': s.total_cost,
                     'coverage': s.coverage, 'size': len(s.targets)}
                    for s in constrained
                ],
                'constraint_stats': constraint_stats,
                'diversity_unconstrained': div_unconstrained,
                'diversity_constrained': div_constrained,
                'top1_changed': top1_changed,
                'top1_jaccard_dist': top1_jaccard,
            })

        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback; traceback.print_exc()
            continue

    # ── Save results ────────────────────────────────────────────────────
    df = pd.DataFrame(summary_rows)
    df.to_csv(f"{OUTPUT_DIR}/mhs_nonuniqueness_summary.csv", index=False)
    print(f"\nSaved: {OUTPUT_DIR}/mhs_nonuniqueness_summary.csv")

    with open(f"{OUTPUT_DIR}/mhs_nonuniqueness_detail.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Saved: {OUTPUT_DIR}/mhs_nonuniqueness_detail.json")

    # ── Print summary table ─────────────────────────────────────────────
    print("\n" + "=" * 120)
    print("SUMMARY")
    print("=" * 120)
    print(f"{'Cancer Type':<38} {'#Unc':>4} {'#Con':>4} {'UncGenes':>8} "
          f"{'ConGenes':>8} {'MnJacc':>6} {'Changed':>7} {'JaccDist':>8} "
          f"{'PanEss':>6} {'PathRed':>7}")
    print("-" * 120)

    n_changed = 0
    for _, r in df.iterrows():
        changed_str = "YES" if r['top1_changed'] else "no"
        if r['top1_changed']:
            n_changed += 1
        print(f"{r['cancer_type']:<38} {r['n_unconstrained']:>4} {r['n_constrained']:>4} "
              f"{r['unc_unique_genes']:>8} {r['con_unique_genes']:>8} "
              f"{r['unc_mean_jaccard']:>6.3f} {changed_str:>7} "
              f"{r['top1_jaccard_dist']:>8.3f} "
              f"{r['removed_pan_essential']:>6} {r['removed_pathway_redundant']:>7}")

    print("-" * 120)
    n_total = len(df)
    print(f"\nTop-1 changed under constraints: {n_changed}/{n_total} "
          f"({100*n_changed/n_total:.1f}%) cancer types")
    print(f"Mean unconstrained solutions: {df['n_unconstrained'].mean():.1f}")
    print(f"Mean constrained solutions: {df['n_constrained'].mean():.1f}")
    print(f"Mean Jaccard diversity: {df['unc_mean_jaccard'].mean():.3f}")
    print(f"Mean genes lost to pan-essential filter: {df['removed_pan_essential'].mean():.1f}")
    print(f"Mean genes lost to pathway-redundancy: {df['removed_pathway_redundant'].mean():.1f}")

    # ── Generate figure ─────────────────────────────────────────────────
    _generate_figure(df)

    return df


def _generate_figure(df: pd.DataFrame):
    """Generate summary figure."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping figure")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Number of solutions before/after constraints
    ax = axes[0, 0]
    x = np.arange(len(df))
    w = 0.35
    ax.bar(x - w/2, df['n_unconstrained'], w, label='Unconstrained', color='#4C72B0')
    ax.bar(x + w/2, df['n_constrained'], w, label='With constraints', color='#DD8452')
    ax.set_xticks(x)
    ax.set_xticklabels([ct[:12] for ct in df['cancer_type']], rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Number of near-optimal MHS')
    ax.set_title('A. Solution multiplicity')
    ax.legend(fontsize=8)

    # 2. Jaccard diversity distribution
    ax = axes[0, 1]
    ax.bar(x, df['unc_mean_jaccard'], color='#4C72B0', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([ct[:12] for ct in df['cancer_type']], rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Mean pairwise Jaccard distance')
    ax.set_title('B. Solution diversity (unconstrained)')
    ax.axhline(y=df['unc_mean_jaccard'].mean(), color='red', linestyle='--', alpha=0.7,
               label=f"Mean={df['unc_mean_jaccard'].mean():.2f}")
    ax.legend(fontsize=8)

    # 3. Constraint removal breakdown
    ax = axes[1, 0]
    categories = ['Druggability', 'Pan-essential', 'Toxicity', 'Path redundancy']
    cols = ['removed_druggability', 'removed_pan_essential', 'removed_toxicity', 'removed_pathway_redundant']
    means = [df[c].mean() for c in cols]
    colors = ['#55A868', '#C44E52', '#8172B2', '#CCB974']
    ax.bar(categories, means, color=colors)
    ax.set_ylabel('Mean solutions removed per cancer')
    ax.set_title('C. Constraint impact')
    for i, v in enumerate(means):
        ax.text(i, v + 0.1, f'{v:.1f}', ha='center', fontsize=9)

    # 4. Top-1 stability
    ax = axes[1, 1]
    changed = df['top1_changed'].sum()
    unchanged = len(df) - changed
    ax.pie([unchanged, changed], labels=['Unchanged', 'Changed'],
           autopct='%1.1f%%', colors=['#4C72B0', '#DD8452'],
           startangle=90)
    ax.set_title(f'D. Top-1 MHS stability\n({changed}/{len(df)} changed)')

    plt.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/mhs_nonuniqueness.png', dpi=200, bbox_inches='tight')
    fig.savefig(f'{OUTPUT_DIR}/mhs_nonuniqueness.pdf', bbox_inches='tight')
    plt.close()
    print(f"Figure saved: {OUTPUT_DIR}/mhs_nonuniqueness.{{png,pdf}}")


if __name__ == '__main__':
    run_analysis()
