"""Second-layer viability scorecard for ALIN prediction runs.

This module complements the benchmark concordance layer with per-cancer
triple-vs-dual, pharmacological, and third-target-extension evidence.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from alin.constants import GENE_EQUIVALENTS, GENE_TO_DRUGS
from alin.drug_sensitivity import PRISMLoader
from alin.prediction_contract import RankedPrediction, load_ranked_predictions
from benchmarking_module import match_cancer
from outcome_benchmark import ALINScorer, APPROVED_DOUBLETS
from pharmacological_validation import PharmacologicalValidator, ValidationResult


MIN_CELL_LINES = 5


def expand_targets(targets: Iterable[str]) -> frozenset[str]:
    """Expand a target set with benchmark gene-equivalence relationships."""
    expanded = {target for target in targets if target}
    for target in list(expanded):
        expanded.update(GENE_EQUIVALENTS.get(target, set()))
    return frozenset(expanded)


def label_targets(targets: Iterable[str]) -> str:
    """Format a target set deterministically for reports."""
    return '+'.join(sorted({target for target in targets if target}))


def load_top_ranked_predictions(
    triples_csv: str | Path,
    min_size: int = 3,
) -> tuple[Dict[str, RankedPrediction], Path, bool]:
    """Load one top-ranked prediction per cancer, preferring 3-target sets."""
    loaded = load_ranked_predictions(triples_csv, include_legacy_best_combo=True)
    top_predictions: Dict[str, RankedPrediction] = {}
    for cancer_type, predictions in loaded.predictions_by_cancer.items():
        if not predictions:
            continue
        chosen = next(
            (prediction for prediction in predictions if len(prediction.targets) >= min_size),
            predictions[0],
        )
        top_predictions[cancer_type] = chosen
    return top_predictions, loaded.resolved_path, loaded.used_legacy_best_combo


def _candidate_genes_from_scorer(scorer: ALINScorer) -> list[str]:
    genes = set(scorer.adj.keys())
    for targets in scorer.adj.values():
        genes.update(targets)
    return sorted(genes)


def evaluate_third_target_extension(
    cancer_type: str,
    triple_targets: Sequence[str],
    scorer: ALINScorer,
) -> Dict[str, object]:
    """Evaluate whether a prediction recovers a curated doublet-extension pattern."""
    matching_entry = next(
        (
            entry
            for entry in APPROVED_DOUBLETS
            if match_cancer(cancer_type, entry['cancer']) or match_cancer(entry['cancer'], cancer_type)
        ),
        None,
    )
    if matching_entry is None:
        return {
            'third_target_evaluable': False,
            'doublet': '',
            'known_third_targets': '',
            'predicted_third_targets': '',
            'doublet_overlap_count': 0,
            'doublet_fully_recovered': False,
            'third_target_match': False,
            'third_target_extension_recovered': False,
            'known_third_best_rank': None,
            'predicted_third_best_rank': None,
        }

    triple_expanded = expand_targets(triple_targets)
    doublet_expanded = expand_targets(matching_entry['doublet'])
    predicted_thirds = triple_expanded - doublet_expanded
    known_thirds = [item['target'] for item in matching_entry['known_third_targets']]
    known_thirds_expanded = expand_targets(known_thirds)
    overlap_count = len(triple_expanded & doublet_expanded)
    doublet_fully_recovered = overlap_count >= len(matching_entry['doublet'])
    third_target_match = bool(predicted_thirds & known_thirds_expanded)

    known_rank = None
    predicted_rank = None
    cancer_lines = scorer.get_cancer_lines(cancer_type)
    candidate_ranks: Dict[str, int] = {}
    if cancer_lines:
        doublet_escape = scorer.escape_routes(set(matching_entry['doublet']), cancer_lines)
        ranked_candidates = []
        for gene in _candidate_genes_from_scorer(scorer):
            if gene in matching_entry['doublet'] or gene not in scorer.gene_map:
                continue
            triple_escape = scorer.escape_routes(set(matching_entry['doublet']) | {gene}, cancer_lines)
            ranked_candidates.append((gene, doublet_escape - triple_escape))
        ranked_candidates.sort(key=lambda item: (-item[1], item[0]))
        candidate_ranks = {gene: index + 1 for index, (gene, _) in enumerate(ranked_candidates)}

    known_rank_candidates = [candidate_ranks[gene] for gene in known_thirds_expanded if gene in candidate_ranks]
    predicted_rank_candidates = [candidate_ranks[gene] for gene in predicted_thirds if gene in candidate_ranks]
    if known_rank_candidates:
        known_rank = min(known_rank_candidates)
    if predicted_rank_candidates:
        predicted_rank = min(predicted_rank_candidates)

    return {
        'third_target_evaluable': True,
        'doublet': label_targets(matching_entry['doublet']),
        'known_third_targets': label_targets(known_thirds),
        'predicted_third_targets': label_targets(predicted_thirds),
        'doublet_overlap_count': overlap_count,
        'doublet_fully_recovered': doublet_fully_recovered,
        'third_target_match': third_target_match,
        'third_target_extension_recovered': doublet_fully_recovered and third_target_match,
        'known_third_best_rank': known_rank,
        'predicted_third_best_rank': predicted_rank,
    }


def _series_from_primary_prism(profile, cancer_lines: Sequence[str]) -> Optional[pd.Series]:
    """Convert a PRISM primary profile into fractional viability on overlapping lines."""
    if profile is None or not profile.cell_lines or not profile.ic50_values:
        return None
    series = pd.Series(profile.ic50_values, index=profile.cell_lines, dtype='float64')
    series = series.groupby(level=0).mean().dropna()
    common_lines = sorted(set(series.index) & set(cancer_lines))
    if len(common_lines) < MIN_CELL_LINES:
        return None
    viability = (2 ** series.loc[common_lines]).clip(lower=0.0, upper=2.0)
    viability.name = getattr(profile, 'drug_name', 'prism')
    return viability


def _crispr_viability_proxy(
    scorer: ALINScorer,
    gene: str,
    cancer_lines: Sequence[str],
) -> Optional[pd.Series]:
    """Fallback viability proxy when no PRISM profile is available for a target."""
    gene_column = scorer.gene_map.get(gene)
    if gene_column is None or not cancer_lines:
        return None
    values = scorer.crispr.loc[scorer.crispr.index.isin(cancer_lines), gene_column].dropna()
    if len(values) < MIN_CELL_LINES:
        return None
    viability = 1 / (1 + np.exp(-2 * values.astype(float)))
    return pd.Series(viability, index=values.index, dtype='float64')


def _select_gene_viability_profile(
    gene: str,
    cancer_lines: Sequence[str],
    prism_loader: PRISMLoader,
    scorer: ALINScorer,
) -> Optional[Dict[str, object]]:
    """Select the best available per-gene viability profile for combination modeling."""
    best_profile = None
    for drug in GENE_TO_DRUGS.get(gene, []):
        profile = prism_loader.get_drug_sensitivity(drug)
        viability = _series_from_primary_prism(profile, cancer_lines)
        if viability is None:
            continue
        candidate = {
            'gene': gene,
            'source': 'prism',
            'drug': drug,
            'series': viability,
        }
        if best_profile is None or len(viability) > len(best_profile['series']):
            best_profile = candidate

    if best_profile is not None:
        return best_profile

    viability_proxy = _crispr_viability_proxy(scorer, gene, cancer_lines)
    if viability_proxy is None:
        return None
    return {
        'gene': gene,
        'source': 'crispr_proxy',
        'drug': '',
        'series': viability_proxy,
    }


def evaluate_bliss_delta(
    cancer_lines: Sequence[str],
    triple_targets: Sequence[str],
    prism_loader: PRISMLoader,
    scorer: ALINScorer,
) -> Dict[str, object]:
    """Estimate triple-vs-best-dual viability using PRISM or CRISPR proxies."""
    profiles = {
        gene: _select_gene_viability_profile(gene, cancer_lines, prism_loader, scorer)
        for gene in triple_targets
    }
    dual_stats: Dict[str, Dict[str, object]] = {}
    for pair in combinations(sorted(triple_targets), 2):
        pair_profiles = [profiles[gene] for gene in pair]
        if any(profile is None for profile in pair_profiles):
            continue
        common_lines = sorted(set(pair_profiles[0]['series'].index).intersection(pair_profiles[1]['series'].index))
        if len(common_lines) < MIN_CELL_LINES:
            continue
        bliss = np.ones(len(common_lines), dtype='float64')
        sources = []
        drugs = []
        for profile in pair_profiles:
            bliss *= profile['series'].loc[common_lines].to_numpy(dtype='float64')
            sources.append(f"{profile['gene']}:{profile['source']}")
            if profile['drug']:
                drugs.append(f"{profile['gene']}:{profile['drug']}")
        dual_stats[label_targets(pair)] = {
            'median_viability': float(np.median(bliss)),
            'n_common_lines': len(common_lines),
            'profile_sources': '; '.join(sources),
            'selected_drugs': '; '.join(drugs),
        }

    triple_stat = None
    triple_profiles = [profiles[gene] for gene in triple_targets]
    if all(profile is not None for profile in triple_profiles):
        common_lines = set(triple_profiles[0]['series'].index)
        for profile in triple_profiles[1:]:
            common_lines &= set(profile['series'].index)
        common_lines = sorted(common_lines)
        if len(common_lines) >= MIN_CELL_LINES:
            bliss = np.ones(len(common_lines), dtype='float64')
            for profile in triple_profiles:
                bliss *= profile['series'].loc[common_lines].to_numpy(dtype='float64')
            triple_stat = {
                'median_viability': float(np.median(bliss)),
                'n_common_lines': len(common_lines),
            }

    best_dual_label = None
    best_dual_stat = None
    if dual_stats:
        best_dual_label, best_dual_stat = min(dual_stats.items(), key=lambda item: item[1]['median_viability'])

    gene_source_map = '; '.join(
        f'{gene}:{profiles[gene]["source"] if profiles[gene] is not None else "none"}'
        for gene in sorted(triple_targets)
    )
    gene_drug_map = '; '.join(
        f'{gene}:{profiles[gene]["drug"]}'
        for gene in sorted(triple_targets)
        if profiles[gene] is not None and profiles[gene]['drug']
    )
    delta = None
    if triple_stat is not None and best_dual_stat is not None:
        delta = triple_stat['median_viability'] - best_dual_stat['median_viability']

    return {
        'bliss_evaluable': triple_stat is not None and best_dual_stat is not None,
        'bliss_profiled_targets': sum(profile is not None for profile in profiles.values()),
        'bliss_gene_sources': gene_source_map,
        'bliss_gene_drugs': gene_drug_map,
        'best_dual_by_bliss': best_dual_label or '',
        'best_dual_median_bliss_viability': None if best_dual_stat is None else best_dual_stat['median_viability'],
        'triple_median_bliss_viability': None if triple_stat is None else triple_stat['median_viability'],
        'delta_bliss_viability_vs_best_dual': delta,
        'bliss_common_lines': None if triple_stat is None else triple_stat['n_common_lines'],
    }


def _dual_score_rows(
    scorer: ALINScorer,
    cancer_type: str,
    triple_targets: Sequence[str],
) -> list[Dict[str, object]]:
    rows = []
    for pair in combinations(sorted(triple_targets), 2):
        score = scorer.score_combination(pair, cancer_type)
        rows.append(
            {
                'label': label_targets(pair),
                'alin_composite': score.get('alin_composite'),
                'escape_routes': score.get('escape_routes'),
                'escape_route_ratio': score.get('escape_route_ratio'),
            }
        )
    return rows


def _mean_concordance_score(validation_result: ValidationResult) -> float:
    scores = [gene_result.concordance_score for gene_result in validation_result.gene_concordances.values()]
    return float(np.mean(scores)) if scores else 0.0


def _safe_float(value: object) -> Optional[float]:
    try:
        if value is None or value == '':
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_negative(value: object) -> bool:
    numeric = _safe_float(value)
    return numeric is not None and numeric < 0.0


def classify_second_layer_support(row: Dict[str, object]) -> Dict[str, object]:
    """Classify one scorecard row into an explicit support label."""
    evidence_tier = _safe_float(row.get('evidence_tier'))
    escape_delta = _safe_float(row.get('delta_escape_routes_vs_best_dual'))
    bliss_delta = _safe_float(row.get('delta_bliss_viability_vs_best_dual'))
    pharmacology_fraction = _safe_float(row.get('pharmacological_support_fraction')) or 0.0
    concordance_score = _safe_float(row.get('mean_target_concordance_score')) or 0.0
    n_prism_concordant = int(_safe_float(row.get('n_prism_concordant_targets')) or 0)

    escape_evaluable = bool(row.get('escape_evaluable')) and escape_delta is not None
    bliss_evaluable = bool(row.get('bliss_evaluable')) and bliss_delta is not None
    positive_escape = escape_evaluable and escape_delta < 0.0
    positive_bliss = bliss_evaluable and bliss_delta < 0.0
    negative_escape = escape_evaluable and not positive_escape
    negative_bliss = bliss_evaluable and not positive_bliss

    direct_benefit_supported = positive_escape or positive_bliss
    direct_benefit_conflict = direct_benefit_supported and (negative_escape or negative_bliss)

    supportive_tier = evidence_tier is not None and evidence_tier <= 3.0
    third_target_evaluable = bool(row.get('third_target_evaluable'))
    third_target_match = bool(row.get('third_target_match'))
    third_target_extension_recovered = bool(row.get('third_target_extension_recovered'))

    support_unknown_evaluable = (
        (not escape_evaluable or not bliss_evaluable)
        and not direct_benefit_supported
        and not negative_escape
        and not negative_bliss
    )
    concordance_non_regression = supportive_tier or n_prism_concordant > 0 or pharmacology_fraction > 0.0
    curated_support = third_target_match or third_target_extension_recovered
    compensating_support = supportive_tier or curated_support or concordance_non_regression or direct_benefit_supported

    if supportive_tier and direct_benefit_supported and (not third_target_evaluable or third_target_match):
        support_label = 'strong'
    elif (negative_escape or negative_bliss or (escape_evaluable or bliss_evaluable)) and not direct_benefit_supported and not compensating_support:
        support_label = 'weak'
    else:
        support_label = 'mixed'

    reasons = []
    if evidence_tier is not None:
        tier = int(evidence_tier)
        if supportive_tier:
            reasons.append(f'evidence tier {tier} is within the supportive 1-3 range')
        else:
            reasons.append(f'evidence tier {tier} is below the supportive 1-3 range')

    if positive_escape:
        reasons.append(f'escape routes improve versus the best dual ({escape_delta:.3f})')
    elif negative_escape:
        reasons.append(f'escape routes do not improve versus the best dual ({escape_delta:.3f})')
    else:
        reasons.append('escape-route comparison unavailable')

    if positive_bliss:
        reasons.append(f'Bliss viability improves versus the best dual ({bliss_delta:.3f})')
    elif negative_bliss:
        reasons.append(f'Bliss viability does not improve versus the best dual ({bliss_delta:.3f})')
    else:
        reasons.append('Bliss viability comparison unavailable')

    if third_target_evaluable:
        if third_target_extension_recovered:
            reasons.append('curated third-target extension is fully recovered')
        elif third_target_match:
            reasons.append('curated third-target match is recovered without full doublet recovery')
        else:
            reasons.append('curated third-target case does not recover the known third target')

    if direct_benefit_conflict:
        reasons.append('available direct-benefit signals are mixed across escape and Bliss')
    elif not direct_benefit_supported and concordance_non_regression:
        reasons.append(
            f'concordance is retained as a non-regression guardrail '
            f'(mean={concordance_score:.3f}, fraction={pharmacology_fraction:.3f})'
        )

    if support_unknown_evaluable:
        reasons.append('missing direct-benefit data prevents a clean strong-versus-weak call')

    if support_label == 'weak':
        reasons.append('no compensating pharmacology or curated-support signal offsets the negative direct evidence')

    return {
        'support_label': support_label,
        'support_reasons': '; '.join(reasons),
        'support_unknown_evaluable': support_unknown_evaluable,
        'concordance_non_regression': concordance_non_regression,
    }


def build_second_layer_scorecard(
    triples_csv: str | Path,
    depmap_dir: str | Path,
    drug_dir: str | Path,
    scorer: Optional[ALINScorer] = None,
    validator: Optional[PharmacologicalValidator] = None,
    prism_loader: Optional[PRISMLoader] = None,
) -> tuple[list[Dict[str, object]], Dict[str, object]]:
    """Build a per-cancer second-layer scorecard for one prediction run."""
    scorer = scorer or ALINScorer()
    validator = validator or PharmacologicalValidator(str(depmap_dir), str(drug_dir))
    prism_loader = prism_loader or PRISMLoader(str(drug_dir))
    top_predictions, resolved_path, used_legacy_best_combo = load_top_ranked_predictions(triples_csv)

    rows = []
    for cancer_type in sorted(top_predictions):
        prediction = top_predictions[cancer_type]
        triple_targets = tuple(sorted(prediction.targets))
        cancer_lines = scorer.get_cancer_lines(cancer_type)
        has_triple = len(triple_targets) >= 3

        dual_rows = _dual_score_rows(scorer, cancer_type, triple_targets) if has_triple else []
        triple_score = scorer.score_combination(triple_targets, cancer_type) if has_triple else {}

        best_dual_by_composite = max(dual_rows, key=lambda row: row['alin_composite'], default=None)
        best_dual_by_escape = min(dual_rows, key=lambda row: row['escape_routes'], default=None)

        pharmacology = validator.validate_predictions(
            cancer_type=cancer_type,
            predicted_targets=tuple(triple_targets),
            cell_line_ids=list(cancer_lines),
            n_cell_lines=len(cancer_lines),
        )
        bliss = evaluate_bliss_delta(cancer_lines, triple_targets, prism_loader, scorer) if has_triple else {
            'bliss_evaluable': False,
            'bliss_profiled_targets': 0,
            'bliss_gene_sources': '',
            'bliss_gene_drugs': '',
            'best_dual_by_bliss': '',
            'best_dual_median_bliss_viability': None,
            'triple_median_bliss_viability': None,
            'delta_bliss_viability_vs_best_dual': None,
            'bliss_common_lines': None,
        }
        extension = evaluate_third_target_extension(cancer_type, triple_targets, scorer) if has_triple else {
            'third_target_evaluable': False,
            'doublet': '',
            'known_third_targets': '',
            'predicted_third_targets': '',
            'doublet_overlap_count': 0,
            'doublet_fully_recovered': False,
            'third_target_match': False,
            'third_target_extension_recovered': False,
            'known_third_best_rank': None,
            'predicted_third_best_rank': None,
        }

        row = {
            'cancer_type': cancer_type,
            'predictions_source': str(resolved_path),
            'used_legacy_best_combo': used_legacy_best_combo,
            'prediction_rank': prediction.rank,
            'prediction_source': prediction.source,
            'prediction_size': len(triple_targets),
            'prediction_targets': label_targets(triple_targets),
            'n_depmap_lines': len(cancer_lines),
            'escape_evaluable': has_triple and bool(dual_rows),
            'triple_alin_composite': triple_score.get('alin_composite') if has_triple else None,
            'best_dual_by_composite': '' if best_dual_by_composite is None else best_dual_by_composite['label'],
            'best_dual_alin_composite': None if best_dual_by_composite is None else best_dual_by_composite['alin_composite'],
            'delta_alin_composite_vs_best_dual': None if best_dual_by_composite is None else triple_score['alin_composite'] - best_dual_by_composite['alin_composite'],
            'best_dual_by_escape': '' if best_dual_by_escape is None else best_dual_by_escape['label'],
            'best_dual_escape_routes': None if best_dual_by_escape is None else best_dual_by_escape['escape_routes'],
            'triple_escape_routes': triple_score.get('escape_routes') if has_triple else None,
            'delta_escape_routes_vs_best_dual': None if best_dual_by_escape is None else triple_score['escape_routes'] - best_dual_by_escape['escape_routes'],
            'triple_escape_route_ratio': triple_score.get('escape_route_ratio') if has_triple else None,
            'best_dual_escape_route_ratio': None if best_dual_by_escape is None else best_dual_by_escape['escape_route_ratio'],
            'delta_escape_ratio_vs_best_dual': None if best_dual_by_escape is None else triple_score['escape_route_ratio'] - best_dual_by_escape['escape_route_ratio'],
            'evidence_tier': pharmacology.evidence_tier.tier,
            'evidence_tier_label': pharmacology.evidence_tier.tier_label,
            'evidence_tier_reasons': '; '.join(pharmacology.evidence_tier.reasons),
            'n_prism_concordant_targets': pharmacology.evidence_tier.n_concordant_targets,
            'pharmacological_support_fraction': pharmacology.evidence_tier.concordance_fraction,
            'mean_target_concordance_score': _mean_concordance_score(pharmacology),
            'strongest_target_concordance': max(
                (gene_result.concordance_score for gene_result in pharmacology.gene_concordances.values()),
                default=0.0,
            ),
            **bliss,
            **extension,
        }
        row.update(classify_second_layer_support(row))
        rows.append(row)

    return rows, summarize_second_layer_scorecard(rows)


def summarize_second_layer_scorecard(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """Summarize per-cancer second-layer scorecard rows."""
    annotated_rows = []
    for row in rows:
        annotated = dict(row)
        if (
            'support_label' not in annotated
            or 'support_unknown_evaluable' not in annotated
            or 'concordance_non_regression' not in annotated
        ):
            annotated.update(classify_second_layer_support(annotated))
        annotated_rows.append(annotated)

    evidence_tier_counts = Counter(
        int(row['evidence_tier'])
        for row in annotated_rows
        if row.get('evidence_tier') is not None
    )
    support_label_counts = Counter(row.get('support_label', 'mixed') for row in annotated_rows)
    n_rows = len(annotated_rows)

    support_label_rates = {
        label: (support_label_counts.get(label, 0) / n_rows) if n_rows else 0.0
        for label in ('strong', 'mixed', 'weak')
    }

    summary = {
        'n_cancers': n_rows,
        'n_three_target_predictions': sum(int(row.get('prediction_size', 0)) >= 3 for row in annotated_rows),
        'n_escape_evaluable': sum(bool(row.get('escape_evaluable')) for row in annotated_rows),
        'n_positive_escape_benefit': sum(_is_negative(row.get('delta_escape_routes_vs_best_dual')) for row in annotated_rows),
        'n_positive_composite_benefit': sum(
            row.get('delta_alin_composite_vs_best_dual') is not None and float(row['delta_alin_composite_vs_best_dual']) > 0.0
            for row in annotated_rows
        ),
        'n_bliss_evaluable': sum(bool(row.get('bliss_evaluable')) for row in annotated_rows),
        'n_positive_bliss_benefit': sum(_is_negative(row.get('delta_bliss_viability_vs_best_dual')) for row in annotated_rows),
        'n_third_target_evaluable': sum(bool(row.get('third_target_evaluable')) for row in annotated_rows),
        'n_third_target_match': sum(bool(row.get('third_target_match')) for row in annotated_rows),
        'n_third_target_extension_recovered': sum(bool(row.get('third_target_extension_recovered')) for row in annotated_rows),
        'n_support_strong': support_label_counts.get('strong', 0),
        'n_support_mixed': support_label_counts.get('mixed', 0),
        'n_support_weak': support_label_counts.get('weak', 0),
        'support_label_counts': {label: support_label_counts.get(label, 0) for label in ('strong', 'mixed', 'weak')},
        'support_label_rates': support_label_rates,
        'n_unknown_evaluable': sum(bool(row.get('support_unknown_evaluable')) for row in annotated_rows),
        'unknown_evaluable_rate': (
            sum(bool(row.get('support_unknown_evaluable')) for row in annotated_rows) / n_rows
        ) if n_rows else 0.0,
        'n_concordance_non_regression': sum(bool(row.get('concordance_non_regression')) for row in annotated_rows),
        'concordance_non_regression_rate': (
            sum(bool(row.get('concordance_non_regression')) for row in annotated_rows) / n_rows
        ) if n_rows else 0.0,
        'evidence_tier_counts': {str(tier): count for tier, count in sorted(evidence_tier_counts.items())},
        'mean_target_concordance_score': float(
            np.mean([float(row.get('mean_target_concordance_score', 0.0)) for row in annotated_rows])
        ) if annotated_rows else 0.0,
    }
    return summary


__all__ = [
    'build_second_layer_scorecard',
    'classify_second_layer_support',
    'evaluate_bliss_delta',
    'evaluate_third_target_extension',
    'expand_targets',
    'label_targets',
    'load_top_ranked_predictions',
    'summarize_second_layer_scorecard',
]