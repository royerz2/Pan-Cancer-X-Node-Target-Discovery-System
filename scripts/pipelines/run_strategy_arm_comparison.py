#!/usr/bin/env python3
"""Run, benchmark, and compare explicit strategy arms against historical baselines."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alin.drug_sensitivity import PRISMLoader
from alin.strategy_arms import SUPPORTED_STRATEGY_ARMS, infer_strategy_arm_from_scoring_mode
from alin.viability_scorecard import build_second_layer_scorecard
from benchmarking_module import (
    export_benchmark,
    run_benchmark as run_primary_benchmark,
    run_target_prioritization_benchmark,
)
from gold_standard import run_benchmark as run_expanded_gold_benchmark
from outcome_benchmark import ALINScorer
from pharmacological_validation import PharmacologicalValidator


# Use the active interpreter by default and allow explicit override when needed.
PYTHON = Path(os.environ.get('ALIN_PYTHON', sys.executable or 'python'))
DEFAULT_ARMS = list(SUPPORTED_STRATEGY_ARMS)
DEFAULT_MODES = ['actionable', 'exploratory']
DECLARED_PRIMARY_METRIC_LABEL = '21-entry exact-combination recall'
CURRENT_COMPARISON_TUPLE_LABEL = (
    '21-entry pair-overlap, then 43-entry any-overlap, then 21-entry exact, '
    'then 43-entry precision, then mean matched rank'
)
SECOND_LAYER_POLICY_LABEL = (
    'strong-support share, then weak-support share, then unknown direct-benefit burden, '
    'then escape/Bliss/curated-extension support, then concordance non-regression, '
    'then the current comparison tuple'
)
DEPMAP_DIR = ROOT / 'depmap_data'
DRUG_DIR = ROOT / 'drug_sensitivity_data'

HISTORICAL_RESULT_DIRS: Dict[str, Dict[str, Path]] = {
    'v6.2': {
        'actionable': ROOT / 'archive' / 'cleanup_20260305_235148' / 'results' / 'results_v6.2_actionable',
        'exploratory': ROOT / 'archive' / 'cleanup_20260305_235148' / 'results' / 'results_v6.2_exploratory',
    },
    'v7.1': {
        'actionable': ROOT / 'archive' / 'cleanup_20260305_235148' / 'results' / 'results_v7.1_actionable',
        'exploratory': ROOT / 'archive' / 'cleanup_20260305_235148' / 'results' / 'results_v7.1_exploratory',
    },
    'v7.2': {
        'actionable': ROOT / 'archive' / 'cleanup_20260305_235148' / 'results' / 'results_v7.2_actionable',
        'exploratory': ROOT / 'archive' / 'cleanup_20260305_235148' / 'results' / 'results_v7.2_exploratory',
    },
    'v10e': {
        'actionable': ROOT / 'outputs' / 'current' / 'v10e' / 'actionable' / 'results_v10e_actionable',
        'exploratory': ROOT / 'outputs' / 'current' / 'v10e' / 'exploratory' / 'results_v10e_exploratory',
    },
}


@dataclass(frozen=True)
class EvaluationTarget:
    label: str
    mode: str
    results_dir: Path
    source_kind: str
    strategy_arm: str = ''


@dataclass
class SecondLayerResources:
    scorer: ALINScorer
    validator: PharmacologicalValidator
    prism_loader: PRISMLoader


class Logger:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self._handle = log_path.open('w', encoding='utf-8')

    def close(self) -> None:
        self._handle.close()

    def log(self, message: str, echo: bool = True) -> None:
        timestamp = datetime.now().strftime('%H:%M:%S')
        line = f'[{timestamp}] {message}'
        if echo:
            print(line, flush=True)
        self._handle.write(line + '\n')
        self._handle.flush()


def json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    if hasattr(value, 'item'):
        return value.item()
    return str(value)


def safe_float(value) -> Optional[float]:
    try:
        if value is None or value == '':
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def ensure_results_dir(output_base: Path, mode: str) -> Path:
    if (output_base / 'triple_combinations.csv').exists():
        return output_base
    suffixed = Path(f'{output_base}_{mode}')
    if (suffixed / 'triple_combinations.csv').exists():
        return suffixed
    return output_base


def run_command(
    logger: Logger,
    label: str,
    cmd: Sequence[str],
    timeout: Optional[int],
    stream_child_output: bool,
) -> None:
    logger.log(f'START {label}')
    logger.log('CMD   ' + ' '.join(str(part) for part in cmd))
    started = time.time()
    env = {**os.environ, 'PYTHONIOENCODING': 'utf-8'}
    process = subprocess.Popen(
        [str(part) for part in cmd],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        errors='replace',
        bufsize=1,
        env=env,
    )
    assert process.stdout is not None
    try:
        while True:
            line = process.stdout.readline()
            if line:
                logger.log(f'  | {line.rstrip()}', echo=stream_child_output)
            elif process.poll() is not None:
                break
            else:
                time.sleep(0.25)
            if timeout and (time.time() - started) > timeout:
                process.kill()
                raise TimeoutError(f'{label} exceeded timeout of {timeout}s')
        return_code = process.wait()
    finally:
        process.stdout.close()
    elapsed = time.time() - started
    if return_code != 0:
        raise RuntimeError(f'{label} failed with exit code {return_code} after {elapsed:.0f}s')
    logger.log(f'DONE  {label} ({elapsed:.0f}s)')


def normalize_arm_series(df: pd.DataFrame) -> pd.Series:
    if 'Strategy_Arm' in df.columns:
        arms = df['Strategy_Arm'].fillna('').astype(str).str.strip()
        if (arms != '').any():
            return arms.replace('', 'default')
    if 'Scoring_Mode' in df.columns:
        return df['Scoring_Mode'].map(infer_strategy_arm_from_scoring_mode)
    return pd.Series(['default'] * len(df), index=df.index, dtype='object')


def load_predictions_frame(results_dir: Path) -> pd.DataFrame:
    triples_csv = results_dir / 'triple_combinations.csv'
    if not triples_csv.exists():
        raise FileNotFoundError(f'Missing triples CSV: {triples_csv}')
    df = pd.read_csv(triples_csv)
    if df.empty:
        raise ValueError(f'No predictions found in {triples_csv}')
    return df


def extract_combo(row: pd.Series) -> frozenset[str]:
    targets = []
    for column in ('Target_1', 'Target_2', 'Target_3'):
        value = row.get(column, '')
        if isinstance(value, str) and value.strip():
            targets.append(value.strip())
    return frozenset(targets)


def summarize_predictions(results_dir: Path) -> Dict[str, object]:
    df = load_predictions_frame(results_dir)
    strategy_arms = normalize_arm_series(df)
    combos = [tuple(sorted(extract_combo(row))) for _, row in df.iterrows()]
    combo_counter = Counter(combos)
    gene_counter: Counter[str] = Counter()
    for combo in combos:
        gene_counter.update(combo)

    combined_scores = [value for value in (safe_float(v) for v in df.get('Combined_Score', [])) if value is not None]
    synergy_scores = [value for value in (safe_float(v) for v in df.get('Synergy_Score', [])) if value is not None]
    resistance_scores = [value for value in (safe_float(v) for v in df.get('Resistance_Score', [])) if value is not None]

    def mean_or_none(values: List[float]) -> Optional[float]:
        return round(sum(values) / len(values), 6) if values else None

    return {
        'results_dir': str(results_dir),
        'n_rows': int(len(df)),
        'n_cancers': int(df['Cancer_Type'].nunique()) if 'Cancer_Type' in df.columns else 0,
        'n_unique_combos': len(combo_counter),
        'n_unique_genes': len(gene_counter),
        'mean_combined_score': mean_or_none(combined_scores),
        'mean_synergy_score': mean_or_none(synergy_scores),
        'mean_resistance_score': mean_or_none(resistance_scores),
        'strategy_arm_distribution': dict(sorted(Counter(strategy_arms).items())),
        'top_genes': [
            {'gene': gene, 'count': count}
            for gene, count in gene_counter.most_common(10)
        ],
        'top_combos': [
            {'combo': ' + '.join(combo), 'count': count}
            for combo, count in combo_counter.most_common(10)
        ],
    }


def load_top_predictions(results_dir: Path) -> Dict[str, frozenset[str]]:
    df = load_predictions_frame(results_dir).copy()
    if 'Rank' in df.columns:
        df['_rank'] = pd.to_numeric(df['Rank'], errors='coerce').fillna(float('inf'))
        df = df.sort_values(['Cancer_Type', '_rank'], ascending=[True, True])
    elif 'Combined_Score' in df.columns:
        df['_score'] = pd.to_numeric(df['Combined_Score'], errors='coerce').fillna(float('inf'))
        df = df.sort_values(['Cancer_Type', '_score'], ascending=[True, True])
    top_map: Dict[str, frozenset[str]] = {}
    for _, row in df.iterrows():
        cancer_type = str(row.get('Cancer_Type', '')).strip()
        if not cancer_type or cancer_type in top_map:
            continue
        combo = extract_combo(row)
        if combo:
            top_map[cancer_type] = combo
    return top_map


def compare_prediction_maps(left: Dict[str, frozenset[str]], right: Dict[str, frozenset[str]]) -> Dict[str, object]:
    shared_cancers = sorted(set(left) & set(right))
    exact = 0
    pair_overlap = 0
    any_overlap = 0
    jaccards: List[float] = []
    for cancer_type in shared_cancers:
        left_combo = left[cancer_type]
        right_combo = right[cancer_type]
        intersection = len(left_combo & right_combo)
        union = len(left_combo | right_combo)
        if left_combo == right_combo:
            exact += 1
        if intersection >= 2:
            pair_overlap += 1
        if intersection >= 1:
            any_overlap += 1
        if union:
            jaccards.append(intersection / union)
    shared_count = len(shared_cancers)
    return {
        'shared_cancers': shared_count,
        'left_only_cancers': len(set(left) - set(right)),
        'right_only_cancers': len(set(right) - set(left)),
        'exact_overlap_rate': (exact / shared_count) if shared_count else None,
        'pair_overlap_rate': (pair_overlap / shared_count) if shared_count else None,
        'any_overlap_rate': (any_overlap / shared_count) if shared_count else None,
        'mean_jaccard': (sum(jaccards) / len(jaccards)) if jaccards else None,
    }


def rate_or_none(numerator: object, denominator: object) -> Optional[float]:
    numerator_value = safe_float(numerator)
    denominator_value = safe_float(denominator)
    if numerator_value is None or denominator_value in (None, 0.0):
        return None
    return numerator_value / denominator_value


def numeric_series(df: pd.DataFrame, column: str, default: float) -> pd.Series:
    if column in df.columns:
        return pd.to_numeric(df[column], errors='coerce').fillna(default)
    return pd.Series([default] * len(df), index=df.index, dtype='float64')


def display_label(row: pd.Series) -> str:
    strategy_arm = str(row.get('strategy_arm', '')).strip()
    if strategy_arm:
        return strategy_arm
    return str(row.get('label', 'unknown')).strip() or 'unknown'


def format_percent(value: object) -> str:
    numeric = safe_float(value)
    if numeric is None:
        return 'n/a'
    return f'{numeric:.1%}'


def support_profile_summary(row: pd.Series) -> str:
    clauses = [
        f'strong={format_percent(row.get("support_strong_rate"))}',
        f'weak={format_percent(row.get("support_weak_rate"))}',
        f'guardrail={format_percent(row.get("concordance_non_regression_rate"))}',
    ]
    for label, column in (
        ('escape+', 'escape_benefit_rate'),
        ('Bliss+', 'bliss_benefit_rate'),
        ('third+', 'third_target_recovery_rate'),
    ):
        value = safe_float(row.get(column))
        if value is not None:
            clauses.append(f'{label}={value:.1%}')
    return ', '.join(clauses)


def support_advantage_clauses(candidate: pd.Series, baseline: pd.Series) -> List[str]:
    clauses: List[str] = []

    def delta(column: str) -> float:
        return (safe_float(candidate.get(column)) or 0.0) - (safe_float(baseline.get(column)) or 0.0)

    strong_delta = delta('support_strong_rate')
    weak_delta = delta('support_weak_rate')
    unknown_delta = delta('support_unknown_rate')
    escape_delta = delta('escape_benefit_rate')
    bliss_delta = delta('bliss_benefit_rate')
    third_delta = delta('third_target_recovery_rate')
    guardrail_delta = delta('concordance_non_regression_rate')

    if strong_delta > 0.001:
        clauses.append(
            f'higher strong-support share ({format_percent(candidate.get("support_strong_rate"))} '
            f'vs {format_percent(baseline.get("support_strong_rate"))})'
        )
    if weak_delta < -0.001:
        clauses.append(
            f'lower weak-support share ({format_percent(candidate.get("support_weak_rate"))} '
            f'vs {format_percent(baseline.get("support_weak_rate"))})'
        )
    if escape_delta > 0.001:
        clauses.append(
            f'better escape-route benefit coverage ({format_percent(candidate.get("escape_benefit_rate"))} '
            f'vs {format_percent(baseline.get("escape_benefit_rate"))})'
        )
    if bliss_delta > 0.001:
        clauses.append(
            f'better Bliss benefit coverage ({format_percent(candidate.get("bliss_benefit_rate"))} '
            f'vs {format_percent(baseline.get("bliss_benefit_rate"))})'
        )
    if third_delta > 0.001:
        clauses.append(
            f'better curated third-target recovery ({format_percent(candidate.get("third_target_recovery_rate"))} '
            f'vs {format_percent(baseline.get("third_target_recovery_rate"))})'
        )
    if guardrail_delta > 0.001:
        clauses.append(
            f'better concordance non-regression ({format_percent(candidate.get("concordance_non_regression_rate"))} '
            f'vs {format_percent(baseline.get("concordance_non_regression_rate"))})'
        )
    if unknown_delta < -0.001:
        clauses.append(
            f'fewer unknown direct-benefit cases ({format_percent(candidate.get("support_unknown_rate"))} '
            f'vs {format_percent(baseline.get("support_unknown_rate"))})'
        )
    return clauses


def rank_mode_df_by_current_tuple(mode_df: pd.DataFrame) -> pd.DataFrame:
    if mode_df.empty:
        return mode_df.copy()
    ranked = mode_df.copy()
    ranked['_tuple_pair'] = numeric_series(ranked, 'primary_recall_pair_overlap', -1.0)
    ranked['_tuple_gold_any'] = numeric_series(ranked, 'gold_any_overlap', -1.0)
    ranked['_tuple_exact'] = numeric_series(ranked, 'primary_recall_exact', -1.0)
    ranked['_tuple_precision'] = numeric_series(ranked, 'gold_precision', -1.0)
    ranked['_tuple_rank'] = numeric_series(ranked, 'primary_mean_rank_when_matched', float('inf'))
    return ranked.sort_values(
        ['_tuple_pair', '_tuple_gold_any', '_tuple_exact', '_tuple_precision', '_tuple_rank'],
        ascending=[False, False, False, False, True],
    )


def pick_mode_tuple_leader(mode_df: pd.DataFrame) -> Optional[pd.Series]:
    ranked = rank_mode_df_by_current_tuple(mode_df)
    if ranked.empty:
        return None
    return ranked.iloc[0]


def rank_mode_df_by_policy(mode_df: pd.DataFrame) -> pd.DataFrame:
    if mode_df.empty:
        return mode_df.copy()
    ranked = rank_mode_df_by_current_tuple(mode_df)
    ranked['_support_strong_rate'] = numeric_series(ranked, 'support_strong_rate', -1.0)
    ranked['_support_weak_rate'] = numeric_series(ranked, 'support_weak_rate', 1.0)
    ranked['_support_unknown_rate'] = numeric_series(ranked, 'support_unknown_rate', 1.0)
    ranked['_escape_benefit_rate'] = numeric_series(ranked, 'escape_benefit_rate', -1.0)
    ranked['_bliss_benefit_rate'] = numeric_series(ranked, 'bliss_benefit_rate', -1.0)
    ranked['_third_target_recovery_rate'] = numeric_series(ranked, 'third_target_recovery_rate', -1.0)
    ranked['_guardrail_rate'] = numeric_series(ranked, 'concordance_non_regression_rate', -1.0)
    ranked['_support_concordance'] = numeric_series(ranked, 'second_layer_mean_target_concordance_score', -1.0)
    return ranked.sort_values(
        [
            '_support_strong_rate',
            '_support_weak_rate',
            '_support_unknown_rate',
            '_escape_benefit_rate',
            '_bliss_benefit_rate',
            '_third_target_recovery_rate',
            '_guardrail_rate',
            '_support_concordance',
            '_tuple_pair',
            '_tuple_gold_any',
            '_tuple_exact',
            '_tuple_precision',
            '_tuple_rank',
        ],
        ascending=[False, True, True, False, False, False, False, False, False, False, False, False, True],
    )


def select_mode_scope(summary_df: pd.DataFrame, mode: str) -> tuple[pd.DataFrame, str]:
    mode_df = summary_df[summary_df['mode'] == mode].copy()
    if mode_df.empty:
        return mode_df, 'evaluated runs'
    fresh_df = mode_df[mode_df['source_kind'] == 'fresh'].copy()
    if not fresh_df.empty:
        return fresh_df, 'fresh arms'
    return mode_df, 'evaluated runs'


def evaluate_target(
    logger: Logger,
    target: EvaluationTarget,
    eval_root: Path,
    second_layer_resources: Optional[SecondLayerResources] = None,
) -> Dict[str, object]:
    logger.log(f'EVALUATE {target.label} ({target.mode}) from {target.results_dir}')
    results_dir = target.results_dir
    triples_csv = results_dir / 'triple_combinations.csv'
    summary_csv = results_dir / 'pan_cancer_summary.csv'
    ranked_csv = results_dir / 'ranked_triple_combinations.csv'
    output_dir = eval_root / f'{target.label}_{target.mode}'

    # Cache: if all expected output files exist, reload from disk
    _expected = [
        'primary_benchmark/benchmark_metrics.json',
        'expanded_gold_standard.json',
        'prediction_summary.json',
        'second_layer_scorecard_summary.json',
    ]
    if all((output_dir / f).exists() for f in _expected):
        logger.log(f'CACHED {target.label} ({target.mode}) — reusing {output_dir}')
        with (output_dir / 'primary_benchmark' / 'benchmark_metrics.json').open(encoding='utf-8') as fh:
            primary_metrics = json.load(fh)
        target_prioritization = primary_metrics.pop('target_prioritization', {})
        with (output_dir / 'expanded_gold_standard.json').open(encoding='utf-8') as fh:
            gold = json.load(fh)
        with (output_dir / 'prediction_summary.json').open(encoding='utf-8') as fh:
            summary = json.load(fh)
        with (output_dir / 'second_layer_scorecard_summary.json').open(encoding='utf-8') as fh:
            second_layer_summary = json.load(fh)
        return {
            'label': target.label,
            'mode': target.mode,
            'source_kind': target.source_kind,
            'strategy_arm': target.strategy_arm,
            'results_dir': str(results_dir),
            'triples_csv': str(triples_csv),
            'ranked_csv': str(ranked_csv) if ranked_csv.exists() else '',
            'primary_metrics': primary_metrics,
            'target_prioritization': target_prioritization,
            'expanded_gold_standard': {
                'recall': gold.get('recall', {}),
                'gold_standard_stats': gold.get('gold_standard_stats', {}),
                'predictions_source': gold.get('predictions_source', ''),
                'used_legacy_best_combo': gold.get('used_legacy_best_combo', False),
            },
            'prediction_summary': summary,
            'second_layer_summary': second_layer_summary,
            'evaluation_dir': str(output_dir),
        }

    output_dir.mkdir(parents=True, exist_ok=True)

    primary_results, primary_metrics = run_primary_benchmark(
        str(triples_csv),
        str(summary_csv) if summary_csv.exists() else None,
    )
    target_prioritization = run_target_prioritization_benchmark(str(triples_csv))
    export_benchmark(
        primary_results,
        primary_metrics,
        output_dir / 'primary_benchmark',
        target_prioritization=target_prioritization,
    )

    gold = run_expanded_gold_benchmark(
        str(triples_csv),
        tier1=True,
        tier2=False,
        verbose=False,
    )
    with (output_dir / 'expanded_gold_standard.json').open('w', encoding='utf-8') as handle:
        json.dump(gold, handle, indent=2, default=json_default)
    pd.DataFrame(gold.get('results', [])).to_csv(
        output_dir / 'expanded_gold_standard_results.csv',
        index=False,
    )

    summary = summarize_predictions(results_dir)
    with (output_dir / 'prediction_summary.json').open('w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2, default=json_default)

    second_layer_rows, second_layer_summary = build_second_layer_scorecard(
        str(triples_csv),
        depmap_dir=DEPMAP_DIR,
        drug_dir=DRUG_DIR,
        scorer=None if second_layer_resources is None else second_layer_resources.scorer,
        validator=None if second_layer_resources is None else second_layer_resources.validator,
        prism_loader=None if second_layer_resources is None else second_layer_resources.prism_loader,
    )
    pd.DataFrame(second_layer_rows).to_csv(output_dir / 'second_layer_scorecard.csv', index=False)
    with (output_dir / 'second_layer_scorecard_summary.json').open('w', encoding='utf-8') as handle:
        json.dump(second_layer_summary, handle, indent=2, default=json_default)

    return {
        'label': target.label,
        'mode': target.mode,
        'source_kind': target.source_kind,
        'strategy_arm': target.strategy_arm,
        'results_dir': str(results_dir),
        'triples_csv': str(triples_csv),
        'ranked_csv': str(ranked_csv) if ranked_csv.exists() else '',
        'primary_metrics': primary_metrics,
        'target_prioritization': target_prioritization,
        'expanded_gold_standard': {
            'recall': gold.get('recall', {}),
            'gold_standard_stats': gold.get('gold_standard_stats', {}),
            'predictions_source': gold.get('predictions_source', ''),
            'used_legacy_best_combo': gold.get('used_legacy_best_combo', False),
        },
        'prediction_summary': summary,
        'second_layer_summary': second_layer_summary,
        'evaluation_dir': str(output_dir),
    }


def flatten_evaluation(evaluation: Dict[str, object]) -> Dict[str, object]:
    primary = evaluation.get('primary_metrics', {})
    gold = evaluation.get('expanded_gold_standard', {}).get('recall', {})
    prediction_summary = evaluation.get('prediction_summary', {})
    second_layer_summary = evaluation.get('second_layer_summary', {})
    support_label_rates = second_layer_summary.get('support_label_rates', {})
    target_prioritization = evaluation.get('target_prioritization', {})
    return {
        'label': evaluation['label'],
        'mode': evaluation['mode'],
        'source_kind': evaluation['source_kind'],
        'strategy_arm': evaluation.get('strategy_arm', ''),
        'results_dir': evaluation['results_dir'],
        'primary_total_gold_standard': primary.get('total_gold_standard'),
        'primary_recall_exact': primary.get('recall_exact'),
        'primary_recall_superset': primary.get('recall_superset_or_better'),
        'primary_recall_pair_overlap': primary.get('recall_pair_overlap_or_better', primary.get('recall_pairwise_or_better')),
        'primary_recall_any_overlap': primary.get('recall_any_overlap_or_better'),
        'primary_mean_rank_when_matched': primary.get('mean_rank_when_matched'),
        'target_hit_rate': target_prioritization.get('hit_rate'),
        'gold_exact': gold.get('exact'),
        'gold_superset': gold.get('superset'),
        'gold_pair_overlap': gold.get('pair_overlap', gold.get('pairwise')),
        'gold_any_overlap': gold.get('any_overlap', gold.get('pair_overlap', gold.get('pairwise'))),
        'gold_precision': gold.get('precision'),
        'gold_testable_any_overlap': gold.get('testable_any_overlap'),
        'n_cancers': prediction_summary.get('n_cancers'),
        'n_rows': prediction_summary.get('n_rows'),
        'n_unique_combos': prediction_summary.get('n_unique_combos'),
        'n_unique_genes': prediction_summary.get('n_unique_genes'),
        'mean_combined_score': prediction_summary.get('mean_combined_score'),
        'mean_synergy_score': prediction_summary.get('mean_synergy_score'),
        'mean_resistance_score': prediction_summary.get('mean_resistance_score'),
        'strategy_arm_distribution': json.dumps(prediction_summary.get('strategy_arm_distribution', {}), sort_keys=True),
        'support_strong_count': second_layer_summary.get('n_support_strong'),
        'support_mixed_count': second_layer_summary.get('n_support_mixed'),
        'support_weak_count': second_layer_summary.get('n_support_weak'),
        'support_strong_rate': support_label_rates.get('strong'),
        'support_mixed_rate': support_label_rates.get('mixed'),
        'support_weak_rate': support_label_rates.get('weak'),
        'support_unknown_count': second_layer_summary.get('n_unknown_evaluable'),
        'support_unknown_rate': second_layer_summary.get('unknown_evaluable_rate'),
        'concordance_non_regression_count': second_layer_summary.get('n_concordance_non_regression'),
        'concordance_non_regression_rate': second_layer_summary.get('concordance_non_regression_rate'),
        'second_layer_mean_target_concordance_score': second_layer_summary.get('mean_target_concordance_score'),
        'positive_escape_benefit_count': second_layer_summary.get('n_positive_escape_benefit'),
        'escape_benefit_rate': rate_or_none(
            second_layer_summary.get('n_positive_escape_benefit'),
            second_layer_summary.get('n_escape_evaluable'),
        ),
        'positive_bliss_benefit_count': second_layer_summary.get('n_positive_bliss_benefit'),
        'bliss_benefit_rate': rate_or_none(
            second_layer_summary.get('n_positive_bliss_benefit'),
            second_layer_summary.get('n_bliss_evaluable'),
        ),
        'third_target_recovery_count': second_layer_summary.get('n_third_target_extension_recovered'),
        'third_target_recovery_rate': rate_or_none(
            second_layer_summary.get('n_third_target_extension_recovered'),
            second_layer_summary.get('n_third_target_evaluable'),
        ),
    }


def pick_mode_winner(mode_df: pd.DataFrame) -> Optional[pd.Series]:
    ranked = rank_mode_df_by_policy(mode_df)
    if ranked.empty:
        return None
    return ranked.iloc[0]


def build_recommendations(summary_df: pd.DataFrame) -> List[str]:
    recommendations: List[str] = []
    winners: Dict[str, pd.Series] = {}
    for mode in DEFAULT_MODES:
        mode_df, scope_label = select_mode_scope(summary_df, mode)
        winner = pick_mode_winner(mode_df)
        if winner is None:
            continue
        winners[mode] = winner
        tuple_leader = pick_mode_tuple_leader(mode_df)
        winner_name = display_label(winner)
        tuple_name = display_label(tuple_leader) if tuple_leader is not None else ''
        default_df = mode_df[mode_df['strategy_arm'] == 'default']
        default_row = default_df.iloc[0] if not default_df.empty else None
        if default_row is None:
            message = f'{mode}: policy leader={winner_name} among {scope_label} ({support_profile_summary(winner)}).'
            if tuple_leader is not None and tuple_name != winner_name:
                message += f' The old comparison tuple instead favored {tuple_name} on overlap metrics alone.'
            recommendations.append(message)
            continue

        reason_clauses = support_advantage_clauses(winner, default_row)
        reason_text = '; '.join(reason_clauses[:4]) if reason_clauses else support_profile_summary(winner)
        strong_delta = (safe_float(winner.get('support_strong_rate')) or 0.0) - (
            safe_float(default_row.get('support_strong_rate')) or 0.0
        )
        weak_delta = (safe_float(winner.get('support_weak_rate')) or 0.0) - (
            safe_float(default_row.get('support_weak_rate')) or 0.0
        )

        if winner_name == 'default':
            message = f'{mode}: keep default; it retains the strongest second-layer support profile ({support_profile_summary(default_row)}).'
            if tuple_leader is not None and tuple_name != 'default':
                message += f' The old comparison tuple instead favored {tuple_name}, but without better viability support.'
            recommendations.append(message)
        elif strong_delta >= -0.001 and weak_delta <= 0.001:
            message = f'{mode}: prefer {winner_name} over default for now; {reason_text}.'
            if tuple_leader is not None and tuple_name != winner_name:
                message += f' The old comparison tuple instead favored {tuple_name} on overlap metrics alone.'
            recommendations.append(message)
        else:
            message = f'{mode}: {winner_name} ranks first on the second-layer policy, but the tradeoff versus default is mixed ({reason_text}); treat as inconclusive.'
            if tuple_leader is not None and tuple_name != winner_name:
                message += f' The old comparison tuple instead favored {tuple_name}.'
            recommendations.append(message)

    actionable_winner = winners.get('actionable')
    exploratory_winner = winners.get('exploratory')
    if actionable_winner is not None and exploratory_winner is not None:
        actionable_name = display_label(actionable_winner)
        exploratory_name = display_label(exploratory_winner)
        if actionable_name == exploratory_name:
            recommendations.append(
                f'cross-mode: {actionable_name} is the policy leader across actionable and exploratory.'
            )
        else:
            recommendations.append(
                f'cross-mode: actionable favors {actionable_name}, exploratory favors {exploratory_name}; keep the modes split.'
            )
    return recommendations


def write_report(
    output_root: Path,
    summary_df: pd.DataFrame,
    overlap_df: pd.DataFrame,
    recommendations: List[str],
) -> None:
    report_path = output_root / 'comparison_report.md'
    has_overlap_rows = not overlap_df.empty and 'mode' in overlap_df.columns

    def fmt(value) -> str:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return 'n/a'
        if isinstance(value, float):
            return f'{value:.1%}' if 0.0 <= value <= 1.0 else f'{value:.3f}'
        return str(value)

    lines = [
        '# Strategy Arm Comparison',
        '',
        f'Generated: {datetime.now().isoformat(timespec="seconds")}',
        '',
        '## Metric Notes',
        '',
        f'- The benchmark module declares {DECLARED_PRIMARY_METRIC_LABEL} as the primary metric.',
        f'- Ordering below follows the post-scorecard policy: {SECOND_LAYER_POLICY_LABEL}.',
        f'- The old comparison tuple remains a secondary tie-break and disagreement monitor: {CURRENT_COMPARISON_TUPLE_LABEL}.',
        '',
        '## Recommendations',
        '',
    ]
    for line in recommendations:
        lines.append(f'- {line}')

    for mode in DEFAULT_MODES:
        mode_df = summary_df[summary_df['mode'] == mode].copy()
        if mode_df.empty:
            continue
        scoped_mode_df, scope_label = select_mode_scope(summary_df, mode)
        policy_winner = pick_mode_winner(scoped_mode_df)
        tuple_leader = pick_mode_tuple_leader(scoped_mode_df)
        mode_df = rank_mode_df_by_policy(mode_df)
        lines.extend([
            '',
            f'## {mode.title()}',
            '',
        ])
        if policy_winner is not None:
            lines.append(f'- Policy winner ({scope_label}): {display_label(policy_winner)} [{support_profile_summary(policy_winner)}]')
        if tuple_leader is not None:
            lines.append(f'- Current tuple leader ({scope_label}): {display_label(tuple_leader)}')
        lines.extend([
            '',
            '| Label | Source | Arm | Strong | Weak | Unknown | Guardrail | Escape+ | Bliss+ | Third+ | 21-entry Pair | 21-entry Exact | 43-entry Any | 43-entry Precision |',
            '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |',
        ])
        for _, row in mode_df.iterrows():
            lines.append(
                '| {label} | {source} | {arm} | {strong} | {weak} | {unknown} | {guardrail} | {escape_rate} | {bliss_rate} | {third_rate} | {pair} | {exact} | {gold_any} | {precision} |'.format(
                    label=row['label'],
                    source=row['source_kind'],
                    arm=row.get('strategy_arm', '') or 'legacy',
                    strong=fmt(row.get('support_strong_rate')),
                    weak=fmt(row.get('support_weak_rate')),
                    unknown=fmt(row.get('support_unknown_rate')),
                    guardrail=fmt(row.get('concordance_non_regression_rate')),
                    escape_rate=fmt(row.get('escape_benefit_rate')),
                    bliss_rate=fmt(row.get('bliss_benefit_rate')),
                    third_rate=fmt(row.get('third_target_recovery_rate')),
                    pair=fmt(row.get('primary_recall_pair_overlap')),
                    exact=fmt(row.get('primary_recall_exact')),
                    gold_any=fmt(row.get('gold_any_overlap')),
                    precision=fmt(row.get('gold_precision')),
                )
            )

        mode_overlap = overlap_df[overlap_df['mode'] == mode] if has_overlap_rows else pd.DataFrame()
        if not mode_overlap.empty:
            lines.extend([
                '',
                f'### {mode.title()} Pairwise Overlap',
                '',
                '| Left | Right | Shared Cancers | Exact | Pair | Any | Mean Jaccard |',
                '| --- | --- | --- | --- | --- | --- | --- |',
            ])
            for _, row in mode_overlap.iterrows():
                lines.append(
                    '| {left} | {right} | {shared} | {exact} | {pair} | {any_ov} | {jaccard} |'.format(
                        left=row['left_label'],
                        right=row['right_label'],
                        shared=row['shared_cancers'],
                        exact=fmt(row['exact_overlap_rate']),
                        pair=fmt(row['pair_overlap_rate']),
                        any_ov=fmt(row['any_overlap_rate']),
                        jaccard=fmt(row['mean_jaccard']),
                    )
                )

    report_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run and compare explicit ALIN strategy arms')
    parser.add_argument('--arms', nargs='+', default=DEFAULT_ARMS, choices=SUPPORTED_STRATEGY_ARMS)
    parser.add_argument('--modes', nargs='+', default=DEFAULT_MODES, choices=DEFAULT_MODES)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--top-n', type=int, default=None,
                        help='Optional top-N cancer cap passed through to pan_cancer_xnode.py')
    parser.add_argument('--no-api', action='store_true',
                        help='Disable API calls during fresh runs')
    parser.add_argument('--skip-pipelines', action='store_true',
                        help='Do not run fresh pipelines; only evaluate existing outputs and history')
    parser.add_argument('--skip-historical', action='store_true',
                        help='Do not re-score historical baselines')
    parser.add_argument('--historical-baselines', nargs='+',
                        default=['v6.2', 'v7.1', 'v7.2', 'v10e'],
                        choices=sorted(HISTORICAL_RESULT_DIRS.keys()))
    parser.add_argument('--output-root', type=str, default='',
                        help='Optional comparison output root. Defaults to a timestamped outputs/comparisons directory.')
    parser.add_argument('--force', action='store_true',
                        help='Re-run fresh pipelines even when outputs already exist in the target directory')
    parser.add_argument('--timeout', type=int, default=0,
                        help='Per-pipeline timeout in seconds. 0 disables timeouts.')
    parser.add_argument('--stop-on-error', action='store_true',
                        help='Abort after the first pipeline or evaluation failure instead of continuing')
    parser.add_argument('--stream-subprocess-output', action='store_true',
                        help='Echo every child pipeline log line to the console. Detailed logs always go to comparison.log.')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_root = Path(args.output_root) if args.output_root else (
        ROOT / 'outputs' / 'comparisons' / f'strategy_arm_head_to_head_{timestamp}'
    )
    output_root.mkdir(parents=True, exist_ok=True)
    fresh_root = output_root / 'fresh_runs'
    eval_root = output_root / 'evaluations'
    eval_root.mkdir(parents=True, exist_ok=True)
    logger = Logger(output_root / 'comparison.log')

    try:
        logger.log('Strategy-arm comparison starting')
        logger.log(f'Output root: {output_root}')
        logger.log(f'Arms: {args.arms}')
        logger.log(f'Modes: {args.modes}')

        evaluation_targets: List[EvaluationTarget] = []
        failures: List[Dict[str, str]] = []

        if not args.skip_pipelines:
            fresh_root.mkdir(parents=True, exist_ok=True)
            for mode in args.modes:
                for arm in args.arms:
                    label = f'{arm}_{mode}'
                    output_base = fresh_root / label
                    results_dir = ensure_results_dir(output_base, mode)
                    triples_csv = results_dir / 'triple_combinations.csv'
                    if args.force or not triples_csv.exists():
                        cmd = [
                            str(PYTHON),
                            'pan_cancer_xnode.py',
                            '--mode', mode,
                            '--all-cancers',
                            '--triples',
                            '--strategy-arm', arm,
                            '--output', str(output_base),
                            '--workers', str(args.workers),
                        ]
                        if args.top_n is not None:
                            cmd.extend(['--top-n', str(args.top_n)])
                        if args.no_api:
                            cmd.append('--no-api')
                        try:
                            run_command(
                                logger,
                                f'pipeline:{label}',
                                cmd,
                                args.timeout or None,
                                stream_child_output=args.stream_subprocess_output,
                            )
                        except Exception as exc:
                            logger.log(f'FAIL  pipeline:{label}: {exc}')
                            failures.append({'stage': 'pipeline', 'label': label, 'error': str(exc)})
                            if args.stop_on_error:
                                raise
                            continue
                    results_dir = ensure_results_dir(output_base, mode)
                    if triples_csv.exists() or (results_dir / 'triple_combinations.csv').exists():
                        evaluation_targets.append(EvaluationTarget(
                            label=arm,
                            mode=mode,
                            results_dir=results_dir,
                            source_kind='fresh',
                            strategy_arm=arm,
                        ))
                    else:
                        message = f'Fresh run completed without triples CSV: {results_dir}'
                        logger.log(f'FAIL  {message}')
                        failures.append({'stage': 'pipeline', 'label': label, 'error': message})
                        if args.stop_on_error:
                            raise FileNotFoundError(message)
        else:
            # --skip-pipelines: discover any pre-existing fresh outputs
            if fresh_root.is_dir():
                for mode in args.modes:
                    for arm in args.arms:
                        label = f'{arm}_{mode}'
                        output_base = fresh_root / label
                        results_dir = ensure_results_dir(output_base, mode)
                        triples_csv = results_dir / 'triple_combinations.csv'
                        if triples_csv.exists():
                            evaluation_targets.append(EvaluationTarget(
                                label=arm,
                                mode=mode,
                                results_dir=results_dir,
                                source_kind='fresh',
                                strategy_arm=arm,
                            ))
                            logger.log(f'REUSE fresh:{label} from {results_dir}')

        if not args.skip_historical:
            for version in args.historical_baselines:
                for mode in args.modes:
                    results_dir = HISTORICAL_RESULT_DIRS[version][mode]
                    if (results_dir / 'triple_combinations.csv').exists():
                        evaluation_targets.append(EvaluationTarget(
                            label=version,
                            mode=mode,
                            results_dir=results_dir,
                            source_kind='historical',
                            strategy_arm='',
                        ))
                    else:
                        logger.log(f'SKIP  historical:{version}_{mode} missing {results_dir / "triple_combinations.csv"}')

        second_layer_resources: Optional[SecondLayerResources] = None
        if evaluation_targets:
            logger.log('Loading shared second-layer resources')
            second_layer_resources = SecondLayerResources(
                scorer=ALINScorer(),
                validator=PharmacologicalValidator(str(DEPMAP_DIR), str(DRUG_DIR)),
                prism_loader=PRISMLoader(str(DRUG_DIR)),
            )

        evaluations: List[Dict[str, object]] = []
        for target in evaluation_targets:
            try:
                evaluations.append(
                    evaluate_target(
                        logger,
                        target,
                        eval_root,
                        second_layer_resources=second_layer_resources,
                    )
                )
            except Exception as exc:
                logger.log(f'FAIL  evaluate:{target.label}_{target.mode}: {exc}')
                failures.append({'stage': 'evaluation', 'label': f'{target.label}_{target.mode}', 'error': str(exc)})
                if args.stop_on_error:
                    raise

        if not evaluations:
            logger.log('No successful evaluations were produced')
            manifest = {
                'output_root': str(output_root),
                'failures': failures,
                'evaluations': [],
            }
            with (output_root / 'manifest.json').open('w', encoding='utf-8') as handle:
                json.dump(manifest, handle, indent=2, default=json_default)
            return 1 if failures else 0

        summary_rows = [flatten_evaluation(evaluation) for evaluation in evaluations]
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(output_root / 'benchmark_summary.csv', index=False)

        prediction_maps = {
            (evaluation['label'], evaluation['mode']): load_top_predictions(Path(evaluation['results_dir']))
            for evaluation in evaluations
        }
        overlap_rows: List[Dict[str, object]] = []
        for left in evaluations:
            for right in evaluations:
                if left['mode'] != right['mode']:
                    continue
                if left['label'] >= right['label']:
                    continue
                comparison = compare_prediction_maps(
                    prediction_maps[(left['label'], left['mode'])],
                    prediction_maps[(right['label'], right['mode'])],
                )
                overlap_rows.append({
                    'mode': left['mode'],
                    'left_label': left['label'],
                    'right_label': right['label'],
                    'left_source': left['source_kind'],
                    'right_source': right['source_kind'],
                    **comparison,
                })
        overlap_df = pd.DataFrame(overlap_rows, columns=[
            'mode',
            'left_label',
            'right_label',
            'left_source',
            'right_source',
            'shared_cancers',
            'left_only_cancers',
            'right_only_cancers',
            'exact_overlap_rate',
            'pair_overlap_rate',
            'any_overlap_rate',
            'mean_jaccard',
        ])
        overlap_df.to_csv(output_root / 'pairwise_overlap.csv', index=False)

        recommendations = build_recommendations(summary_df)
        write_report(output_root, summary_df, overlap_df, recommendations)

        manifest = {
            'output_root': str(output_root),
            'fresh_arms': args.arms,
            'modes': args.modes,
            'historical_baselines': [] if args.skip_historical else args.historical_baselines,
            'evaluations': evaluations,
            'failures': failures,
            'recommendations': recommendations,
        }
        with (output_root / 'manifest.json').open('w', encoding='utf-8') as handle:
            json.dump(manifest, handle, indent=2, default=json_default)

        logger.log(f'Wrote summary CSV: {output_root / "benchmark_summary.csv"}')
        logger.log(f'Wrote overlap CSV: {output_root / "pairwise_overlap.csv"}')
        logger.log(f'Wrote report: {output_root / "comparison_report.md"}')
        if failures:
            logger.log(f'Completed with {len(failures)} failure(s); see manifest.json')
            return 1
        logger.log('Comparison completed successfully')
        return 0
    finally:
        logger.close()


if __name__ == '__main__':
    sys.exit(main())