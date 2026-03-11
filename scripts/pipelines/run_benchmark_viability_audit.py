#!/usr/bin/env python3
"""Run a benchmark viability audit for one ALIN prediction set."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alin.viability_scorecard import build_second_layer_scorecard
from benchmarking_module import (
    audit_benchmark_matches,
    export_benchmark,
    run_baseline_calibration,
    run_benchmark,
    run_target_prioritization_benchmark,
)
from gold_standard import run_benchmark as run_expanded_gold_benchmark


def json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    if hasattr(value, 'item'):
        return value.item()
    return str(value)


def fmt_rate(value) -> str:
    if value is None:
        return 'n/a'
    try:
        return f'{float(value):.1%}'
    except (TypeError, ValueError):
        return str(value)


def baseline_notes(row: Dict[str, object]) -> str:
    parts: List[str] = []
    if row.get('n_trials'):
        parts.append(f'n_trials={row["n_trials"]}')
    if row.get('n_cancers_with_data'):
        parts.append(f'cancers_with_data={row["n_cancers_with_data"]}')
    if row.get('n_cancers_with_drivers'):
        parts.append(f'cancers_with_drivers={row["n_cancers_with_drivers"]}')
    if row.get('median_pool_size'):
        parts.append(f'median_pool_size={row["median_pool_size"]}')
    if row.get('error'):
        parts.append(f'error={row["error"]}')
    return '; '.join(parts) if parts else 'n/a'


def metrics_table_row(label: str, metrics: Dict[str, object]) -> str:
    return '| {label} | {exact} | {superset} | {pair} | {any_overlap} | {none} |'.format(
        label=label,
        exact=fmt_rate(metrics.get('recall_exact')),
        superset=fmt_rate(metrics.get('recall_superset_or_better')),
        pair=fmt_rate(metrics.get('recall_pair_overlap_or_better', metrics.get('recall_pairwise_or_better'))),
        any_overlap=fmt_rate(metrics.get('recall_any_overlap_or_better')),
        none=metrics.get('no_match', 'n/a'),
    )


def _format_tier_counts(tier_counts: Dict[str, object]) -> str:
    if not tier_counts:
        return 'n/a'
    return ', '.join(
        f'Tier {tier}: {count}'
        for tier, count in sorted(tier_counts.items(), key=lambda item: item[0])
    )


def _bool_label(value: object) -> str:
    return 'yes' if value else 'no'


def write_report(
    output_root: Path,
    primary_metrics: Dict[str, object],
    target_prioritization: Dict[str, object],
    audit_summary: Dict[str, object],
    baseline_rows: List[Dict[str, object]],
    expanded_gold: Optional[Dict[str, object]],
    second_layer_rows: List[Dict[str, object]],
    second_layer_summary: Optional[Dict[str, object]],
) -> None:
    report_path = output_root / 'benchmark_viability_report.md'
    lines = [
        '# Benchmark Viability Audit',
        '',
        f'Generated: {datetime.now().isoformat(timespec="seconds")}',
        '',
        '## Metric Notes',
        '',
        '- The 21-entry benchmark module declares exact-combination recall as the primary metric.',
        '- The 21-entry implementation currently counts the lowest-rank matched prediction.',
        '- The expanded 43-entry benchmark counts the strongest available match tier, then rank.',
        '',
        '## 21-Entry Current Benchmark',
        '',
        f'- Predictions source: {primary_metrics.get("predictions_source", "n/a")}',
        f'- Used legacy best-combo metadata: {primary_metrics.get("used_legacy_best_combo", False)}',
        f'- Exact recall: {fmt_rate(primary_metrics.get("recall_exact"))}',
        f'- Pair-overlap recall: {fmt_rate(primary_metrics.get("recall_pair_overlap_or_better", primary_metrics.get("recall_pairwise_or_better")))}',
        f'- Any-overlap recall: {fmt_rate(primary_metrics.get("recall_any_overlap_or_better"))}',
        f'- Target prioritization hit rate: {fmt_rate(target_prioritization.get("hit_rate"))} ({target_prioritization.get("hits", 0)}/{target_prioritization.get("total", 0)})',
        '',
        '## Semantics Audit',
        '',
        '| View | Exact | Superset+ | Pair+ | Any+ | No Match |',
        '| --- | --- | --- | --- | --- | --- |',
        metrics_table_row('Top prediction', audit_summary.get('top_prediction_metrics', {})),
        metrics_table_row('Current 21-entry implementation', audit_summary.get('current_benchmark_metrics', {})),
        metrics_table_row('Strongest available', audit_summary.get('strongest_available_metrics', {})),
        '',
        f'- Rank semantic conflicts: {audit_summary.get("rank_semantic_conflicts", 0)}/{audit_summary.get("total_gold_standard", 0)}',
        f'- Exact available off-rank: {audit_summary.get("exact_available_off_rank", 0)}',
        f'- Top prediction missed but a later ranked match exists: {audit_summary.get("top_prediction_missed_but_later_match_exists", 0)}',
        '',
        '### Gap Classification',
        '',
        '| Classification | Count |',
        '| --- | --- |',
    ]

    for classification, count in audit_summary.get('gap_classification_counts', {}).items():
        lines.append(f'| {classification} | {count} |')

    if second_layer_summary:
        lines.extend([
            '',
            '## Second-Layer Scorecard',
            '',
            f'- Three-target predictions scored: {second_layer_summary.get("n_three_target_predictions", 0)}/{second_layer_summary.get("n_cancers", 0)}',
            f'- Escape-route evaluable: {second_layer_summary.get("n_escape_evaluable", 0)}; positive triple-vs-best-dual route reduction: {second_layer_summary.get("n_positive_escape_benefit", 0)}',
            f'- PRISM/CRISPR Bliss evaluable: {second_layer_summary.get("n_bliss_evaluable", 0)}; lower triple viability than best dual: {second_layer_summary.get("n_positive_bliss_benefit", 0)}',
            (
                f'- Support labels: strong={second_layer_summary.get("n_support_strong", 0)}, '
                f'mixed={second_layer_summary.get("n_support_mixed", 0)}, '
                f'weak={second_layer_summary.get("n_support_weak", 0)}; '
                f'unknown direct-benefit cases={second_layer_summary.get("n_unknown_evaluable", 0)}'
            ),
            (
                f'- Concordance non-regression guardrail: '
                f'{second_layer_summary.get("n_concordance_non_regression", 0)}/'
                f'{second_layer_summary.get("n_cancers", 0)} '
                f'({second_layer_summary.get("concordance_non_regression_rate", 0.0):.1%})'
            ),
            f'- Mean target concordance score: {second_layer_summary.get("mean_target_concordance_score", 0.0):.3f}',
            f'- Pharmacological evidence tiers: {_format_tier_counts(second_layer_summary.get("evidence_tier_counts", {}))}',
            f'- Curated third-target extension cases evaluable: {second_layer_summary.get("n_third_target_evaluable", 0)}; strict extension recoveries: {second_layer_summary.get("n_third_target_extension_recovered", 0)}',
        ])

        extension_rows = [row for row in second_layer_rows if row.get('third_target_evaluable')]
        if extension_rows:
            lines.extend([
                '',
                '### Curated Third-Target Extensions',
                '',
                '| Cancer | Predicted Triple | Doublet | Predicted Third | Known Thirds | Full Doublet | Third Match | Strict Extension |',
                '| --- | --- | --- | --- | --- | --- | --- | --- |',
            ])
            for row in extension_rows:
                lines.append(
                    '| {cancer} | {triple} | {doublet} | {predicted_third} | {known_thirds} | {full_doublet} | {third_match} | {strict_match} |'.format(
                        cancer=row.get('cancer_type', 'n/a'),
                        triple=row.get('prediction_targets', 'n/a'),
                        doublet=row.get('doublet', 'n/a'),
                        predicted_third=row.get('predicted_third_targets', 'n/a'),
                        known_thirds=row.get('known_third_targets', 'n/a'),
                        full_doublet=_bool_label(row.get('doublet_fully_recovered')),
                        third_match=_bool_label(row.get('third_target_match')),
                        strict_match=_bool_label(row.get('third_target_extension_recovered')),
                    )
                )

        lines.extend([
            '',
            'Detailed per-cancer second-layer outputs are written to `second_layer_scorecard.csv` and `second_layer_scorecard_summary.json` in this audit directory.',
        ])

    if baseline_rows:
        lines.extend([
            '',
            '## Baseline Calibration',
            '',
            '| Method | Metric Source | Exact | Superset+ | Pair+ | Any+ | Notes |',
            '| --- | --- | --- | --- | --- | --- | --- |',
        ])
        for row in baseline_rows:
            lines.append(
                '| {method} | {source} | {exact} | {superset} | {pair} | {any_overlap} | {notes} |'.format(
                    method=row.get('method', 'unknown'),
                    source=row.get('metric_source', 'n/a'),
                    exact=fmt_rate(row.get('recall_exact')),
                    superset=fmt_rate(row.get('recall_superset_or_better')),
                    pair=fmt_rate(row.get('recall_pair_overlap_or_better')),
                    any_overlap=fmt_rate(row.get('recall_any_overlap_or_better')),
                    notes=baseline_notes(row),
                )
            )

    if expanded_gold:
        recall = expanded_gold.get('recall', {})
        lines.extend([
            '',
            '## Expanded Gold Reference',
            '',
            '| Exact | Superset+ | Pair+ | Any+ | Precision | Testable Any+ |',
            '| --- | --- | --- | --- | --- | --- |',
            '| {exact} | {superset} | {pair} | {any_overlap} | {precision} | {testable_any} |'.format(
                exact=fmt_rate(recall.get('exact')),
                superset=fmt_rate(recall.get('superset')),
                pair=fmt_rate(recall.get('pair_overlap', recall.get('pairwise'))),
                any_overlap=fmt_rate(recall.get('any_overlap', recall.get('pair_overlap', recall.get('pairwise')))),
                precision=fmt_rate(recall.get('precision')),
                testable_any=fmt_rate(recall.get('testable_any_overlap')),
            ),
        ])

    report_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run a benchmark viability audit for one prediction set')
    parser.add_argument('--triples', type=str, default='results/triple_combinations.csv')
    parser.add_argument('--summary', type=str, default='results/pan_cancer_summary.csv')
    parser.add_argument('--output-root', type=str, default='')
    parser.add_argument('--reports-dir', type=str, default='')
    parser.add_argument('--depmap-dir', type=str, default=str(ROOT / 'depmap_data'))
    parser.add_argument('--drug-dir', type=str, default=str(ROOT / 'drug_sensitivity_data'))
    parser.add_argument('--n-trials', type=int, default=1000)
    parser.add_argument('--skip-baselines', action='store_true')
    parser.add_argument('--skip-expanded', action='store_true')
    parser.add_argument('--skip-second-layer', action='store_true')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_root = Path(args.output_root) if args.output_root else (
        ROOT / 'outputs' / 'benchmark_audits' / f'benchmark_viability_{timestamp}'
    )
    output_root.mkdir(parents=True, exist_ok=True)

    triples_path = Path(args.triples)
    summary_path = Path(args.summary) if args.summary else None
    reports_dir = args.reports_dir or str(triples_path.parent)

    print(f'[audit] output_root={output_root}')
    print(f'[audit] triples={triples_path}')

    primary_results, primary_metrics = run_benchmark(
        str(triples_path),
        str(summary_path) if summary_path and summary_path.exists() else None,
    )
    target_prioritization = run_target_prioritization_benchmark(str(triples_path))
    export_benchmark(
        primary_results,
        primary_metrics,
        output_root / 'primary_benchmark',
        target_prioritization=target_prioritization,
    )

    audit_rows, audit_summary = audit_benchmark_matches(
        str(triples_path),
        str(summary_path) if summary_path and summary_path.exists() else None,
    )
    pd.DataFrame(audit_rows).to_csv(output_root / 'benchmark_audit.csv', index=False)
    (output_root / 'benchmark_audit_summary.json').write_text(
        json.dumps(audit_summary, indent=2, default=json_default) + '\n',
        encoding='utf-8',
    )

    baseline_rows: List[Dict[str, object]] = []
    if not args.skip_baselines:
        baseline_rows = run_baseline_calibration(
            str(triples_path),
            n_trials=args.n_trials,
            reports_dir=reports_dir,
            depmap_dir=args.depmap_dir,
        )
        pd.DataFrame(baseline_rows).to_csv(output_root / 'baseline_calibration.csv', index=False)
        (output_root / 'baseline_calibration.json').write_text(
            json.dumps(baseline_rows, indent=2, default=json_default) + '\n',
            encoding='utf-8',
        )

    expanded_gold = None
    if not args.skip_expanded:
        expanded_gold = run_expanded_gold_benchmark(
            str(triples_path),
            tier1=True,
            tier2=False,
            verbose=False,
        )
        (output_root / 'expanded_gold_standard.json').write_text(
            json.dumps(expanded_gold, indent=2, default=json_default) + '\n',
            encoding='utf-8',
        )
        pd.DataFrame(expanded_gold.get('results', [])).to_csv(
            output_root / 'expanded_gold_standard_results.csv',
            index=False,
        )

    second_layer_rows: List[Dict[str, object]] = []
    second_layer_summary: Optional[Dict[str, object]] = None
    if not args.skip_second_layer:
        second_layer_rows, second_layer_summary = build_second_layer_scorecard(
            str(triples_path),
            depmap_dir=args.depmap_dir,
            drug_dir=args.drug_dir,
        )
        pd.DataFrame(second_layer_rows).to_csv(output_root / 'second_layer_scorecard.csv', index=False)
        (output_root / 'second_layer_scorecard_summary.json').write_text(
            json.dumps(second_layer_summary, indent=2, default=json_default) + '\n',
            encoding='utf-8',
        )

    write_report(
        output_root,
        primary_metrics,
        target_prioritization,
        audit_summary,
        baseline_rows,
        expanded_gold,
        second_layer_rows,
        second_layer_summary,
    )

    print('[audit] wrote primary benchmark, audit tables, and markdown report')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())