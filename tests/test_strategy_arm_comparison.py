import pandas as pd

from scripts.pipelines.run_strategy_arm_comparison import (
    build_recommendations,
    flatten_evaluation,
    pick_mode_winner,
    write_report,
)


def _policy_row(**overrides):
    row = {
        'label': 'default_actionable',
        'mode': 'actionable',
        'source_kind': 'fresh',
        'strategy_arm': 'default',
        'support_strong_rate': 0.10,
        'support_weak_rate': 0.30,
        'support_unknown_rate': 0.10,
        'escape_benefit_rate': 0.10,
        'bliss_benefit_rate': 0.05,
        'third_target_recovery_rate': 0.00,
        'concordance_non_regression_rate': 0.90,
        'second_layer_mean_target_concordance_score': 0.50,
        'primary_recall_pair_overlap': 0.40,
        'gold_any_overlap': 0.35,
        'primary_recall_exact': 0.0,
        'gold_precision': 0.20,
        'primary_mean_rank_when_matched': 1.0,
        'n_unique_combos': 10,
        'n_unique_genes': 12,
    }
    row.update(overrides)
    return row


def test_pick_mode_winner_prefers_second_layer_support_over_current_tuple():
    summary_df = pd.DataFrame([
        _policy_row(),
        _policy_row(
            label='pdac_actionable',
            strategy_arm='liaki_pdac_template',
            support_strong_rate=0.35,
            support_weak_rate=0.05,
            support_unknown_rate=0.00,
            escape_benefit_rate=0.25,
            bliss_benefit_rate=0.20,
            concordance_non_regression_rate=0.85,
            primary_recall_pair_overlap=0.25,
            gold_any_overlap=0.30,
        ),
    ])

    winner = pick_mode_winner(summary_df)

    assert winner is not None
    assert winner['strategy_arm'] == 'liaki_pdac_template'


def test_build_recommendations_reports_policy_over_tuple_disagreement():
    summary_df = pd.DataFrame([
        _policy_row(),
        _policy_row(
            label='pdac_actionable',
            strategy_arm='liaki_pdac_template',
            support_strong_rate=0.35,
            support_weak_rate=0.05,
            support_unknown_rate=0.00,
            escape_benefit_rate=0.25,
            bliss_benefit_rate=0.20,
            concordance_non_regression_rate=0.85,
            primary_recall_pair_overlap=0.25,
            gold_any_overlap=0.30,
        ),
    ])

    recommendations = build_recommendations(summary_df)
    joined = ' '.join(recommendations)

    assert 'prefer liaki_pdac_template over default' in joined
    assert 'favored default on overlap metrics alone' in joined


def test_build_recommendations_handles_historical_only_scope():
    summary_df = pd.DataFrame([
        _policy_row(
            label='v6.2',
            source_kind='historical',
            strategy_arm='',
            support_strong_rate=0.15,
            support_weak_rate=0.25,
            primary_recall_pair_overlap=0.45,
            gold_any_overlap=0.40,
        ),
        _policy_row(
            label='v10e',
            source_kind='historical',
            strategy_arm='',
            support_strong_rate=0.30,
            support_weak_rate=0.10,
            support_unknown_rate=0.00,
            escape_benefit_rate=0.20,
            bliss_benefit_rate=0.15,
            concordance_non_regression_rate=0.95,
            primary_recall_pair_overlap=0.30,
            gold_any_overlap=0.32,
        ),
    ])

    recommendations = build_recommendations(summary_df)
    joined = ' '.join(recommendations)

    assert 'policy leader=v10e among evaluated runs' in joined
    assert 'favored v6.2 on overlap metrics alone' in joined


def test_flatten_evaluation_includes_second_layer_summary_fields():
    row = flatten_evaluation(
        {
            'label': 'v10e',
            'mode': 'actionable',
            'source_kind': 'historical',
            'strategy_arm': '',
            'results_dir': 'results_v10e_actionable',
            'primary_metrics': {
                'total_gold_standard': 21,
                'recall_exact': 0.0,
                'recall_superset_or_better': 0.1,
                'recall_pair_overlap_or_better': 0.2,
                'recall_any_overlap_or_better': 0.3,
                'mean_rank_when_matched': 1.0,
            },
            'target_prioritization': {'hit_rate': 0.2},
            'expanded_gold_standard': {'recall': {'any_overlap': 0.4, 'precision': 0.5}},
            'prediction_summary': {
                'n_cancers': 76,
                'n_rows': 76,
                'n_unique_combos': 36,
                'n_unique_genes': 22,
                'mean_combined_score': 1.0,
                'mean_synergy_score': 0.5,
                'mean_resistance_score': 0.4,
                'strategy_arm_distribution': {'default': 76},
            },
            'second_layer_summary': {
                'n_support_strong': 10,
                'n_support_mixed': 50,
                'n_support_weak': 16,
                'support_label_rates': {'strong': 10 / 76, 'mixed': 50 / 76, 'weak': 16 / 76},
                'n_unknown_evaluable': 8,
                'unknown_evaluable_rate': 8 / 76,
                'n_concordance_non_regression': 60,
                'concordance_non_regression_rate': 60 / 76,
                'mean_target_concordance_score': 0.61,
                'n_positive_escape_benefit': 12,
                'n_escape_evaluable': 60,
                'n_positive_bliss_benefit': 15,
                'n_bliss_evaluable': 50,
                'n_third_target_extension_recovered': 2,
                'n_third_target_evaluable': 10,
            },
        }
    )

    assert row['support_strong_count'] == 10
    assert row['support_strong_rate'] == 10 / 76
    assert row['support_weak_rate'] == 16 / 76
    assert row['support_unknown_rate'] == 8 / 76
    assert row['concordance_non_regression_rate'] == 60 / 76
    assert row['escape_benefit_rate'] == 12 / 60
    assert row['bliss_benefit_rate'] == 15 / 50
    assert row['third_target_recovery_rate'] == 2 / 10


def test_write_report_uses_benchmark_family_labels(tmp_path):
    summary_df = pd.DataFrame([
        _policy_row()
    ])

    write_report(tmp_path, summary_df, pd.DataFrame(), ['test recommendation'])
    report = (tmp_path / 'comparison_report.md').read_text(encoding='utf-8')

    assert 'Strong' in report
    assert 'Guardrail' in report
    assert '21-entry Pair' in report
    assert '21-entry Exact' in report
    assert '43-entry Any' in report
    assert 'Primary Pair' not in report