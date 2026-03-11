from types import SimpleNamespace

import pandas as pd
import pytest

from alin.viability_scorecard import (
    build_second_layer_scorecard,
    classify_second_layer_support,
    load_top_ranked_predictions,
    summarize_second_layer_scorecard,
)


class _StubScorer:
    def __init__(self):
        self.adj = {
            'KRAS': {'STAT3', 'PIK3CA'},
            'EGFR': {'STAT3'},
            'STAT3': set(),
            'PIK3CA': set(),
        }
        self.gene_map = {gene: gene for gene in ['KRAS', 'EGFR', 'STAT3', 'PIK3CA']}
        self.crispr = pd.DataFrame(
            {
                'KRAS': [-0.9, -0.8, -0.85, -0.88, -0.82],
                'EGFR': [-0.7, -0.72, -0.69, -0.71, -0.68],
                'STAT3': [-0.6, -0.58, -0.61, -0.59, -0.6],
                'PIK3CA': [-0.3, -0.31, -0.29, -0.28, -0.32],
            },
            index=['CL1', 'CL2', 'CL3', 'CL4', 'CL5'],
        )

    def get_cancer_lines(self, cancer_type):
        return ['CL1', 'CL2', 'CL3', 'CL4', 'CL5']

    def escape_routes(self, targets, cancer_lines):
        targets = set(targets)
        if targets == {'KRAS', 'EGFR'}:
            return 5
        if targets == {'EGFR', 'STAT3'}:
            return 3
        if targets == {'KRAS', 'STAT3'}:
            return 2
        if targets == {'KRAS', 'EGFR', 'STAT3'}:
            return 1
        if 'PIK3CA' in targets:
            return 4
        return 6

    def score_combination(self, targets, cancer_name):
        label = '+'.join(sorted(set(targets)))
        composite = {
            'EGFR+KRAS': 0.40,
            'EGFR+STAT3': 0.55,
            'KRAS+STAT3': 0.62,
            'EGFR+KRAS+STAT3': 0.81,
        }
        escape_routes = {
            'EGFR+KRAS': 5,
            'EGFR+STAT3': 3,
            'KRAS+STAT3': 2,
            'EGFR+KRAS+STAT3': 1,
        }
        escape_ratio = {
            'EGFR+KRAS': 0.50,
            'EGFR+STAT3': 0.30,
            'KRAS+STAT3': 0.20,
            'EGFR+KRAS+STAT3': 0.10,
        }
        return {
            'alin_composite': composite[label],
            'escape_routes': escape_routes[label],
            'escape_route_ratio': escape_ratio[label],
        }


class _StubValidator:
    def validate_predictions(self, cancer_type, predicted_targets, cell_line_ids, n_cell_lines):
        tier = SimpleNamespace(
            tier=2,
            tier_label='PRISM pharmacologically supported',
            reasons=['2/3 targets PRISM-concordant'],
            n_concordant_targets=2,
            concordance_fraction=2 / 3,
        )
        gene_concordances = {
            'KRAS': SimpleNamespace(concordance_score=0.8),
            'EGFR': SimpleNamespace(concordance_score=0.7),
            'STAT3': SimpleNamespace(concordance_score=0.4),
        }
        return SimpleNamespace(
            evidence_tier=tier,
            gene_concordances=gene_concordances,
        )


class _StubProfile:
    def __init__(self, drug_name, values):
        self.drug_name = drug_name
        self.cell_lines = ['CL1', 'CL2', 'CL3', 'CL4', 'CL5']
        self.ic50_values = values


class _StubPrismLoader:
    def __init__(self):
        self._profiles = {
            'sotorasib': _StubProfile('sotorasib', [-1.1, -1.0, -1.05, -1.0, -1.08]),
            'erlotinib': _StubProfile('erlotinib', [-0.8, -0.82, -0.78, -0.79, -0.81]),
            'TTI-101': _StubProfile('TTI-101', [-0.6, -0.58, -0.62, -0.59, -0.61]),
        }

    def get_drug_sensitivity(self, drug_name):
        return self._profiles.get(drug_name)


def _support_row(**overrides):
    row = {
        'escape_evaluable': True,
        'delta_escape_routes_vs_best_dual': 1.0,
        'bliss_evaluable': True,
        'delta_bliss_viability_vs_best_dual': 0.1,
        'evidence_tier': 4,
        'n_prism_concordant_targets': 0,
        'pharmacological_support_fraction': 0.0,
        'mean_target_concordance_score': 0.0,
        'third_target_evaluable': False,
        'third_target_match': False,
        'third_target_extension_recovered': False,
    }
    row.update(overrides)
    row.update(classify_second_layer_support(row))
    return row


def test_load_top_ranked_predictions_prefers_three_target_prediction(tmp_path):
    triple_path = tmp_path / 'triple_combinations.csv'
    pd.DataFrame(
        [
            {
                'Cancer_Type': 'Melanoma',
                'Rank': 1,
                'Target_1': 'BRAF',
                'Target_2': 'STAT3',
                'Target_3': 'AKT1',
                'Best_Combo_1': 'BRAF',
                'Best_Combo_2': 'MAP2K1',
                'Best_Combo_3': '',
            },
        ]
    ).to_csv(triple_path, index=False)

    top_predictions, _, used_legacy_best_combo = load_top_ranked_predictions(triple_path)

    assert used_legacy_best_combo is True
    assert top_predictions['Melanoma'].targets == frozenset({'BRAF', 'STAT3', 'AKT1'})
    assert len(top_predictions['Melanoma'].targets) == 3


def test_build_second_layer_scorecard_reports_marginal_benefit(tmp_path):
    triple_path = tmp_path / 'ranked_triple_combinations.csv'
    pd.DataFrame(
        [
            {
                'Cancer_Type': 'Pancreatic Adenocarcinoma',
                'Rank': 1,
                'Target_1': 'KRAS',
                'Target_2': 'EGFR',
                'Target_3': 'STAT3',
            },
        ]
    ).to_csv(triple_path, index=False)

    rows, summary = build_second_layer_scorecard(
        triple_path,
        depmap_dir=tmp_path,
        drug_dir=tmp_path,
        scorer=_StubScorer(),
        validator=_StubValidator(),
        prism_loader=_StubPrismLoader(),
    )

    assert len(rows) == 1
    row = rows[0]
    assert row['prediction_targets'] == 'EGFR+KRAS+STAT3'
    assert row['delta_escape_routes_vs_best_dual'] == -1
    assert row['delta_alin_composite_vs_best_dual'] > 0
    assert row['bliss_evaluable'] is True
    assert row['delta_bliss_viability_vs_best_dual'] < 0
    assert row['third_target_evaluable'] is True
    assert row['third_target_match'] is True
    assert row['third_target_extension_recovered'] is True
    assert row['known_third_best_rank'] == 1
    assert row['support_label'] == 'strong'
    assert row['support_unknown_evaluable'] is False
    assert row['concordance_non_regression'] is True

    assert summary['n_cancers'] == 1
    assert summary['n_positive_escape_benefit'] == 1
    assert summary['n_positive_bliss_benefit'] == 1
    assert summary['n_third_target_extension_recovered'] == 1
    assert summary['evidence_tier_counts'] == {'2': 1}
    assert summary['support_label_counts'] == {'strong': 1, 'mixed': 0, 'weak': 0}
    assert summary['n_unknown_evaluable'] == 0
    assert summary['n_concordance_non_regression'] == 1


def test_classify_second_layer_support_marks_missing_direct_evidence_as_unknown_evaluable():
    row = _support_row(
        escape_evaluable=False,
        delta_escape_routes_vs_best_dual=None,
        bliss_evaluable=False,
        delta_bliss_viability_vs_best_dual=None,
        evidence_tier=2,
        n_prism_concordant_targets=2,
        pharmacological_support_fraction=2 / 3,
        mean_target_concordance_score=0.63,
    )

    assert row['support_label'] == 'mixed'
    assert row['support_unknown_evaluable'] is True
    assert row['concordance_non_regression'] is True
    assert 'missing direct-benefit data prevents a clean strong-versus-weak call' in row['support_reasons']


def test_classify_second_layer_support_requires_curated_match_for_strong_label():
    row = _support_row(
        escape_evaluable=True,
        delta_escape_routes_vs_best_dual=-1.0,
        bliss_evaluable=False,
        delta_bliss_viability_vs_best_dual=None,
        evidence_tier=2,
        n_prism_concordant_targets=2,
        pharmacological_support_fraction=2 / 3,
        mean_target_concordance_score=0.63,
        third_target_evaluable=True,
        third_target_match=False,
        third_target_extension_recovered=False,
    )

    assert row['support_label'] == 'mixed'
    assert row['support_unknown_evaluable'] is False
    assert 'curated third-target case does not recover the known third target' in row['support_reasons']


def test_classify_second_layer_support_marks_uncompensated_negative_evidence_as_weak():
    row = _support_row(
        escape_evaluable=True,
        delta_escape_routes_vs_best_dual=1.0,
        bliss_evaluable=True,
        delta_bliss_viability_vs_best_dual=0.2,
        evidence_tier=4,
        n_prism_concordant_targets=0,
        pharmacological_support_fraction=0.0,
        mean_target_concordance_score=0.0,
    )

    assert row['support_label'] == 'weak'
    assert row['support_unknown_evaluable'] is False
    assert row['concordance_non_regression'] is False
    assert 'no compensating pharmacology or curated-support signal offsets the negative direct evidence' in row['support_reasons']


def test_summarize_second_layer_scorecard_reports_support_distribution():
    rows = [
        _support_row(
            escape_evaluable=True,
            delta_escape_routes_vs_best_dual=-1.0,
            bliss_evaluable=False,
            delta_bliss_viability_vs_best_dual=None,
            evidence_tier=2,
            n_prism_concordant_targets=2,
            pharmacological_support_fraction=2 / 3,
            mean_target_concordance_score=0.63,
        ),
        _support_row(
            escape_evaluable=False,
            delta_escape_routes_vs_best_dual=None,
            bliss_evaluable=False,
            delta_bliss_viability_vs_best_dual=None,
            evidence_tier=2,
            n_prism_concordant_targets=1,
            pharmacological_support_fraction=1 / 3,
            mean_target_concordance_score=0.25,
        ),
        _support_row(
            escape_evaluable=True,
            delta_escape_routes_vs_best_dual=1.0,
            bliss_evaluable=True,
            delta_bliss_viability_vs_best_dual=0.2,
            evidence_tier=4,
            n_prism_concordant_targets=0,
            pharmacological_support_fraction=0.0,
            mean_target_concordance_score=0.0,
        ),
    ]

    summary = summarize_second_layer_scorecard(rows)

    assert summary['support_label_counts'] == {'strong': 1, 'mixed': 1, 'weak': 1}
    assert summary['n_support_strong'] == 1
    assert summary['n_support_mixed'] == 1
    assert summary['n_support_weak'] == 1
    assert summary['n_unknown_evaluable'] == 1
    assert summary['n_concordance_non_regression'] == 2
    assert summary['support_label_rates']['strong'] == pytest.approx(1 / 3)
    assert summary['support_label_rates']['mixed'] == pytest.approx(1 / 3)
    assert summary['support_label_rates']['weak'] == pytest.approx(1 / 3)
    assert summary['unknown_evaluable_rate'] == pytest.approx(1 / 3)
    assert summary['concordance_non_regression_rate'] == pytest.approx(2 / 3)