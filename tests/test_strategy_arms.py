import pandas as pd

import pan_cancer_xnode
from alin.strategy_arms import (
    DEFAULT_STRATEGY_ARM,
    LIAKI_PDAC_TEMPLATE_ARM,
    LIAKI_ROLE_ARM,
    default_strategy_arm_for_mode,
    infer_strategy_arm_from_scoring_mode,
    is_structural_strategy_arm,
    normalize_strategy_arm,
)
from alin.structural_triples import get_structural_arm_config
from core.data_structures import CancerTypeAnalysis, TripleCombination


class DummyDrugDB:
    def get_drug_info(self, _gene):
        return None


def _make_combo(
    strategy_arm=LIAKI_ROLE_ARM,
    scoring_mode='structural:liaki_role',
    role_assignments=None,
):
    return TripleCombination(
        targets=('EGFR', 'KRAS', 'STAT3'),
        total_cost=1.0,
        synergy_score=0.8,
        resistance_score=0.2,
        pathway_coverage={'MAPK': 0.5},
        coverage=0.7,
        druggable_count=3,
        combined_score=0.42,
        strategy_arm=strategy_arm,
        scoring_mode=scoring_mode,
        role_assignments=role_assignments or {
            'feeder': 'EGFR',
            'driver': 'KRAS',
            'escape': 'STAT3',
        },
    )


def _make_analysis(combo):
    return CancerTypeAnalysis(
        cancer_type='Pancreatic Adenocarcinoma',
        lineage='Pancreas',
        n_cell_lines=20,
        cell_line_ids=['CL1', 'CL2'],
        driver_mutations={'KRAS': 0.9},
        essential_genes={'KRAS': -1.2},
        viability_paths=[],
        minimal_hitting_sets=[],
        top_x_node_sets=[],
        recommended_combination=list(combo.targets),
        triple_combinations=[combo],
        best_triple=combo,
        best_combination=combo,
    )


def test_structural_alias_normalizes_to_liaki_role():
    assert normalize_strategy_arm(DEFAULT_STRATEGY_ARM, structural_mode=True) == LIAKI_ROLE_ARM


def test_default_strategy_arm_for_mode_is_mode_aware():
    assert default_strategy_arm_for_mode('actionable') == LIAKI_ROLE_ARM
    assert default_strategy_arm_for_mode('exploratory') == DEFAULT_STRATEGY_ARM
    assert default_strategy_arm_for_mode(None) == DEFAULT_STRATEGY_ARM


def test_normalize_strategy_arm_uses_mode_default_when_omitted():
    assert normalize_strategy_arm(None, run_mode='actionable') == LIAKI_ROLE_ARM
    assert normalize_strategy_arm(None, run_mode='exploratory') == DEFAULT_STRATEGY_ARM
    assert normalize_strategy_arm(DEFAULT_STRATEGY_ARM, run_mode='actionable') == DEFAULT_STRATEGY_ARM


def test_pdac_template_is_structural():
    assert is_structural_strategy_arm(LIAKI_PDAC_TEMPLATE_ARM) is True
    assert is_structural_strategy_arm(DEFAULT_STRATEGY_ARM) is False


def test_infer_strategy_arm_from_legacy_scoring_mode():
    assert infer_strategy_arm_from_scoring_mode('structural:liaki_pdac_template+synergy') == LIAKI_PDAC_TEMPLATE_ARM
    assert infer_strategy_arm_from_scoring_mode('structural+synergy') == LIAKI_ROLE_ARM
    assert infer_strategy_arm_from_scoring_mode('multiplicative') == DEFAULT_STRATEGY_ARM


def test_get_structural_arm_config_for_pdac_template():
    config = get_structural_arm_config(LIAKI_PDAC_TEMPLATE_ARM)

    assert config.arm_name == LIAKI_PDAC_TEMPLATE_ARM
    assert config.prioritize_role_templates is True
    assert config.scoring_label == 'structural:liaki_pdac_template'
    assert config.driver_priors['KRAS'] == 1.0
    assert config.escape_priors['STAT3'] == 1.0


def test_resolve_strategy_arm_prefers_explicit_metadata():
    combo = _make_combo(
        strategy_arm=LIAKI_PDAC_TEMPLATE_ARM,
        scoring_mode='multiplicative',
    )

    assert pan_cancer_xnode.resolve_strategy_arm(combo) == LIAKI_PDAC_TEMPLATE_ARM


def test_generate_triple_summary_table_exports_explicit_arm_and_roles(monkeypatch):
    monkeypatch.setattr(pan_cancer_xnode, 'DrugTargetDB', DummyDrugDB)
    combo = _make_combo(
        strategy_arm=LIAKI_PDAC_TEMPLATE_ARM,
        scoring_mode='multiplicative',
    )
    analysis = _make_analysis(combo)

    df = pan_cancer_xnode.generate_triple_summary_table({'PAAD': analysis})

    assert isinstance(df, pd.DataFrame)
    assert df.iloc[0]['Strategy_Arm'] == LIAKI_PDAC_TEMPLATE_ARM
    assert df.iloc[0]['Role Feeder'] == 'EGFR'
    assert df.iloc[0]['Role Driver'] == 'KRAS'
    assert df.iloc[0]['Role Escape'] == 'STAT3'