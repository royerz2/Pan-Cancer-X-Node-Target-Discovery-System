"""Tests for the dual-mode configuration helpers."""

import pytest

from alin.run_modes import ModeConfig, RunMode, actionable_config, exploratory_config, get_config


def test_mode_config_defaults_to_actionable():
    cfg = ModeConfig()
    assert cfg.mode == RunMode.ACTIONABLE


def test_actionable_config_matches_public_defaults():
    cfg = actionable_config()
    assert cfg.mode == RunMode.ACTIONABLE
    assert cfg.W_DRUGGABLE == 0.09
    assert cfg.prefer_druggable is True
    assert cfg.cost_gamma == 0.3
    assert cfg.W_ESSENTIALITY == 0.0
    assert cfg.W_MUTATION == 0.0
    assert cfg.W_CENTRALITY == 0.0
    assert cfg.output_suffix == '_actionable'


def test_exploratory_config_matches_public_defaults():
    cfg = exploratory_config()
    assert cfg.mode == RunMode.EXPLORATORY
    assert cfg.W_DRUGGABLE == 0.0
    assert cfg.prefer_druggable is False
    assert cfg.cost_gamma == 0.0
    assert cfg.W_ESSENTIALITY > 0
    assert cfg.W_MUTATION > 0
    assert cfg.W_CENTRALITY > 0
    assert cfg.DRUGGABILITY_THRESHOLD == 0.0
    assert cfg.druggable_pool_threshold == 0.0
    assert cfg.inject_druggable_path_genes is False
    assert cfg.drug_overrides_genomic_filter is False
    assert cfg.flat_protein_druggability is True
    assert cfg.output_suffix == '_exploratory'


def test_get_config_dispatches():
    assert get_config('actionable').mode == RunMode.ACTIONABLE
    assert get_config('exploratory').mode == RunMode.EXPLORATORY


def test_get_config_rejects_unknown_mode():
    with pytest.raises(ValueError, match='Unknown mode'):
        get_config('magic')


def test_to_dict_serializes_mode_and_excluded_genes():
    cfg = exploratory_config()
    data = cfg.to_dict()
    assert data['mode'] == 'exploratory'
    assert isinstance(data['excluded_genes'], list)


def test_actionable_and_exploratory_gates_diverge_as_expected():
    act = actionable_config()
    exp = exploratory_config()

    assert act.W_DRUGGABLE > 0
    assert exp.W_DRUGGABLE == 0.0
    assert act.cost_gamma > 0
    assert exp.cost_gamma == 0.0
    assert act.prefer_druggable is True
    assert exp.prefer_druggable is False
    assert act.flat_protein_druggability is False
    assert exp.flat_protein_druggability is True


def test_exploratory_config_can_be_customized_independently():
    first = exploratory_config()
    second = exploratory_config()

    first.W_ESSENTIALITY = 0.24

    assert first.W_ESSENTIALITY == 0.24
    assert second.W_ESSENTIALITY != 0.24