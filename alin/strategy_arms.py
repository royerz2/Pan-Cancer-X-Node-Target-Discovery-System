"""Strategy-arm helpers for ALIN comparison workflows."""

from __future__ import annotations

from typing import Optional

DEFAULT_STRATEGY_ARM = 'default'
LIAKI_ROLE_ARM = 'liaki_role'
LIAKI_PDAC_TEMPLATE_ARM = 'liaki_pdac_template'
DEFAULT_STRATEGY_ARM_BY_MODE = {
    'actionable': LIAKI_ROLE_ARM,
    'exploratory': DEFAULT_STRATEGY_ARM,
}

SUPPORTED_STRATEGY_ARMS = (
    DEFAULT_STRATEGY_ARM,
    LIAKI_ROLE_ARM,
    LIAKI_PDAC_TEMPLATE_ARM,
)

STRUCTURAL_STRATEGY_ARMS = frozenset({
    LIAKI_ROLE_ARM,
    LIAKI_PDAC_TEMPLATE_ARM,
})


def default_strategy_arm_for_mode(run_mode: Optional[str]) -> str:
    """Return the implicit strategy arm for a run mode."""
    normalized_mode = str(run_mode or '').strip().lower()
    return DEFAULT_STRATEGY_ARM_BY_MODE.get(normalized_mode, DEFAULT_STRATEGY_ARM)


def normalize_strategy_arm(
    strategy_arm: Optional[str],
    structural_mode: bool = False,
    run_mode: Optional[str] = None,
) -> str:
    """Resolve CLI and legacy boolean selection into one explicit arm label."""
    normalized = str(strategy_arm or '').strip().lower()
    if not normalized:
        normalized = default_strategy_arm_for_mode(run_mode)
    if structural_mode and normalized == DEFAULT_STRATEGY_ARM:
        normalized = LIAKI_ROLE_ARM
    if normalized not in SUPPORTED_STRATEGY_ARMS:
        supported = ', '.join(SUPPORTED_STRATEGY_ARMS)
        raise ValueError(f"Unknown strategy arm '{strategy_arm}'. Expected one of: {supported}")
    return normalized


def is_structural_strategy_arm(strategy_arm: str) -> bool:
    """Return True when an arm uses structural upstream/driver/escape logic."""
    return strategy_arm in STRUCTURAL_STRATEGY_ARMS


def infer_strategy_arm_from_scoring_mode(scoring_mode: Optional[str]) -> str:
    """Best-effort fallback for legacy results that only encoded the scoring mode."""
    normalized = str(scoring_mode or '').strip().lower()
    if normalized.startswith('structural:liaki_pdac_template'):
        return LIAKI_PDAC_TEMPLATE_ARM
    if normalized.startswith('structural'):
        return LIAKI_ROLE_ARM
    return DEFAULT_STRATEGY_ARM