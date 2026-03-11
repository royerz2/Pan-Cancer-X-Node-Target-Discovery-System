"""
Run-mode configuration for the ALIN pipeline.

Two fully divergent pipelines:
  - **actionable**: Translational / clinical-trial-ready.  Uses *multiplicative*
    scoring so a zero-druggability triple scores zero.  Hard-excludes undruggable
    pan-essential genes (TP53).  Adds a clinical-readiness multiplier based on
    drug approval stage.  Strict 50% coverage.
  - **exploratory**: Discovery / biology-first.  Keeps additive scoring with
    biology rewards.  Adds a novelty bonus for triples not in the gold standard.
    Network topology candidate selection: betweenness centrality instead of
    path-frequency.  Relaxed 30% default coverage.

Dominance is broken structurally via per-cancer independent ranking rather
than cross-cancer penalties (V8: removed pan-essential dampener and V6.2
essentiality factor that caused whack-a-mole gene dominance).

Both modes apply a **resistance hard-gate**: triples where
resistance >= synergy are discarded (net-negative therapeutic window).

The weights for each mode can be overridden individually, and a sensitivity-
analysis sweep can be run to find optimal exploratory settings.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, Dict, FrozenSet, Set


class RunMode(str, Enum):
    ACTIONABLE = "actionable"
    EXPLORATORY = "exploratory"


@dataclass
class ModeConfig:
    """All tuneable parameters that differ between Actionable and Exploratory."""

    mode: RunMode = RunMode.ACTIONABLE

    # ── Combo-level scoring weights ────────────────────────────────
    W_COST: float = 0.20
    W_SYNERGY: float = 0.16
    W_RESISTANCE: float = 0.16
    W_COVERAGE: float = 0.13
    W_COMBO_TOX: float = 0.16
    W_DRUGGABLE: float = 0.09          # bonus per druggable target
    W_SELECTIVITY: float = 0.18        # cancer-selectivity bonus
    W_GENOMIC: float = 0.50            # TCGA mutation-relevance weight
    PERTURBATION_SCALING: float = 0.12
    HUB_PENALTY_MULTIPLIER: float = 1.5

    # ── Biology reward (Exploratory only) ──────────────────────────
    #: Weight for an *essentiality bonus* — rewards targets that are
    #: strongly essential in this cancer type's cell lines (DepMap).
    W_ESSENTIALITY: float = 0.0
    #: Weight for a *mutation-frequency bonus* — rewards targets that
    #: are commonly mutated in TCGA patients of this cancer type.
    W_MUTATION: float = 0.0
    #: Weight for a *pathway-centrality bonus* — rewards targets that
    #: appear in many viability paths (network centrality).
    W_CENTRALITY: float = 0.0

    # ── V7: New data-integration weights ───────────────────────────
    #: Weight for *copy-number amplification bonus* — rewards targets
    #: recurrently amplified in this cancer type's cell lines.
    W_COPY_NUMBER: float = 0.0
    #: Weight for *drug-sensitivity bonus* — rewards targets whose
    #: drugs show actual cell-killing in GDSC2/PRISM screens.
    W_DRUG_SENSITIVITY: float = 0.0
    #: Weight for *RNAi concordance bonus* — rewards targets whose
    #: essentiality is confirmed by an orthogonal RNAi screen.
    W_RNAI_CONCORDANCE: float = 0.0

    # ── Per-gene cost parameters ───────────────────────────────────
    #: gamma term in NodeCost.total_cost — reduction for druggable genes
    cost_gamma: float = 0.3

    # ── Candidate pool behaviour ──────────────────────────────────
    #: Whether to hard-filter candidates to druggable-only
    prefer_druggable: bool = True
    #: Minimum druggability score to enter the candidate pool
    druggable_pool_threshold: float = 0.4
    #: Minimum druggability score to count a target as "druggable" in
    #: the combo-level scoring
    DRUGGABILITY_THRESHOLD: float = 0.6
    #: Whether to inject druggable-only path genes into candidates
    inject_druggable_path_genes: bool = True
    #: Whether approved-drug status overrides the genomic-filter
    drug_overrides_genomic_filter: bool = True

    # ── Protein-scoring blending ──────────────────────────────────
    #: If True, pass a flat druggability function (lambda g: 0.5) to
    #: ProteinDruggabilityScorer so protein-structural properties
    #: dominate rather than clinical-stage information.
    flat_protein_druggability: bool = False

    # ── V6: Mode-divergent parameters ─────────────────────────────

    # Resistance hard-gate (mode-agnostic): discard triples where
    # resistance >= synergy (net-negative therapeutic window).
    resistance_hard_gate: bool = True

    # Scoring mode: 'additive' (original) or 'multiplicative'
    # (translational — zero druggability → zero score).
    scoring_mode: str = 'additive'

    # Clinical readiness multiplier (translational only):
    # 3/3 approved → 1.0, 2/3 → 0.7, 1/3 → 0.4, 0/3 → 0.1
    use_clinical_readiness: bool = False

    # Minimum drug stage for a candidate to be primary
    # (translational: 'phase2'; exploratory: None = no filter)
    min_drug_stage: Optional[str] = None

    # Pan-essential dampener: REMOVED in V8.  Per-cancer ranking breaks
    # dominance structurally without cross-cancer penalties.
    # Kept at 0.0 for backward compatibility with serialized configs.
    pan_essential_dampener: float = 0.0   # 0 = off (V8: always off)
    # Set of genes to hard-exclude from candidate pool.
    # Translational: undruggable pan-essentials like TP53.
    excluded_genes: FrozenSet[str] = field(default_factory=frozenset)

    # Novelty bonus (exploratory): reward triples not in gold standard.
    use_novelty_bonus: bool = False
    novelty_bonus_weight: float = 0.0

    # Network topology candidate selection (exploratory): use
    # betweenness centrality ranking instead of path-frequency.
    use_network_topology_candidates: bool = False

    # Default coverage threshold for triple scoring.
    default_min_coverage: float = 0.50

    # ── Output ────────────────────────────────────────────────────
    #: Suffix appended to output directory (e.g. results_exploratory/)
    output_suffix: str = ""

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["mode"] = self.mode.value
        # frozenset is not JSON serializable
        d["excluded_genes"] = sorted(self.excluded_genes)
        return d


# ── Pre-built configurations ──────────────────────────────────────────


def actionable_config() -> ModeConfig:
    """
    Translational pipeline — drug-focused, clinically actionable.

    Key properties:
    - *Multiplicative* scoring: druggability_fraction * synergy * (1-resistance)
      so zero-druggability triples score zero.
    - Hard-exclude undruggable pan-essentials (TP53, CDKN2A): forces the
      algorithm to find actionable alternatives.
    - Clinical readiness multiplier: 3 approved drugs → 1.0x, 2 → 0.7x.
    - Strict 50% coverage threshold.
    - Resistance hard-gate with synergy net-positive requirement.
    - No cross-cancer dominance penalties (V8: per-cancer ranking approach).
    """
    return ModeConfig(
        mode=RunMode.ACTIONABLE,
        # Multiplicative scoring for translational
        scoring_mode='multiplicative',
        use_clinical_readiness=True,
        # Hard-exclude undruggable pan-essentials
        excluded_genes=frozenset({'TP53', 'CDKN2A', 'ARID1A', 'RB1'}),
        # Minimum drug stage for primary candidates
        min_drug_stage='phase2',
        # V7.2: Restored pre-amplifier base weights (V7.1 over-corrected by
        # lowering weights AND removing *5 amplifiers; the amplifiers were the
        # sole problem).  These are the original V7 base weights without amplifiers.
        W_COPY_NUMBER=0.10,
        W_DRUG_SENSITIVITY=0.15,
        W_RNAI_CONCORDANCE=0.08,
        # Strict coverage
        default_min_coverage=0.50,
        # Resistance hard-gate (both modes)
        resistance_hard_gate=True,
        output_suffix="_actionable",
    )


def exploratory_config() -> ModeConfig:
    """
    Biology-first discovery pipeline.

    Key differences from actionable:
    - *Additive* scoring with biology reward signals.
    - Novelty bonus: triples absent from the gold standard are rewarded.
    - Network topology candidate selection: betweenness centrality
      instead of path-frequency, finding vulnerabilities the solver misses.
    - Relaxed 30% coverage for cancer types with fragmented paths.
    - No cross-cancer dominance penalties (V8: per-cancer ranking
      approach replaces the old pan-essential dampener).
    """
    return ModeConfig(
        mode=RunMode.EXPLORATORY,
        # Additive scoring with biology rewards
        scoring_mode='additive',
        W_DRUGGABLE=0.0,
        DRUGGABILITY_THRESHOLD=0.0,
        prefer_druggable=False,
        druggable_pool_threshold=0.0,
        inject_druggable_path_genes=False,
        drug_overrides_genomic_filter=False,
        cost_gamma=0.0,
        flat_protein_druggability=True,
        # Biology reward signals
        W_ESSENTIALITY=0.12,
        W_MUTATION=0.10,
        W_CENTRALITY=0.08,
        # V7.2: Restored pre-amplifier base weights (proportionally lower
        # than actionable — discovery should not be dominated by drug evidence).
        W_COPY_NUMBER=0.08,
        W_DRUG_SENSITIVITY=0.10,
        W_RNAI_CONCORDANCE=0.08,
        # Novelty bonus for non-gold-standard triples
        use_novelty_bonus=True,
        novelty_bonus_weight=0.15,
        # Network topology candidate selection
        use_network_topology_candidates=True,
        # Relaxed coverage for discovery
        default_min_coverage=0.30,
        # Resistance hard-gate (both modes)
        resistance_hard_gate=True,
        # Increase path-coverage weight (biology cares about breadth)
        W_COVERAGE=0.18,
        W_COST=0.20,
        W_SYNERGY=0.16,
        W_RESISTANCE=0.16,
        W_COMBO_TOX=0.16,
        W_SELECTIVITY=0.20,
        W_GENOMIC=0.55,
        output_suffix="_exploratory",
    )


def get_config(mode: str) -> ModeConfig:
    """Get config by mode name string."""
    if mode == "actionable":
        return actionable_config()
    elif mode == "exploratory":
        return exploratory_config()
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Use 'actionable' or 'exploratory'.")
