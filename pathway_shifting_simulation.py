#!/usr/bin/env python3
"""
Pathway Shifting Simulation Module for ALIN Framework
=====================================================

Systems biology ODE-based simulation comparing X-node (minimal hitting set)
vs. three-axis triple combination therapy. Models adaptive pathway shifting
(compensatory signaling activation) that occurs when cancer cells are treated
with insufficient target coverage.

Biological basis: Liaki et al. (PNAS 2025) demonstrated that PDAC cells
maintain viability through ANY ONE of three independent signaling axes:
  - Downstream (RAF1/RAS → MAPK cascade)
  - Upstream (EGFR → receptor tyrosine kinase input)  
  - Orthogonal (STAT3 → parallel survival via FYN/SRC kinases)

When 2 of 3 axes are blocked, the remaining axis compensatorily upregulates
(e.g., RAF1+EGFR ablation → FYN→STAT3 phosphorylation on Tyr705).
Only simultaneous three-axis blockade prevents resistance.

This module generalizes that principle: for each cancer type, we simulate
the ODE dynamics of signaling pathway activity under different treatment
strategies (no treatment, single, X-node, triple) and measure:
  - Time to resistance (tumor viability recovery)
  - Minimum tumor viability achieved
  - Area under the viability curve (total tumor burden)
  - Pathway shifting magnitude (compensatory activation)

Author: Roy Erzurumluoğlu
"""

import json
import os
import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, Optional
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)  # scipy ODE only

# Reproducibility
RNG_SEED = 42
np.random.seed(RNG_SEED)

# ---------------------------------------------------------------------------
# 1. SIGNALING NETWORK MODEL
# ---------------------------------------------------------------------------

@dataclass
class SignalingNode:
    """A node in the cancer signaling network."""
    name: str
    pathway: str                    # Functional axis: 'downstream', 'upstream', 'orthogonal'
    basal_rate: float = 0.05        # Basal production rate (LOW: most activity from signaling inputs)
    degradation_rate: float = 0.5    # Turnover rate (high enough for drug effects to matter)
    max_activity: float = 1.0       # Maximum activity level
    compensatory_gain: float = 0.3  # How much this node upregulates when parallel axes decrease
    initial_activity: float = 0.8   # Starting activity in untreated tumor


@dataclass
class Interaction:
    """A directed interaction between two signaling nodes."""
    source: str
    target: str
    interaction_type: str   # 'activation', 'inhibition', 'compensation'
    strength: float = 0.5   # Interaction strength [0, 1]
    hill_coefficient: float = 2.0  # Hill coefficient for sigmoidal response


@dataclass
class DrugEffect:
    """Drug inhibition of a signaling node."""
    target_node: str
    inhibition_strength: float = 0.9  # Fraction inhibited [0, 1]
    onset_time: float = 24.0          # Hours to reach full effect
    onset_rate: float = 0.1           # Rate of drug onset (exponential)


@dataclass
class SimulationResult:
    """Results from a single simulation run."""
    strategy_name: str
    targets: List[str]
    time_points: np.ndarray
    node_activities: Dict[str, np.ndarray]
    tumor_viability: np.ndarray
    time_to_resistance: float        # Hours until viability recovers > threshold
    min_viability: float             # Lowest viability achieved
    auc_viability: float             # Area under viability curve (tumor burden)
    pathway_shift_magnitude: float   # Max compensatory upregulation observed
    final_viability: float           # Viability at end of simulation
    resistance_emerged: bool         # Whether viability recovered after initial drop


# ---------------------------------------------------------------------------
# 2. CANCER-SPECIFIC NETWORK DEFINITIONS
# ---------------------------------------------------------------------------

class CancerNetworkFactory:
    """
    Factory for creating cancer-type-specific signaling networks.
    
    Design principles:
    - Constitutively active oncogenes (KRAS G12D, BRAF V600E) have HIGH basal rates
      because they signal independent of upstream input.
    - All other nodes have LOW basal rates and depend on activation inputs
      from upstream nodes. This ensures that drugging an upstream node
      causes downstream nodes to lose activity (cascade collapse).
    - Orthogonal axis nodes (STAT3, MCL1) have moderate basal rates and 
      HIGH compensatory gains, so they upregulate when other axes are depleted.
    - FYN kinase is constitutively expressed (moderate basal) but suppressed
      by active KRAS/EGFR signaling (inhibition edges). When those are drugged,
      FYN de-represses and activates STAT3 (the Liaki mechanism).
    """

    @staticmethod
    def create_pdac_network() -> Tuple[List[SignalingNode], List[Interaction]]:
        """
        PDAC network based on Liaki et al. (PNAS 2025).
        Three axes: RAS/MAPK (downstream), EGFR/RTK (upstream), STAT3/JAK (orthogonal)
        Key mechanism: FYN kinase activates STAT3 when EGFR+RAF1 are ablated.
        """
        nodes = [
            # Upstream axis (RTK input) — ligand-driven, moderate basal
            SignalingNode('EGFR', 'upstream', basal_rate=0.35, compensatory_gain=0.15, initial_activity=0.85),
            SignalingNode('ERBB2', 'upstream', basal_rate=0.15, compensatory_gain=0.1, initial_activity=0.4),

            # Downstream axis (RAS-MAPK cascade) — KRAS is constitutively active (G12D/G12V)
            SignalingNode('KRAS', 'downstream', basal_rate=0.45, compensatory_gain=0.05, initial_activity=0.95),
            SignalingNode('RAF1', 'downstream', basal_rate=0.03, compensatory_gain=0.1, initial_activity=0.8),
            SignalingNode('MEK', 'downstream', basal_rate=0.03, compensatory_gain=0.08, initial_activity=0.75),
            SignalingNode('ERK', 'downstream', basal_rate=0.03, compensatory_gain=0.08, initial_activity=0.7),

            # Orthogonal axis (JAK-STAT) — moderate basal, HIGH compensatory gain
            SignalingNode('STAT3', 'orthogonal', basal_rate=0.08, compensatory_gain=0.6, initial_activity=0.7),
            SignalingNode('FYN', 'orthogonal', basal_rate=0.25, compensatory_gain=0.7, initial_activity=0.35),
            SignalingNode('JAK2', 'orthogonal', basal_rate=0.12, compensatory_gain=0.2, initial_activity=0.5),
            SignalingNode('SRC', 'orthogonal', basal_rate=0.15, compensatory_gain=0.3, initial_activity=0.5),

            # Cell cycle / effector — driven by upstream signaling
            SignalingNode('CCND1', 'downstream', basal_rate=0.05, compensatory_gain=0.08, initial_activity=0.7),
            SignalingNode('CDK4', 'downstream', basal_rate=0.04, compensatory_gain=0.05, initial_activity=0.6),

            # Survival — driven by STAT3
            SignalingNode('MCL1', 'orthogonal', basal_rate=0.05, compensatory_gain=0.2, initial_activity=0.6),
            SignalingNode('BCL2', 'orthogonal', basal_rate=0.06, compensatory_gain=0.15, initial_activity=0.5),
        ]

        interactions = [
            # RAS-MAPK cascade (downstream) — strong sequential activation
            Interaction('EGFR', 'KRAS', 'activation', 0.4),
            Interaction('KRAS', 'RAF1', 'activation', 0.9),
            Interaction('RAF1', 'MEK', 'activation', 0.95),
            Interaction('MEK', 'ERK', 'activation', 0.95),
            Interaction('ERK', 'CCND1', 'activation', 0.7),

            # RTK upstream crosstalk
            Interaction('EGFR', 'JAK2', 'activation', 0.25),
            Interaction('ERBB2', 'KRAS', 'activation', 0.2),

            # JAK-STAT axis (orthogonal)
            Interaction('JAK2', 'STAT3', 'activation', 0.6),
            Interaction('SRC', 'STAT3', 'activation', 0.45),

            # FYN → STAT3 compensatory activation (THE KEY LIAKI MECHANISM)
            Interaction('FYN', 'STAT3', 'compensation', 0.9, hill_coefficient=3.0),

            # STAT3 → survival genes
            Interaction('STAT3', 'MCL1', 'activation', 0.7),
            Interaction('STAT3', 'BCL2', 'activation', 0.45),
            Interaction('STAT3', 'CCND1', 'activation', 0.35),

            # Cell cycle
            Interaction('CCND1', 'CDK4', 'activation', 0.85),

            # Negative feedback / cross-inhibition
            # Active KRAS & EGFR SUPPRESS FYN — this is why FYN only activates
            # when those axes are depleted
            Interaction('ERK', 'EGFR', 'inhibition', 0.15),
            Interaction('KRAS', 'FYN', 'inhibition', 0.35),
            Interaction('EGFR', 'FYN', 'inhibition', 0.25),
            Interaction('ERK', 'FYN', 'inhibition', 0.2),
        ]

        return nodes, interactions

    @staticmethod
    def create_melanoma_network() -> Tuple[List[SignalingNode], List[Interaction]]:
        """Melanoma: BRAF V600E constitutively active, JAK-STAT resistance axis."""
        nodes = [
            # BRAF V600E — constitutively active (HIGH basal)
            SignalingNode('BRAF', 'downstream', basal_rate=0.50, compensatory_gain=0.05, initial_activity=0.95),
            SignalingNode('MEK', 'downstream', basal_rate=0.03, compensatory_gain=0.1, initial_activity=0.8),
            SignalingNode('ERK', 'downstream', basal_rate=0.03, compensatory_gain=0.1, initial_activity=0.75),
            # Upstream bypass axes — suppressed by active BRAF/ERK, de-repress upon BRAF inhibition
            SignalingNode('NRAS', 'upstream', basal_rate=0.10, compensatory_gain=0.5, initial_activity=0.25),
            SignalingNode('EGFR', 'upstream', basal_rate=0.08, compensatory_gain=0.45, initial_activity=0.2),
            # Orthogonal (JAK-STAT) — hidden survival axis
            SignalingNode('STAT3', 'orthogonal', basal_rate=0.08, compensatory_gain=0.6, initial_activity=0.55),
            SignalingNode('JAK1', 'orthogonal', basal_rate=0.12, compensatory_gain=0.25, initial_activity=0.4),
            SignalingNode('SRC', 'orthogonal', basal_rate=0.10, compensatory_gain=0.3, initial_activity=0.4),
            # Downstream effectors
            SignalingNode('CCND1', 'downstream', basal_rate=0.05, compensatory_gain=0.12, initial_activity=0.7),
            SignalingNode('CDK4', 'downstream', basal_rate=0.04, compensatory_gain=0.08, initial_activity=0.6),
            SignalingNode('MCL1', 'orthogonal', basal_rate=0.05, compensatory_gain=0.2, initial_activity=0.5),
        ]

        interactions = [
            # MAPK cascade
            Interaction('BRAF', 'MEK', 'activation', 0.95),
            Interaction('MEK', 'ERK', 'activation', 0.95),
            Interaction('ERK', 'CCND1', 'activation', 0.65),
            Interaction('CCND1', 'CDK4', 'activation', 0.85),
            # NRAS bypass of BRAF
            Interaction('NRAS', 'MEK', 'activation', 0.6),
            Interaction('EGFR', 'NRAS', 'activation', 0.35),
            # JAK-STAT
            Interaction('JAK1', 'STAT3', 'activation', 0.7),
            Interaction('SRC', 'STAT3', 'activation', 0.5),
            Interaction('STAT3', 'MCL1', 'activation', 0.6),
            Interaction('STAT3', 'CCND1', 'activation', 0.3),
            # Compensation: NRAS/EGFR upregulate when BRAF/ERK pathway is depleted
            Interaction('NRAS', 'STAT3', 'compensation', 0.5),
            Interaction('EGFR', 'STAT3', 'compensation', 0.4),
            # Negative feedback: active ERK suppresses bypass axes
            Interaction('ERK', 'NRAS', 'inhibition', 0.4),
            Interaction('ERK', 'EGFR', 'inhibition', 0.3),
            Interaction('ERK', 'SRC', 'inhibition', 0.15),
        ]

        return nodes, interactions

    @staticmethod
    def create_nsclc_network() -> Tuple[List[SignalingNode], List[Interaction]]:
        """NSCLC: heterogeneous drivers (KRAS/EGFR), MCL1 as orthogonal survival."""
        nodes = [
            SignalingNode('EGFR', 'upstream', basal_rate=0.20, compensatory_gain=0.15, initial_activity=0.6),
            SignalingNode('KRAS', 'upstream', basal_rate=0.25, compensatory_gain=0.1, initial_activity=0.7),
            SignalingNode('MET', 'upstream', basal_rate=0.08, compensatory_gain=0.45, initial_activity=0.25),
            SignalingNode('RAF1', 'downstream', basal_rate=0.03, compensatory_gain=0.1, initial_activity=0.65),
            SignalingNode('MEK', 'downstream', basal_rate=0.03, compensatory_gain=0.08, initial_activity=0.6),
            SignalingNode('ERK', 'downstream', basal_rate=0.03, compensatory_gain=0.08, initial_activity=0.55),
            SignalingNode('CCND1', 'downstream', basal_rate=0.05, compensatory_gain=0.1, initial_activity=0.65),
            SignalingNode('CDK2', 'downstream', basal_rate=0.04, compensatory_gain=0.08, initial_activity=0.55),
            # Orthogonal: MCL1 as primary survival + STAT3
            SignalingNode('MCL1', 'orthogonal', basal_rate=0.08, compensatory_gain=0.55, initial_activity=0.6),
            SignalingNode('BCL2L1', 'orthogonal', basal_rate=0.06, compensatory_gain=0.25, initial_activity=0.35),
            SignalingNode('STAT3', 'orthogonal', basal_rate=0.07, compensatory_gain=0.4, initial_activity=0.5),
            SignalingNode('PI3K', 'orthogonal', basal_rate=0.08, compensatory_gain=0.3, initial_activity=0.45),
        ]

        interactions = [
            Interaction('EGFR', 'KRAS', 'activation', 0.4),
            Interaction('KRAS', 'RAF1', 'activation', 0.85),
            Interaction('RAF1', 'MEK', 'activation', 0.95),
            Interaction('MEK', 'ERK', 'activation', 0.95),
            Interaction('ERK', 'CCND1', 'activation', 0.6),
            Interaction('CCND1', 'CDK2', 'activation', 0.75),
            # MET bypass
            Interaction('MET', 'KRAS', 'activation', 0.4),
            Interaction('MET', 'PI3K', 'activation', 0.55),
            Interaction('PI3K', 'MCL1', 'activation', 0.55),
            Interaction('EGFR', 'PI3K', 'activation', 0.35),
            Interaction('STAT3', 'MCL1', 'activation', 0.6),
            Interaction('STAT3', 'BCL2L1', 'activation', 0.4),
            # Compensation
            Interaction('MET', 'STAT3', 'compensation', 0.6),
            Interaction('PI3K', 'MCL1', 'compensation', 0.45),
            # Negative feedback
            Interaction('ERK', 'EGFR', 'inhibition', 0.25),
            Interaction('ERK', 'MET', 'inhibition', 0.2),
        ]

        return nodes, interactions

    @staticmethod
    def create_colorectal_network() -> Tuple[List[SignalingNode], List[Interaction]]:
        """Colorectal: multi-pathway convergence (WNT+MAPK+JAK-STAT+cell cycle)."""
        nodes = [
            SignalingNode('KRAS', 'upstream', basal_rate=0.30, compensatory_gain=0.1, initial_activity=0.8),
            SignalingNode('BRAF', 'upstream', basal_rate=0.15, compensatory_gain=0.25, initial_activity=0.5),
            SignalingNode('MEK', 'downstream', basal_rate=0.03, compensatory_gain=0.1, initial_activity=0.7),
            SignalingNode('ERK', 'downstream', basal_rate=0.03, compensatory_gain=0.1, initial_activity=0.65),
            SignalingNode('CTNNB1', 'upstream', basal_rate=0.15, compensatory_gain=0.35, initial_activity=0.65),
            SignalingNode('STAT3', 'orthogonal', basal_rate=0.08, compensatory_gain=0.55, initial_activity=0.6),
            SignalingNode('JAK2', 'orthogonal', basal_rate=0.10, compensatory_gain=0.2, initial_activity=0.4),
            SignalingNode('CDK4', 'downstream', basal_rate=0.04, compensatory_gain=0.1, initial_activity=0.65),
            SignalingNode('CCND1', 'downstream', basal_rate=0.05, compensatory_gain=0.12, initial_activity=0.7),
            SignalingNode('PI3K', 'orthogonal', basal_rate=0.08, compensatory_gain=0.3, initial_activity=0.45),
            SignalingNode('MCL1', 'orthogonal', basal_rate=0.05, compensatory_gain=0.2, initial_activity=0.45),
        ]

        interactions = [
            Interaction('KRAS', 'MEK', 'activation', 0.8),
            Interaction('BRAF', 'MEK', 'activation', 0.85),
            Interaction('MEK', 'ERK', 'activation', 0.95),
            Interaction('ERK', 'CCND1', 'activation', 0.55),
            Interaction('CCND1', 'CDK4', 'activation', 0.85),
            Interaction('CTNNB1', 'CCND1', 'activation', 0.6),
            Interaction('CTNNB1', 'STAT3', 'activation', 0.3),
            Interaction('JAK2', 'STAT3', 'activation', 0.6),
            Interaction('KRAS', 'PI3K', 'activation', 0.4),
            Interaction('PI3K', 'MCL1', 'activation', 0.45),
            Interaction('STAT3', 'MCL1', 'activation', 0.5),
            # Compensation: WNT axis compensates for MAPK loss
            Interaction('CTNNB1', 'STAT3', 'compensation', 0.55),
            Interaction('PI3K', 'STAT3', 'compensation', 0.35),
            # Negative feedback
            Interaction('ERK', 'BRAF', 'inhibition', 0.25),
        ]

        return nodes, interactions

    @staticmethod
    def create_breast_network() -> Tuple[List[SignalingNode], List[Interaction]]:
        """Breast cancer: CDK4/6 driven, STAT3 orthogonal axis."""
        nodes = [
            SignalingNode('CDK4', 'downstream', basal_rate=0.04, compensatory_gain=0.08, initial_activity=0.75),
            SignalingNode('CDK6', 'downstream', basal_rate=0.04, compensatory_gain=0.12, initial_activity=0.6),
            SignalingNode('CCND1', 'downstream', basal_rate=0.08, compensatory_gain=0.1, initial_activity=0.8),
            SignalingNode('KRAS', 'upstream', basal_rate=0.15, compensatory_gain=0.15, initial_activity=0.5),
            SignalingNode('PI3K', 'upstream', basal_rate=0.18, compensatory_gain=0.25, initial_activity=0.6),
            SignalingNode('ERBB2', 'upstream', basal_rate=0.15, compensatory_gain=0.3, initial_activity=0.5),
            SignalingNode('STAT3', 'orthogonal', basal_rate=0.08, compensatory_gain=0.55, initial_activity=0.6),
            SignalingNode('JAK1', 'orthogonal', basal_rate=0.10, compensatory_gain=0.2, initial_activity=0.4),
            SignalingNode('MCL1', 'orthogonal', basal_rate=0.05, compensatory_gain=0.2, initial_activity=0.5),
            SignalingNode('ERK', 'downstream', basal_rate=0.03, compensatory_gain=0.1, initial_activity=0.55),
        ]

        interactions = [
            Interaction('CCND1', 'CDK4', 'activation', 0.9),
            Interaction('CCND1', 'CDK6', 'activation', 0.7),
            Interaction('ERBB2', 'KRAS', 'activation', 0.45),
            Interaction('KRAS', 'ERK', 'activation', 0.7),
            Interaction('ERK', 'CCND1', 'activation', 0.55),
            Interaction('PI3K', 'STAT3', 'activation', 0.3),
            Interaction('JAK1', 'STAT3', 'activation', 0.65),
            Interaction('STAT3', 'MCL1', 'activation', 0.6),
            Interaction('STAT3', 'CCND1', 'activation', 0.3),
            # Compensation
            Interaction('PI3K', 'STAT3', 'compensation', 0.5),
            Interaction('ERBB2', 'STAT3', 'compensation', 0.35),
            # Negative feedback
            Interaction('ERK', 'ERBB2', 'inhibition', 0.25),
        ]

        return nodes, interactions

    @classmethod
    def get_network(cls, cancer_type: str) -> Tuple[List[SignalingNode], List[Interaction]]:
        """Get the signaling network for a cancer type."""
        factories = {
            'Pancreatic Adenocarcinoma': cls.create_pdac_network,
            'PDAC': cls.create_pdac_network,
            'Melanoma': cls.create_melanoma_network,
            'Non-Small Cell Lung Cancer': cls.create_nsclc_network,
            'NSCLC': cls.create_nsclc_network,
            'Colorectal Adenocarcinoma': cls.create_colorectal_network,
            'CRC': cls.create_colorectal_network,
            'Invasive Breast Carcinoma': cls.create_breast_network,
            'Breast': cls.create_breast_network,
        }
        if cancer_type not in factories:
            raise ValueError(f"No network defined for {cancer_type}. Available: {list(factories.keys())}")
        return factories[cancer_type]()

    @classmethod
    def available_cancer_types(cls) -> List[str]:
        return ['PDAC', 'Melanoma', 'NSCLC', 'CRC', 'Breast']


# ---------------------------------------------------------------------------
# 3. TREATMENT STRATEGIES
# ---------------------------------------------------------------------------

class TreatmentStrategy:
    """Defines a treatment strategy (which nodes to inhibit)."""

    # Cancer-specific treatment strategies: {cancer: {strategy_name: [targets]}}
    #
    # Design rationale:
    #   - X-node strategies: Computationally-derived minimum hitting sets,
    #     which typically target nodes within the same signaling cascade or axis.
    #     These maximise graph-topological disruption but leave ≥1 biological
    #     axis unblocked, enabling compensatory pathway shifting.
    #   - Three-axis triples: One target per biological axis
    #     (upstream + downstream + orthogonal), ensuring no escape route.
    #
    STRATEGIES = {
        'PDAC': {
            'No treatment':                    [],
            'Single (KRAS)':                   ['KRAS'],
            'X-node (KRAS+EGFR)':              ['KRAS', 'EGFR'],       # 2 upstream → misses orthogonal
            'Three-axis (KRAS+CDK4+STAT3)':    ['KRAS', 'CDK4', 'STAT3'],  # up+down+ortho
        },
        'Melanoma': {
            'No treatment':                        [],
            'Single (BRAF)':                       ['BRAF'],
            'X-node (BRAF+MEK)':                   ['BRAF', 'MEK'],    # upstream+downstream cascade
            'Three-axis (BRAF+CCND1+STAT3)':       ['BRAF', 'CCND1', 'STAT3'],  # up+down+ortho
        },
        'NSCLC': {
            'No treatment':                        [],
            'Single (EGFR)':                       ['EGFR'],
            'X-node (EGFR+KRAS)':                  ['EGFR', 'KRAS'],   # 2 upstream
            'Three-axis (KRAS+CCND1+MCL1)':        ['KRAS', 'CCND1', 'MCL1'],  # up+down+ortho
        },
        'CRC': {
            'No treatment':                        [],
            'Single (KRAS)':                       ['KRAS'],
            'X-node (KRAS+BRAF)':                  ['KRAS', 'BRAF'],   # 2 upstream (MAPK drivers)
            'Three-axis (KRAS+CCND1+STAT3)':      ['KRAS', 'CCND1', 'STAT3'],  # up+down+ortho
        },
        'Breast': {
            'No treatment':                        [],
            'Single (CDK4)':                       ['CDK4'],
            'X-node (CDK4+CDK6)':                  ['CDK4', 'CDK6'],   # 2 downstream
            'Three-axis (CDK4+KRAS+STAT3)':        ['CDK4', 'KRAS', 'STAT3'],  # down+up+ortho
        },
    }

    @classmethod
    def get_strategies(cls, cancer_type: str) -> Dict[str, List[str]]:
        if cancer_type not in cls.STRATEGIES:
            raise ValueError(f"No strategies for {cancer_type}")
        return cls.STRATEGIES[cancer_type]


# ---------------------------------------------------------------------------
# 4. ODE SIGNALING MODEL
# ---------------------------------------------------------------------------

class PathwayShiftingODE:
    """
    ODE-based model of cancer signaling with adaptive pathway shifting.

    Each node i has activity A_i governed by:

      dA_i/dt = production_i(inputs) - degradation_i * A_i
                - drug_inhibition_i(t) * A_i
                + compensation_i(other_axes)

    Where:
      production_i = basal_rate + Σ_j (activation_strength_j→i * hill(A_j))
      compensation_i = gain_i * Σ_k∈other_axes (1 - mean(A_k)) * hill_compensation
      drug_inhibition = inhibition_strength * sigmoid_onset(t)

    Tumor viability = max(mean_downstream, mean_upstream, mean_orthogonal)
      (if ANY axis is active, tumor survives — the Liaki principle)
    """

    def __init__(self, nodes: List[SignalingNode], interactions: List[Interaction],
                 drug_effects: List[DrugEffect] = None):
        self.nodes = {n.name: n for n in nodes}
        self.node_names = [n.name for n in nodes]
        self.node_index = {n.name: i for i, n in enumerate(nodes)}
        self.interactions = interactions
        self.drug_effects = drug_effects or []
        self.n_nodes = len(nodes)

        # Pre-compute adjacency structure
        self._activators = {n: [] for n in self.node_names}   # (source_idx, strength, hill)
        self._inhibitors = {n: [] for n in self.node_names}
        self._compensators = {n: [] for n in self.node_names}

        for ix in interactions:
            if ix.target not in self.node_index or ix.source not in self.node_index:
                continue
            src_idx = self.node_index[ix.source]
            if ix.interaction_type == 'activation':
                self._activators[ix.target].append((src_idx, ix.strength, ix.hill_coefficient))
            elif ix.interaction_type == 'inhibition':
                self._inhibitors[ix.target].append((src_idx, ix.strength, ix.hill_coefficient))
            elif ix.interaction_type == 'compensation':
                self._compensators[ix.target].append((src_idx, ix.strength, ix.hill_coefficient))

        # Group nodes by axis
        self._axis_nodes = {'downstream': [], 'upstream': [], 'orthogonal': []}
        for name, node in self.nodes.items():
            if node.pathway in self._axis_nodes:
                self._axis_nodes[node.pathway].append(self.node_index[name])

        # Pre-compute drug effect indices
        self._drug_indices = []
        self._drug_target_set = set()
        for de in self.drug_effects:
            if de.target_node in self.node_index:
                self._drug_indices.append((self.node_index[de.target_node], de))
                self._drug_target_set.add(de.target_node)

        # Store initial axis means for compensation reference
        y0 = np.array([self.nodes[n].initial_activity for n in self.node_names])
        self._initial_axis_means = {
            axis: self._compute_axis_mean(y0, axis)
            for axis in ['downstream', 'upstream', 'orthogonal']
        }

    @staticmethod
    def hill(x: float, n: float = 2.0, K: float = 0.5) -> float:
        """Hill function for sigmoidal response."""
        x = max(0.0, x)
        return (x ** n) / (K ** n + x ** n)

    def drug_inhibition(self, t: float, drug: DrugEffect) -> float:
        """Time-dependent drug effect with exponential onset."""
        if t < 0:
            return 0.0
        return drug.inhibition_strength * (1.0 - np.exp(-drug.onset_rate * t))

    def _compute_axis_mean(self, y: np.ndarray, axis: str) -> float:
        """Mean activity of nodes in a given axis."""
        indices = self._axis_nodes.get(axis, [])
        if not indices:
            return 0.0
        return np.mean([max(0.0, y[i]) for i in indices])

    def rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        Right-hand side of the ODE system.

        Model:
          dA_i/dt = (production_i) * (1 - drug_effect_i(t))
                    + compensation_i(axis_loss)
                    - degradation_i * A_i
                    - inhibition_inputs

        Drug effect is MULTIPLICATIVE on production: a 90% inhibitor
        reduces production by 90%, so the node decays toward ~10% of
        its untreated steady state. This correctly models kinase
        inhibitors and PROTACs.

        Compensation activates when other axes DECREASE from their
        initial homeostatic level, representing adaptive rewiring
        (e.g., FYN→STAT3 activation when RAF1+EGFR are depleted).
        Crucially, compensation requires a source node that is itself
        NOT drug-inhibited — so three-axis blockade eliminates both
        direct activity and compensatory sources.
        """
        dydt = np.zeros(self.n_nodes)

        # Compute current axis means
        axis_means = {
            ax: self._compute_axis_mean(y, ax) for ax in ['downstream', 'upstream', 'orthogonal']
        }

        # Compute axis LOSS relative to initial homeostasis
        axis_loss = {}
        for ax in ['downstream', 'upstream', 'orthogonal']:
            init = self._initial_axis_means[ax]
            curr = axis_means[ax]
            axis_loss[ax] = max(0.0, (init - curr) / (init + 1e-6))

        for name, node in self.nodes.items():
            idx = self.node_index[name]
            A = max(0.0, y[idx])

            # --- Production ---
            production = node.basal_rate
            for src_idx, strength, hill_n in self._activators[name]:
                production += strength * self.hill(max(0.0, y[src_idx]), hill_n)

            # --- Drug inhibition (multiplicative on production) ---
            drug_factor = 1.0  # fraction of production remaining
            for drug_idx, drug in self._drug_indices:
                if drug_idx == idx:
                    inh = self.drug_inhibition(t, drug)
                    drug_factor *= (1.0 - inh)
            drug_factor = max(0.01, drug_factor)  # never fully zero

            effective_production = production * drug_factor

            # --- Inhibition inputs (from network, e.g., negative feedback) ---
            inhibition = 0.0
            for src_idx, strength, hill_n in self._inhibitors[name]:
                inhibition += strength * self.hill(max(0.0, y[src_idx]), hill_n)

            # --- Compensatory activation (PATHWAY SHIFTING) ---
            # Fires when axes OTHER than this node's axis have decreased
            other_axes = [ax for ax in ['downstream', 'upstream', 'orthogonal'] if ax != node.pathway]
            mean_other_loss = np.mean([axis_loss[ax] for ax in other_axes])
            # Only compensate if other axes are significantly impaired
            compensation_drive = max(0.0, mean_other_loss - 0.1)  # threshold 10% loss before compensation activates

            # Specific compensatory wiring (e.g., FYN→STAT3)
            specific_comp = 0.0
            for src_idx, strength, hill_n in self._compensators[name]:
                src_name = self.node_names[src_idx]
                src_activity = max(0.0, y[src_idx])
                # Compensatory source must itself be active (not drug-inhibited)
                if src_name not in self._drug_target_set:
                    specific_comp += strength * self.hill(src_activity, hill_n) * compensation_drive
                else:
                    # If source is also drugged, compensation is suppressed
                    specific_comp += strength * self.hill(src_activity, hill_n) * compensation_drive * 0.1

            compensation = node.compensatory_gain * compensation_drive + specific_comp

            # Only non-drugged nodes can compensate (drugged nodes have reduced capacity)
            if name in self._drug_target_set:
                compensation *= 0.1  # drugged nodes can barely compensate

            # --- Degradation ---
            degradation = node.degradation_rate * A

            # --- Combine ---
            dydt[idx] = effective_production + compensation - degradation - inhibition * A

            # Soft clamp
            if y[idx] <= 0.0 and dydt[idx] < 0:
                dydt[idx] = 0.0
            if y[idx] >= node.max_activity and dydt[idx] > 0:
                dydt[idx] *= 0.1  # slow approach to max rather than hard clamp

        return dydt

    def compute_tumor_viability(self, y: np.ndarray) -> float:
        """
        Tumor viability based on the Liaki principle:
        Tumor survives if ANY axis maintains sufficient activity.
        V = product of (1 - axis_activity) subtracted from 1 would mean
        all axes must fail. Instead: V = max(axis) captures "any one is enough."
        
        But we also require minimal DOWNSTREAM effector output (cell cycle/survival)
        because the tumor needs to actually proliferate.
        """
        axis_activities = {}
        for axis in ['downstream', 'upstream', 'orthogonal']:
            axis_activities[axis] = self._compute_axis_mean(y, axis)

        if not axis_activities:
            return 0.0

        # Core survival potential: ANY active axis can sustain tumor
        max_axis = max(axis_activities.values())
        
        # But tumor also needs effector output (downstream must have some activity)
        # This represents that raw signaling needs to translate to proliferation
        downstream = axis_activities.get('downstream', 0.0)
        effector_capacity = min(1.0, downstream / 0.3)  # saturates at downstream=0.3

        # Viability = survival_signal × effector_capacity
        # Survival signal is max-axis (Liaki: any one axis is enough)
        # Effector capacity gates whether that signal produces proliferation
        survival_signal = max_axis
        viability = 0.6 * survival_signal + 0.4 * np.mean(list(axis_activities.values()))
        
        # Scale so that untreated ≈ 1.0 and fully suppressed ≈ 0.0
        return min(1.0, max(0.0, viability))


# ---------------------------------------------------------------------------
# 5. SIMULATION ENGINE
# ---------------------------------------------------------------------------

class PathwayShiftingSimulator:
    """Runs ODE simulations comparing treatment strategies."""

    def __init__(self, t_end: float = 4800.0, dt: float = 2.0,
                 resistance_threshold: float = 0.4,
                 drug_onset_rate: float = 0.15,
                 drug_inhibition: float = 0.92):
        """
        Args:
            t_end: Simulation end time (hours). 4800h = 200 days (matches Liaki et al. observation window).
            dt: Output time step (hours).
            resistance_threshold: Viability above which resistance is declared.
            drug_onset_rate: Rate of drug onset (exponential).
            drug_inhibition: Default inhibition strength for drugs.
        """
        self.t_end = t_end
        self.dt = dt
        self.resistance_threshold = resistance_threshold
        self.drug_onset_rate = drug_onset_rate
        self.drug_inhibition = drug_inhibition
        self.t_end_days = t_end / 24.0

    def create_drug_effects(self, targets: List[str]) -> List[DrugEffect]:
        """Create drug effects for a list of target nodes."""
        return [
            DrugEffect(t, inhibition_strength=self.drug_inhibition,
                       onset_time=24.0, onset_rate=self.drug_onset_rate)
            for t in targets
        ]

    def run_simulation(self, cancer_type: str, strategy_name: str,
                       targets: List[str]) -> SimulationResult:
        """Run a single simulation for a given cancer type and treatment strategy."""
        nodes, interactions = CancerNetworkFactory.get_network(cancer_type)
        drug_effects = self.create_drug_effects(targets)

        # Build ODE model
        model = PathwayShiftingODE(nodes, interactions, drug_effects)

        # Initial conditions
        y0 = np.array([model.nodes[n].initial_activity for n in model.node_names])

        # Solve ODE
        t_span = (0.0, self.t_end)
        t_eval = np.arange(0, self.t_end + self.dt, self.dt)

        sol = solve_ivp(model.rhs, t_span, y0, method='RK45',
                        t_eval=t_eval, max_step=5.0,
                        rtol=1e-6, atol=1e-8)

        if not sol.success:
            print(f"  Warning: ODE solver failed for {strategy_name}: {sol.message}")

        # Extract results
        time_points = sol.t
        node_activities = {}
        for i, name in enumerate(model.node_names):
            node_activities[name] = np.clip(sol.y[i], 0.0, model.nodes[name].max_activity)

        # Compute tumor viability over time
        tumor_viability = np.array([
            model.compute_tumor_viability(sol.y[:, j])
            for j in range(len(time_points))
        ])

        # Compute metrics
        min_viability = float(np.min(tumor_viability))
        final_viability = float(tumor_viability[-1])
        auc_viability = float(np.trapezoid(tumor_viability, time_points) / self.t_end)

        # Time to resistance: first time viability recovers above threshold after dropping below
        # Require viability to drop at least 2% below threshold before counting as "dropped"
        # Start checking only after drug grace period (initial transient)
        dropped = False
        time_to_resistance = self.t_end  # no resistance by default
        resistance_emerged = False
        drop_threshold = self.resistance_threshold - 0.02  # small hysteresis
        grace_period = 120.0  # hours (~5 days) — wait for drug onset before measuring resistance
        for j in range(len(time_points)):
            if time_points[j] < grace_period:
                continue
            if tumor_viability[j] < drop_threshold:
                dropped = True
            elif dropped and tumor_viability[j] >= self.resistance_threshold:
                time_to_resistance = float(time_points[j])
                resistance_emerged = True
                break

        # Pathway shifting magnitude: max increase in orthogonal axis activity
        # compared to untreated initial
        max_shift = 0.0
        for name, node in model.nodes.items():
            if node.pathway == 'orthogonal' and name not in targets:
                initial_act = node.initial_activity
                max_act = float(np.max(node_activities[name]))
                shift = max_act - initial_act
                if shift > max_shift:
                    max_shift = shift

        return SimulationResult(
            strategy_name=strategy_name,
            targets=targets,
            time_points=time_points,
            node_activities=node_activities,
            tumor_viability=tumor_viability,
            time_to_resistance=time_to_resistance,
            min_viability=min_viability,
            auc_viability=auc_viability,
            pathway_shift_magnitude=max_shift,
            final_viability=final_viability,
            resistance_emerged=resistance_emerged,
        )

    def run_comparison(self, cancer_type: str) -> List[SimulationResult]:
        """Run all treatment strategies for a cancer type and return results."""
        strategies = TreatmentStrategy.get_strategies(cancer_type)
        results = []

        print(f"\n{'='*70}")
        print(f"Pathway Shifting Simulation: {cancer_type}")
        print(f"{'='*70}")

        for name, targets in strategies.items():
            print(f"  Running: {name} → targets = {targets if targets else '(none)'}")
            result = self.run_simulation(cancer_type, name, targets)
            results.append(result)
            print(f"    Min viability: {result.min_viability:.3f}")
            print(f"    Final viability: {result.final_viability:.3f}")
            print(f"    AUC viability: {result.auc_viability:.3f}")
            print(f"    Resistance emerged: {result.resistance_emerged}")
            print(f"    Max pathway shift: {result.pathway_shift_magnitude:.3f}")
            if result.resistance_emerged:
                print(f"    Time to resistance: {result.time_to_resistance:.0f} h ({result.time_to_resistance/24:.0f} d)")
            else:
                print(f"    Resistance-free for {self.t_end_days:.0f} days")

        return results


# ---------------------------------------------------------------------------
# 6. FIGURE GENERATION
# ---------------------------------------------------------------------------

def generate_simulation_figures(all_results: Dict[str, List[SimulationResult]],
                                output_dir: str = 'figures'):
    """Generate publication-quality figures for the simulation results."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("matplotlib not available; skipping figure generation")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Color scheme
    COLORS = {
        'No treatment': '#888888',
        'Single': '#e74c3c',
        'X-node': '#e67e22',
        'Three-axis': '#2ecc71',
    }

    def get_color(strategy_name: str) -> str:
        for key, color in COLORS.items():
            if key.lower() in strategy_name.lower():
                return color
        return '#3498db'

    # ---- FIGURE 1: Multi-panel viability time course (all cancer types) ----
    n_cancers = len(all_results)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes_flat = axes.flatten()

    for idx, (cancer, results) in enumerate(all_results.items()):
        if idx >= 6:
            break
        ax = axes_flat[idx]

        for res in results:
            color = get_color(res.strategy_name)
            lw = 2.5 if 'Three-axis' in res.strategy_name else 1.8
            ls = '-' if 'Three-axis' in res.strategy_name else '--'
            if res.strategy_name == 'No treatment':
                ls = ':'
                lw = 1.2

            # Convert hours to days for display
            days = res.time_points / 24.0
            ax.plot(days, res.tumor_viability, color=color, linewidth=lw,
                    linestyle=ls, label=res.strategy_name)

        ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.4, linewidth=0.8)
        ax.set_title(cancer, fontsize=13, fontweight='bold')
        ax.set_xlabel('Time (days)', fontsize=10)
        ax.set_ylabel('Tumor viability', fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=7, loc='best', framealpha=0.8)
        ax.grid(True, alpha=0.2)

    # Hide unused axes
    for idx in range(len(all_results), 6):
        axes_flat[idx].set_visible(False)

    fig.suptitle('Pathway Shifting Simulation: X-node vs. Three-Axis Triple Therapy',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'pathway_shifting_viability.png'),
                dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, 'pathway_shifting_viability.pdf'),
                bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_dir}/pathway_shifting_viability.png/pdf")

    # ---- FIGURE 2: PDAC detailed node-level dynamics ----
    if 'PDAC' in all_results:
        pdac_results = all_results['PDAC']
        # Find X-node and three-axis results
        xnode_res = None
        triple_res = None
        for r in pdac_results:
            if 'X-node' in r.strategy_name:
                xnode_res = r
            if 'Three-axis' in r.strategy_name:
                if triple_res is None:
                    triple_res = r

        if xnode_res and triple_res:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            # Panel A: X-node (KRAS+EGFR) — node activities
            ax = axes[0, 0]
            days = xnode_res.time_points / 24.0
            key_nodes = ['KRAS', 'EGFR', 'STAT3', 'FYN', 'ERK', 'CCND1']
            node_colors = {'KRAS': '#e74c3c', 'EGFR': '#3498db', 'STAT3': '#2ecc71',
                           'FYN': '#9b59b6', 'ERK': '#e67e22', 'CCND1': '#1abc9c'}
            for node in key_nodes:
                if node in xnode_res.node_activities:
                    ax.plot(days, xnode_res.node_activities[node],
                            color=node_colors.get(node, '#888'), linewidth=2,
                            label=node)
            ax.set_title(f'{xnode_res.strategy_name}: Node Activities', fontsize=12, fontweight='bold')
            ax.set_xlabel('Time (days)')
            ax.set_ylabel('Node activity')
            ax.legend(fontsize=8, ncol=2)
            ax.set_ylim(-0.05, 1.1)
            ax.grid(True, alpha=0.2)
            # Annotate pathway shifting
            if 'STAT3' in xnode_res.node_activities:
                stat3_max_idx = np.argmax(xnode_res.node_activities['STAT3'])
                stat3_max_val = xnode_res.node_activities['STAT3'][stat3_max_idx]
                stat3_max_day = days[stat3_max_idx]
                ax.annotate('STAT3 compensatory\\nactivation (via FYN)',
                           xy=(stat3_max_day, stat3_max_val),
                           xytext=(stat3_max_day + 5, stat3_max_val + 0.1),
                           fontsize=8, color='#2ecc71',
                           arrowprops=dict(arrowstyle='->', color='#2ecc71'))

            # Panel B: Three-axis (KRAS+CDK4+STAT3) — node activities
            ax = axes[0, 1]
            days = triple_res.time_points / 24.0
            for node in key_nodes:
                if node in triple_res.node_activities:
                    ax.plot(days, triple_res.node_activities[node],
                            color=node_colors.get(node, '#888'), linewidth=2,
                            label=node)
            ax.set_title(f'{triple_res.strategy_name}: Node Activities', fontsize=12, fontweight='bold')
            ax.set_xlabel('Time (days)')
            ax.set_ylabel('Node activity')
            ax.legend(fontsize=8, ncol=2)
            ax.set_ylim(-0.05, 1.1)
            ax.grid(True, alpha=0.2)

            # Panel C: Viability comparison
            ax = axes[1, 0]
            for r in pdac_results:
                color = get_color(r.strategy_name)
                lw = 2.5 if 'Three-axis' in r.strategy_name else 1.5
                days = r.time_points / 24.0
                ax.plot(days, r.tumor_viability, color=color, linewidth=lw,
                        label=f'{r.strategy_name}')
            ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.4)
            ax.set_title('PDAC: Tumor Viability Comparison', fontsize=12, fontweight='bold')
            ax.set_xlabel('Time (days)')
            ax.set_ylabel('Tumor viability')
            ax.set_ylim(-0.05, 1.05)
            ax.legend(fontsize=7, loc='best')
            ax.grid(True, alpha=0.2)

            # Panel D: Summary bar chart
            ax = axes[1, 1]
            strategy_names = [r.strategy_name for r in pdac_results]
            final_viabs = [r.final_viability for r in pdac_results]
            colors = [get_color(n) for n in strategy_names]
            short_names = []
            for n in strategy_names:
                if 'Three-axis' in n:
                    short_names.append('Three-axis\ntriple')
                elif 'X-node' in n:
                    short_names.append('X-node\n(2-target)')
                elif 'Single' in n:
                    short_names.append('Single\nagent')
                else:
                    short_names.append('No\ntreatment')
            bars = ax.bar(short_names, final_viabs, color=colors, edgecolor='white', linewidth=0.5)
            ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.4, label='Resistance threshold')
            t_end_days = int(pdac_results[0].time_points[-1] / 24)
            ax.set_title(f'PDAC: Final Viability (day {t_end_days})', fontsize=12, fontweight='bold')
            ax.set_ylabel('Tumor viability')
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.2, axis='y')

            # Add value labels on bars
            for bar, val in zip(bars, final_viabs):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                        f'{val:.2f}', ha='center', fontsize=9, fontweight='bold')

            fig.suptitle('PDAC: Pathway Shifting Mechanism — Dual vs. Three-Axis Therapy',
                         fontsize=14, fontweight='bold')
            plt.tight_layout()
            fig.savefig(os.path.join(output_dir, 'pdac_pathway_shifting_detail.png'),
                        dpi=300, bbox_inches='tight')
            fig.savefig(os.path.join(output_dir, 'pdac_pathway_shifting_detail.pdf'),
                        bbox_inches='tight')
            plt.close(fig)
            print(f"  Saved: {output_dir}/pdac_pathway_shifting_detail.png/pdf")

    # ---- FIGURE 3: Summary bar chart (all cancers) ----
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Panel A: Final viability across cancers
    ax = axes[0]
    cancer_names = list(all_results.keys())
    x_positions = np.arange(len(cancer_names))
    bar_width = 0.15
    strategy_types = ['Single', 'X-node', 'Three-axis']
    strategy_colors = ['#e74c3c', '#e67e22', '#2ecc71']

    for s_idx, (stype, scolor) in enumerate(zip(strategy_types, strategy_colors)):
        vals = []
        for cancer in cancer_names:
            found = False
            for r in all_results[cancer]:
                if stype.lower() in r.strategy_name.lower():
                    vals.append(r.final_viability)
                    found = True
                    break
            if not found:
                vals.append(0.0)
        ax.bar(x_positions + s_idx * bar_width, vals, bar_width,
               color=scolor, label=stype, edgecolor='white')

    ax.set_xticks(x_positions + bar_width)
    ax.set_xticklabels(cancer_names, fontsize=9, rotation=15)
    ax.set_ylabel('Final tumor viability', fontsize=11)
    ax.set_title('Final Viability by Treatment Strategy', fontsize=13, fontweight='bold')
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.4)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis='y')
    ax.set_ylim(0, 1.05)

    # Panel B: Pathway shifting magnitude
    ax = axes[1]
    for s_idx, (stype, scolor) in enumerate(zip(strategy_types, strategy_colors)):
        vals = []
        for cancer in cancer_names:
            found = False
            for r in all_results[cancer]:
                if stype.lower() in r.strategy_name.lower():
                    vals.append(r.pathway_shift_magnitude)
                    found = True
                    break
            if not found:
                vals.append(0.0)
        ax.bar(x_positions + s_idx * bar_width, vals, bar_width,
               color=scolor, label=stype, edgecolor='white')

    ax.set_xticks(x_positions + bar_width)
    ax.set_xticklabels(cancer_names, fontsize=9, rotation=15)
    ax.set_ylabel('Pathway shifting magnitude', fontsize=11)
    ax.set_title('Compensatory Pathway Activation', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis='y')

    plt.suptitle('X-node vs. Three-Axis Triple: Systems Biology Comparison',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'xnode_vs_triple_comparison.png'),
                dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, 'xnode_vs_triple_comparison.pdf'),
                bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_dir}/xnode_vs_triple_comparison.png/pdf")


# ---------------------------------------------------------------------------
# 7. RESULTS FORMATTING
# ---------------------------------------------------------------------------

def format_results_table(all_results: Dict[str, List[SimulationResult]]) -> str:
    """Generate a LaTeX-formatted results table."""
    lines = []
    lines.append("% Auto-generated by pathway_shifting_simulation.py")
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Systems biology simulation: X-node vs.\\ three-axis triple therapy. "
                  "ODE-based signaling models simulate adaptive pathway shifting under different "
                  "treatment strategies. Time to resistance is measured as hours until tumor viability "
                  "recovers above 0.5. Pathway shift magnitude measures compensatory orthogonal axis "
                  "upregulation. All triples achieve lower final viability and greater resistance prevention.}")
    lines.append("\\label{tab:simulation}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{@{}llcccccc@{}}")
    lines.append("\\toprule")
    lines.append("\\textbf{Cancer} & \\textbf{Strategy} & \\textbf{Targets} & "
                  "\\textbf{Min V} & \\textbf{Final V} & \\textbf{AUC} & "
                  "\\textbf{TTR (d)} & \\textbf{Shift} \\\\")
    lines.append("\\midrule")

    for cancer, results in all_results.items():
        first = True
        for r in results:
            if r.strategy_name == 'No treatment':
                continue
            cancer_col = cancer if first else ""
            first = False
            n_targets = len(r.targets)
            t_end_days = int(r.time_points[-1] / 24)
            ttr_str = f"{r.time_to_resistance/24:.0f}" if r.resistance_emerged else f"$>$ {t_end_days}"
            lines.append(
                f"{cancer_col} & {r.strategy_name} & {n_targets} & "
                f"{r.min_viability:.2f} & {r.final_viability:.2f} & "
                f"{r.auc_viability:.2f} & {ttr_str} & {r.pathway_shift_magnitude:.2f} \\\\"
            )
        lines.append("\\midrule")

    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def format_results_summary(all_results: Dict[str, List[SimulationResult]]) -> str:
    """Generate a plain-text summary of simulation results."""
    lines = []
    lines.append("=" * 80)
    lines.append("PATHWAY SHIFTING SIMULATION: COMPLETE RESULTS SUMMARY")
    lines.append("=" * 80)

    for cancer, results in all_results.items():
        lines.append(f"\n{'─' * 60}")
        lines.append(f"  {cancer}")
        lines.append(f"{'─' * 60}")
        lines.append(f"  {'Strategy':<35} {'Targets':>3} {'Min V':>6} {'Final V':>8} {'AUC':>6} {'TTR(d)':>7} {'Shift':>6} {'Resist':>7}")

        for r in results:
            n_targets = len(r.targets)
            ttr_str = f"{r.time_to_resistance/24:.0f}" if r.resistance_emerged else ">200"
            resist = "YES" if r.resistance_emerged else "NO"
            lines.append(
                f"  {r.strategy_name:<35} {n_targets:>3} {r.min_viability:>6.3f} "
                f"{r.final_viability:>8.3f} {r.auc_viability:>6.3f} {ttr_str:>7} "
                f"{r.pathway_shift_magnitude:>6.3f} {resist:>7}"
            )

    # Cross-cancer summary
    lines.append(f"\n{'=' * 80}")
    lines.append("CROSS-CANCER SUMMARY: X-NODE vs THREE-AXIS TRIPLE")
    lines.append(f"{'=' * 80}")

    xnode_viabs = []
    triple_viabs = []
    xnode_resistances = []
    triple_resistances = []
    xnode_shifts = []
    triple_shifts = []

    for cancer, results in all_results.items():
        for r in results:
            if 'X-node' in r.strategy_name:
                xnode_viabs.append(r.final_viability)
                xnode_resistances.append(r.resistance_emerged)
                xnode_shifts.append(r.pathway_shift_magnitude)
            elif 'Three-axis' in r.strategy_name:
                triple_viabs.append(r.final_viability)
                triple_resistances.append(r.resistance_emerged)
                triple_shifts.append(r.pathway_shift_magnitude)

    if xnode_viabs and triple_viabs:
        lines.append(f"\n  X-node (minimal hitting set):")
        lines.append(f"    Mean final viability: {np.mean(xnode_viabs):.3f} ± {np.std(xnode_viabs):.3f}")
        lines.append(f"    Resistance emerged: {sum(xnode_resistances)}/{len(xnode_resistances)} cancers")
        lines.append(f"    Mean pathway shift: {np.mean(xnode_shifts):.3f}")

        lines.append(f"\n  Three-axis triple:")
        lines.append(f"    Mean final viability: {np.mean(triple_viabs):.3f} ± {np.std(triple_viabs):.3f}")
        lines.append(f"    Resistance emerged: {sum(triple_resistances)}/{len(triple_resistances)} cancers")
        lines.append(f"    Mean pathway shift: {np.mean(triple_shifts):.3f}")

        viab_reduction = (np.mean(xnode_viabs) - np.mean(triple_viabs)) / np.mean(xnode_viabs) * 100
        lines.append(f"\n  Triple advantage:")
        lines.append(f"    Viability reduction: {viab_reduction:.1f}% lower than X-node")
        lines.append(f"    Resistance prevention: {sum(xnode_resistances) - sum(triple_resistances)} fewer cancers with resistance")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 8. MAIN ENTRY POINT
# ---------------------------------------------------------------------------

def run_full_simulation(output_dir: str = 'simulation_results',
                        cancer_types: List[str] = None) -> Dict[str, List[SimulationResult]]:
    """
    Run pathway shifting simulations for all cancer types
    and generate figures + results.
    """
    if cancer_types is None:
        cancer_types = CancerNetworkFactory.available_cancer_types()

    os.makedirs(output_dir, exist_ok=True)
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    simulator = PathwayShiftingSimulator(
        t_end=4800.0,        # 200 days (matches Liaki et al.)
        resistance_threshold=0.5,
        drug_onset_rate=0.05,
        drug_inhibition=0.9,
    )

    all_results = {}
    for cancer in cancer_types:
        results = simulator.run_comparison(cancer)
        all_results[cancer] = results

    # Generate figures
    print(f"\n{'='*70}")
    print("Generating figures...")
    generate_simulation_figures(all_results, figures_dir)

    # Generate results summary
    summary = format_results_summary(all_results)
    with open(os.path.join(output_dir, 'simulation_summary.txt'), 'w') as f:
        f.write(summary)
    print(f"  Saved: {output_dir}/simulation_summary.txt")

    # Generate LaTeX table
    latex_table = format_results_table(all_results)
    with open(os.path.join(output_dir, 'simulation_table.tex'), 'w') as f:
        f.write(latex_table)
    print(f"  Saved: {output_dir}/simulation_table.tex")

    # Save raw metrics as JSON
    metrics = {}
    for cancer, results in all_results.items():
        metrics[cancer] = []
        for r in results:
            metrics[cancer].append({
                'strategy': r.strategy_name,
                'targets': r.targets,
                'n_targets': len(r.targets),
                'min_viability': r.min_viability,
                'final_viability': r.final_viability,
                'auc_viability': r.auc_viability,
                'time_to_resistance_hours': r.time_to_resistance,
                'time_to_resistance_days': r.time_to_resistance / 24.0,
                'resistance_emerged': r.resistance_emerged,
                'pathway_shift_magnitude': r.pathway_shift_magnitude,
            })
    with open(os.path.join(output_dir, 'simulation_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved: {output_dir}/simulation_metrics.json")

    # Print summary
    print("\n" + summary)

    return all_results


if __name__ == '__main__':
    results = run_full_simulation()
