"""
ALIN Framework Core Modules
===========================
Modular components for the Adaptive Lethal Intersection Network framework.

This package contains:
- data_structures: Core data classes (TargetNode, NodeCost, ViabilityPath, HittingSet, etc.)
- data_loader: DepMap and OmniPath data loading utilities
- path_inference: Viability path inference algorithms
- hitting_set: Minimal hitting set solver
- scoring: Synergy, resistance, and combination scoring
- statistics: Statistical corrections and confidence intervals
"""

from .data_structures import (
    TargetNode,
    NodeCost,
    ViabilityPath,
    HittingSet,
    CancerTypeAnalysis,
    DrugTarget,
    TripleCombination,
)

from .statistics import (
    apply_fdr_correction,
    compute_confidence_interval,
    bootstrap_confidence_interval,
    sensitivity_analysis,
)

__all__ = [
    # Data structures
    'TargetNode',
    'NodeCost',
    'ViabilityPath',
    'HittingSet',
    'CancerTypeAnalysis',
    'DrugTarget',
    'TripleCombination',
    # Statistics
    'apply_fdr_correction',
    'compute_confidence_interval',
    'bootstrap_confidence_interval',
    'sensitivity_analysis',
]

__version__ = '1.0.0'
