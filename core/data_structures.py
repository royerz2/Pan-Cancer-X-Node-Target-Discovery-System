"""
Core Data Structures for ALIN Framework
=======================================
Dataclasses representing targets, costs, paths, and combinations.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Tuple, Optional, FrozenSet


@dataclass(frozen=True)
class TargetNode:
    """Single target node in the network"""
    gene_symbol: str
    entrez_id: Optional[int] = None
    
    def __hash__(self):
        return hash(self.gene_symbol)
    
    def __eq__(self, other):
        if not isinstance(other, TargetNode):
            return False
        return self.gene_symbol == other.gene_symbol


@dataclass
class NodeCost:
    """
    Cost function for a target node.
    
    Attributes:
        gene: Gene symbol
        toxicity_score: 0-1, higher = more toxic
        tumor_specificity: 0-1, higher = more tumor-specific
        druggability_score: 0-1, higher = more druggable
        protein_druggability_score: 0-1, protein-level druggability
            (structural + abundance + degradability + PPI). None if not computed.
        pan_essential_penalty: Penalty if gene is pan-essential
        base_penalty: Base cost per node
    """
    gene: str
    toxicity_score: float
    tumor_specificity: float
    druggability_score: float
    protein_druggability_score: Optional[float] = None
    pan_essential_penalty: float = 0.0
    base_penalty: float = 1.0
    
    @property
    def effective_druggability(self) -> float:
        """Blended druggability: gene-level + protein-level when available.
        
        Uses adaptive blending with discordance penalty (v2):
        - When gene and protein scores diverge by >0.4, switches to
          geometric mean so low protein truly penalises.
        - Low-protein veto: caps blended at 0.5 if protein < 0.25.
        Falls back to d_gene(g) when protein score is absent.
        """
        if self.protein_druggability_score is not None:
            from alin.protein_scoring import _adaptive_blend
            return _adaptive_blend(self.druggability_score,
                                   self.protein_druggability_score)
        return self.druggability_score
    
    def total_cost(self, alpha=1.0, beta=0.5, gamma=0.3, delta=2.0, lambda_base=1.0) -> float:
        """
        Compute weighted total cost.
        
        Formula: c(g,c) = α*τ(g) - β*s(g,c) - γ*d_eff(g) + δ*pan + λ
        
        Uses effective_druggability (blended gene + protein score) when
        protein scoring data is available; otherwise uses gene-level only.
        
        Args:
            alpha: Weight for toxicity (penalize toxic targets)
            beta: Weight for tumor specificity (reward specific targets)
            gamma: Weight for druggability (reward druggable targets)
            delta: Weight for pan-essential penalty (strongly penalize)
            lambda_base: Base penalty per node
            
        Returns:
            Total cost score (lower is better for target selection)
        """
        return (
            alpha * self.toxicity_score 
            - beta * self.tumor_specificity 
            - gamma * self.effective_druggability 
            + delta * self.pan_essential_penalty
            + lambda_base * self.base_penalty
        )


@dataclass
class ViabilityPath:
    """
    A functional path that supports tumor viability.
    
    Attributes:
        path_id: Unique identifier for this path
        nodes: Set of genes in this path
        context: Cancer type or cell line context
        confidence: Confidence score (0-1)
        path_type: Type of path (essential_module, signaling_path, etc.)
    """
    path_id: str
    nodes: FrozenSet[str]
    context: str
    confidence: float = 1.0
    path_type: str = "essential_module"
    
    def __hash__(self):
        return hash((self.path_id, self.nodes))
    
    def __contains__(self, gene: str) -> bool:
        """Check if gene is in this path"""
        return gene in self.nodes
    
    def intersects(self, genes: Set[str]) -> bool:
        """Check if path intersects with a set of genes"""
        return bool(self.nodes & genes)


@dataclass
class HittingSet:
    """
    A candidate X-node target set.
    
    Attributes:
        targets: Set of gene targets
        total_cost: Sum of individual target costs
        coverage: Fraction of viability paths hit
        paths_covered: IDs of covered paths
    """
    targets: FrozenSet[str]
    total_cost: float
    coverage: float
    paths_covered: Set[str]
    
    def __len__(self):
        return len(self.targets)
    
    def __hash__(self):
        return hash(self.targets)
    
    def __contains__(self, gene: str) -> bool:
        return gene in self.targets
    
    def hits_path(self, path: ViabilityPath) -> bool:
        """Check if this hitting set covers a path"""
        return bool(self.targets & path.nodes)


@dataclass
class DrugTarget:
    """Druggable target with clinical context"""
    gene: str
    available_drugs: List[str]
    clinical_stage: str  # "approved", "phase3", "phase2", "phase1", "preclinical"
    known_toxicities: List[str]
    
    @property
    def is_approved(self) -> bool:
        return self.clinical_stage == "approved"
    
    @property
    def is_clinical(self) -> bool:
        return self.clinical_stage in ("approved", "phase3", "phase2", "phase1")


@dataclass
class TripleCombination:
    """
    Triple drug combination with comprehensive scoring.
    
    Attributes:
        targets: Tuple of three gene targets
        total_cost: Sum of individual costs
        synergy_score: Predicted synergy (0-1, higher is better)
        resistance_score: Resistance probability (0-1, lower is better)
        pathway_coverage: Dict of pathway names to coverage fractions
        coverage: Fraction of viability paths covered
        druggable_count: Number of targets with approved drugs
        combined_score: Overall score (lower is better)
        drug_info: Dict of gene to DrugTarget info
        confidence_interval: Optional 95% CI for combined score
    """
    targets: Tuple[str, ...]  # 2 or 3 gene targets
    total_cost: float
    synergy_score: float
    resistance_score: float
    pathway_coverage: Dict[str, float]
    coverage: float
    druggable_count: int
    combined_score: float
    drug_info: Dict[str, Optional[DrugTarget]] = field(default_factory=dict)
    combo_tox_score: float = 0.0  # Combination-level toxicity (DDI, overlapping toxicities)
    combo_tox_details: Dict = field(default_factory=dict)  # Breakdown of combo toxicity
    confidence_interval: Optional[Tuple[float, float]] = None
    resistance_implausible: bool = False  # True when resistance_score < 0.05 (biologically implausible)
    druggability_tier: str = ''  # Per-combo tier: 'immediate', 'partial', 'research'
    evidence_power: str = ''  # 'robust', 'adequate', 'suggestive', 'hypothesis'
    # V6 fields
    clinical_readiness: float = 0.0   # Translational: multiplier based on drug approval stage
    novelty_score: float = 0.0        # Exploratory: bonus for non-gold-standard triples
    pan_essential_penalty: float = 0.0 # Exploratory: penalty for over-represented genes
    scoring_mode: str = 'additive'    # 'additive' or 'multiplicative'
    strategy_arm: str = 'default'     # Comparison arm label for exports and reports
    role_assignments: Dict[str, str] = field(default_factory=dict)  # structural role metadata
    
    def __lt__(self, other):
        return self.combined_score < other.combined_score
    
    def __contains__(self, gene: str) -> bool:
        return gene in self.targets
    
    @property
    def drugs(self) -> List[str]:
        """Get list of available drugs for this combination"""
        drugs = []
        for target in self.targets:
            info = self.drug_info.get(target)
            if info and info.available_drugs:
                drugs.append(info.available_drugs[0])
            else:
                drugs.append(f"({target} inhibitor)")
        return drugs


@dataclass
class CancerTypeAnalysis:
    """Complete analysis for one cancer type"""
    cancer_type: str
    lineage: str
    n_cell_lines: int
    cell_line_ids: List[str]
    driver_mutations: Dict[str, float]
    essential_genes: Dict[str, float]
    viability_paths: List[ViabilityPath]
    minimal_hitting_sets: List[HittingSet]
    top_x_node_sets: List[Tuple[FrozenSet[str], float]]
    recommended_combination: Optional[List[str]]
    triple_combinations: List[TripleCombination] = field(default_factory=list)
    best_triple: Optional[TripleCombination] = None
    best_combination: Optional[TripleCombination] = None  # Best of any size (2 or 3)
    statistics: Dict[str, Any] = field(default_factory=dict)
    pharmacological_validation: Optional[Dict[str, Any]] = None  # From PharmacologicalValidator
    
    @property
    def has_predictions(self) -> bool:
        """Check if analysis produced predictions"""
        return bool(self.triple_combinations) or bool(self.recommended_combination)
