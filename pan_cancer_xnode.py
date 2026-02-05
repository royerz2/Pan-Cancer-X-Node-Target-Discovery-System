#!/usr/bin/env python3
"""
ALIN Framework — Adaptive Lethal Intersection Network
====================================================
Generalized minimal hitting set framework for discovering optimal drug combination targets
across all cancer types using DepMap + OmniPath integration.

Based on the methodology from:
"A targeted combination therapy achieves effective pancreatic cancer regression 
and prevents tumor resistance"

Theoretical foundation:
- X-node targets = minimal hitting sets of tumor viability/resistance networks
- Penalty for each additional node (toxicity, side effects)
- Works across ALL cancer types in DepMap

Key Improvements over stub version:
- Loads REAL DepMap data (CRISPRGeneEffect.csv, Model.csv)
- Proper cancer type matching via OncotreePrimaryDisease
- Statistical filtering (pan-essential genes, significance testing)
- OmniPath API integration for signaling networks
- Biologically-sound viability path inference

Usage:
    python pan_cancer_xnode.py --cancer-type "Pancreatic Adenocarcinoma" --output results/
    python pan_cancer_xnode.py --cancer-type PAAD --output results/
    python pan_cancer_xnode.py --all-cancers --top-n 20
"""

import json
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Set, Tuple, Optional, FrozenSet, Union
from enum import Enum
from collections import defaultdict
import logging
from itertools import combinations
from pathlib import Path
from scipy import stats
import warnings
import re

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not installed
    def tqdm(iterable, *args, **kwargs):
        return iterable

from alin.utils import sanitize_cancer_name


def _progress(msg: str, step: str = "") -> None:
    """Print progress message so user always sees what's happening (flush immediately)."""
    if step:
        print(f"    -> {msg} [{step}]", flush=True)
    else:
        print(f"    -> {msg}...", flush=True)

# Import validation module
try:
    from alin.validation import (
        ValidationEngine,
        CombinationValidation,
        generate_validation_report,
        export_validation_results
    )
    VALIDATION_AVAILABLE = True
except ImportError:
    VALIDATION_AVAILABLE = False
    logging.warning("Validation module not available. Install dependencies or check import.")

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass(frozen=True)
class TargetNode:
    """Single target node in the network"""
    gene_symbol: str
    entrez_id: Optional[int] = None
    
    def __hash__(self):
        return hash(self.gene_symbol)
    
    def __eq__(self, other):
        return self.gene_symbol == other.gene_symbol

@dataclass
class NodeCost:
    """Cost function for a target node"""
    gene: str
    toxicity_score: float  # 0-1, higher = more toxic
    tumor_specificity: float  # 0-1, higher = more tumor-specific
    druggability_score: float  # 0-1, higher = more druggable
    pan_essential_penalty: float = 0.0  # Penalty if gene is pan-essential
    base_penalty: float = 1.0  # Base cost per node
    
    def total_cost(self, alpha=1.0, beta=0.5, gamma=0.3, delta=2.0, lambda_base=1.0) -> float:
        """
        Compute weighted total cost
        
        Args:
            alpha: Weight for toxicity (penalize toxic targets)
            beta: Weight for tumor specificity (reward specific targets)
            gamma: Weight for druggability (reward druggable targets)
            delta: Weight for pan-essential penalty (strongly penalize)
            lambda_base: Base penalty per node
        """
        return (
            alpha * self.toxicity_score 
            - beta * self.tumor_specificity 
            - gamma * self.druggability_score 
            + delta * self.pan_essential_penalty
            + lambda_base * self.base_penalty
        )

@dataclass
class ViabilityPath:
    """A functional path that supports tumor viability"""
    path_id: str
    nodes: FrozenSet[str]  # Genes in this path
    context: str  # Which cell lines / conditions this path is active in
    confidence: float = 1.0
    path_type: str = "essential_module"  # "essential_module", "signaling_path", "synthetic_lethal"
    
    def __hash__(self):
        return hash((self.path_id, self.nodes))

@dataclass
class HittingSet:
    """A candidate X-node target set"""
    targets: FrozenSet[str]
    total_cost: float
    coverage: float  # Fraction of viability paths hit
    paths_covered: Set[str]
    
    def __len__(self):
        return len(self.targets)
    
    def __hash__(self):
        return hash(self.targets)

@dataclass
class CancerTypeAnalysis:
    """Complete analysis for one cancer type"""
    cancer_type: str
    lineage: str
    n_cell_lines: int
    cell_line_ids: List[str]
    driver_mutations: Dict[str, float]  # Gene -> frequency
    essential_genes: Dict[str, float]  # Gene -> mean dependency score
    viability_paths: List[ViabilityPath]
    minimal_hitting_sets: List[HittingSet]
    top_x_node_sets: List[Tuple[FrozenSet[str], float]]  # (targets, cost)
    recommended_combination: Optional[List[str]]
    triple_combinations: List['TripleCombination'] = field(default_factory=list)  # Systems biology triples
    best_triple: Optional['TripleCombination'] = None
    statistics: Dict[str, any] = field(default_factory=dict)
    
@dataclass
class DrugTarget:
    """Druggable target with clinical context"""
    gene: str
    available_drugs: List[str]
    clinical_stage: str  # "approved", "phase3", "phase2", "phase1", "preclinical"
    known_toxicities: List[str]

# ============================================================================
# CANCER TYPE MAPPING
# ============================================================================

# Map common abbreviations to full OncoTree disease names
CANCER_TYPE_ALIASES = {
    'PAAD': 'Pancreatic Adenocarcinoma',
    'PDAC': 'Pancreatic Adenocarcinoma',
    'LUAD': 'Lung Adenocarcinoma',
    'LUSC': 'Lung Squamous Cell Carcinoma',
    'NSCLC': 'Non-Small Cell Lung Cancer',
    'BRCA': 'Invasive Breast Carcinoma',
    'COAD': 'Colon Adenocarcinoma',
    'READ': 'Rectal Adenocarcinoma',
    'CRC': 'Colorectal Adenocarcinoma',
    'SKCM': 'Melanoma',
    'MEL': 'Melanoma',
    'GBM': 'Glioblastoma',
    'OV': 'Ovarian Epithelial Tumor',
    'HGSOC': 'High-Grade Serous Ovarian Cancer',
    'PRAD': 'Prostate Adenocarcinoma',
    'STAD': 'Stomach Adenocarcinoma',
    'ESCA': 'Esophageal Adenocarcinoma',
    'LIHC': 'Hepatocellular Carcinoma',
    'HCC': 'Hepatocellular Carcinoma',
    'BLCA': 'Bladder Urothelial Carcinoma',
    'KIRC': 'Renal Clear Cell Carcinoma',
    'CCRCC': 'Renal Clear Cell Carcinoma',
    'AML': 'Acute Myeloid Leukemia',
    'ALL': 'Acute Lymphoblastic Leukemia',
    'MM': 'Plasma Cell Myeloma',
    'PCM': 'Plasma Cell Myeloma',
}

def normalize_cancer_type(cancer_type: str) -> str:
    """Normalize cancer type to full OncoTree name"""
    # Check if it's an alias
    upper_type = cancer_type.upper().strip()
    if upper_type in CANCER_TYPE_ALIASES:
        return CANCER_TYPE_ALIASES[upper_type]
    # Return as-is (assume it's already the full name)
    return cancer_type.strip()

# ============================================================================
# REAL DEPMAP DATA LOADER
# ============================================================================

class DepMapLoader:
    """
    Load REAL DepMap data from local CSV files
    
    Expected files in data_dir:
    - Model.csv: Cell line metadata with OncotreePrimaryDisease
    - CRISPRGeneEffect.csv: CRISPR dependency scores (Chronos algorithm)
    - SubtypeMatrix.csv: Binary subtype feature matrix (optional)
    """
    
    def __init__(self, data_dir: str = "./depmap_data"):
        self.data_dir = Path(data_dir)
        self._model_df = None
        self._crispr_df = None
        self._subtype_df = None
        self._expression_df = None
        self._gene_name_map = {}  # Entrez ID -> Gene symbol
        logger.info(f"DepMap loader initialized. Data dir: {self.data_dir}")
        
    def _load_model_metadata(self) -> pd.DataFrame:
        """Load cell line metadata"""
        if self._model_df is not None:
            return self._model_df
            
        model_path = self.data_dir / "Model.csv"
        if not model_path.exists():
            raise FileNotFoundError(f"Model.csv not found in {self.data_dir}")
        
        logger.info("Loading cell line metadata from Model.csv")
        self._model_df = pd.read_csv(model_path, index_col='ModelID')
        logger.info(f"Loaded metadata for {len(self._model_df)} cell lines")
        return self._model_df
    
    def _parse_gene_column(self, col: str) -> Tuple[str, Optional[int]]:
        """Parse gene column format 'GENE (ENTREZ)' -> (gene_symbol, entrez_id)"""
        match = re.match(r'^([A-Z0-9\-]+)\s*\((\d+)\)$', col)
        if match:
            return match.group(1), int(match.group(2))
        return col, None
        
    def load_crispr_dependencies(self) -> pd.DataFrame:
        """
        Load CRISPR gene dependency scores (Chronos algorithm)
        
        Returns:
            DataFrame: rows=ModelID (ACH-...), cols=gene symbols, values=dependency_score
                      (lower/more negative = more essential)
        """
        if self._crispr_df is not None:
            return self._crispr_df
            
        crispr_path = self.data_dir / "CRISPRGeneEffect.csv"
        if not crispr_path.exists():
            raise FileNotFoundError(f"CRISPRGeneEffect.csv not found in {self.data_dir}")
        
        logger.info("Loading CRISPR dependency matrix from CRISPRGeneEffect.csv")
        
        # Load with first column as index
        df = pd.read_csv(crispr_path, index_col=0)
        
        # Parse column names to get gene symbols (remove Entrez IDs)
        new_columns = {}
        for col in df.columns:
            gene_symbol, entrez_id = self._parse_gene_column(col)
            new_columns[col] = gene_symbol
            if entrez_id:
                self._gene_name_map[entrez_id] = gene_symbol
        
        df = df.rename(columns=new_columns)
        
        # Handle duplicate columns by keeping the first
        df = df.loc[:, ~df.columns.duplicated()]
        
        self._crispr_df = df
        logger.info(f"Loaded {len(df)} cell lines × {len(df.columns)} genes")
        return self._crispr_df
    
    def load_lineage_annotations(self) -> pd.DataFrame:
        """Load cancer lineage/subtype for each cell line"""
        model_df = self._load_model_metadata()
        
        # Select relevant columns
        cols = ['OncotreeLineage', 'OncotreePrimaryDisease', 'OncotreeSubtype', 
                'OncotreeCode', 'CellLineName']
        available_cols = [c for c in cols if c in model_df.columns]
        
        return model_df[available_cols].copy()
    
    def get_cell_lines_for_cancer(self, cancer_type: str) -> List[str]:
        """Get ModelIDs for a specific cancer type"""
        cancer_type = normalize_cancer_type(cancer_type)
        model_df = self._load_model_metadata()
        
        # Search in OncotreePrimaryDisease (exact match first)
        matches = model_df[model_df['OncotreePrimaryDisease'] == cancer_type].index.tolist()
        
        # If no exact match, try partial/case-insensitive match
        if len(matches) == 0:
            mask = model_df['OncotreePrimaryDisease'].str.lower().str.contains(
                cancer_type.lower(), na=False
            )
            matches = model_df[mask].index.tolist()
        
        # Also check OncotreeCode
        if len(matches) == 0:
            mask = model_df['OncotreeCode'] == cancer_type.upper()
            matches = model_df[mask].index.tolist()
        
        return matches
    
    def get_available_cancer_types(self) -> List[Tuple[str, int]]:
        """Get all available cancer types with cell line counts"""
        model_df = self._load_model_metadata()
        counts = model_df['OncotreePrimaryDisease'].value_counts()
        return [(name, count) for name, count in counts.items() if pd.notna(name)]
    
    def load_subtype_features(self) -> Optional[pd.DataFrame]:
        """Load binary subtype feature matrix"""
        subtype_path = self.data_dir / "SubtypeMatrix.csv"
        if not subtype_path.exists():
            return None
        
        if self._subtype_df is not None:
            return self._subtype_df
            
        logger.info("Loading subtype features from SubtypeMatrix.csv")
        self._subtype_df = pd.read_csv(subtype_path, index_col=0)
        return self._subtype_df
    
    def load_expression(self) -> Optional[pd.DataFrame]:
        """
        Load optional CCLE expression data for expression-filtered essentiality.
        Looks for CCLE_expression.csv or CCLE_RNAseq_reads.csv in depmap_data/.
        Returns None if not found. Rows=cell lines (ModelID), cols=genes.
        """
        if self._expression_df is not None:
            return self._expression_df
        for fname in ('CCLE_expression.csv', 'CCLE_RNAseq_reads.csv', 'OmicsExpressionProteinCodingGenesTPMLogp1.csv'):
            expr_path = self.data_dir / fname
            if expr_path.exists():
                logger.info(f"Loading expression from {fname}")
                self._expression_df = pd.read_csv(expr_path, index_col=0)
                return self._expression_df
        return None
    
    def get_pan_essential_genes(self, threshold: float = 0.9, show_progress: bool = False) -> Set[str]:
        """
        Identify pan-essential genes (essential in >threshold fraction of all cell lines)
        
        These should be filtered out as they are required for all cells, not cancer-specific
        """
        crispr = self.load_crispr_dependencies()
        
        # A gene is essential if dependency < -0.5
        if show_progress:
            logger.info("Computing pan-essential genes...")
        essential_matrix = crispr < -0.5
        essential_fraction = essential_matrix.mean(axis=0)
        
        pan_essential = set(essential_fraction[essential_fraction > threshold].index)
        logger.info(f"Identified {len(pan_essential)} pan-essential genes (>{threshold*100:.0f}% of lines)")
        
        return pan_essential

# ============================================================================
# OMNIPATH NETWORK LOADER
# ============================================================================

class OmniPathLoader:
    """
    Load signaling network from OmniPath
    
    Can use:
    1. Local cache file (omnipath_network.csv)
    2. API call to OmniPath (if requests is available)
    3. Built-in cancer signaling network as fallback
    """
    
    def __init__(self, cache_dir: str = "./depmap_data"):
        self.cache_dir = Path(cache_dir)
        self._network_df = None
        logger.info("OmniPath loader initialized")
    
    def _get_builtin_cancer_network(self) -> pd.DataFrame:
        """
        Built-in cancer signaling network based on KEGG/Reactome
        Covers major cancer pathways: MAPK, PI3K/AKT, JAK/STAT, Wnt, Notch, p53
        """
        edges = [
            # RTK signaling
            ('EGFR', 'KRAS', 'activation', 'KEGG'),
            ('EGFR', 'PIK3CA', 'activation', 'Reactome'),
            ('EGFR', 'STAT3', 'activation', 'SIGNOR'),
            ('EGFR', 'SRC', 'activation', 'PhosphoSite'),
            ('ERBB2', 'EGFR', 'activation', 'KEGG'),
            ('ERBB2', 'PIK3CA', 'activation', 'Reactome'),
            ('ERBB3', 'PIK3CA', 'activation', 'Reactome'),
            ('MET', 'KRAS', 'activation', 'SIGNOR'),
            ('MET', 'PIK3CA', 'activation', 'Reactome'),
            ('MET', 'STAT3', 'activation', 'SIGNOR'),
            ('FGFR1', 'KRAS', 'activation', 'KEGG'),
            ('FGFR1', 'PIK3CA', 'activation', 'KEGG'),
            ('FGFR2', 'KRAS', 'activation', 'KEGG'),
            ('IGF1R', 'PIK3CA', 'activation', 'Reactome'),
            ('IGF1R', 'KRAS', 'activation', 'SIGNOR'),
            ('AXL', 'PIK3CA', 'activation', 'SIGNOR'),
            ('AXL', 'STAT3', 'activation', 'SIGNOR'),
            
            # RAS/MAPK pathway
            ('KRAS', 'RAF1', 'activation', 'KEGG'),
            ('KRAS', 'BRAF', 'activation', 'KEGG'),
            ('KRAS', 'PIK3CA', 'activation', 'Reactome'),
            ('NRAS', 'RAF1', 'activation', 'KEGG'),
            ('NRAS', 'BRAF', 'activation', 'KEGG'),
            ('HRAS', 'RAF1', 'activation', 'KEGG'),
            ('BRAF', 'MAP2K1', 'activation', 'KEGG'),
            ('RAF1', 'MAP2K1', 'activation', 'KEGG'),
            ('MAP2K1', 'MAPK1', 'activation', 'KEGG'),
            ('MAP2K1', 'MAPK3', 'activation', 'KEGG'),
            ('MAP2K2', 'MAPK1', 'activation', 'KEGG'),
            ('MAPK1', 'MYC', 'activation', 'Reactome'),
            ('MAPK1', 'ELK1', 'activation', 'KEGG'),
            ('MAPK3', 'MYC', 'activation', 'Reactome'),
            
            # PI3K/AKT/mTOR pathway
            ('PIK3CA', 'AKT1', 'activation', 'KEGG'),
            ('PIK3CA', 'AKT2', 'activation', 'KEGG'),
            ('AKT1', 'MTOR', 'activation', 'Reactome'),
            ('AKT1', 'GSK3B', 'inhibition', 'KEGG'),
            ('AKT1', 'FOXO3', 'inhibition', 'Reactome'),
            ('AKT1', 'BAD', 'inhibition', 'KEGG'),
            ('MTOR', 'RPS6KB1', 'activation', 'KEGG'),
            ('MTOR', 'EIF4EBP1', 'inhibition', 'KEGG'),
            ('PTEN', 'PIK3CA', 'inhibition', 'KEGG'),
            ('PTEN', 'AKT1', 'inhibition', 'KEGG'),
            
            # JAK/STAT pathway
            ('JAK1', 'STAT3', 'activation', 'KEGG'),
            ('JAK2', 'STAT3', 'activation', 'KEGG'),
            ('JAK2', 'STAT5A', 'activation', 'KEGG'),
            ('TYK2', 'STAT3', 'activation', 'KEGG'),
            ('STAT3', 'MYC', 'activation', 'Reactome'),
            ('STAT3', 'BCL2', 'activation', 'Reactome'),
            ('STAT3', 'MCL1', 'activation', 'Reactome'),
            ('STAT3', 'CCND1', 'activation', 'Reactome'),
            
            # SRC family kinases
            ('SRC', 'STAT3', 'activation', 'PhosphoSite'),
            ('SRC', 'FAK', 'activation', 'PhosphoSite'),
            ('SRC', 'PIK3CA', 'activation', 'SIGNOR'),
            ('FYN', 'STAT3', 'activation', 'PhosphoSite'),
            ('FYN', 'PIK3CA', 'activation', 'SIGNOR'),
            ('YES1', 'STAT3', 'activation', 'SIGNOR'),
            ('LYN', 'STAT3', 'activation', 'SIGNOR'),
            ('LCK', 'STAT3', 'activation', 'KEGG'),
            
            # Cell cycle
            ('MYC', 'CDK4', 'activation', 'Reactome'),
            ('MYC', 'CDK2', 'activation', 'Reactome'),
            ('MYC', 'CCND1', 'activation', 'Reactome'),
            ('CCND1', 'CDK4', 'activation', 'KEGG'),
            ('CCND1', 'CDK6', 'activation', 'KEGG'),
            ('CDK4', 'RB1', 'inhibition', 'KEGG'),
            ('CDK6', 'RB1', 'inhibition', 'KEGG'),
            ('RB1', 'E2F1', 'inhibition', 'KEGG'),
            ('CDKN2A', 'CDK4', 'inhibition', 'KEGG'),
            ('CDKN2A', 'CDK6', 'inhibition', 'KEGG'),
            ('CDKN1A', 'CDK2', 'inhibition', 'KEGG'),
            ('CDKN1B', 'CDK2', 'inhibition', 'KEGG'),
            
            # p53 pathway
            ('TP53', 'CDKN1A', 'activation', 'KEGG'),
            ('TP53', 'BAX', 'activation', 'KEGG'),
            ('TP53', 'BBC3', 'activation', 'Reactome'),
            ('TP53', 'MDM2', 'activation', 'KEGG'),
            ('MDM2', 'TP53', 'inhibition', 'KEGG'),
            
            # Apoptosis
            ('BCL2', 'BAX', 'inhibition', 'KEGG'),
            ('BCL2L1', 'BAX', 'inhibition', 'KEGG'),
            ('MCL1', 'BAX', 'inhibition', 'KEGG'),
            ('BAX', 'CASP3', 'activation', 'KEGG'),
            ('CASP8', 'CASP3', 'activation', 'KEGG'),
            
            # Wnt pathway
            ('WNT1', 'FZD1', 'activation', 'KEGG'),
            ('FZD1', 'DVL1', 'activation', 'KEGG'),
            ('DVL1', 'GSK3B', 'inhibition', 'KEGG'),
            ('CTNNB1', 'MYC', 'activation', 'KEGG'),
            ('CTNNB1', 'CCND1', 'activation', 'KEGG'),
            ('APC', 'CTNNB1', 'inhibition', 'KEGG'),
            
            # Notch pathway
            ('NOTCH1', 'HES1', 'activation', 'KEGG'),
            ('NOTCH1', 'MYC', 'activation', 'Reactome'),
            
            # NF-kB pathway
            ('NFKB1', 'BCL2', 'activation', 'KEGG'),
            ('NFKB1', 'CCND1', 'activation', 'Reactome'),
            ('IKBKB', 'NFKB1', 'activation', 'KEGG'),
            
            # Hippo pathway
            ('YAP1', 'MYC', 'activation', 'Reactome'),
            ('YAP1', 'CTGF', 'activation', 'Reactome'),
            ('LATS1', 'YAP1', 'inhibition', 'KEGG'),
            ('LATS2', 'YAP1', 'inhibition', 'KEGG'),
        ]
        
        df = pd.DataFrame(edges, columns=['source', 'target', 'interaction_type', 'database'])
        return df
    
    def load_signaling_network(self, use_api: bool = False) -> pd.DataFrame:
        """
        Load directed signaling network
        
        Returns:
            DataFrame with columns: source, target, interaction_type, database
        """
        if self._network_df is not None:
            return self._network_df
        
        # Try to load from cache
        cache_path = self.cache_dir / "omnipath_network.csv"
        if cache_path.exists():
            logger.info(f"Loading OmniPath network from cache: {cache_path}")
            self._network_df = pd.read_csv(cache_path)
            logger.info(f"Loaded {len(self._network_df)} edges from cache")
            return self._network_df
        
        # Try API if requested
        if use_api:
            try:
                import requests
                logger.info("Fetching network from OmniPath API...")
                url = "https://omnipathdb.org/interactions"
                params = {
                    'fields': 'sources,references',
                    'genesymbols': 1,
                    'datasets': 'omnipath,pathwayextra,kinaseextra',
                    'types': 'post_translational'
                }
                response = requests.get(url, params=params, timeout=60)
                if response.status_code == 200:
                    from io import StringIO
                    self._network_df = pd.read_csv(StringIO(response.text), sep='\t')
                    # Rename columns to match our format
                    self._network_df = self._network_df.rename(columns={
                        'source_genesymbol': 'source',
                        'target_genesymbol': 'target',
                        'is_stimulation': 'stimulation',
                        'is_inhibition': 'inhibition'
                    })
                    # Determine interaction type
                    self._network_df['interaction_type'] = self._network_df.apply(
                        lambda r: 'activation' if r.get('stimulation', 0) == 1 else 
                                  ('inhibition' if r.get('inhibition', 0) == 1 else 'unknown'),
                        axis=1
                    )
                    self._network_df['database'] = 'OmniPath'
                    # Save to cache
                    self._network_df.to_csv(cache_path, index=False)
                    logger.info(f"Loaded {len(self._network_df)} edges from OmniPath API")
                    return self._network_df
            except Exception as e:
                logger.warning(f"Failed to fetch from OmniPath API: {e}")
        
        # Fallback to built-in network
        logger.info("Using built-in cancer signaling network")
        self._network_df = self._get_builtin_cancer_network()
        logger.info(f"Loaded {len(self._network_df)} edges in signaling network")
        return self._network_df
    
    def get_downstream_targets(self, gene: str, depth: int = 2) -> Set[str]:
        """Get all downstream targets of a gene up to given depth"""
        network = self.load_signaling_network()
        
        visited = {gene}
        frontier = {gene}
        
        for _ in range(depth):
            new_frontier = set()
            for g in frontier:
                targets = network[network['source'] == g]['target'].tolist()
                new_frontier.update(targets)
            new_frontier -= visited
            visited.update(new_frontier)
            frontier = new_frontier
            if not frontier:
                break
        
        visited.discard(gene)
        return visited
    
    def get_upstream_regulators(self, gene: str, depth: int = 2) -> Set[str]:
        """Get all upstream regulators of a gene up to given depth"""
        network = self.load_signaling_network()
        
        visited = {gene}
        frontier = {gene}
        
        for _ in range(depth):
            new_frontier = set()
            for g in frontier:
                sources = network[network['target'] == g]['source'].tolist()
                new_frontier.update(sources)
            new_frontier -= visited
            visited.update(new_frontier)
            frontier = new_frontier
            if not frontier:
                break
        
        visited.discard(gene)
        return visited

# ============================================================================
# DRUG TARGET DATABASE
# ============================================================================

class DrugTargetDB:
    """
    Drug target and toxicity database
    Based on DGIdb, ChEMBL, and clinical data
    """
    
    # Known druggable targets with clinical information
    DRUG_DB = {
        # EGFR family
        'EGFR': {'drugs': ['erlotinib', 'gefitinib', 'osimertinib', 'afatinib'], 
                 'stage': 'approved', 'toxicity': 0.6, 'toxicities': ['rash', 'diarrhea', 'ILD']},
        'ERBB2': {'drugs': ['trastuzumab', 'pertuzumab', 'lapatinib', 'tucatinib'], 
                  'stage': 'approved', 'toxicity': 0.5, 'toxicities': ['cardiotoxicity']},
        
        # KRAS
        'KRAS': {'drugs': ['sotorasib', 'adagrasib'], 
                 'stage': 'approved', 'toxicity': 0.5, 'toxicities': ['diarrhea', 'hepatotoxicity']},
        
        # BRAF/MEK
        'BRAF': {'drugs': ['vemurafenib', 'dabrafenib', 'encorafenib'], 
                 'stage': 'approved', 'toxicity': 0.5, 'toxicities': ['rash', 'photosensitivity']},
        'MAP2K1': {'drugs': ['trametinib', 'cobimetinib', 'binimetinib'], 
                   'stage': 'approved', 'toxicity': 0.5, 'toxicities': ['rash', 'retinopathy']},
        'MAP2K2': {'drugs': ['trametinib', 'cobimetinib'], 'stage': 'approved', 'toxicity': 0.5, 'toxicities': ['rash']},
        
        # PI3K/AKT/mTOR
        'PIK3CA': {'drugs': ['alpelisib', 'idelalisib', 'copanlisib'], 
                   'stage': 'approved', 'toxicity': 0.6, 'toxicities': ['hyperglycemia', 'rash', 'diarrhea']},
        'AKT1': {'drugs': ['capivasertib', 'ipatasertib'], 
                 'stage': 'phase3', 'toxicity': 0.5, 'toxicities': ['hyperglycemia', 'rash']},
        'MTOR': {'drugs': ['everolimus', 'temsirolimus'], 
                 'stage': 'approved', 'toxicity': 0.6, 'toxicities': ['mucositis', 'pneumonitis']},
        
        # CDK
        'CDK4': {'drugs': ['palbociclib', 'ribociclib', 'abemaciclib'], 
                 'stage': 'approved', 'toxicity': 0.4, 'toxicities': ['neutropenia']},
        'CDK6': {'drugs': ['palbociclib', 'ribociclib', 'abemaciclib'], 
                 'stage': 'approved', 'toxicity': 0.4, 'toxicities': ['neutropenia']},
        'CDK2': {'drugs': ['dinaciclib'], 'stage': 'phase2', 'toxicity': 0.5, 'toxicities': ['myelosuppression']},
        
        # JAK/STAT
        'JAK1': {'drugs': ['ruxolitinib', 'tofacitinib', 'baricitinib'], 
                 'stage': 'approved', 'toxicity': 0.5, 'toxicities': ['infections', 'thrombosis']},
        'JAK2': {'drugs': ['ruxolitinib', 'fedratinib'], 
                 'stage': 'approved', 'toxicity': 0.6, 'toxicities': ['anemia', 'thrombocytopenia']},
        'STAT3': {'drugs': ['napabucasin', 'TTI-101'], 
                  'stage': 'phase2', 'toxicity': 0.3, 'toxicities': ['GI toxicity']},
        
        # SRC family - key for PDAC X-node therapy
        'SRC': {'drugs': ['dasatinib', 'bosutinib', 'saracatinib'], 
                'stage': 'approved', 'toxicity': 0.7, 'toxicities': ['pleural effusion', 'myelosuppression']},
        'FYN': {'drugs': ['dasatinib', 'saracatinib'], 
                'stage': 'approved', 'toxicity': 0.5, 'toxicities': ['myelosuppression']},
        'YES1': {'drugs': ['dasatinib'], 'stage': 'approved', 'toxicity': 0.6, 'toxicities': ['pleural effusion']},
        'LYN': {'drugs': ['dasatinib', 'bafetinib'], 'stage': 'approved', 'toxicity': 0.5, 'toxicities': ['myelosuppression']},
        
        # Other RTKs
        'MET': {'drugs': ['capmatinib', 'tepotinib', 'crizotinib'], 
                'stage': 'approved', 'toxicity': 0.5, 'toxicities': ['edema', 'nausea']},
        'ALK': {'drugs': ['crizotinib', 'alectinib', 'brigatinib', 'lorlatinib'], 
                'stage': 'approved', 'toxicity': 0.4, 'toxicities': ['visual disturbances']},
        'ROS1': {'drugs': ['crizotinib', 'entrectinib'], 'stage': 'approved', 'toxicity': 0.4, 'toxicities': ['edema']},
        'RET': {'drugs': ['selpercatinib', 'pralsetinib'], 'stage': 'approved', 'toxicity': 0.4, 'toxicities': ['hypertension']},
        'FGFR1': {'drugs': ['erdafitinib', 'pemigatinib'], 'stage': 'approved', 'toxicity': 0.5, 'toxicities': ['hyperphosphatemia']},
        'FGFR2': {'drugs': ['erdafitinib', 'pemigatinib'], 'stage': 'approved', 'toxicity': 0.5, 'toxicities': ['hyperphosphatemia']},
        'AXL': {'drugs': ['bemcentinib', 'gilteritinib'], 'stage': 'phase2', 'toxicity': 0.4, 'toxicities': ['fatigue']},
        
        # BCL2 family
        'BCL2': {'drugs': ['venetoclax', 'navitoclax'], 
                 'stage': 'approved', 'toxicity': 0.6, 'toxicities': ['tumor lysis syndrome', 'neutropenia']},
        'BCL2L1': {'drugs': ['navitoclax'], 'stage': 'phase2', 'toxicity': 0.7, 'toxicities': ['thrombocytopenia']},
        'MCL1': {'drugs': ['AMG-176', 'S64315'], 'stage': 'phase1', 'toxicity': 0.6, 'toxicities': ['cardiotoxicity']},
        
        # PARP
        'PARP1': {'drugs': ['olaparib', 'rucaparib', 'niraparib', 'talazoparib'], 
                  'stage': 'approved', 'toxicity': 0.5, 'toxicities': ['anemia', 'nausea']},
        
        # IDH
        'IDH1': {'drugs': ['ivosidenib'], 'stage': 'approved', 'toxicity': 0.4, 'toxicities': ['differentiation syndrome']},
        'IDH2': {'drugs': ['enasidenib'], 'stage': 'approved', 'toxicity': 0.4, 'toxicities': ['differentiation syndrome']},
        
        # Undruggable / difficult targets
        'TP53': {'drugs': [], 'stage': 'preclinical', 'toxicity': 0.9, 'toxicities': ['unknown']},
        'MYC': {'drugs': ['OMO-103'], 'stage': 'phase1', 'toxicity': 0.8, 'toxicities': ['unknown']},
        'RB1': {'drugs': [], 'stage': 'preclinical', 'toxicity': 0.9, 'toxicities': ['unknown']},
    }
    
    def get_druggability_score(self, gene: str) -> float:
        """Get druggability score (0=undruggable, 1=many approved drugs)"""
        if gene not in self.DRUG_DB:
            return 0.2  # Unknown genes get low score
        
        info = self.DRUG_DB[gene]
        stage_scores = {'approved': 1.0, 'phase3': 0.8, 'phase2': 0.6, 'phase1': 0.4, 'preclinical': 0.2}
        base = stage_scores.get(info['stage'], 0.2)
        
        # Bonus for multiple drugs
        n_drugs = len(info.get('drugs', []))
        bonus = min(0.2, n_drugs * 0.05)
        
        return min(1.0, base + bonus)
    
    def get_toxicity_score(self, gene: str) -> float:
        """Get toxicity score (0=safe, 1=highly toxic)"""
        if gene not in self.DRUG_DB:
            return 0.5  # Unknown
        return self.DRUG_DB[gene].get('toxicity', 0.5)
    
    def get_drug_info(self, gene: str) -> Optional[DrugTarget]:
        """Get full drug target information"""
        if gene not in self.DRUG_DB:
            return None
        
        info = self.DRUG_DB[gene]
        return DrugTarget(
            gene=gene,
            available_drugs=info.get('drugs', []),
            clinical_stage=info.get('stage', 'unknown'),
            known_toxicities=info.get('toxicities', [])
        )

# ============================================================================
# VIABILITY PATH INFERENCE
# ============================================================================

class ViabilityPathInference:
    """
    Infer viability paths P from DepMap + OmniPath
    
    Methods:
    1. Essential gene modules per cell line (from CRISPR)
    2. Signaling pathway dependencies (from network + CRISPR)
    3. Cancer-specific dependencies (vs all other cancers)
    """
    
    def __init__(self, depmap: DepMapLoader, omnipath: OmniPathLoader):
        self.depmap = depmap
        self.omnipath = omnipath
        self._pan_essential = None
        
    def _get_pan_essential(self) -> Set[str]:
        """Get cached pan-essential genes"""
        if self._pan_essential is None:
            self._pan_essential = self.depmap.get_pan_essential_genes(threshold=0.9)
        return self._pan_essential
        
    def infer_essential_modules(self, cancer_type: str, 
                                 dependency_threshold: float = -0.5,
                                 min_cell_lines: int = 3,
                                 min_selectivity_fraction: float = 0.3,
                                 expression_threshold: float = 1.0) -> List[ViabilityPath]:
        """
        Infer essential gene modules for a cancer type (refined).
        
        Refinements:
        1. Selectivity: only genes essential in >min_selectivity_fraction of cancer cell lines
        2. Co-essentiality clustering: genes essential together = same pathway (clustered)
        3. Expression filter: if expression data available, only count essential if expressed
        """
        logger.info(f"Inferring essential modules for {cancer_type} (selectivity>{min_selectivity_fraction:.0%})")
        
        crispr = self.depmap.load_crispr_dependencies()
        cell_lines = self.depmap.get_cell_lines_for_cancer(cancer_type)
        pan_essential = self._get_pan_essential()
        
        if len(cell_lines) == 0:
            logger.warning(f"No cell lines found for {cancer_type}")
            return []
        
        available_lines = [cl for cl in cell_lines if cl in crispr.index]
        if len(available_lines) < min_cell_lines:
            logger.warning(f"Too few cell lines ({len(available_lines)}) for {cancer_type}")
            return []
            
        crispr_subset = crispr.loc[available_lines]
        n_lines = len(available_lines)
        min_lines_essential = max(1, int(n_lines * min_selectivity_fraction))
        
        # Optional expression filter
        expr_df = self.depmap.load_expression()
        expr_available = expr_df is not None and len(set(expr_df.index) & set(available_lines)) > 0
        
        def _is_essential_in_line(gene: str, cl: str) -> bool:
            if gene not in crispr_subset.columns or cl not in crispr_subset.index:
                return False
            if crispr_subset.loc[cl, gene] >= dependency_threshold:
                return False
            if expr_available and gene in expr_df.columns:
                # Only count essential if expressed in tumor (TPM > threshold)
                try:
                    expr_val = expr_df.loc[cl, gene] if cl in expr_df.index else np.nan
                    if pd.isna(expr_val) or expr_val < expression_threshold:
                        return False
                except (KeyError, TypeError):
                    pass
            return True
        
        # Build per-line essential sets (with selectivity + expression filter)
        line_essential_sets = {}
        for cl in available_lines:
            essential_genes = [
                g for g in crispr_subset.columns
                if g not in pan_essential and _is_essential_in_line(g, cl)
            ]
            if len(essential_genes) >= 2:
                line_essential_sets[cl] = set(essential_genes)
        
        # Selectivity filter: only genes essential in >= min_lines_essential cell lines
        gene_essential_count = defaultdict(int)
        for ess_set in line_essential_sets.values():
            for g in ess_set:
                gene_essential_count[g] += 1
        
        selective_genes = {g for g, c in gene_essential_count.items() if c >= min_lines_essential}
        if len(selective_genes) < 2:
            logger.info(f"Few selective genes ({len(selective_genes)}); falling back to consensus")
            mean_dep = crispr_subset.mean(axis=0)
            selective_genes = set(mean_dep[mean_dep < dependency_threshold].index) - pan_essential
        
        # Co-essentiality clustering: genes that co-occur in essential sets = pathway modules
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform
        
        selective_list = list(selective_genes)
        if len(selective_list) < 2:
            return []
        
        # Co-occurrence matrix (Jaccard-like: fraction of lines where both essential)
        n_genes = len(selective_list)
        co_essential = np.zeros((n_genes, n_genes))
        for i, g1 in enumerate(selective_list):
            for j, g2 in enumerate(selective_list):
                if i == j:
                    co_essential[i, j] = 0
                    continue
                co_count = sum(1 for ess in line_essential_sets.values() 
                              if g1 in ess and g2 in ess)
                co_essential[i, j] = co_count / max(1, n_lines)
        
        # Convert to distance (1 - similarity) for linkage; ensure symmetry
        dist = 1 - co_essential
        dist = (dist + dist.T) / 2
        np.fill_diagonal(dist, 0)
        
        # Hierarchical clustering; cut to get ~5-15 clusters
        n_clusters = min(15, max(3, n_genes // 5))
        try:
            from scipy.spatial.distance import squareform
            from scipy.cluster.hierarchy import fcluster, linkage
            condensed = squareform(dist, checks=False)
            Z = linkage(condensed, method='average')
            clusters = fcluster(Z, n_clusters, criterion='maxclust')
            cluster_to_genes = defaultdict(set)
            for gene, c in zip(selective_list, clusters):
                cluster_to_genes[c].add(gene)
        except Exception as e:
            logger.debug(f"Co-essentiality clustering failed ({e}), using single module")
            cluster_to_genes = {0: selective_genes}
        
        paths = []
        for cid, genes in cluster_to_genes.items():
            if len(genes) >= 2:
                path = ViabilityPath(
                    path_id=f"{cancer_type}_coessential_cluster_{cid}",
                    nodes=frozenset(genes),
                    context=cancer_type,
                    confidence=0.9,
                    path_type="co_essential_module"
                )
                paths.append(path)
        
        # Consensus path (all selective genes) for full coverage
        if len(selective_genes) >= 2:
            paths.append(ViabilityPath(
                path_id=f"{cancer_type}_consensus_essential",
                nodes=frozenset(selective_genes),
                context=cancer_type,
                confidence=1.0,
                path_type="essential_module"
            ))
        
        logger.info(f"Inferred {len(paths)} essential module paths ({len(selective_genes)} selective genes)")
        return paths
    
    def infer_signaling_paths(self, cancer_type: str,
                              dependency_threshold: float = -0.5,
                              max_path_length: int = 4,
                              min_confidence: float = 0.5) -> List[ViabilityPath]:
        """
        Infer active signaling paths using NetworkX all_simple_paths.
        
        Refinements:
        1. Use NetworkX all_simple_paths() with length limits (2-4 hops)
        2. Score paths by mean dependency in cancer type (stronger dep = higher confidence)
        3. Prune low-confidence paths (confidence < min_confidence)
        """
        logger.info(f"Inferring signaling paths for {cancer_type} (max_len={max_path_length})")
        
        try:
            import networkx as nx
        except ImportError:
            logger.warning("NetworkX not installed; falling back to 2-hop paths")
            return self._infer_signaling_paths_legacy(cancer_type, dependency_threshold)
        
        network = self.omnipath.load_signaling_network()
        crispr = self.depmap.load_crispr_dependencies()
        cell_lines = self.depmap.get_cell_lines_for_cancer(cancer_type)
        pan_essential = self._get_pan_essential()
        
        available_lines = [cl for cl in cell_lines if cl in crispr.index]
        if len(available_lines) == 0:
            return []
        
        crispr_subset = crispr.loc[available_lines]
        mean_dep = crispr_subset.mean(axis=0)
        
        essential_genes = set(mean_dep[mean_dep < dependency_threshold].index)
        essential_genes -= pan_essential
        
        drivers = {'KRAS', 'BRAF', 'EGFR', 'ERBB2', 'MET', 'PIK3CA', 'TP53', 
                   'NRAS', 'HRAS', 'FGFR1', 'FGFR2', 'ALK', 'ROS1', 'RET'}
        effectors = {'MYC', 'CCND1', 'CDK4', 'CDK6', 'BCL2', 'MCL1', 'STAT3',
                     'MTOR', 'RPS6KB1', 'E2F1'}
        
        # Build directed graph
        G = nx.DiGraph()
        for _, row in network.iterrows():
            src, tgt = row['source'], row['target']
            if src in crispr.columns and tgt in crispr.columns:
                G.add_edge(src, tgt)
        
        paths = []
        seen_nodes = set()
        
        for driver in drivers:
            if driver not in G or G.out_degree(driver) == 0:
                continue
            
            for effector in effectors:
                if effector not in G or effector not in essential_genes:
                    continue
                
                try:
                    for path_nodes in nx.all_simple_paths(
                            G, driver, effector, cutoff=max_path_length):
                        if len(path_nodes) < 2:
                            continue
                        
                        # Score by mean dependency (more negative = higher confidence)
                        path_deps = [mean_dep.get(g, 0) for g in path_nodes if g in mean_dep.index]
                        mean_path_dep = np.mean(path_deps) if path_deps else 0
                        # Convert: dep < -0.5 -> high conf; dep > 0 -> low conf
                        confidence = max(0, min(1, 0.5 - mean_path_dep))
                        
                        if confidence < min_confidence:
                            continue
                        
                        path_id = f"{cancer_type}_{'_'.join(path_nodes[:5])}"
                        if len(path_nodes) > 5:
                            path_id += "_trunc"
                        
                        path = ViabilityPath(
                            path_id=path_id,
                            nodes=frozenset(path_nodes),
                            context=cancer_type,
                            confidence=round(confidence, 2),
                            path_type="signaling_path"
                        )
                        paths.append(path)
                        seen_nodes.update(path_nodes)
                        
                except nx.NetworkXNoPath:
                    pass
        
        logger.info(f"Inferred {len(paths)} signaling paths (confidence >= {min_confidence})")
        return paths
    
    def _infer_signaling_paths_legacy(self, cancer_type: str, 
                                      dependency_threshold: float) -> List[ViabilityPath]:
        """Fallback 2-hop paths when NetworkX unavailable."""
        network = self.omnipath.load_signaling_network()
        crispr = self.depmap.load_crispr_dependencies()
        cell_lines = self.depmap.get_cell_lines_for_cancer(cancer_type)
        pan_essential = self._get_pan_essential()
        available_lines = [cl for cl in cell_lines if cl in crispr.index]
        if not available_lines:
            return []
        
        crispr_subset = crispr.loc[available_lines]
        mean_dep = crispr_subset.mean(axis=0)
        essential_genes = set(mean_dep[mean_dep < dependency_threshold].index) - pan_essential
        drivers = {'KRAS', 'BRAF', 'EGFR', 'ERBB2', 'MET', 'PIK3CA', 'TP53', 
                   'NRAS', 'HRAS', 'FGFR1', 'FGFR2', 'ALK', 'ROS1', 'RET'}
        effectors = {'MYC', 'CCND1', 'CDK4', 'CDK6', 'BCL2', 'MCL1', 'STAT3',
                     'MTOR', 'RPS6KB1', 'E2F1'}
        
        paths = []
        for driver in drivers:
            if driver not in network['source'].values:
                continue
            direct_targets = set(network[network['source'] == driver]['target'])
            for target in direct_targets:
                if target in effectors and target in essential_genes:
                    paths.append(ViabilityPath(
                        path_id=f"{cancer_type}_{driver}_to_{target}",
                        nodes=frozenset([driver, target]),
                        context=cancer_type, confidence=0.8, path_type="signaling_path"))
                second_hop = set(network[network['source'] == target]['target'])
                for effector in second_hop:
                    if effector in effectors and effector in essential_genes:
                        paths.append(ViabilityPath(
                            path_id=f"{cancer_type}_{driver}_via_{target}_to_{effector}",
                            nodes=frozenset([driver, target, effector]),
                            context=cancer_type, confidence=0.6, path_type="signaling_path"))
        return paths
    
    def infer_cancer_specific_dependencies(self, cancer_type: str,
                                            p_value_threshold: float = 0.05,
                                            effect_threshold: float = 0.3) -> List[ViabilityPath]:
        """
        Find genes that are significantly MORE essential in this cancer vs others
        Uses t-test to compare dependency scores
        """
        logger.info(f"Finding cancer-specific dependencies for {cancer_type}")
        
        crispr = self.depmap.load_crispr_dependencies()
        cell_lines = self.depmap.get_cell_lines_for_cancer(cancer_type)
        pan_essential = self._get_pan_essential()
        
        available_lines = [cl for cl in cell_lines if cl in crispr.index]
        if len(available_lines) < 3:
            logger.warning(f"Too few cell lines ({len(available_lines)}) for statistical comparison")
            return []
        
        other_lines = [cl for cl in crispr.index if cl not in available_lines]
        
        cancer_specific_genes = []
        
        # Filter genes to test (exclude pan-essential)
        genes_to_test = [g for g in crispr.columns if g not in pan_essential]
        
        for gene in genes_to_test:
            cancer_scores = crispr.loc[available_lines, gene].dropna()
            other_scores = crispr.loc[other_lines, gene].dropna()
            
            if len(cancer_scores) < 3 or len(other_scores) < 10:
                continue
            
            # T-test: is this gene more essential in this cancer?
            t_stat, p_value = stats.ttest_ind(cancer_scores, other_scores)
            
            # Check if significantly MORE essential (lower scores = more essential)
            effect_size = other_scores.mean() - cancer_scores.mean()  # Positive = more essential in cancer
            
            if p_value < p_value_threshold and effect_size > effect_threshold:
                cancer_specific_genes.append({
                    'gene': gene,
                    'effect_size': effect_size,
                    'p_value': p_value,
                    'cancer_mean': cancer_scores.mean()
                })
        
        if len(cancer_specific_genes) >= 2:
            # Sort by effect size and take top genes
            cancer_specific_genes.sort(key=lambda x: x['effect_size'], reverse=True)
            top_genes = [g['gene'] for g in cancer_specific_genes[:20]]
            
            path = ViabilityPath(
                path_id=f"{cancer_type}_specific_dependencies",
                nodes=frozenset(top_genes),
                context=cancer_type,
                confidence=0.95,
                path_type="cancer_specific"
            )
            
            logger.info(f"Found {len(top_genes)} cancer-specific essential genes")
            return [path]
        
        return []
    
    def infer_perturbation_response_paths(self, cancer_type: str,
                                          dependency_threshold: float = -0.5,
                                          min_overlap: int = 2) -> List[ViabilityPath]:
        """
        Infer viability paths from perturbation-induced signaling changes.
        
        Uses curated phosphoproteomics and transcriptional response signatures
        to find essential genes that respond to target inhibition.
        
        This captures dynamic pathway relationships that static co-essentiality
        and network topology miss.
        """
        logger.info(f"Inferring perturbation response paths for {cancer_type}")
        
        try:
            from alin.perturbation import (
                build_perturbation_response_paths,
                get_perturbation_signature,
            )
        except ImportError:
            logger.warning("Perturbation module not available; skipping perturbation paths")
            return []
        
        crispr = self.depmap.load_crispr_dependencies()
        cell_lines = self.depmap.get_cell_lines_for_cancer(cancer_type)
        pan_essential = self._get_pan_essential()
        
        available_lines = [cl for cl in cell_lines if cl in crispr.index]
        if len(available_lines) == 0:
            return []
        
        crispr_subset = crispr.loc[available_lines]
        mean_dep = crispr_subset.mean(axis=0)
        
        # Get essential genes (excluding pan-essential)
        essential_genes = set(mean_dep[mean_dep < dependency_threshold].index)
        essential_genes -= pan_essential
        
        # Build paths from perturbation signatures
        pert_paths = build_perturbation_response_paths(
            essential_genes=essential_genes,
            min_overlap=min_overlap,
        )
        
        paths = []
        for target, path_genes, confidence in pert_paths:
            if len(path_genes) >= 2:
                path = ViabilityPath(
                    path_id=f"{cancer_type}_perturbation_{target}",
                    nodes=frozenset(path_genes),
                    context=cancer_type,
                    confidence=confidence,
                    path_type="perturbation_response"
                )
                paths.append(path)
        
        logger.info(f"Inferred {len(paths)} perturbation response paths")
        return paths
    
    def infer_all_paths(self, cancer_type: str, min_confidence: float = 0.5) -> List[ViabilityPath]:
        """Combine all path inference methods; prune paths with confidence < min_confidence."""
        paths = []
        
        _progress("Essential modules (co-essentiality)", step="")
        paths.extend(self.infer_essential_modules(cancer_type))
        _progress("Signaling paths (OmniPath)", step="")
        paths.extend(self.infer_signaling_paths(cancer_type, min_confidence=min_confidence))
        _progress("Cancer-specific dependencies", step="")
        paths.extend(self.infer_cancer_specific_dependencies(cancer_type))
        _progress("Perturbation response paths", step="")
        paths.extend(self.infer_perturbation_response_paths(cancer_type))
        
        # Prune low-confidence paths
        paths = [p for p in paths if p.confidence >= min_confidence]
        
        # Deduplicate by path_id
        seen = set()
        unique_paths = []
        for path in paths:
            if path.path_id not in seen:
                seen.add(path.path_id)
                unique_paths.append(path)
        
        logger.info(f"Total: {len(unique_paths)} unique viability paths for {cancer_type} (conf >= {min_confidence})")
        return unique_paths

# ============================================================================
# COST FUNCTION
# ============================================================================

class CostFunction:
    """
    Compute node costs based on toxicity, specificity, druggability.
    
    Toxicity sources:
    1. DrugTargetDB (built-in clinical data)
    2. OpenTargets API (off-target safety liabilities)
    3. Tissue expression weight (OpenTargets baseline expression; higher expression in healthy tissue increases weight)
    4. FDA FAERS (OpenFDA API) for known ADRs used when assessing drug safety
    """
    
    def __init__(self, depmap: DepMapLoader, drug_db: DrugTargetDB,
                 toxicity_cache_dir: Optional[str] = None):
        self.depmap = depmap
        self.drug_db = drug_db
        self.toxicity_cache_dir = toxicity_cache_dir
        self._pan_essential = None
        
    def _get_pan_essential(self) -> Set[str]:
        if self._pan_essential is None:
            self._pan_essential = self.depmap.get_pan_essential_genes()
        return self._pan_essential
    
    def _get_toxicity_score(self, gene: str) -> float:
        """Get toxicity score, enhanced by OpenTargets if available."""
        base_toxicity = self.drug_db.get_toxicity_score(gene)
        try:
            from alin.toxicity import (
                get_opentargets_toxicity,
                get_tissue_expression_weight,
            )
            ot_tox = get_opentargets_toxicity(gene, self.toxicity_cache_dir)
            if ot_tox is not None:
                base_toxicity = 0.6 * base_toxicity + 0.4 * ot_tox
            tissue_weight = get_tissue_expression_weight(gene)
            base_toxicity *= tissue_weight
        except ImportError:
            pass
        return max(0, min(1, base_toxicity))
        
    def compute_cost(self, gene: str, cancer_type: str) -> NodeCost:
        """Compute comprehensive cost for a gene in a cancer context"""
        
        # Druggability
        druggability = self.drug_db.get_druggability_score(gene)
        
        # Toxicity (enhanced by OpenTargets, tissue expression)
        toxicity = self._get_toxicity_score(gene)
        
        # Pan-essential penalty
        pan_essential = self._get_pan_essential()
        pan_penalty = 1.0 if gene in pan_essential else 0.0
        
        # Tumor specificity (based on dependency)
        crispr = self.depmap.load_crispr_dependencies()
        cell_lines = self.depmap.get_cell_lines_for_cancer(cancer_type)
        available_lines = [cl for cl in cell_lines if cl in crispr.index]
        
        if gene in crispr.columns and len(available_lines) > 0:
            cancer_dep = crispr.loc[available_lines, gene].mean()
            all_dep = crispr[gene].mean()
            # More negative in cancer = more specific
            specificity = max(0, min(1, (all_dep - cancer_dep)))
        else:
            specificity = 0.5
        
        return NodeCost(
            gene=gene,
            toxicity_score=toxicity,
            tumor_specificity=specificity,
            druggability_score=druggability,
            pan_essential_penalty=pan_penalty,
            base_penalty=1.0
        )

# ============================================================================
# MINIMAL HITTING SET SOLVER
# ============================================================================

class MinimalHittingSetSolver:
    """
    Solve weighted minimal hitting set problem
    
    Given:
    - Set of viability paths P
    - Cost function c(v)
    
    Find:
    - Minimal-cost set T such that every path in P intersects T
    """
    
    def __init__(self, cost_function: CostFunction):
        self.cost_fn = cost_function
    
    def solve(self, paths: List[ViabilityPath], cancer_type: str, 
              max_size: int = 4, min_coverage: float = 0.8) -> List[HittingSet]:
        """
        Find optimal hitting sets using greedy + enumeration
        
        Args:
            paths: Viability paths to cover
            cancer_type: For computing costs
            max_size: Maximum cardinality of hitting set
            min_coverage: Minimum fraction of paths to cover
        
        Returns:
            List of HittingSet solutions sorted by cost
        """
        if len(paths) == 0:
            return []
        
        logger.info(f"Solving hitting set: {len(paths)} paths, max_size={max_size}")
        
        # Extract all genes from paths
        all_genes = set()
        for path in paths:
            all_genes.update(path.nodes)
        
        # Compute costs
        gene_costs = {}
        for gene in all_genes:
            cost_obj = self.cost_fn.compute_cost(gene, cancer_type)
            gene_costs[gene] = cost_obj.total_cost()
        
        solutions = []
        
        # Greedy solution
        greedy = self._solve_greedy(paths, gene_costs, max_size)
        if greedy:
            solutions.append(greedy)
        
        # Exhaustive for small sets
        if len(all_genes) <= 25:
            exhaustive = self._solve_exhaustive(paths, gene_costs, max_size, min_coverage)
            solutions.extend(exhaustive)
        
        # Deduplicate and sort
        seen = set()
        unique_solutions = []
        for sol in solutions:
            if sol.targets not in seen:
                seen.add(sol.targets)
                unique_solutions.append(sol)
        
        unique_solutions.sort(key=lambda x: (len(x.targets), x.total_cost))
        
        logger.info(f"Found {len(unique_solutions)} hitting set solutions")
        return unique_solutions[:20]
    
    def _solve_greedy(self, paths: List[ViabilityPath], gene_costs: Dict[str, float],
                      max_size: int) -> Optional[HittingSet]:
        """Greedy: pick gene with best coverage/cost ratio"""
        
        all_genes = set(gene_costs.keys())
        uncovered = set(paths)
        selected = set()
        total_cost = 0.0
        
        while uncovered and len(selected) < max_size:
            best_gene = None
            best_ratio = -np.inf
            
            for gene in all_genes - selected:
                hits = sum(1 for p in uncovered if gene in p.nodes)
                cost = gene_costs[gene]
                
                if hits > 0:
                    ratio = hits / (cost + 0.01)
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_gene = gene
            
            if best_gene is None:
                break
            
            selected.add(best_gene)
            total_cost += gene_costs[best_gene]
            uncovered = {p for p in uncovered if best_gene not in p.nodes}
        
        coverage = 1.0 - len(uncovered) / len(paths)
        paths_covered = {p.path_id for p in paths if any(g in selected for g in p.nodes)}
        
        return HittingSet(
            targets=frozenset(selected),
            total_cost=total_cost,
            coverage=coverage,
            paths_covered=paths_covered
        )
    
    def _solve_exhaustive(self, paths: List[ViabilityPath], gene_costs: Dict[str, float],
                          max_size: int, min_coverage: float) -> List[HittingSet]:
        """Enumerate all small hitting sets"""
        
        all_genes = list(gene_costs.keys())
        solutions = []
        
        for size in range(1, min(max_size + 1, len(all_genes) + 1)):
            combos = list(combinations(all_genes, size))
            for subset in combos:
                subset_set = set(subset)
                
                # Count coverage
                covered = sum(1 for p in paths if any(g in subset_set for g in p.nodes))
                coverage = covered / len(paths)
                
                if coverage >= min_coverage:
                    total_cost = sum(gene_costs[g] for g in subset)
                    paths_covered = {p.path_id for p in paths if any(g in subset_set for g in p.nodes)}
                    
                    solutions.append(HittingSet(
                        targets=frozenset(subset),
            total_cost=total_cost,
            coverage=coverage,
            paths_covered=paths_covered
                    ))
        
        return solutions

# ============================================================================
# NETWORK TOPOLOGY ANALYZER - Systems Biology X-Node Discovery
# ============================================================================

class XNodeNetworkAnalyzer:
    """
    High-throughput systems biology analysis for finding X-nodes (convergence points)
    Based on network topology metrics used in the PDAC paper:
    - Betweenness centrality (information flow bottlenecks)
    - In-degree (convergence of upstream signals)
    - PageRank (importance in signal propagation)
    - Pathway membership overlap
    """
    
    def __init__(self, omnipath: OmniPathLoader):
        self.omnipath = omnipath
        self._centrality_cache = None
        
    def compute_network_centrality(self) -> Dict[str, Dict[str, float]]:
        """Compute multiple centrality metrics for all genes in network"""
        if self._centrality_cache is not None:
            return self._centrality_cache
            
        network = self.omnipath.load_signaling_network()
        
        # Build adjacency for analysis
        out_edges = defaultdict(set)  # gene -> downstream targets
        in_edges = defaultdict(set)   # gene -> upstream regulators
        
        for _, row in network.iterrows():
            source = row['source']
            target = row['target']
            out_edges[source].add(target)
            in_edges[target].add(source)
        
        all_genes = set(out_edges.keys()) | set(in_edges.keys())
        
        centrality = {}
        for gene in all_genes:
            # In-degree: convergence of upstream signals
            in_degree = len(in_edges[gene])
            
            # Out-degree: broadcast to downstream
            out_degree = len(out_edges[gene])
            
            # Approximated betweenness: genes that connect many upstream to many downstream
            betweenness_approx = in_degree * out_degree
            
            # Hub score: genes that regulate many important genes
            downstream_importance = sum(len(in_edges[t]) for t in out_edges[gene])
            
            centrality[gene] = {
                'in_degree': in_degree,
                'out_degree': out_degree,
                'betweenness_approx': betweenness_approx,
                'downstream_importance': downstream_importance,
                'xnode_score': (in_degree * 2 + out_degree + betweenness_approx * 0.1 + 
                               downstream_importance * 0.05)
            }
        
        self._centrality_cache = centrality
        return centrality
    
    def get_top_xnodes(self, n: int = 50) -> List[Tuple[str, float]]:
        """Get top X-node candidates by network topology"""
        centrality = self.compute_network_centrality()
        
        # Sort by X-node score
        ranked = [(gene, metrics['xnode_score']) for gene, metrics in centrality.items()]
        ranked.sort(key=lambda x: x[1], reverse=True)
        
        return ranked[:n]
    
    def get_pathway_coverage(self, genes: Set[str]) -> Dict[str, float]:
        """Calculate which pathways are covered by a set of genes"""
        # Define major cancer pathways
        PATHWAYS = {
            'RAS_MAPK': {'KRAS', 'NRAS', 'HRAS', 'BRAF', 'RAF1', 'MAP2K1', 'MAP2K2', 'MAPK1', 'MAPK3'},
            'PI3K_AKT_MTOR': {'PIK3CA', 'PIK3CB', 'AKT1', 'AKT2', 'MTOR', 'PTEN', 'TSC1', 'TSC2'},
            'JAK_STAT': {'JAK1', 'JAK2', 'TYK2', 'STAT3', 'STAT5A', 'STAT5B', 'SOCS1', 'SOCS3'},
            'SRC_FAMILY': {'SRC', 'FYN', 'YES1', 'LYN', 'LCK', 'HCK', 'FGR'},
            'CELL_CYCLE': {'CDK1', 'CDK2', 'CDK4', 'CDK6', 'CCND1', 'CCNE1', 'RB1', 'E2F1'},
            'APOPTOSIS': {'BCL2', 'BCL2L1', 'MCL1', 'BAX', 'BAK1', 'BID', 'CASP3', 'CASP8'},
            'P53': {'TP53', 'MDM2', 'CDKN1A', 'CDKN2A', 'BBC3', 'PMAIP1'},
            'WNT': {'WNT1', 'WNT3A', 'FZD1', 'CTNNB1', 'APC', 'GSK3B', 'AXIN1'},
            'NOTCH': {'NOTCH1', 'NOTCH2', 'JAG1', 'DLL1', 'HES1', 'HEY1'},
            'NF_KB': {'NFKB1', 'RELA', 'IKBKB', 'IKBKG', 'NFKBIA'},
            'HIPPO': {'YAP1', 'WWTR1', 'LATS1', 'LATS2', 'MST1', 'MST2'},
            'DNA_REPAIR': {'BRCA1', 'BRCA2', 'ATM', 'ATR', 'PARP1', 'RAD51'},
            'RECEPTOR_TYROSINE_KINASES': {'EGFR', 'ERBB2', 'MET', 'FGFR1', 'FGFR2', 'ALK', 'RET'},
        }
        
        coverage = {}
        for pathway, pathway_genes in PATHWAYS.items():
            covered = len(genes & pathway_genes)
            total = len(pathway_genes)
            coverage[pathway] = covered / total if total > 0 else 0
        
        return coverage
    
    def find_pathway_bridges(self) -> List[str]:
        """Find genes that bridge multiple pathways (key convergence points)"""
        PATHWAYS = {
            'RAS_MAPK': {'KRAS', 'NRAS', 'BRAF', 'MAP2K1', 'MAPK1', 'MAPK3'},
            'PI3K_AKT': {'PIK3CA', 'AKT1', 'MTOR'},
            'JAK_STAT': {'JAK1', 'JAK2', 'STAT3', 'STAT5A'},
            'SRC_FAMILY': {'SRC', 'FYN', 'YES1'},
            'CELL_CYCLE': {'CDK4', 'CDK6', 'CDK2', 'CCND1'},
            'APOPTOSIS': {'BCL2', 'MCL1', 'BAX'},
        }
        
        network = self.omnipath.load_signaling_network()
        gene_pathways = defaultdict(set)
        
        # Map genes to pathways
        for pathway, genes in PATHWAYS.items():
            for gene in genes:
                gene_pathways[gene].add(pathway)
        
        # Check network connections
        for _, row in network.iterrows():
            source = row['source']
            target = row['target']
            
            for pathway, genes in PATHWAYS.items():
                if source in genes:
                    gene_pathways[target].add(f"{pathway}_downstream")
                if target in genes:
                    gene_pathways[source].add(f"{pathway}_upstream")
        
        # Find genes in multiple pathways
        bridges = []
        for gene, pathways in gene_pathways.items():
            # Count unique pathway families
            unique_pathways = set(p.split('_')[0] if '_' in p else p for p in pathways)
            if len(unique_pathways) >= 2:
                bridges.append((gene, len(unique_pathways), pathways))
        
        bridges.sort(key=lambda x: x[1], reverse=True)
        return [b[0] for b in bridges[:30]]


# ============================================================================
# SYNERGY SCORER - Predict Drug Combination Synergies
# ============================================================================

class SynergyScorer:
    """
    Estimate synergy between drug targets based on:
    - Pathway complementarity (hitting independent pathways)
    - Synthetic lethality relationships
    - Known clinical combination data
    - Network distance
    """
    
    # Known synergistic combinations from clinical trials
    KNOWN_SYNERGIES = {
        frozenset({'BRAF', 'MAP2K1'}): 0.9,  # BRAF + MEK (clinical standard)
        frozenset({'CDK4', 'EGFR'}): 0.8,
        frozenset({'SRC', 'STAT3'}): 0.85,   # From PDAC paper
        frozenset({'FYN', 'STAT3'}): 0.85,   # From PDAC paper
        frozenset({'PIK3CA', 'MTOR'}): 0.7,
        frozenset({'BCL2', 'MCL1'}): 0.9,    # Double BCL2 family
        frozenset({'CDK4', 'CDK6'}): 0.6,    # Same target class
        frozenset({'JAK1', 'STAT3'}): 0.8,
        frozenset({'EGFR', 'MET'}): 0.85,    # Bypass resistance
        frozenset({'KRAS', 'SRC'}): 0.75,
        frozenset({'KRAS', 'STAT3'}): 0.8,
        frozenset({'SRC', 'FYN'}): 0.7,      # SFK family
        frozenset({'SRC', 'FYN', 'STAT3'}): 0.95,  # Paper's triple
    }
    
    # Pathway assignments for complementarity scoring
    PATHWAY_ASSIGNMENT = {
        'KRAS': 'RAS', 'NRAS': 'RAS', 'HRAS': 'RAS', 'BRAF': 'MAPK', 'MAP2K1': 'MAPK',
        'PIK3CA': 'PI3K', 'AKT1': 'PI3K', 'MTOR': 'PI3K', 'PTEN': 'PI3K',
        'JAK1': 'JAK_STAT', 'JAK2': 'JAK_STAT', 'STAT3': 'JAK_STAT', 'STAT5A': 'JAK_STAT',
        'SRC': 'SRC', 'FYN': 'SRC', 'YES1': 'SRC', 'LYN': 'SRC',
        'CDK4': 'CELL_CYCLE', 'CDK6': 'CELL_CYCLE', 'CDK2': 'CELL_CYCLE', 'CCND1': 'CELL_CYCLE',
        'BCL2': 'APOPTOSIS', 'MCL1': 'APOPTOSIS', 'BCL2L1': 'APOPTOSIS',
        'EGFR': 'RTK', 'ERBB2': 'RTK', 'MET': 'RTK', 'ALK': 'RTK',
        'TP53': 'P53', 'MDM2': 'P53', 'CDKN2A': 'P53',
        'CTNNB1': 'WNT', 'APC': 'WNT', 'GSK3B': 'WNT',
    }
    
    def __init__(self, omnipath: OmniPathLoader):
        self.omnipath = omnipath
        
    def compute_synergy_score(self, genes: Set[str]) -> float:
        """
        Compute synergy score for a combination of genes
        
        Returns:
            Score 0-1 where higher = more synergistic
        """
        if len(genes) < 2:
            return 0.0
        
        genes_frozen = frozenset(genes)
        
        # Check known synergies
        if genes_frozen in self.KNOWN_SYNERGIES:
            return self.KNOWN_SYNERGIES[genes_frozen]
        
        # Check pairwise known synergies
        known_pair_score = 0.0
        pair_count = 0
        for g1, g2 in combinations(genes, 2):
            pair = frozenset({g1, g2})
            if pair in self.KNOWN_SYNERGIES:
                known_pair_score += self.KNOWN_SYNERGIES[pair]
                pair_count += 1
        
        if pair_count > 0:
            known_pair_score /= pair_count
        
        # Pathway complementarity score
        pathways = set()
        for gene in genes:
            if gene in self.PATHWAY_ASSIGNMENT:
                pathways.add(self.PATHWAY_ASSIGNMENT[gene])
        
        # More diverse pathways = better complementarity
        pathway_diversity = len(pathways) / max(len(genes), 1)
        
        # Combine scores
        synergy = (
            known_pair_score * 0.4 +
            pathway_diversity * 0.6
        )
        
        return min(1.0, synergy)
    
    def get_synergistic_partners(self, gene: str, candidates: Set[str], top_n: int = 10) -> List[Tuple[str, float]]:
        """Find most synergistic partners for a gene"""
        scores = []
        for candidate in candidates:
            if candidate != gene:
                synergy = self.compute_synergy_score({gene, candidate})
                scores.append((candidate, synergy))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_n]


# ============================================================================
# RESISTANCE PROBABILITY ESTIMATOR
# ============================================================================

class ResistanceProbabilityEstimator:
    """
    Estimate probability of resistance emergence
    Based on:
    - Number of alternative pathways available
    - Mutation frequency of bypass genes
    - Historical resistance mechanisms
    """
    
    # Known resistance mechanisms
    RESISTANCE_MECHANISMS = {
        'EGFR': ['MET', 'ERBB2', 'KRAS', 'PIK3CA'],  # EGFR TKI resistance
        'BRAF': ['MAP2K1', 'NRAS', 'PIK3CA'],         # BRAF inhibitor resistance
        'ALK': ['EGFR', 'MET', 'SRC'],
        'KRAS': ['PIK3CA', 'BRAF', 'MET'],
        'BCL2': ['MCL1', 'BCL2L1'],
        'CDK4': ['CDK6', 'CCNE1', 'CDK2'],
        'MTOR': ['PIK3CA', 'AKT1', 'MAPK1'],
        'SRC': ['FYN', 'YES1', 'LYN'],               # SFK compensation
        'STAT3': ['STAT5A', 'NFKB1'],                # Transcription factor bypass
    }
    
    def __init__(self, omnipath: OmniPathLoader, depmap: DepMapLoader):
        self.omnipath = omnipath
        self.depmap = depmap
        
    def estimate_resistance_probability(self, targets: Set[str], cancer_type: str) -> float:
        """
        Estimate probability of resistance for a target combination
        
        Lower is better (more likely to prevent resistance)
        """
        if len(targets) == 0:
            return 1.0
        
        # Count uncovered resistance mechanisms
        all_resistance_genes = set()
        for target in targets:
            if target in self.RESISTANCE_MECHANISMS:
                all_resistance_genes.update(self.RESISTANCE_MECHANISMS[target])
        
        # Subtract genes already targeted
        uncovered = all_resistance_genes - targets
        
        # Base probability from uncovered mechanisms
        base_prob = len(uncovered) / (len(all_resistance_genes) + 1)
        
        # Adjust based on number of targets (more targets = harder to develop resistance)
        n_targets = len(targets)
        target_modifier = 1.0 / (1.0 + n_targets * 0.3)
        
        # Check if we're targeting compensatory family members
        src_family = {'SRC', 'FYN', 'YES1', 'LYN', 'LCK'}
        bcl2_family = {'BCL2', 'MCL1', 'BCL2L1'}
        
        family_coverage_bonus = 0.0
        if len(targets & src_family) >= 2:
            family_coverage_bonus -= 0.2
        if len(targets & bcl2_family) >= 2:
            family_coverage_bonus -= 0.2
        
        resistance_prob = base_prob * target_modifier + family_coverage_bonus
        
        return max(0.0, min(1.0, resistance_prob))


# ============================================================================
# TRIPLE COMBINATION FINDER - Main Systems Biology Engine
# ============================================================================

@dataclass
class TripleCombination:
    """Triple drug combination with comprehensive scoring"""
    targets: Tuple[str, str, str]
    total_cost: float
    synergy_score: float
    resistance_score: float  # Lower is better
    pathway_coverage: Dict[str, float]
    coverage: float  # Fraction of viability paths covered
    druggable_count: int  # How many have approved drugs
    combined_score: float  # Overall score (lower is better for therapeutic potential)
    drug_info: Dict[str, Optional[DrugTarget]] = field(default_factory=dict)
    combo_tox_score: float = 0.0  # Combination-level toxicity (DDI, overlapping toxicities)
    combo_tox_details: Dict = field(default_factory=dict)  # Breakdown of combo toxicity
    
    def __lt__(self, other):
        return self.combined_score < other.combined_score


class TripleCombinationFinder:
    """
    High-throughput systems biology approach to find optimal triple combinations
    
    Methodology from the PDAC paper:
    1. Identify X-nodes (network convergence points)
    2. Score synergistic interactions
    3. Minimize resistance probability
    4. Maximize pathway coverage
    5. Prioritize druggable targets
    """
    
    def __init__(self, depmap: DepMapLoader, omnipath: OmniPathLoader, drug_db: DrugTargetDB,
                 toxicity_cache_dir: Optional[str] = None):
        self.depmap = depmap
        self.omnipath = omnipath
        self.drug_db = drug_db
        self.network_analyzer = XNodeNetworkAnalyzer(omnipath)
        self.synergy_scorer = SynergyScorer(omnipath)
        self.resistance_estimator = ResistanceProbabilityEstimator(omnipath, depmap)
        self.cost_fn = CostFunction(depmap, drug_db, toxicity_cache_dir=toxicity_cache_dir)
        
    def find_triple_combinations(self, 
                                  paths: List[ViabilityPath], 
                                  cancer_type: str,
                                  top_n: int = 20,
                                  min_coverage: float = 0.7,
                                  prefer_druggable: bool = True) -> List[TripleCombination]:
        """
        Find optimal triple combinations using systems biology scoring
        
        Args:
            paths: Viability paths to cover
            cancer_type: For computing cancer-specific costs
            top_n: Number of top combinations to return
            min_coverage: Minimum fraction of paths to cover
            prefer_druggable: Prioritize druggable targets
        
        Returns:
            List of TripleCombination objects sorted by combined_score
        """
        if len(paths) == 0:
            logger.warning("No viability paths provided")
            return []
        
        logger.info(f"Finding triple combinations for {cancer_type} ({len(paths)} paths)")
        
        # Extract all candidate genes from paths
        all_genes = set()
        for path in paths:
            all_genes.update(path.nodes)
        
        # Get top X-nodes from network analysis
        xnodes = self.network_analyzer.get_top_xnodes(n=30)
        xnode_genes = {g for g, _ in xnodes}
        
        # Get pathway bridge genes
        bridge_genes = set(self.network_analyzer.find_pathway_bridges())
        
        # Prioritize candidates: X-nodes > bridges > path genes
        priority_genes = (xnode_genes & all_genes) | (bridge_genes & all_genes)
        
        if len(priority_genes) < 10:
            # Add essential genes from paths
            gene_frequency = defaultdict(int)
            for path in paths:
                for gene in path.nodes:
                    gene_frequency[gene] += 1
            
            frequent_genes = sorted(gene_frequency.items(), key=lambda x: x[1], reverse=True)
            priority_genes.update(g for g, _ in frequent_genes[:30])
        
        # Filter to druggable if preferred
        if prefer_druggable:
            druggable = {g for g in priority_genes if self.drug_db.get_druggability_score(g) >= 0.4}
            if len(druggable) >= 6:
                candidate_genes = list(druggable)[:50]
            else:
                # Mix druggable and non-druggable
                candidate_genes = list(druggable) + [g for g in priority_genes if g not in druggable][:30]
        else:
            candidate_genes = list(priority_genes)[:50]
        
        logger.info(f"Evaluating combinations from {len(candidate_genes)} candidate genes")
        
        # Compute individual costs
        gene_costs = {}
        for gene in candidate_genes:
            cost_obj = self.cost_fn.compute_cost(gene, cancer_type)
            gene_costs[gene] = cost_obj.total_cost()
        
        # Enumerate and score triple combinations
        triple_combinations = []
        
        combos = list(combinations(candidate_genes, 3))
        for triple in tqdm(combos, desc="Scoring triples", leave=False, mininterval=0.5, miniters=10):
            triple_set = set(triple)
            
            # Calculate coverage
            covered = sum(1 for p in paths if any(g in triple_set for g in p.nodes))
            coverage = covered / len(paths)
            
            if coverage < min_coverage:
                continue
            
            # Total cost
            total_cost = sum(gene_costs.get(g, 1.0) for g in triple)
            
            # Synergy score
            synergy = self.synergy_scorer.compute_synergy_score(triple_set)
            
            # Resistance probability
            resistance = self.resistance_estimator.estimate_resistance_probability(triple_set, cancer_type)
            
            # Pathway coverage
            pathway_cov = self.network_analyzer.get_pathway_coverage(triple_set)
            
            # Count druggable targets
            druggable_count = sum(1 for g in triple if self.drug_db.get_druggability_score(g) >= 0.6)
            
            # Get drug info
            drug_info = {g: self.drug_db.get_drug_info(g) for g in triple}
            
            # Compute combination-level toxicity (DDI, overlapping toxicities, FAERS signals)
            combo_tox_score = 0.0
            combo_tox_details = {}
            try:
                from alin.toxicity import compute_combo_toxicity_score
                combo_tox_result = compute_combo_toxicity_score(list(triple), use_faers=False)
                combo_tox_score = combo_tox_result['combo_tox_score']
                combo_tox_details = combo_tox_result
            except ImportError:
                pass
            
            # Compute perturbation response score (bonus for targeting feedback/resistance genes)
            perturbation_bonus = 0.0
            try:
                from alin.perturbation import score_combination_by_perturbation
                essential_genes = set()
                for p in paths:
                    essential_genes.update(p.nodes)
                pert_result = score_combination_by_perturbation(list(triple), essential_genes)
                # Bonus if combination targets feedback genes (resistance prevention)
                perturbation_bonus = pert_result.get('feedback_coverage', 0) * 0.1
            except ImportError:
                pass
            
            # Combined score (lower is better)
            # Weights: cost (0.22), synergy (-0.18), resistance (0.18), coverage (-0.14), combo_tox (0.18)
            combined_score = (
                total_cost * 0.22 +
                (1 - synergy) * 0.18 +  # Invert synergy (higher synergy = better)
                resistance * 0.18 +
                (1 - coverage) * 0.14 +
                combo_tox_score * 0.18 -  # Penalty for combination toxicity
                druggable_count * 0.1 -   # Bonus for druggability
                perturbation_bonus        # Bonus for targeting feedback genes
            )
            
            triple_combinations.append(TripleCombination(
                targets=tuple(sorted(triple)),
                total_cost=total_cost,
                synergy_score=synergy,
                resistance_score=resistance,
                pathway_coverage=pathway_cov,
                coverage=coverage,
                druggable_count=druggable_count,
                combined_score=combined_score,
                drug_info=drug_info,
                combo_tox_score=combo_tox_score,
                combo_tox_details=combo_tox_details,
            ))
        
        # Sort by combined score
        triple_combinations.sort(key=lambda x: x.combined_score)
        
        logger.info(f"Found {len(triple_combinations)} valid triple combinations")
        
        return triple_combinations[:top_n]
    
    def find_best_triple_for_pathways(self, 
                                       paths: List[ViabilityPath],
                                       cancer_type: str,
                                       target_pathways: List[str]) -> Optional[TripleCombination]:
        """Find best triple that covers specific pathways"""
        all_triples = self.find_triple_combinations(paths, cancer_type, top_n=100)
        
        for triple in all_triples:
            covered_pathways = [p for p, cov in triple.pathway_coverage.items() if cov > 0]
            if all(tp in covered_pathways for tp in target_pathways):
                return triple
        
        return all_triples[0] if all_triples else None
    
    def generate_triple_report(self, triple: TripleCombination, cancer_type: str) -> str:
        """Generate detailed report for a triple combination"""
        lines = [
            f"{'='*80}",
            f"TRIPLE COMBINATION REPORT: {cancer_type}",
            f"{'='*80}",
            f"",
            f"TARGETS: {', '.join(triple.targets)}",
            f"",
            f"SCORES:",
            f"  Combined Score: {triple.combined_score:.3f} (lower is better)",
            f"  Synergy Score: {triple.synergy_score:.2f} (higher is better)",
            f"  Resistance Score: {triple.resistance_score:.2f} (lower is better)",
            f"  Combo Toxicity: {triple.combo_tox_score:.2f} (lower is better)",
            f"  Path Coverage: {triple.coverage*100:.1f}%",
            f"  Total Cost: {triple.total_cost:.2f}",
            f"  Druggable Targets: {triple.druggable_count}/3",
            f"",
            f"DRUG INFORMATION:",
        ]
        
        for target in triple.targets:
            drug_info = triple.drug_info.get(target)
            if drug_info and drug_info.available_drugs:
                drugs_str = ', '.join(drug_info.available_drugs[:3])
                lines.append(f"  {target}:")
                lines.append(f"    Drugs: {drugs_str}")
                lines.append(f"    Stage: {drug_info.clinical_stage}")
                if drug_info.known_toxicities:
                    lines.append(f"    Toxicities: {', '.join(drug_info.known_toxicities[:3])}")
            else:
                lines.append(f"  {target}: No approved drugs (research target)")
        
        # Add combo toxicity details
        lines.extend([f"", f"COMBINATION TOXICITY ASSESSMENT:"])
        if triple.combo_tox_details:
            details = triple.combo_tox_details
            if details.get('ddi_penalties'):
                lines.append(f"  Known Drug-Drug Interactions:")
                for ddi in details['ddi_penalties']:
                    lines.append(f"    - {ddi['drugs'][0]} + {ddi['drugs'][1]}: {ddi['severity']} ({ddi['mechanism']})")
            else:
                lines.append(f"  No known major drug-drug interactions")
            
            if details.get('overlapping_toxicities'):
                lines.append(f"  Overlapping Toxicity Classes:")
                for tox_class, count in details['overlapping_toxicities'].items():
                    lines.append(f"    - {tox_class}: {count} drugs share this toxicity")
            else:
                lines.append(f"  No major overlapping toxicity classes")
            
            comp = details.get('component_scores', {})
            lines.append(f"  Component Scores: DDI={comp.get('ddi', 0):.2f}, Overlap={comp.get('overlap', 0):.2f}")
        else:
            lines.append(f"  Combo toxicity data unavailable")
        
        lines.extend([
            f"",
            f"PATHWAY COVERAGE:",
        ])
        
        for pathway, cov in sorted(triple.pathway_coverage.items(), key=lambda x: x[1], reverse=True):
            if cov > 0:
                lines.append(f"  {pathway}: {cov*100:.0f}%")
        
        lines.extend([
            f"",
            f"RATIONALE:",
            f"  This triple combination targets multiple signaling nodes to achieve",
            f"  complete pathway inhibition while minimizing resistance probability.",
            f"  The combination covers {sum(1 for c in triple.pathway_coverage.values() if c > 0)} pathways",
            f"  with {triple.druggable_count} clinically druggable targets.",
            f"{'='*80}",
        ])
        
        return '\n'.join(lines)


# ============================================================================
# VALIDATION INTEGRATION
# ============================================================================

class XNodeValidationIntegrator:
    """
    Integrates X-Node Discovery with Validation Module
    Runs validation on discovered combinations and generates combined reports
    """
    
    def __init__(self, validation_data_dir: str = "./validation_data"):
        self.validation_data_dir = validation_data_dir
        self.validator = None
        
        if VALIDATION_AVAILABLE:
            try:
                self.validator = ValidationEngine(data_dir=validation_data_dir)
                logger.info("Validation engine initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize validation engine: {e}")
        else:
            logger.warning("Validation module not available")
    
    def validate_analysis(self, analysis: 'CancerTypeAnalysis', 
                          enable_api_calls: bool = True) -> Optional[CombinationValidation]:
        """
        Validate the recommended combination from an analysis
        
        Args:
            analysis: CancerTypeAnalysis object
            enable_api_calls: Whether to make external API calls
            
        Returns:
            CombinationValidation object or None
        """
        if not self.validator:
            logger.warning("Validation engine not available")
            return None
        
        if not analysis.recommended_combination:
            logger.warning(f"No combination to validate for {analysis.cancer_type}")
            return None
        
        logger.info(f"Validating combination for {analysis.cancer_type}: {analysis.recommended_combination}")
        
        validation = self.validator.validate_combination(
            targets=analysis.recommended_combination,
            cancer_type=analysis.cancer_type,
            enable_api_calls=enable_api_calls
        )
        
        return validation
    
    def validate_triple(self, triple: 'TripleCombination', cancer_type: str,
                        enable_api_calls: bool = True) -> Optional[CombinationValidation]:
        """
        Validate a triple combination
        
        Args:
            triple: TripleCombination object
            cancer_type: Cancer type name
            enable_api_calls: Whether to make external API calls
            
        Returns:
            CombinationValidation object or None
        """
        if not self.validator:
            logger.warning("Validation engine not available")
            return None
        
        logger.info(f"Validating triple for {cancer_type}: {triple.targets}")
        
        validation = self.validator.validate_combination(
            targets=list(triple.targets),
            cancer_type=cancer_type,
            enable_api_calls=enable_api_calls
        )
        
        return validation
    
    def validate_all_results(self, results: Dict[str, 'CancerTypeAnalysis'],
                             enable_api_calls: bool = True,
                             validate_triples: bool = True) -> Dict[str, CombinationValidation]:
        """
        Validate all combinations from pan-cancer analysis
        
        Args:
            results: Dictionary of cancer_type -> CancerTypeAnalysis
            enable_api_calls: Whether to make external API calls
            validate_triples: Whether to validate triple combinations specifically
            
        Returns:
            Dictionary of cancer_type -> CombinationValidation
        """
        if not self.validator:
            logger.warning("Validation engine not available")
            return {}
        
        validations = {}
        
        items = list(results.items())
        for cancer_type, analysis in tqdm(items, desc="Validating combinations", unit="cancer"):
            # Prefer validating triples if available
            if validate_triples and analysis.best_triple:
                validation = self.validate_triple(
                    analysis.best_triple, 
                    cancer_type,
                    enable_api_calls=enable_api_calls
                )
            elif analysis.recommended_combination:
                validation = self.validate_analysis(
                    analysis, 
                    enable_api_calls=enable_api_calls
                )
            else:
                validation = None
            
            if validation:
                validations[cancer_type] = validation
        
        return validations
    
    def generate_combined_report(self, analysis: 'CancerTypeAnalysis', 
                                 validation: CombinationValidation) -> str:
        """Generate combined X-Node + Validation report"""
        
        report = f"""
{'='*80}
COMBINED X-NODE DISCOVERY & VALIDATION REPORT
{'='*80}
Cancer Type: {analysis.cancer_type}
Cell Lines: {analysis.n_cell_lines}
Lineage: {analysis.lineage}

{'='*80}
X-NODE DISCOVERY RESULTS
{'='*80}
Recommended Combination: {', '.join(analysis.recommended_combination) if analysis.recommended_combination else 'None'}
"""
        
        if analysis.best_triple:
            bt = analysis.best_triple
            report += f"""
Best Triple Combination: {', '.join(bt.targets)}
  Synergy Score: {bt.synergy_score:.2f}
  Resistance Score: {bt.resistance_score:.2f}
  Path Coverage: {bt.coverage*100:.1f}%
  Druggable Targets: {bt.druggable_count}/3
"""
        
        report += f"""
{'='*80}
VALIDATION RESULTS
{'='*80}
Overall Validation Score: {validation.validation_score:.3f}
Confidence Level: {validation.confidence_level}

Literature Evidence:
  PubMed Publications: {validation.pubmed_mentions}
  Clinical Trials: {len(validation.clinical_trials)}

Network Evidence:
  Protein Interaction Confidence: {validation.ppi_confidence:.3f}
  Pathway Overlap Score: {validation.pathway_overlap:.3f}

Detailed Evidence ({len(validation.all_evidence)} sources):
"""
        
        for i, ev in enumerate(sorted(validation.all_evidence, key=lambda x: -x.score)[:5], 1):
            report += f"  {i}. [{ev.source}] {ev.details} (score: {ev.score:.2f})\n"
        
        report += f"""
{'='*80}
CLINICAL ACTIONABILITY ASSESSMENT
{'='*80}
"""
        
        # Assess actionability
        drug_db = DrugTargetDB()
        if analysis.recommended_combination:
            approved_drugs = []
            research_targets = []
            for target in analysis.recommended_combination:
                info = drug_db.get_drug_info(target)
                if info and info.available_drugs:
                    approved_drugs.append(f"{target}: {info.available_drugs[0]} ({info.clinical_stage})")
                else:
                    research_targets.append(target)
            
            report += f"Targets with Approved Drugs ({len(approved_drugs)}):\n"
            for d in approved_drugs:
                report += f"  • {d}\n"
            
            if research_targets:
                report += f"\nResearch Targets (no approved drugs): {', '.join(research_targets)}\n"
            
            # Overall actionability
            actionability = len(approved_drugs) / len(analysis.recommended_combination)
            if actionability >= 0.8 and validation.validation_score >= 0.5:
                status = "HIGH - Ready for clinical consideration"
            elif actionability >= 0.5 or validation.validation_score >= 0.5:
                status = "MEDIUM - Promising but needs further validation"
            else:
                status = "LOW - Early research stage"
            
            report += f"\nClinical Actionability: {status}\n"
        
        report += f"{'='*80}\n"
        
        return report


# ============================================================================
# PAN-CANCER ANALYSIS ENGINE
# ============================================================================

class PanCancerXNodeAnalyzer:
    """Main analysis engine for all cancer types"""
    
    def __init__(self, data_dir: str = "./depmap_data", validation_data_dir: str = "./validation_data",
                 toxicity_cache_dir: Optional[str] = None):
        self.depmap = DepMapLoader(data_dir)
        self.omnipath = OmniPathLoader(data_dir)
        self.drug_db = DrugTargetDB()
        self.path_inference = ViabilityPathInference(self.depmap, self.omnipath)
        self.cost_fn = CostFunction(self.depmap, self.drug_db, toxicity_cache_dir=toxicity_cache_dir)
        self.solver = MinimalHittingSetSolver(self.cost_fn)
        self.triple_finder = TripleCombinationFinder(
            self.depmap, self.omnipath, self.drug_db,
            toxicity_cache_dir=toxicity_cache_dir
        )
        self.validation_integrator = XNodeValidationIntegrator(validation_data_dir)
        
    def analyze_cancer_type(self, cancer_type: str) -> CancerTypeAnalysis:
        """Run complete X-node analysis for one cancer type"""
        
        cancer_type_normalized = normalize_cancer_type(cancer_type)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Analyzing {cancer_type_normalized}")
        logger.info(f"{'='*80}")
        
        # Get cell lines
        cell_lines = self.depmap.get_cell_lines_for_cancer(cancer_type_normalized)
        n_cell_lines = len(cell_lines)
        
        if n_cell_lines == 0:
            logger.warning(f"No cell lines found for '{cancer_type_normalized}'")
            # List available cancer types
            available = self.depmap.get_available_cancer_types()[:10]
            logger.info(f"Available cancer types (top 10): {[a[0] for a in available]}")
            
            return CancerTypeAnalysis(
                cancer_type=cancer_type_normalized,
                lineage="Unknown",
                n_cell_lines=0,
                cell_line_ids=[],
                driver_mutations={},
                essential_genes={},
                viability_paths=[],
                minimal_hitting_sets=[],
                top_x_node_sets=[],
                recommended_combination=None,
                statistics={'error': f'No cell lines found for {cancer_type_normalized}'}
            )
        
        _progress(f"{cancer_type_normalized}: {n_cell_lines} cell lines")
        
        # Get lineage
        lineage_df = self.depmap.load_lineage_annotations()
        available_lines = [cl for cl in cell_lines if cl in lineage_df.index]
        if available_lines:
            lineage = lineage_df.loc[available_lines[0], 'OncotreeLineage']
        else:
            lineage = "Unknown"
        
        # Get essential genes
        crispr = self.depmap.load_crispr_dependencies()
        available_crispr = [cl for cl in cell_lines if cl in crispr.index]
        
        if len(available_crispr) > 0:
            crispr_subset = crispr.loc[available_crispr]
            mean_dep = crispr_subset.mean(axis=0).sort_values()
            essential_genes = mean_dep[mean_dep < -0.5].head(50).to_dict()
            
            # Get driver mutations from subtype features if available
            subtype_df = self.depmap.load_subtype_features()
            driver_mutations = {}
            if subtype_df is not None:
                driver_cols = [c for c in subtype_df.columns if any(
                    d in c for d in ['KRAS', 'BRAF', 'TP53', 'EGFR', 'PIK3CA', 'PTEN', 'CDKN2A']
                )]
                if driver_cols and len(available_crispr) > 0:
                    available_subtype = [cl for cl in available_crispr if cl in subtype_df.index]
                    if available_subtype:
                        for col in driver_cols:
                            freq = subtype_df.loc[available_subtype, col].mean()
                            if freq > 0:
                                driver_mutations[col] = float(freq)
        else:
            essential_genes = {}
            driver_mutations = {}
        
        # Infer viability paths
        _progress("Inferring viability paths (essential + signaling + cancer-specific)", step="")
        all_paths = self.path_inference.infer_all_paths(cancer_type_normalized)
        _progress(f"Inferred {len(all_paths)} viability paths", step="done")
        
        if len(all_paths) == 0:
            logger.warning(f"No viability paths found for {cancer_type_normalized}")
            return CancerTypeAnalysis(
                cancer_type=cancer_type_normalized,
                lineage=lineage,
                n_cell_lines=n_cell_lines,
                cell_line_ids=cell_lines,
                driver_mutations=driver_mutations,
                essential_genes=essential_genes,
                viability_paths=[],
                minimal_hitting_sets=[],
                top_x_node_sets=[],
                recommended_combination=None,
                statistics={'warning': 'No viability paths found'}
            )
        
        # Solve minimal hitting set
        _progress("Solving minimal hitting set", step="")
        hitting_sets = self.solver.solve(all_paths, cancer_type_normalized, max_size=4)
        _progress(f"Found {len(hitting_sets)} hitting set solutions", step="done")
        
        # Extract top combinations
        top_sets = [(hs.targets, hs.total_cost) for hs in hitting_sets[:10]]
        
        # Find triple combinations using systems biology approach
        _progress("Scoring triple combinations", step="")
        triple_combinations = self.triple_finder.find_triple_combinations(
            all_paths, cancer_type_normalized, top_n=10, min_coverage=0.5
        )
        best_triple = triple_combinations[0] if triple_combinations else None
        if triple_combinations:
            _progress(f"Best triple: {' + '.join(sorted(best_triple.targets))}", step="done")
        
        # Best recommendation (prefer triple from systems biology analysis)
        if best_triple:
            recommended = list(best_triple.targets)
        elif hitting_sets:
            # Fallback to hitting set if no triples found
            good_sets = [hs for hs in hitting_sets if len(hs.targets) <= 3 and hs.coverage >= 0.7]
            if good_sets:
                best = min(good_sets, key=lambda x: x.total_cost)
            else:
                best = hitting_sets[0]
            recommended = list(best.targets)
        else:
            recommended = None
        
        return CancerTypeAnalysis(
            cancer_type=cancer_type_normalized,
            lineage=lineage,
            n_cell_lines=n_cell_lines,
            cell_line_ids=cell_lines,
            driver_mutations=driver_mutations,
            essential_genes=essential_genes,
            viability_paths=all_paths,
            minimal_hitting_sets=hitting_sets[:10],
            top_x_node_sets=top_sets,
            recommended_combination=recommended,
            triple_combinations=triple_combinations,
            best_triple=best_triple,
            statistics={
                'n_paths': len(all_paths),
                'n_unique_genes': len(set(g for p in all_paths for g in p.nodes)),
                'best_coverage': hitting_sets[0].coverage if hitting_sets else 0,
                'n_triple_combinations': len(triple_combinations),
                'best_triple_synergy': best_triple.synergy_score if best_triple else 0,
                'best_triple_resistance': best_triple.resistance_score if best_triple else 1
            }
        )
    
    def analyze_all_cancers(self, top_n: int = 20) -> Dict[str, CancerTypeAnalysis]:
        """Run analysis across top cancer types by cell line count"""
        
        cancer_types = self.depmap.get_available_cancer_types()
        logger.info(f"Found {len(cancer_types)} cancer types in DepMap")
        
        # Sort by cell line count and take top N
        cancer_types = sorted(cancer_types, key=lambda x: -x[1])[:top_n]
        
        results = {}
        valid_cancers = [(ct, c) for ct, c in cancer_types if pd.notna(ct)]
        n_total = len(valid_cancers)
        print(f"\n  Analyzing {n_total} cancer types (progress below):\n", flush=True)
        
        # Reduce logging verbosity during batch analysis
        original_level = logger.level
        logger.setLevel(logging.WARNING)
        
        for idx, (cancer_type, count) in enumerate(tqdm(valid_cancers, desc="Cancers", unit="", leave=True), start=1):
            try:
                print(f"\n  [{idx}/{n_total}] {cancer_type}", flush=True)
                analysis = self.analyze_cancer_type(cancer_type)
                results[cancer_type] = analysis
            except Exception as e:
                print(f"    (skipped: {e})", flush=True)
                continue
        
        # Restore logging
        logger.setLevel(original_level)
        
        return results

# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_cancer_report(analysis: CancerTypeAnalysis) -> str:
    """Generate detailed clinical report for any cancer type"""
    
    report = f"""
{'='*80}
X-NODE TARGET DISCOVERY REPORT
Cancer Type: {analysis.cancer_type}
Lineage: {analysis.lineage}
Cell Lines Analyzed: {analysis.n_cell_lines}
Viability Paths Identified: {len(analysis.viability_paths)}
{'='*80}

DRIVER MUTATION LANDSCAPE:
{'-'*80}
"""
    
    if analysis.driver_mutations:
        for gene, freq in sorted(analysis.driver_mutations.items(), key=lambda x: -x[1])[:10]:
            report += f"{gene}: {freq:.1%} of cell lines\n"
    else:
        report += "No mutation data available\n"
    
    report += f"""
{'='*80}
TOP ESSENTIAL GENES (Cancer-Specific):
{'-'*80}
"""
    
    if analysis.essential_genes:
        for gene, score in list(analysis.essential_genes.items())[:10]:
            report += f"{gene}: {score:.3f} (dependency score)\n"
    else:
        report += "No CRISPR data available\n"
    
    report += f"""
{'='*80}
RECOMMENDED X-NODE TARGET COMBINATION:
{'-'*80}
"""
    
    if analysis.recommended_combination:
        report += f"Optimal {len(analysis.recommended_combination)}-node combination:\n"
        for gene in analysis.recommended_combination:
            # Get drug info
            drug_db = DrugTargetDB()
            info = drug_db.get_drug_info(gene)
            if info:
                drugs = ', '.join(info.available_drugs[:3]) if info.available_drugs else 'No approved drugs'
                report += f"  • {gene} ({info.clinical_stage}): {drugs}\n"
            else:
                report += f"  • {gene}\n"
        
        if analysis.minimal_hitting_sets:
            best_hs = analysis.minimal_hitting_sets[0]
            report += f"\nCost Score: {best_hs.total_cost:.2f}\n"
            report += f"Path Coverage: {best_hs.coverage:.1%}\n"
            report += f"Paths Covered: {len(best_hs.paths_covered)}/{len(analysis.viability_paths)}\n"
    else:
        report += "No viable combination found.\n"
    
    report += f"""
{'='*80}
ALTERNATIVE X-NODE COMBINATIONS:
{'-'*80}
"""
    
    for i, (targets, cost) in enumerate(analysis.top_x_node_sets[1:6], 1):
        report += f"\nOption {i}: {{{', '.join(sorted(targets))}}}\n"
        report += f"  Cost: {cost:.2f}, Size: {len(targets)} targets\n"
    
    report += f"""
{'='*80}
SYSTEMS BIOLOGY TRIPLE COMBINATIONS:
{'-'*80}
"""
    
    if analysis.best_triple:
        bt = analysis.best_triple
        report += f"""
BEST TRIPLE COMBINATION: {', '.join(bt.targets)}
  Combined Score: {bt.combined_score:.3f} (lower is better)
  Synergy Score: {bt.synergy_score:.2f} (higher is better)
  Resistance Score: {bt.resistance_score:.2f} (lower is better)
  Path Coverage: {bt.coverage*100:.1f}%
  Druggable Targets: {bt.druggable_count}/3

Drug Details:
"""
        drug_db = DrugTargetDB()
        for target in bt.targets:
            info = drug_db.get_drug_info(target)
            if info and info.available_drugs:
                drugs = ', '.join(info.available_drugs[:3])
                report += f"  • {target} ({info.clinical_stage}): {drugs}\n"
            else:
                report += f"  • {target}: Research target (no approved drugs)\n"
        
        report += "\nPathway Coverage:\n"
        for pathway, cov in sorted(bt.pathway_coverage.items(), key=lambda x: x[1], reverse=True):
            if cov > 0:
                report += f"  {pathway}: {cov*100:.0f}%\n"
        
        if len(analysis.triple_combinations) > 1:
            report += "\nAlternative Triple Combinations:\n"
            for i, tc in enumerate(analysis.triple_combinations[1:5], 1):
                report += f"  {i}. {', '.join(tc.targets)} (score: {tc.combined_score:.3f}, synergy: {tc.synergy_score:.2f})\n"
    else:
        report += "No triple combinations found (insufficient data or coverage)\n"
    
    report += f"""
{'='*80}
VIABILITY PATH SUMMARY:
{'-'*80}
Total paths: {len(analysis.viability_paths)}
"""
    
    if analysis.viability_paths:
        unique_genes = set(g for p in analysis.viability_paths for g in p.nodes)
        report += f"Unique gene nodes: {len(unique_genes)}\n"
        
        # Group by path type
        path_types = {}
        for p in analysis.viability_paths:
            ptype = p.path_type
            if ptype not in path_types:
                path_types[ptype] = 0
            path_types[ptype] += 1
        
        report += "\nPaths by type:\n"
        for ptype, count in path_types.items():
            report += f"  {ptype}: {count}\n"
        
        report += "\nSample paths:\n"
    for path in analysis.viability_paths[:5]:
            nodes = ', '.join(list(path.nodes)[:6])
            if len(path.nodes) > 6:
                nodes += '...'
            report += f"  [{path.path_type}] {path.path_id}: {{{nodes}}}\n"
    
    report += f"{'='*80}\n"
    
    return report

def export_comprehensive_findings(results: Dict[str, CancerTypeAnalysis], output_path: Path):
    """Export all findings to comprehensive files"""
    
    drug_db = DrugTargetDB()
    
    # 1. Detailed X-Node combinations with drug info
    rows = []
    for cancer_type, analysis in results.items():
        if not analysis.recommended_combination:
            continue
            
        for i, target in enumerate(analysis.recommended_combination):
            drug_info = drug_db.get_drug_info(target)
            rows.append({
                'Cancer_Type': cancer_type,
                'Lineage': analysis.lineage,
                'Cell_Lines': analysis.n_cell_lines,
                'Combination_Size': len(analysis.recommended_combination),
                'Target_Position': i + 1,
                'Target_Gene': target,
                'Clinical_Stage': drug_info.clinical_stage if drug_info else 'unknown',
                'Available_Drugs': '; '.join(drug_info.available_drugs[:5]) if drug_info and drug_info.available_drugs else '',
                'Known_Toxicities': '; '.join(drug_info.known_toxicities) if drug_info else '',
                'Full_Combination': ' + '.join(sorted(analysis.recommended_combination)),
                'Path_Coverage': f"{analysis.minimal_hitting_sets[0].coverage:.1%}" if analysis.minimal_hitting_sets else '',
                'Cost_Score': f"{analysis.minimal_hitting_sets[0].total_cost:.2f}" if analysis.minimal_hitting_sets else '',
            })
    
    if rows:
        df_detailed = pd.DataFrame(rows)
        df_detailed.to_csv(output_path / "xnode_combinations_detailed.csv", index=False)
    
    # 2. Summary by target frequency across cancers
    target_counts = {}
    for cancer_type, analysis in results.items():
        if analysis.recommended_combination:
            for target in analysis.recommended_combination:
                if target not in target_counts:
                    target_counts[target] = {'count': 0, 'cancers': []}
                target_counts[target]['count'] += 1
                target_counts[target]['cancers'].append(cancer_type)
    
    target_rows = []
    for target, info in sorted(target_counts.items(), key=lambda x: -x[1]['count']):
        drug_info = drug_db.get_drug_info(target)
        target_rows.append({
            'Target_Gene': target,
            'Cancer_Types_Count': info['count'],
            'Clinical_Stage': drug_info.clinical_stage if drug_info else 'unknown',
            'Available_Drugs': '; '.join(drug_info.available_drugs[:3]) if drug_info and drug_info.available_drugs else '',
            'Cancer_Types': '; '.join(info['cancers'][:10]) + ('...' if len(info['cancers']) > 10 else '')
        })
    
    if target_rows:
        df_targets = pd.DataFrame(target_rows)
        df_targets.to_csv(output_path / "target_frequency_summary.csv", index=False)
    
    # 3. Drug combination protocols
    protocol_rows = []
    for cancer_type, analysis in results.items():
        if not analysis.recommended_combination:
            continue
        
        drugs_for_combo = []
        for target in analysis.recommended_combination:
            drug_info = drug_db.get_drug_info(target)
            if drug_info and drug_info.available_drugs:
                drugs_for_combo.append(f"{target}: {drug_info.available_drugs[0]}")
            else:
                drugs_for_combo.append(f"{target}: [no approved drug]")
        
        protocol_rows.append({
            'Cancer_Type': cancer_type,
            'X_Node_Targets': ' + '.join(sorted(analysis.recommended_combination)),
            'Suggested_Drug_Protocol': ' + '.join(drugs_for_combo),
            'All_Targets_Druggable': all(
                drug_db.get_drug_info(t) and drug_db.get_drug_info(t).available_drugs 
                for t in analysis.recommended_combination
            ),
            'Cell_Lines_Analyzed': analysis.n_cell_lines,
            'Viability_Paths': len(analysis.viability_paths),
        })
    
    if protocol_rows:
        df_protocols = pd.DataFrame(protocol_rows)
        df_protocols.to_csv(output_path / "drug_protocols.csv", index=False)
    
    # 4. Triple Combinations Export (Systems Biology)
    triple_rows = []
    for cancer_type, analysis in results.items():
        if not analysis.best_triple:
            continue
        
        bt = analysis.best_triple
        triple_rows.append({
            'Cancer_Type': cancer_type,
            'Lineage': analysis.lineage,
            'Cell_Lines': analysis.n_cell_lines,
            'Triple_Targets': ' + '.join(bt.targets),
            'Target_1': bt.targets[0],
            'Target_2': bt.targets[1],
            'Target_3': bt.targets[2],
            'Combined_Score': f"{bt.combined_score:.3f}",
            'Synergy_Score': f"{bt.synergy_score:.2f}",
            'Resistance_Score': f"{bt.resistance_score:.2f}",
            'Path_Coverage': f"{bt.coverage:.1%}",
            'Druggable_Count': bt.druggable_count,
            'Drug_1': (drug_db.get_drug_info(bt.targets[0]).available_drugs[0] 
                      if drug_db.get_drug_info(bt.targets[0]) and drug_db.get_drug_info(bt.targets[0]).available_drugs 
                      else 'N/A'),
            'Drug_2': (drug_db.get_drug_info(bt.targets[1]).available_drugs[0] 
                      if drug_db.get_drug_info(bt.targets[1]) and drug_db.get_drug_info(bt.targets[1]).available_drugs 
                      else 'N/A'),
            'Drug_3': (drug_db.get_drug_info(bt.targets[2]).available_drugs[0] 
                      if drug_db.get_drug_info(bt.targets[2]) and drug_db.get_drug_info(bt.targets[2]).available_drugs 
                      else 'N/A'),
            'Pathways_Covered': sum(1 for c in bt.pathway_coverage.values() if c > 0),
        })
    
    if triple_rows:
        df_triples = pd.DataFrame(triple_rows)
        df_triples = df_triples.sort_values(['Synergy_Score', 'Resistance_Score'], ascending=[False, True])
        df_triples.to_csv(output_path / "triple_combinations.csv", index=False)
    
    # 5. Triple Frequency Summary (Most common targets in triples)
    triple_target_counts = {}
    for cancer_type, analysis in results.items():
        if analysis.best_triple:
            for target in analysis.best_triple.targets:
                if target not in triple_target_counts:
                    triple_target_counts[target] = {'count': 0, 'cancers': [], 'avg_synergy': []}
                triple_target_counts[target]['count'] += 1
                triple_target_counts[target]['cancers'].append(cancer_type)
                triple_target_counts[target]['avg_synergy'].append(analysis.best_triple.synergy_score)
    
    triple_summary_rows = []
    for target, info in sorted(triple_target_counts.items(), key=lambda x: -x[1]['count']):
        drug_info = drug_db.get_drug_info(target)
        triple_summary_rows.append({
            'Target_Gene': target,
            'Appearances_in_Triples': info['count'],
            'Avg_Synergy_Score': f"{np.mean(info['avg_synergy']):.2f}",
            'Clinical_Stage': drug_info.clinical_stage if drug_info else 'unknown',
            'Available_Drugs': '; '.join(drug_info.available_drugs[:3]) if drug_info and drug_info.available_drugs else '',
            'Cancer_Types': '; '.join(info['cancers'][:10]) + ('...' if len(info['cancers']) > 10 else '')
        })
    
    if triple_summary_rows:
        df_triple_summary = pd.DataFrame(triple_summary_rows)
        df_triple_summary.to_csv(output_path / "triple_target_frequency.csv", index=False)
    
    # 6. Complete JSON export
    all_findings = {
        'analysis_date': str(pd.Timestamp.now()),
        'total_cancer_types': len(results),
        'total_with_combinations': sum(1 for a in results.values() if a.recommended_combination),
        'total_with_triples': sum(1 for a in results.values() if a.best_triple),
        'results': {}
    }
    
    for cancer_type, analysis in results.items():
        # Build triple combination info
        triple_info = None
        if analysis.best_triple:
            bt = analysis.best_triple
            triple_info = {
                'targets': list(bt.targets),
                'combined_score': bt.combined_score,
                'synergy_score': bt.synergy_score,
                'resistance_score': bt.resistance_score,
                'path_coverage': bt.coverage,
                'druggable_count': bt.druggable_count,
                'pathways_covered': {k: v for k, v in bt.pathway_coverage.items() if v > 0}
            }
        
        analysis_dict = {
            'cancer_type': analysis.cancer_type,
            'lineage': analysis.lineage,
            'n_cell_lines': analysis.n_cell_lines,
            'driver_mutations': analysis.driver_mutations,
            'recommended_combination': analysis.recommended_combination,
            'combination_cost': analysis.minimal_hitting_sets[0].total_cost if analysis.minimal_hitting_sets else None,
            'path_coverage': analysis.minimal_hitting_sets[0].coverage if analysis.minimal_hitting_sets else None,
            'n_viability_paths': len(analysis.viability_paths),
            'top_essential_genes': dict(list(analysis.essential_genes.items())[:10]),
            'best_triple_combination': triple_info,
            'n_triple_alternatives': len(analysis.triple_combinations),
        }
        all_findings['results'][cancer_type] = analysis_dict
    
    with open(output_path / "all_findings.json", 'w') as f:
        json.dump(all_findings, f, indent=2, default=str)
    
    print(f"\n{'='*80}")
    print("EXPORTED FILES:")
    print(f"{'='*80}")
    print(f"  1. xnode_combinations_detailed.csv - All targets with drug info")
    print(f"  2. target_frequency_summary.csv    - Most common targets across cancers")
    print(f"  3. drug_protocols.csv              - Suggested drug combinations")
    print(f"  4. triple_combinations.csv         - Systems biology triple combinations")
    print(f"  5. triple_target_frequency.csv     - Most common targets in triples")
    print(f"  6. all_findings.json               - Complete analysis data")
    print(f"  7. pan_cancer_summary.csv          - Summary table")
    print(f"  8. [CancerType]_report.txt         - Individual reports")
    print(f"{'='*80}")


def generate_summary_table(results: Dict[str, CancerTypeAnalysis]) -> pd.DataFrame:
    """Generate cross-cancer summary table"""
    
    rows = []
    for cancer_type, analysis in results.items():
        if analysis.recommended_combination:
            combo = ', '.join(sorted(analysis.recommended_combination))
            n_nodes = len(analysis.recommended_combination)
            cost = analysis.minimal_hitting_sets[0].total_cost if analysis.minimal_hitting_sets else 0
            coverage = analysis.minimal_hitting_sets[0].coverage if analysis.minimal_hitting_sets else 0
        else:
            combo = "None found"
            n_nodes = 0
            cost = 0
            coverage = 0
        
        # Add triple information
        if analysis.best_triple:
            triple = ', '.join(analysis.best_triple.targets)
            synergy = analysis.best_triple.synergy_score
            resistance = analysis.best_triple.resistance_score
        else:
            triple = "None"
            synergy = 0
            resistance = 1
        
        rows.append({
            'Cancer Type': cancer_type,
            'Cell Lines': analysis.n_cell_lines,
            'Paths': len(analysis.viability_paths),
            'X-Node Set': combo,
            'Size': n_nodes,
            'Cost': f"{cost:.2f}",
            'Coverage': f"{coverage:.1%}",
            'Best Triple': triple,
            'Synergy': f"{synergy:.2f}",
            'Resist': f"{resistance:.2f}"
        })
    
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values('Cell Lines', ascending=False)
    return df


def generate_triple_summary_table(results: Dict[str, CancerTypeAnalysis]) -> pd.DataFrame:
    """Generate summary table focused on triple combinations"""
    
    rows = []
    for cancer_type, analysis in results.items():
        if not analysis.best_triple:
            continue
            
        bt = analysis.best_triple
        drug_db = DrugTargetDB()
        
        # Get drug names for each target
        drugs = []
        for target in bt.targets:
            info = drug_db.get_drug_info(target)
            if info and info.available_drugs:
                drugs.append(info.available_drugs[0])
            else:
                drugs.append('[research]')
        
        rows.append({
            'Cancer Type': cancer_type,
            'Cell Lines': analysis.n_cell_lines,
            'Target 1': bt.targets[0],
            'Target 2': bt.targets[1],
            'Target 3': bt.targets[2],
            'Drug 1': drugs[0],
            'Drug 2': drugs[1],
            'Drug 3': drugs[2],
            'Synergy': f"{bt.synergy_score:.2f}",
            'Resistance': f"{bt.resistance_score:.2f}",
            'Coverage': f"{bt.coverage:.1%}",
            'Druggable': f"{bt.druggable_count}/3"
        })
    
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(['Synergy', 'Resistance'], ascending=[False, True])
    return df

# ============================================================================
# CLI INTERFACE
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="ALIN Framework (Adaptive Lethal Intersection Network) - High-Throughput Systems Biology",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pan_cancer_xnode.py --cancer-type "Pancreatic Adenocarcinoma" --output results/
  python pan_cancer_xnode.py --cancer-type PAAD --output results/
  python pan_cancer_xnode.py --cancer-type LUAD --output results/
  python pan_cancer_xnode.py --all-cancers --top-n 10
  python pan_cancer_xnode.py --all-cancers --triples  # Focus on triple combinations
  python pan_cancer_xnode.py --all-cancers --validate  # Run with validation
  python pan_cancer_xnode.py --list-cancers

Triple Combination Analysis:
  The --triples flag activates systems biology analysis to find optimal
  triple drug combinations using network topology, synergy scoring, and
  resistance probability estimation, based on the methodology from
  "A targeted combination therapy achieves effective pancreatic cancer
  regression and prevents tumor resistance"

Validation:
  The --validate flag runs discovered combinations through multi-source
  validation including PubMed literature, clinical trials, STRING PPI,
  TCGA patient data, and drug synergy databases.
        """
    )
    parser.add_argument('--cancer-type', type=str, 
                        help='Cancer type to analyze (e.g., "Pancreatic Adenocarcinoma" or "PAAD")')
    parser.add_argument('--all-cancers', action='store_true', 
                        help='Analyze all cancer types')
    parser.add_argument('--top-n', type=int, default=999, 
                        help='Max number of cancer types to analyze (default: all)')
    parser.add_argument('--output', type=str, default='results', 
                        help='Output directory (default: results)')
    parser.add_argument('--list-cancers', action='store_true',
                        help='List all available cancer types')
    parser.add_argument('--data-dir', type=str, default='./depmap_data',
                        help='Path to DepMap data directory')
    parser.add_argument('--triples', action='store_true',
                        help='Focus output on triple combinations (systems biology analysis)')
    parser.add_argument('--validate', action='store_true',
                        help='Run validation on discovered combinations')
    parser.add_argument('--validation-dir', type=str, default='./validation_data',
                        help='Path to validation data directory')
    parser.add_argument('--no-api', action='store_true',
                        help='Disable external API calls during validation (offline mode)')
    parser.add_argument('--validate-only', type=str, metavar='RESULTS_DIR',
                        help='Run validation only on existing results (skip discovery)')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = PanCancerXNodeAnalyzer(
        data_dir=args.data_dir, 
        validation_data_dir=args.validation_dir
    )
    
    # Validate-only mode: load existing results and run validation
    if args.validate_only:
        logger.info(f"Running validation-only mode on existing results: {args.validate_only}")
        
        results_dir = Path(args.validate_only)
        triple_file = results_dir / "triple_combinations.csv"
        
        if not triple_file.exists():
            logger.error(f"Triple combinations file not found: {triple_file}")
            logger.info("Run full analysis first with: --all-cancers --triples")
            exit(1)
        
        # Load existing triple combinations
        df = pd.read_csv(triple_file)
        logger.info(f"Loaded {len(df)} triple combinations from {triple_file}")
        
        # Initialize validation engine
        if not VALIDATION_AVAILABLE:
            logger.error("Validation module not available")
            exit(1)
        
        validator = ValidationEngine(data_dir=args.validation_dir)
        validations = []
        
        # Run validation on each triple
        for _, row in tqdm(df.iterrows(), desc="Validating triples", total=len(df)):
            targets = [row['Target_1'], row['Target_2'], row['Target_3']]
            cancer_type = row['Cancer_Type']
            
            validation = validator.validate_combination(
                targets=targets,
                cancer_type=cancer_type,
                enable_api_calls=not args.no_api
            )
            validations.append(validation)
        
        # Export validation results
        validation_output = results_dir / "validation"
        export_validation_results(validations, validation_output)
        
        # Print summary
        print("\n" + "="*80)
        print("VALIDATION SUMMARY")
        print("="*80)
        
        val_df = pd.DataFrame([
            {
                'Cancer Type': v.cancer_type,
                'Targets': ' + '.join(sorted(v.targets)),
                'Val Score': f"{v.validation_score:.2f}",
                'Confidence': v.confidence_level.split('-')[0].strip(),
                'Evidence': len(v.all_evidence)
            }
            for v in validations
        ])
        val_df = val_df.sort_values('Val Score', ascending=False)
        print(val_df.to_string(index=False))
        
        # Save summary
        val_df.to_csv(results_dir / "validation_overview.csv", index=False)
        
        logger.info(f"Validation complete! Results saved to {validation_output}")
        exit(0)
    
    if args.list_cancers:
        # List available cancer types
        print("\nAvailable Cancer Types in DepMap:")
        print("="*60)
        cancer_types = analyzer.depmap.get_available_cancer_types()
        for cancer_type, count in cancer_types[:50]:
            print(f"  {cancer_type}: {count} cell lines")
        print(f"\n... and {len(cancer_types) - 50} more" if len(cancer_types) > 50 else "")
        
    elif args.cancer_type:
        # Single cancer analysis
        logger.info(f"Running single-cancer analysis: {args.cancer_type}")
        analysis = analyzer.analyze_cancer_type(args.cancer_type)
        
        report = generate_cancer_report(analysis)
        print(report)
        
        # Save results
        output_path = Path(args.output)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # Save report
        safe_name = sanitize_cancer_name(analysis.cancer_type)
        with open(output_path / f"{safe_name}_report.txt", 'w') as f:
            f.write(report)
        
        # Save JSON
        analysis_dict = asdict(analysis)
        # Convert frozensets to lists for JSON serialization
        for hs in analysis_dict.get('minimal_hitting_sets', []):
            hs['targets'] = list(hs['targets'])
            hs['paths_covered'] = list(hs['paths_covered'])
        analysis_dict['top_x_node_sets'] = [(list(t), c) for t, c in analysis_dict.get('top_x_node_sets', [])]
        for p in analysis_dict.get('viability_paths', []):
            p['nodes'] = list(p['nodes'])
        
        with open(output_path / f"{safe_name}_analysis.json", 'w') as f:
            json.dump(analysis_dict, f, default=str, indent=2)
        
        # Run validation if requested
        if args.validate:
            logger.info("Running validation...")
            if VALIDATION_AVAILABLE:
                validation = analyzer.validation_integrator.validate_analysis(
                    analysis, 
                    enable_api_calls=not args.no_api
                )
                
                if validation:
                    # Print validation report
                    val_report = generate_validation_report(validation)
                    print("\n" + val_report)
                    
                    # Save validation report
                    with open(output_path / f"{safe_name}_validation.txt", 'w') as f:
                        f.write(val_report)
                    
                    # Generate combined report
                    combined = analyzer.validation_integrator.generate_combined_report(analysis, validation)
                    with open(output_path / f"{safe_name}_combined_report.txt", 'w') as f:
                        f.write(combined)
                    
                    logger.info(f"Validation results saved to {output_path}")
            else:
                logger.warning("Validation module not available. Skipping validation.")
        
        logger.info(f"Results saved to {output_path}")
    
    elif args.all_cancers:
        # Pan-cancer analysis
        print("\n" + "="*60, flush=True)
        print("  PAN-CANCER DISCOVERY (step 1/4)", flush=True)
        print("  Each cancer: paths -> hitting set -> triple scoring", flush=True)
        print("="*60 + "\n", flush=True)
        logger.info(f"Running pan-cancer analysis (max {args.top_n} cancer types)")
        results = analyzer.analyze_all_cancers(top_n=args.top_n)
        
        # Save results
        output_path = Path(args.output)
        output_path.mkdir(exist_ok=True, parents=True)
        
        for cancer_type, analysis in results.items():
            report = generate_cancer_report(analysis)
            safe_name = sanitize_cancer_name(cancer_type)
            
            with open(output_path / f"{safe_name}_report.txt", 'w') as f:
                f.write(report)
        
            # Generate triple report if available
            if analysis.best_triple:
                triple_report = analyzer.triple_finder.generate_triple_report(
                    analysis.best_triple, analysis.cancer_type
                )
                with open(output_path / f"{safe_name}_triple_report.txt", 'w') as f:
                    f.write(triple_report)
        
        # Generate and print summary
        summary = generate_summary_table(results)
        print("\n" + "="*80)
        print("PAN-CANCER X-NODE SUMMARY")
        print("="*80)
        print(summary.to_string(index=False))
        
        # Print triple combination summary if requested
        if args.triples:
            print("\n" + "="*80)
            print("SYSTEMS BIOLOGY TRIPLE COMBINATIONS")
            print("="*80)
            triple_summary = generate_triple_summary_table(results)
            if not triple_summary.empty:
                print(triple_summary.to_string(index=False))
                print(f"\nTotal cancers with triple combinations: {len(triple_summary)}/{len(results)}")
                
                # Show top synergistic combinations
                print("\n" + "-"*80)
                print("TOP 10 MOST SYNERGISTIC TRIPLES:")
                print("-"*80)
                for i, row in triple_summary.head(10).iterrows():
                    print(f"  {row['Cancer Type']}: {row['Target 1']} + {row['Target 2']} + {row['Target 3']}")
                    print(f"    Drugs: {row['Drug 1']} + {row['Drug 2']} + {row['Drug 3']}")
                    print(f"    Synergy: {row['Synergy']}, Resistance: {row['Resistance']}")
            else:
                print("No triple combinations found.")
        
        summary.to_csv(output_path / "pan_cancer_summary.csv", index=False)
        
        # Export comprehensive findings
        export_comprehensive_findings(results, output_path)
        
        # Run validation if requested
        if args.validate:
            logger.info("\n" + "="*80)
            logger.info("RUNNING VALIDATION PIPELINE")
            logger.info("="*80)
            
            if VALIDATION_AVAILABLE:
                validations = analyzer.validation_integrator.validate_all_results(
                    results, 
                    enable_api_calls=not args.no_api,
                    validate_triples=True
                )
                
                if validations:
                    # Export validation results
                    validation_output = output_path / "validation"
                    export_validation_results(list(validations.values()), validation_output)
                    
                    # Generate combined reports for each cancer
                    for cancer_type, validation in validations.items():
                        analysis = results[cancer_type]
                        combined = analyzer.validation_integrator.generate_combined_report(
                            analysis, validation
                        )
                        safe_name = sanitize_cancer_name(cancer_type)
                        with open(validation_output / f"{safe_name}_combined.txt", 'w') as f:
                            f.write(combined)
                    
                    # Print validation summary
                    print("\n" + "="*80)
                    print("VALIDATION SUMMARY")
                    print("="*80)
                    
                    val_df = pd.DataFrame([
                        {
                            'Cancer Type': v.cancer_type,
                            'Targets': ' + '.join(sorted(v.targets)),
                            'Val Score': f"{v.validation_score:.2f}",
                            'Confidence': v.confidence_level.split('-')[0].strip(),
                            'PubMed': v.pubmed_mentions,
                            'Trials': len(v.clinical_trials),
                            'PPI': f"{v.ppi_confidence:.2f}"
                        }
                        for v in validations.values()
                    ])
                    val_df = val_df.sort_values('Val Score', ascending=False)
                    print(val_df.to_string(index=False))
                    
                    # Save validation summary
                    val_df.to_csv(output_path / "validation_overview.csv", index=False)
                    
                    logger.info(f"Validation results saved to {validation_output}")
                else:
                    logger.warning("Validation module not available. Skipping validation.")
        
        logger.info(f"All results saved to {output_path}")
    
    else:
        # Demo mode
        logger.info("Running demo analysis on Pancreatic Adenocarcinoma")
        analysis = analyzer.analyze_cancer_type("Pancreatic Adenocarcinoma")
        report = generate_cancer_report(analysis)
        print(report)
        
        # Show triple combination for demo
        if analysis.best_triple:
            print("\n")
            print(analyzer.triple_finder.generate_triple_report(
                analysis.best_triple, analysis.cancer_type
            ))
