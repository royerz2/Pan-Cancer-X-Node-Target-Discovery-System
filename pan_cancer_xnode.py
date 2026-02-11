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
from typing import Dict, List, Set, Tuple, Optional, FrozenSet, Union, Any
from enum import Enum
from collections import defaultdict
import logging
from itertools import combinations
from pathlib import Path
from scipy import stats
import warnings
import re

from alin.constants import (
    tqdm, CANCER_TYPE_ALIASES, normalize_cancer_type,
    GENE_TO_DRUGS, GENE_CLINICAL_STAGE, GENE_TOXICITY_SCORES, GENE_TOXICITIES,
    PATHWAYS,
)

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
    # Provide lightweight stubs so type-hint evaluation or late imports won't fail
    ValidationEngine = None
    CombinationValidation = Any
    def generate_validation_report(*args, **kwargs):
        return ""
    def export_validation_results(*args, **kwargs):
        return []
    logging.warning("Validation module not available. Install dependencies or check import.")

warnings.filterwarnings('ignore', category=FutureWarning)

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# DATA STRUCTURES — single source of truth is core/data_structures.py
# ============================================================================
from core.data_structures import (
    TargetNode,
    NodeCost,
    ViabilityPath,
    HittingSet,
    CancerTypeAnalysis,
    DrugTarget,
)

# CANCER_TYPE_ALIASES and normalize_cancer_type imported from alin.constants

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
    Drug target and toxicity database.
    Built from canonical constants in alin.constants — single source of truth.
    """
    
    # Build DRUG_DB at class level from canonical constants
    DRUG_DB = {
        gene: {
            'drugs': drugs,
            'stage': GENE_CLINICAL_STAGE.get(gene, 'preclinical'),
            'toxicity': GENE_TOXICITY_SCORES.get(gene, 0.5),
            'toxicities': GENE_TOXICITIES.get(gene, []),
        }
        for gene, drugs in GENE_TO_DRUGS.items()
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
    
    def __init__(self, depmap: DepMapLoader, omnipath: OmniPathLoader,
                 disable_omnipath: bool = False,
                 disable_perturbation: bool = False,
                 disable_coessentiality: bool = False,
                 disable_statistical: bool = False,
                 use_lineage_aware_statistical: bool = False):
        self.depmap = depmap
        self.omnipath = omnipath
        self.disable_omnipath = disable_omnipath
        self.disable_perturbation = disable_perturbation
        self.disable_coessentiality = disable_coessentiality
        self.disable_statistical = disable_statistical
        self.use_lineage_aware_statistical = use_lineage_aware_statistical
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
        
        # Co-essentiality matrix (Jaccard index: |E_i ∩ E_j| / |E_i ∪ E_j|)
        n_genes = len(selective_list)
        # Precompute per-gene essential line sets for efficient Jaccard
        gene_ess_sets = {}
        for g in selective_list:
            gene_ess_sets[g] = {lid for lid, ess in line_essential_sets.items() if g in ess}
        co_essential = np.zeros((n_genes, n_genes))
        for i, g1 in enumerate(selective_list):
            for j, g2 in enumerate(selective_list):
                if i == j:
                    co_essential[i, j] = 0
                    continue
                intersection = len(gene_ess_sets[g1] & gene_ess_sets[g2])
                union = len(gene_ess_sets[g1] | gene_ess_sets[g2])
                co_essential[i, j] = intersection / max(1, union)
        
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
            Z = linkage(condensed, method='ward')
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
        Find genes that are significantly MORE essential in this cancer vs others.
        Uses Welch t-test with Benjamini-Hochberg FDR correction for multiple
        testing (one test per gene ≈ thousands of tests).
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
        
        # --- Phase 1: collect raw p-values for ALL testable genes ---
        gene_results = []   # list of (gene, t_stat, raw_p, effect_size, cancer_mean)
        
        # Filter genes to test (exclude pan-essential)
        genes_to_test = [g for g in crispr.columns if g not in pan_essential]
        
        for gene in genes_to_test:
            cancer_scores = crispr.loc[available_lines, gene].dropna()
            other_scores = crispr.loc[other_lines, gene].dropna()
            
            if len(cancer_scores) < 3 or len(other_scores) < 10:
                continue
            
            # Welch t-test: is this gene more essential in this cancer?
            t_stat, p_value = stats.ttest_ind(cancer_scores, other_scores,
                                              equal_var=False)
            
            # Effect size (positive = more essential in cancer, lower Chronos = more essential)
            effect_size = other_scores.mean() - cancer_scores.mean()
            
            gene_results.append({
                'gene': gene,
                't_stat': t_stat,
                'p_value': p_value,
                'effect_size': effect_size,
                'cancer_mean': cancer_scores.mean()
            })
        
        if not gene_results:
            return []
        
        # --- Phase 2: apply Benjamini-Hochberg FDR correction ---
        raw_pvals = [r['p_value'] for r in gene_results]
        
        try:
            from core.statistics import apply_fdr_correction
            adj_pvals, reject = apply_fdr_correction(raw_pvals, method='fdr_bh',
                                                     alpha=p_value_threshold)
        except ImportError:
            # Inline BH fallback so the fix is self-contained
            n = len(raw_pvals)
            sorted_idx = np.argsort(raw_pvals)
            adj = np.zeros(n)
            for i, idx in enumerate(sorted_idx):
                adj[idx] = min(raw_pvals[idx] * n / (i + 1), 1.0)
            # enforce monotonicity in sorted order
            for i in range(n - 2, -1, -1):
                si, si1 = sorted_idx[i], sorted_idx[i + 1]
                adj[si] = min(adj[si], adj[si1])
            adj_pvals = adj.tolist()
            reject = [q < p_value_threshold for q in adj_pvals]
        
        # Attach q-values back to results
        for r, q in zip(gene_results, adj_pvals):
            r['q_value'] = q
        
        n_tested = len(gene_results)
        n_raw_sig = sum(1 for r in gene_results if r['p_value'] < p_value_threshold and r['effect_size'] > effect_threshold)
        
        # --- Phase 3: filter by FDR-corrected q-value AND effect size ---
        cancer_specific_genes = [
            r for r in gene_results
            if r['q_value'] < p_value_threshold and r['effect_size'] > effect_threshold
        ]
        
        n_fdr_sig = len(cancer_specific_genes)
        logger.info(f"FDR correction: {n_tested} genes tested, "
                     f"{n_raw_sig} raw-significant, {n_fdr_sig} FDR-significant "
                     f"(q < {p_value_threshold}, Cohen's d > {effect_threshold})")
        
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
            
            logger.info(f"Found {len(top_genes)} cancer-specific essential genes (FDR-corrected)")
            return [path]
        
        return []

    def infer_cancer_specific_lineage_aware(self, cancer_type: str,
                                             p_value_threshold: float = 0.05,
                                             effect_threshold: float = 0.3) -> List[ViabilityPath]:
        """
        Find cancer-specific dependencies after controlling for lineage effects.

        Model per gene:  Chronos_g ~ lineage + is_target_cancer
        The coefficient on is_target_cancer captures cancer-type-specific
        essentiality after removing shared lineage dependencies.
        Uses OLS with lineage dummy variables, then BH-FDR correction.
        """
        logger.info(f"Finding lineage-aware cancer-specific dependencies for {cancer_type}")

        crispr = self.depmap.load_crispr_dependencies()
        cell_lines = self.depmap.get_cell_lines_for_cancer(cancer_type)
        pan_essential = self._get_pan_essential()

        available_lines = [cl for cl in cell_lines if cl in crispr.index]
        if len(available_lines) < 3:
            logger.warning(f"Too few cell lines ({len(available_lines)}) for lineage-aware test")
            return []

        # Build lineage annotation vector for all CRISPR lines
        lineage_df = self.depmap.load_lineage_annotations()
        common_lines = [cl for cl in crispr.index if cl in lineage_df.index]
        if len(common_lines) < 50:
            logger.warning("Too few lines with lineage annotation; falling back to Welch t-test")
            return self.infer_cancer_specific_dependencies(cancer_type, p_value_threshold, effect_threshold)

        lineage_series = lineage_df.loc[common_lines, 'OncotreeLineage'].fillna('Unknown')
        is_cancer = pd.Series(0, index=common_lines, dtype=float)
        avail_set = set(available_lines)
        for cl in common_lines:
            if cl in avail_set:
                is_cancer[cl] = 1.0

        # Build design matrix: lineage dummies + is_cancer indicator
        lineage_dummies = pd.get_dummies(lineage_series, prefix='lin', drop_first=True, dtype=float)
        # Ensure cancer type lineage is not collinear with is_cancer
        # (lineage dummies are already orthogonal enough with drop_first)
        design = pd.concat([lineage_dummies, is_cancer.rename('is_target_cancer')], axis=1)
        design.insert(0, 'intercept', 1.0)

        # Pre-compute pseudoinverse for OLS: beta = (X'X)^{-1} X' y
        X = design.values
        try:
            XtX_inv = np.linalg.pinv(X.T @ X)
        except np.linalg.LinAlgError:
            logger.warning("Singular design matrix; falling back to Welch t-test")
            return self.infer_cancer_specific_dependencies(cancer_type, p_value_threshold, effect_threshold)

        XtX_inv_Xt = XtX_inv @ X.T
        n_params = X.shape[1]
        n_obs = X.shape[0]
        cancer_coef_idx = design.columns.get_loc('is_target_cancer')

        genes_to_test = [g for g in crispr.columns if g not in pan_essential]
        crispr_sub = crispr.loc[common_lines]

        gene_results = []
        for gene in genes_to_test:
            y = crispr_sub[gene].values
            valid = ~np.isnan(y)
            if valid.sum() < n_params + 5:
                continue

            if valid.all():
                beta = XtX_inv_Xt @ y
                residuals = y - X @ beta
                dof = n_obs - n_params
            else:
                X_v = X[valid]
                y_v = y[valid]
                try:
                    XtX_inv_v = np.linalg.pinv(X_v.T @ X_v)
                except np.linalg.LinAlgError:
                    continue
                beta = XtX_inv_v @ X_v.T @ y_v
                residuals = y_v - X_v @ beta
                dof = valid.sum() - n_params

            if dof <= 0:
                continue

            mse = (residuals ** 2).sum() / dof
            se = np.sqrt(np.maximum(mse * XtX_inv[cancer_coef_idx, cancer_coef_idx]
                                    if valid.all()
                                    else mse * XtX_inv_v[cancer_coef_idx, cancer_coef_idx], 1e-30))
            t_stat = beta[cancer_coef_idx] / se
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), dof))
            # Negative coefficient = more essential (lower Chronos) in target cancer
            effect_size = -beta[cancer_coef_idx]

            gene_results.append({
                'gene': gene,
                't_stat': t_stat,
                'p_value': p_value,
                'effect_size': effect_size,
                'cancer_coef': beta[cancer_coef_idx],
            })

        if not gene_results:
            return []

        # BH FDR correction
        raw_pvals = [r['p_value'] for r in gene_results]
        try:
            from core.statistics import apply_fdr_correction
            adj_pvals, reject = apply_fdr_correction(raw_pvals, method='fdr_bh',
                                                     alpha=p_value_threshold)
        except ImportError:
            n = len(raw_pvals)
            sorted_idx = np.argsort(raw_pvals)
            adj = np.zeros(n)
            for i, idx in enumerate(sorted_idx):
                adj[idx] = min(raw_pvals[idx] * n / (i + 1), 1.0)
            for i in range(n - 2, -1, -1):
                si, si1 = sorted_idx[i], sorted_idx[i + 1]
                adj[si] = min(adj[si], adj[si1])
            adj_pvals = adj.tolist()

        for r, q in zip(gene_results, adj_pvals):
            r['q_value'] = q

        n_tested = len(gene_results)
        cancer_specific_genes = [
            r for r in gene_results
            if r['q_value'] < p_value_threshold and r['effect_size'] > effect_threshold
        ]
        n_fdr_sig = len(cancer_specific_genes)
        logger.info(f"Lineage-aware FDR: {n_tested} tested, {n_fdr_sig} significant "
                     f"(q < {p_value_threshold}, effect > {effect_threshold})")

        if len(cancer_specific_genes) >= 2:
            cancer_specific_genes.sort(key=lambda x: x['effect_size'], reverse=True)
            top_genes = [g['gene'] for g in cancer_specific_genes[:20]]
            path = ViabilityPath(
                path_id=f"{cancer_type}_lineage_aware_specific",
                nodes=frozenset(top_genes),
                context=cancer_type,
                confidence=0.95,
                path_type="cancer_specific"
            )
            logger.info(f"Found {len(top_genes)} lineage-aware cancer-specific genes")
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
        
        if not self.disable_coessentiality:
            _progress("Essential modules (co-essentiality)", step="")
            paths.extend(self.infer_essential_modules(cancer_type))
        if not self.disable_omnipath:
            _progress("Signaling paths (OmniPath)", step="")
            paths.extend(self.infer_signaling_paths(cancer_type, min_confidence=min_confidence))
        if not self.disable_statistical:
            if self.use_lineage_aware_statistical:
                _progress("Cancer-specific dependencies (lineage-aware)", step="")
                paths.extend(self.infer_cancer_specific_lineage_aware(cancer_type))
            else:
                _progress("Cancer-specific dependencies", step="")
                paths.extend(self.infer_cancer_specific_dependencies(cancer_type))
        if not self.disable_perturbation:
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
    Solve weighted hitting set problem.
    
    Given:
    - Set of viability paths P
    - Cost function c(v)
    
    Find:
    - Low-cost set T such that every path in P intersects T
    
    Solver hierarchy:
    1. Greedy weighted set cover (always runs; ln(n)-approximation guarantee)
    2. ILP-based exact solver via scipy (candidate pool <= ILP_THRESHOLD)
    3. Exhaustive enumeration (candidate pool <= EXHAUSTIVE_THRESHOLD)
    
    The solver records which method produced each solution so downstream
    code can report whether a result is provably optimal or approximate.
    """
    
    # Thresholds for solver selection
    EXHAUSTIVE_THRESHOLD = 20   # brute-force all subsets up to max_size
    ILP_THRESHOLD = 500         # ILP via scipy.optimize.milp
    PREFILTER_TOP_K = 60        # pre-filter to top-K genes before exhaustive
    
    def __init__(self, cost_function: CostFunction):
        self.cost_fn = cost_function
        self.solver_stats: Dict[str, int] = {
            'greedy': 0, 'ilp': 0, 'exhaustive': 0, 'prefiltered_exhaustive': 0
        }
    
    def solve(self, paths: List[ViabilityPath], cancer_type: str, 
              max_size: int = 4, min_coverage: float = 0.8) -> List[HittingSet]:
        """
        Find hitting sets using a hierarchy of solvers.
        
        Solver selection (applied in order of preference):
        1. Greedy (always): fast ln(n)-approximation.
        2. ILP exact solver (pool <= ILP_THRESHOLD): provably optimal via
           mixed-integer linear programming (scipy.optimize.milp).
        3. Pre-filtered exhaustive (ILP_THRESHOLD < pool, but top-K <= 
           PREFILTER_TOP_K * max_size): exhaustive search on cost-ranked
           subset of candidates.  Not provably optimal over full pool.
        4. Exhaustive (pool <= EXHAUSTIVE_THRESHOLD): brute-force over all
           subsets.  Provably optimal but exponential.
        
        Each returned HittingSet now carries a `solver_method` annotation
        (stored in the `paths_covered` frozenset is unchanged; method is
        logged and tracked in self.solver_stats).
        """
        if len(paths) == 0:
            return []
        
        # Extract all genes from paths
        all_genes = set()
        for path in paths:
            all_genes.update(path.nodes)
        
        n_candidates = len(all_genes)
        logger.info(
            f"Solving hitting set: {len(paths)} paths, "
            f"{n_candidates} candidate genes, max_size={max_size}"
        )
        
        # Compute costs
        gene_costs = {}
        for gene in all_genes:
            cost_obj = self.cost_fn.compute_cost(gene, cancer_type)
            gene_costs[gene] = cost_obj.total_cost()
        
        solutions = []
        methods_used = []
        
        # ---- 1. Greedy (always) ----
        greedy = self._solve_greedy(paths, gene_costs, max_size)
        if greedy:
            solutions.append(greedy)
            methods_used.append('greedy')
            self.solver_stats['greedy'] += 1
        
        # ---- 2. Exact solvers (when feasible) ----
        if n_candidates <= self.EXHAUSTIVE_THRESHOLD:
            # Small enough for brute-force enumeration
            exhaustive = self._solve_exhaustive(
                paths, gene_costs, max_size, min_coverage
            )
            solutions.extend(exhaustive)
            methods_used.append(f'exhaustive(n={n_candidates})')
            self.solver_stats['exhaustive'] += 1
            
        elif n_candidates <= self.ILP_THRESHOLD:
            # Medium pool: use ILP for provably optimal solution
            ilp_sol = self._solve_ilp(paths, gene_costs, max_size, min_coverage)
            if ilp_sol:
                solutions.append(ilp_sol)
                methods_used.append(f'ilp(n={n_candidates})')
                self.solver_stats['ilp'] += 1
        
        else:
            # Large pool: ILP on full set, plus pre-filtered exhaustive
            ilp_sol = self._solve_ilp(paths, gene_costs, max_size, min_coverage)
            if ilp_sol:
                solutions.append(ilp_sol)
                methods_used.append(f'ilp(n={n_candidates})')
                self.solver_stats['ilp'] += 1
            
            # Pre-filter to top-K lowest-cost genes, then exhaustive
            top_k = self.PREFILTER_TOP_K
            if n_candidates > top_k:
                sorted_genes = sorted(gene_costs.keys(), key=lambda g: gene_costs[g])
                prefiltered = set(sorted_genes[:top_k])
                # Also keep any gene that the greedy solver selected
                if greedy:
                    prefiltered.update(greedy.targets)
                prefiltered_costs = {g: gene_costs[g] for g in prefiltered}
                prefiltered_paths = [
                    p for p in paths if any(g in prefiltered for g in p.nodes)
                ]
                if len(prefiltered) <= self.PREFILTER_TOP_K + max_size:
                    pf_solutions = self._solve_exhaustive(
                        prefiltered_paths, prefiltered_costs,
                        max_size, min_coverage
                    )
                    solutions.extend(pf_solutions)
                    methods_used.append(
                        f'prefiltered_exhaustive(k={len(prefiltered)})'
                    )
                    self.solver_stats['prefiltered_exhaustive'] += 1
        
        logger.info(f"Solvers used: {', '.join(methods_used)}")
        
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
        """
        Greedy weighted set cover: pick gene with best coverage/cost ratio.
        
        Provides a ln(n)-approximation to the optimal weighted set cover.
        Fast (O(|genes| * |paths| * max_size)) but NOT provably minimal.
        """
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
    
    def _solve_ilp(self, paths: List[ViabilityPath], gene_costs: Dict[str, float],
                   max_size: int, min_coverage: float) -> Optional[HittingSet]:
        """
        Exact solver via Integer Linear Programming (scipy.optimize.milp).
        
        Formulation (with partial coverage support):
            Variables: x_g in {0,1} for each gene g (selected or not)
                       y_p in {0,1} for each path p (covered or not)
            
            min  sum_g  c(g) * x_g
            s.t. sum_{g in N(p)}  x_g  >= y_p       for each path p
                 sum_p  y_p  >= ceil(min_coverage * |P|)   (coverage)
                 sum_g  x_g  <= max_size              (cardinality)
                 x_g, y_p in {0, 1}
        
        Returns None if scipy is unavailable or the ILP is infeasible.
        """
        try:
            from scipy.optimize import milp, LinearConstraint, Bounds
            from scipy.sparse import csc_matrix, hstack, vstack, eye
            import math
        except ImportError:
            logger.warning("scipy.optimize.milp not available; skipping ILP solver")
            return None
        
        genes = sorted(gene_costs.keys())
        gene_idx = {g: i for i, g in enumerate(genes)}
        n_genes = len(genes)
        n_paths = len(paths)
        
        if n_genes == 0 or n_paths == 0:
            return None
        
        # Total variables: n_genes (x_g) + n_paths (y_p)
        n_vars = n_genes + n_paths
        
        # Objective: minimize total cost of selected genes (y_p has 0 cost)
        c = np.zeros(n_vars, dtype=float)
        c[:n_genes] = [gene_costs[g] for g in genes]
        
        # Build path-gene incidence matrix A (n_paths x n_genes)
        rows, cols = [], []
        for p_idx, path in enumerate(paths):
            for node in path.nodes:
                if node in gene_idx:
                    rows.append(p_idx)
                    cols.append(gene_idx[node])
        
        if not rows:
            return None
        
        A_path_gene = csc_matrix(
            (np.ones(len(rows)), (rows, cols)),
            shape=(n_paths, n_genes)
        )
        
        # Constraint 1: A @ x - y >= 0  (path can only be "covered" if genes hit it)
        # Rewritten: [A | -I] @ [x; y] >= 0
        neg_I = -eye(n_paths, format='csc')
        cov_matrix = hstack([A_path_gene, neg_I], format='csc')
        cov_constraint = LinearConstraint(cov_matrix, lb=np.zeros(n_paths))
        
        # Constraint 2: sum(y_p) >= ceil(min_coverage * n_paths)
        min_covered = math.ceil(min_coverage * n_paths)
        sum_y = csc_matrix(
            np.concatenate([np.zeros(n_genes), np.ones(n_paths)]).reshape(1, -1)
        )
        coverage_constraint = LinearConstraint(sum_y, lb=np.array([min_covered]))
        
        # Constraint 3: sum(x_g) <= max_size
        sum_x = csc_matrix(
            np.concatenate([np.ones(n_genes), np.zeros(n_paths)]).reshape(1, -1)
        )
        card_constraint = LinearConstraint(sum_x, ub=np.array([max_size]))
        
        # Variable bounds and integrality
        bounds = Bounds(lb=np.zeros(n_vars), ub=np.ones(n_vars))
        integrality = np.ones(n_vars)  # all binary
        
        try:
            result = milp(
                c=c,
                constraints=[cov_constraint, coverage_constraint, card_constraint],
                integrality=integrality,
                bounds=bounds,
                options={'time_limit': 30}
            )
        except Exception as e:
            logger.warning(f"ILP solver failed: {e}")
            return None
        
        if not result.success:
            logger.info(f"ILP infeasible or timed out: {result.message}")
            return None
        
        # Extract solution
        x_vals = result.x[:n_genes]
        selected = {genes[i] for i in range(n_genes) if x_vals[i] > 0.5}
        if not selected:
            return None
        
        total_cost = sum(gene_costs[g] for g in selected)
        covered_count = sum(
            1 for p in paths if any(g in selected for g in p.nodes)
        )
        coverage = covered_count / n_paths
        paths_covered = {
            p.path_id for p in paths if any(g in selected for g in p.nodes)
        }
        
        if coverage < min_coverage:
            return None
        
        return HittingSet(
            targets=frozenset(selected),
            total_cost=total_cost,
            coverage=coverage,
            paths_covered=paths_covered
        )
    
    def _solve_exhaustive(self, paths: List[ViabilityPath], gene_costs: Dict[str, float],
                          max_size: int, min_coverage: float) -> List[HittingSet]:
        """
        Enumerate all subsets up to max_size and keep those meeting min_coverage.
        
        Provably optimal (finds the true minimum-cost hitting set of each
        cardinality) but O(C(n, max_size)) — only feasible for small n.
        """
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
        
        coverage = {}
        for pathway, pathway_genes in PATHWAYS.items():
            covered = len(genes & pathway_genes)
            total = len(pathway_genes)
            coverage[pathway] = covered / total if total > 0 else 0
        
        return coverage
    
    def find_pathway_bridges(self) -> List[str]:
        """Find genes that bridge multiple pathways (key convergence points)"""
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
    
    # Known synergistic combinations from clinical trials / validated data
    KNOWN_SYNERGIES = {
        # Clinically validated (FDA-approved combinations)
        frozenset({'BRAF', 'MAP2K1'}): 0.95,  # BRAF + MEK (dabrafenib+trametinib, standard of care)
        frozenset({'EGFR', 'MET'}): 0.90,     # Bypass resistance (capmatinib+gefitinib, FDA approved)
        frozenset({'ERBB2', 'PIK3CA'}): 0.85,  # HER2 + PI3K (validated in breast)
        frozenset({'CDK4', 'ERBB2'}): 0.85,   # Palbociclib + trastuzumab (breast)
        frozenset({'CDK6', 'ERBB2'}): 0.85,   # Ribociclib + trastuzumab (breast)
        frozenset({'BRAF', 'EGFR'}): 0.85,    # Encorafenib + cetuximab (CRC, FDA approved)
        frozenset({'EGFR', 'KRAS'}): 0.80,    # Sotorasib + cetuximab (NSCLC trials)
        frozenset({'KRAS', 'MAP2K1'}): 0.85,  # KRAS + MEK inhibitor (validated)
        frozenset({'BCL2', 'MCL1'}): 0.9,     # Double BCL2 family
        # From PDAC paper
        frozenset({'SRC', 'STAT3'}): 0.85,
        frozenset({'FYN', 'STAT3'}): 0.85,
        frozenset({'SRC', 'FYN', 'STAT3'}): 0.95,  # Paper's triple
        # Established pathway interactions
        frozenset({'PIK3CA', 'MTOR'}): 0.70,
        frozenset({'CDK4', 'EGFR'}): 0.80,
        frozenset({'CDK4', 'CDK6'}): 0.60,    # Same target class
        frozenset({'JAK1', 'STAT3'}): 0.80,
        frozenset({'KRAS', 'SRC'}): 0.75,
        frozenset({'KRAS', 'STAT3'}): 0.80,
        frozenset({'SRC', 'FYN'}): 0.70,      # SFK family
        frozenset({'ERBB2', 'KDR'}): 0.80,   # Trastuzumab + ramucirumab (gastric)
        frozenset({'CDK4', 'MAP2K1'}): 0.75,  # Cell cycle + MAPK
        frozenset({'CDK6', 'MAP2K1'}): 0.75,  # Cell cycle + MAPK
        frozenset({'EGFR', 'ERBB2'}): 0.80,   # Dual HER targeting
        frozenset({'EGFR', 'MAP2K1'}): 0.80,  # EGFR + MEK
        frozenset({'BRAF', 'MET'}): 0.75,     # Cross-pathway
    }
    
    # Pathway assignments derived from canonical PATHWAYS (alin.constants)
    PATHWAY_ASSIGNMENT = {gene: pw for pw, genes in PATHWAYS.items() for gene in genes}
    
    def __init__(self, omnipath: OmniPathLoader, use_known_synergies: bool = True):
        self.omnipath = omnipath
        self.use_known_synergies = use_known_synergies
        
    def compute_synergy_score(self, genes: Set[str], use_known_synergies: Optional[bool] = None) -> float:
        """
        Compute synergy score for a combination of genes.
        
        Args:
            genes: Set of gene symbols
            use_known_synergies: If False, skip KNOWN_SYNERGIES lookup and
                compute synergy purely from pathway diversity + co-essentiality.
                If None, uses the instance default (self.use_known_synergies).
        
        Returns:
            Score 0-1 where higher = more synergistic
        """
        if len(genes) < 2:
            return 0.0
        
        _use_ks = use_known_synergies if use_known_synergies is not None else self.use_known_synergies
        
        genes_frozen = frozenset(genes)
        
        known_pair_score = 0.0
        pair_count = 0
        
        if _use_ks:
            # Check known synergies (exact match on full set)
            if genes_frozen in self.KNOWN_SYNERGIES:
                return self.KNOWN_SYNERGIES[genes_frozen]
            
            # Check pairwise known synergies
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
        
        # Combine scores: weight clinical evidence higher when available
        if pair_count > 0:
            # Clinical evidence dominates when available
            synergy = (
                known_pair_score * 0.7 +
                pathway_diversity * 0.3
            )
        else:
            # Pathway diversity only when no clinical evidence
            synergy = pathway_diversity * 0.6
        
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

# TripleCombination dataclass imported from core.data_structures (single source of truth)
from core.data_structures import TripleCombination


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
                 toxicity_cache_dir: Optional[str] = None,
                 use_known_synergies: bool = True,
                 disable_hub_penalty: bool = False):
        self.depmap = depmap
        self.omnipath = omnipath
        self.drug_db = drug_db
        self.use_known_synergies = use_known_synergies
        self.disable_hub_penalty = disable_hub_penalty
        self.network_analyzer = XNodeNetworkAnalyzer(omnipath)
        self.synergy_scorer = SynergyScorer(omnipath, use_known_synergies=use_known_synergies)
        self.resistance_estimator = ResistanceProbabilityEstimator(omnipath, depmap)
        self.cost_fn = CostFunction(depmap, drug_db, toxicity_cache_dir=toxicity_cache_dir)

        # Evidence-backed hub-penalty exemptions: genes with Tier 1 experimental
        # evidence in a specific cancer type are exempt from the hub penalty
        # when they appear alongside a known synergy partner in the combination.
        # Currently only STAT3 in PDAC (Liaki et al. 2025: KRAS+EGFR+STAT3).
        self.EVIDENCE_EXEMPTIONS = {
            'Pancreatic Adenocarcinoma': {
                'STAT3': frozenset({'KRAS', 'EGFR'}),  # exempt if paired with KRAS or EGFR
            },
        }
        
        # Last-run results for doublets and best-of-any-size
        self._last_doublet_combinations = []
        self._last_best_combination = None
        
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
        
        # ---- EXPANDED CANDIDATE INJECTION ----
        # Add all druggable genes from viability paths (approved or phase3)
        # This ensures cancer-relevant targets like ALK, RET, FGFR2, MTOR,
        # PARP1, IDH1/2 enter the candidate pool even if they are not
        # network hubs. The downstream scoring (synergy, coverage, cost)
        # determines whether they appear in the final triple.
        druggable_path_genes = set()
        for gene in all_genes:
            if gene in self.drug_db.DRUG_DB:
                info = self.drug_db.DRUG_DB[gene]
                if info.get('stage') in ('approved', 'phase3', 'phase2'):
                    druggable_path_genes.add(gene)
        
        priority_genes |= druggable_path_genes
        logger.info(f"Injected {len(druggable_path_genes)} druggable path genes into candidates")
        # ---- END EXPANDED CANDIDATE INJECTION ----
        
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
        
        # Pre-compute per-gene path frequencies for hub penalty
        gene_path_freqs = {}
        for gene in candidate_genes:
            gene_path_freqs[gene] = sum(
                1 for p in paths if gene in p.nodes
            ) / max(len(paths), 1)
        freq_values = sorted(gene_path_freqs.values())
        median_path_freq = freq_values[len(freq_values) // 2] if freq_values else 0.3
        
        # Evidence-backed hub penalty exemptions for this cancer type
        cancer_exemptions = self.EVIDENCE_EXEMPTIONS.get(cancer_type, {})
        
        def _score_combo(combo):
            """Score a combination of any size (2 or 3 genes)."""
            combo_set = set(combo)
            
            # Calculate coverage
            covered = sum(1 for p in paths if any(g in combo_set for g in p.nodes))
            coverage = covered / len(paths)
            
            if coverage < min_coverage:
                return None
            
            # Total cost
            total_cost = sum(gene_costs.get(g, 1.0) for g in combo)
            
            # Synergy score (heuristic)
            synergy_heuristic = self.synergy_scorer.compute_synergy_score(combo_set)

            # Data-driven synergy from co-essentiality (if DepMap data available)
            synergy = synergy_heuristic
            try:
                from pharmacological_validation import CoEssentialityInteractionEstimator
                dd = CoEssentialityInteractionEstimator.score_combination(
                    targets=tuple(sorted(combo)),
                    depmap_df=self.depmap._crispr_df if hasattr(self.depmap, '_crispr_df') and self.depmap._crispr_df is not None else None,
                    cell_lines=[],  # will use all available
                    original_synergy=synergy_heuristic,
                    original_pathway_diversity=len(set(
                        self.synergy_scorer.PATHWAY_ASSIGNMENT.get(g, g) for g in combo
                    )) / max(len(combo), 1),
                )
                synergy = dd.data_driven_synergy
            except (ImportError, Exception):
                pass  # fallback to heuristic
            
            # Resistance probability
            resistance = self.resistance_estimator.estimate_resistance_probability(combo_set, cancer_type)
            
            # Pathway coverage
            pathway_cov = self.network_analyzer.get_pathway_coverage(combo_set)
            
            # Count druggable targets
            druggable_count = sum(1 for g in combo if self.drug_db.get_druggability_score(g) >= 0.6)
            
            # Get drug info
            drug_info = {g: self.drug_db.get_drug_info(g) for g in combo}
            
            # Compute combination-level toxicity (DDI, overlapping toxicities, FAERS signals)
            combo_tox_score = 0.0
            combo_tox_details = {}
            try:
                from alin.toxicity import compute_combo_toxicity_score
                combo_tox_result = compute_combo_toxicity_score(list(combo), use_faers=False)
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
                pert_result = score_combination_by_perturbation(list(combo), essential_genes)
                # Bonus if combination targets feedback genes (resistance prevention)
                perturbation_bonus = pert_result.get('feedback_coverage', 0) * 0.1
            except ImportError:
                pass
            
            # Hub gene specificity penalty: penalise genes that appear in
            # a large fraction of viability paths, proportional to how
            # much they exceed the median gene frequency.  Genes appearing
            # in many paths are pan-cancer hubs (e.g., STAT3) and carry
            # little cancer-specific signal.
            # Evidence-aware: exempt genes with Tier 1 experimental evidence
            # when paired with a known synergy partner in this combination.
            hub_penalty = 0.0
            if not self.disable_hub_penalty:
                for g in combo:
                    # Check if gene is evidence-exempt in this cancer + combination
                    if g in cancer_exemptions:
                        required_partners = cancer_exemptions[g]
                        if required_partners & combo_set:
                            continue  # Skip hub penalty for evidence-backed gene
                    excess = gene_path_freqs.get(g, 0) - median_path_freq
                    if excess > 0:
                        # Proportional penalty: scales with excess above median
                        hub_penalty += excess * 1.5
            
            # Combined score (lower is better)
            # Weights: cost (0.22), synergy (-0.18), resistance (0.18),
            #          coverage (-0.14), combo_tox (0.18), hub (penalty)
            combined_score = (
                total_cost * 0.22 +
                (1 - synergy) * 0.18 +  # Invert synergy (higher synergy = better)
                resistance * 0.18 +
                (1 - coverage) * 0.14 +
                combo_tox_score * 0.18 +
                hub_penalty -             # Penalty for pan-cancer hubs
                druggable_count * 0.1 -   # Bonus for druggability
                perturbation_bonus        # Bonus for targeting feedback genes
            )
            
            return TripleCombination(
                targets=tuple(sorted(combo)),
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
            )
        
        # Enumerate and score triple combinations
        triple_combinations = []
        
        combos = list(combinations(candidate_genes, 3))
        for triple in tqdm(combos, desc="Scoring triples", leave=False, mininterval=0.5, miniters=10):
            result = _score_combo(triple)
            if result is not None:
                triple_combinations.append(result)
        
        # Also enumerate and score doublet (2-gene) combinations
        # Many gold-standard entries are doublets; outputting the best
        # doublet enables exact-match recovery for 2-gene regimens.
        doublet_combinations = []
        
        doublet_combos = list(combinations(candidate_genes, 2))
        for doublet in tqdm(doublet_combos, desc="Scoring doublets", leave=False, mininterval=0.5, miniters=10):
            result = _score_combo(doublet)
            if result is not None:
                doublet_combinations.append(result)
        
        # Sort by combined score
        triple_combinations.sort(key=lambda x: x.combined_score)
        doublet_combinations.sort(key=lambda x: x.combined_score)
        
        # Find best combination of ANY size (2 or 3)
        all_scored = triple_combinations + doublet_combinations
        all_scored.sort(key=lambda x: x.combined_score)
        
        logger.info(f"Found {len(triple_combinations)} valid triples, "
                     f"{len(doublet_combinations)} valid doublets")
        
        # Attach best-of-any-size to self for downstream access
        self._last_doublet_combinations = doublet_combinations[:top_n]
        self._last_best_combination = all_scored[0] if all_scored else None
        
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
                          enable_api_calls: bool = True) -> Optional['CombinationValidation']:
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
                        enable_api_calls: bool = True) -> Optional['CombinationValidation']:
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
                             validate_triples: bool = True) -> Dict[str, 'CombinationValidation']:
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
                                 validation: 'CombinationValidation') -> str:
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
                 toxicity_cache_dir: Optional[str] = None,
                 use_known_synergies: bool = True,
                 disable_omnipath: bool = False,
                 disable_perturbation: bool = False,
                 disable_coessentiality: bool = False,
                 disable_statistical: bool = False,
                 disable_hub_penalty: bool = False,
                 use_lineage_aware_statistical: bool = False):
        self.depmap = DepMapLoader(data_dir)
        self.omnipath = OmniPathLoader(data_dir)
        self.drug_db = DrugTargetDB()
        self.path_inference = ViabilityPathInference(
            self.depmap, self.omnipath,
            disable_omnipath=disable_omnipath,
            disable_perturbation=disable_perturbation,
            disable_coessentiality=disable_coessentiality,
            disable_statistical=disable_statistical,
            use_lineage_aware_statistical=use_lineage_aware_statistical,
        )
        self.cost_fn = CostFunction(self.depmap, self.drug_db, toxicity_cache_dir=toxicity_cache_dir)
        self.solver = MinimalHittingSetSolver(self.cost_fn)
        self.triple_finder = TripleCombinationFinder(
            self.depmap, self.omnipath, self.drug_db,
            toxicity_cache_dir=toxicity_cache_dir,
            use_known_synergies=use_known_synergies,
            disable_hub_penalty=disable_hub_penalty,
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
        best_combination = self.triple_finder._last_best_combination
        if triple_combinations:
            _progress(f"Best triple: {' + '.join(sorted(best_triple.targets))}", step="done")
        if best_combination and len(best_combination.targets) != 3:
            _progress(f"Best combo ({len(best_combination.targets)} genes): {' + '.join(sorted(best_combination.targets))}", step="done")
        
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
            best_combination=best_combination,
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

        # Post-pipeline pharmacological validation
        try:
            from pharmacological_validation import PharmacologicalValidator
            pv = PharmacologicalValidator(
                depmap_dir=str(self.depmap.data_dir) if hasattr(self.depmap, 'data_dir') else './depmap_data',
                drug_dir='./drug_sensitivity_data',
            )
            for cancer_type, analysis in results.items():
                try:
                    vr = pv.validate_predictions(
                        cancer_type=cancer_type,
                        predicted_targets=analysis.best_triple.targets if analysis.best_triple else (),
                        cell_line_ids=analysis.cell_line_ids,
                        n_cell_lines=analysis.n_cell_lines,
                        original_synergy=analysis.best_triple.synergy_score if analysis.best_triple else 0.0,
                    )
                    analysis.pharmacological_validation = {
                        'evidence_tier': vr.evidence_tier.tier,
                        'tier_label': vr.evidence_tier.tier_label,
                        'concordance_fraction': vr.evidence_tier.concordance_fraction,
                        'data_driven_synergy': vr.data_driven_synergy.data_driven_synergy if vr.data_driven_synergy else None,
                        'gene_concordances': {
                            g: {'concordant': gc.concordant, 'score': gc.concordance_score}
                            for g, gc in vr.gene_concordances.items()
                        },
                    }
                except Exception as e:
                    logger.debug(f'Pharmacological validation failed for {cancer_type}: {e}')
            # Summary
            tiers = [a.pharmacological_validation['evidence_tier']
                     for a in results.values() if a.pharmacological_validation]
            if tiers:
                print(f'\n  Pharmacological validation: {len(tiers)} cancers classified')
                for t in range(1, 5):
                    c = tiers.count(t)
                    print(f'    Tier {t}: {c} cancers ({100*c/len(tiers):.0f}%)')
        except ImportError:
            logger.debug('pharmacological_validation module not found; skipping post-pipeline validation')
        except Exception as e:
            logger.warning(f'Post-pipeline pharmacological validation failed: {e}')

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
        
        row = {
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
        }
        
        # Add best-combination-of-any-size columns
        bc = analysis.best_combination
        if bc is not None:
            bc_targets = sorted(bc.targets)
            row['Best_Combo_Size'] = len(bc_targets)
            row['Best_Combo_1'] = bc_targets[0] if len(bc_targets) >= 1 else ''
            row['Best_Combo_2'] = bc_targets[1] if len(bc_targets) >= 2 else ''
            row['Best_Combo_3'] = bc_targets[2] if len(bc_targets) >= 3 else ''
            row['Best_Combo_Score'] = f"{bc.combined_score:.3f}"
        else:
            row['Best_Combo_Size'] = 0
            row['Best_Combo_1'] = ''
            row['Best_Combo_2'] = ''
            row['Best_Combo_3'] = ''
            row['Best_Combo_Score'] = ''
        
        rows.append(row)
    
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
    parser.add_argument('--tune-parameters', action='store_true',
                        help='Run parameter tuning against gold-standard benchmark')
    parser.add_argument('--tune-mode', type=str, default='sweep',
                        choices=['grid', 'sweep', 'calibrate', 'all'],
                        help='Tuning mode: sweep (fast sensitivity), calibrate (cluster), '
                             'grid (full search), all (everything)')
    parser.add_argument('--tune-sample', type=int, default=None,
                        help='For grid tuning: randomly sample N configs (faster)')
    
    args = parser.parse_args()
    
    # Parameter tuning mode
    if args.tune_parameters:
        from parameter_tuning import (
            threshold_sensitivity_sweep, calibrate_cluster_count,
            PipelineEvaluator, GridSearchTuner,
        )
        tune_output = Path(args.output) / "tuning_results"
        tune_output.mkdir(parents=True, exist_ok=True)
        
        if args.tune_mode in ('sweep', 'all'):
            threshold_sensitivity_sweep(args.data_dir, str(tune_output))
        if args.tune_mode in ('calibrate', 'all'):
            calibrate_cluster_count(args.data_dir, str(tune_output))
        if args.tune_mode in ('grid', 'all'):
            evaluator = PipelineEvaluator(
                data_dir=args.data_dir, top_n_cancers=args.top_n,
            )
            tuner = GridSearchTuner(evaluator, str(tune_output))
            tuner.run(stage1_sample=args.tune_sample)
        
        logger.info(f"Tuning results saved to {tune_output}")
        exit(0)
    
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
