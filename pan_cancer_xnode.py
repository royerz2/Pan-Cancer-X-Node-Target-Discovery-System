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
import time
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Set, Tuple, Optional, FrozenSet, Union, Any
# enum.Enum removed — no Enum subclasses in this module
from collections import defaultdict
import logging
from itertools import combinations
from pathlib import Path
from scipy import stats
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
import re

from alin.constants import (
    tqdm, CANCER_TYPE_ALIASES, normalize_cancer_type,
    GENE_TO_DRUGS, GENE_CLINICAL_STAGE, GENE_TOXICITY_SCORES, GENE_TOXICITIES,
    PATHWAYS,
)

from alin.utils import sanitize_cancer_name
from alin.genomic_data import (
    TCGAMutationLoader, compute_genomic_bonus, compute_genomic_candidate_boost,
    get_cancer_driver_genes, get_driver_injection_set,
    filter_genomically_irrelevant, get_combined_alteration_freq,
    W_GENOMIC, MIN_MUTATION_FREQ_RELEVANT,
)
from alin.run_modes import ModeConfig, RunMode, actionable_config
from alin.strategy_arms import (
    DEFAULT_STRATEGY_ARM,
    SUPPORTED_STRATEGY_ARMS,
    infer_strategy_arm_from_scoring_mode,
    is_structural_strategy_arm,
    normalize_strategy_arm,
)


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
# ARTIFACT BLOCKLIST — genes that should never appear as drug targets
# ============================================================================
# These genes recurrently appear in CRISPR screens or viability paths due to
# technical artefacts, extreme protein size, or tissue-specific expression
# patterns that do not reflect actionable cancer vulnerabilities.
#
# Categories:
#   1. Immunoglobulin variable regions (IGHV*, IGLV*, IGKV*) — expressed only
#      in B-cell lineages; their "essentiality" reflects lymphoid biology, not
#      pan-cancer drug targets.
#   2. T-cell receptor loci (TRA*, TRB*) — same reasoning.
#   3. Olfactory receptors (OR*) — artefactual screen hits due to pseudogene
#      co-amplification on chr11; never biologically relevant as targets.
#   4. Giant proteins (TTN, MUC16, OBSCN, SYNE1/2) — falsely enrich in
#      mutation-based analyses due to extreme coding length.
#   5. Known false-positive CRISPR artefacts (VPS13D, PTMA, GTF3C1) — flagged
#      by DepMap QC; their essentiality is driven by copy-number effects.
ARTIFACT_BLOCKLIST_PREFIXES = frozenset({
    'IGHV', 'IGLV', 'IGKV', 'IGHD', 'IGHJ',  # Ig variable, diversity, joining
    'TRA', 'TRB', 'TRD', 'TRG',               # TCR loci
    'OR',                                        # Olfactory receptors (OR1A1, OR2T3, etc.)
})

ARTIFACT_BLOCKLIST_EXACT = frozenset({
    # Giant proteins (mutation length artefacts)
    'TTN', 'MUC16', 'OBSCN', 'SYNE1', 'SYNE2', 'RYR1', 'RYR2', 'RYR3',
    'DNAH5', 'DNAH11', 'DNAH17',
    # CRISPR QC artefacts
    'VPS13D', 'PTMA', 'GTF3C1',
    # Other known false-positives in pan-cancer screens
    'PCLO', 'CSMD3', 'LRP1B', 'ZFHX4', 'FAT3', 'FAT4',
})


def _is_artifact(gene: str) -> bool:
    """Check if a gene is in the artifact blocklist (exact match or prefix)."""
    if gene in ARTIFACT_BLOCKLIST_EXACT:
        return True
    for prefix in ARTIFACT_BLOCKLIST_PREFIXES:
        if gene.startswith(prefix) and len(gene) > len(prefix):
            # Ensure we match "OR1A1" but not "ORC1" — olfactory receptors
            # always have a digit after "OR"
            if prefix == 'OR':
                rest = gene[len(prefix):]
                if rest and rest[0].isdigit():
                    return True
            else:
                return True
    return False


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
    
    # V8.1: Define clinically relevant driver mutations for subtype stratification.
    # Each cancer type maps to a list of (gene, min_fraction) tuples.
    # Only genes mutated in >= min_fraction of that cancer's cell lines get a subtype.
    # This ensures subtypes have enough cell lines for meaningful CRISPR analysis.
    SUBTYPE_DRIVERS: Dict[str, List[Tuple[str, float]]] = {
        'Non-Small Cell Lung Cancer': [('KRAS', 0.15), ('EGFR', 0.03), ('STK11', 0.05)],
        'Invasive Breast Carcinoma':  [('PIK3CA', 0.15)],  # ERBB2 via CN_SUBTYPES
        'Acute Myeloid Leukemia':     [('NRAS', 0.10), ('TP53', 0.15), ('FLT3', 0.0)],
        'Colorectal Adenocarcinoma':  [('BRAF', 0.10), ('KRAS', 0.30), ('PIK3CA', 0.15)],
        'Melanoma':                   [('BRAF', 0.30), ('NRAS', 0.10)],
        'Pancreatic Adenocarcinoma':  [('KRAS', 0.50)],
    }
    # V8.1: Copy-number amplification subtypes (gene, CN_threshold).
    # DepMap OmicsCNGeneWGS values are absolute copy numbers (diploid ≈ 2).
    # HER2+ threshold: CN > 3 gives a clean bimodal split.
    CN_SUBTYPES: Dict[str, List[Tuple[str, float]]] = {
        'Invasive Breast Carcinoma': [('ERBB2', 3.0)],
    }
    MIN_SUBTYPE_CELL_LINES = 5  # Don't create subtype if fewer lines

    def __init__(self, data_dir: str = "./depmap_data"):
        self.data_dir = Path(data_dir)
        self._model_df = None
        self._crispr_df = None
        self._subtype_df = None
        self._expression_df = None
        self._cn_df = None           # V7: copy-number data
        self._rnai_df = None         # V7: RNAi screen data
        self._hotspot_df = None      # V8.1: hotspot mutation matrix
        self._gene_name_map = {}  # Entrez ID -> Gene symbol
        self._subtype_cell_lines: Dict[str, List[str]] = {}  # V8.1: subtype -> ModelIDs
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
        
    def _read_csv_cached(self, csv_path: Path, **kwargs) -> pd.DataFrame:
        """Read CSV with pickle caching and robust parser fallbacks."""
        pkl_path = csv_path.with_suffix('.pkl')
        if pkl_path.exists() and pkl_path.stat().st_mtime >= csv_path.stat().st_mtime:
            logger.info(f"Loading cached {pkl_path.name}")
            try:
                return pd.read_pickle(pkl_path)
            except (MemoryError, SystemError, OSError, ValueError) as exc:
                logger.warning(
                    f"Failed to read cache {pkl_path.name} ({exc}); falling back to CSV"
                )
                try:
                    pkl_path.unlink()
                except OSError:
                    pass
        # Use the default parser first. If it fails, fall back to the pure-Python parser.
        try:
            df = pd.read_csv(csv_path, **kwargs)
        except Exception as exc:
            logger.warning(
                f"Default CSV parser failed for {csv_path.name} ({exc}); "
                "retrying with python engine"
            )
            df = pd.read_csv(csv_path, engine='python', **kwargs)
        df.to_pickle(pkl_path)
        logger.info(f"Cached to {pkl_path.name}")
        return df

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
        df = self._read_csv_cached(crispr_path, index_col=0)
        
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
    
    def load_hotspot_mutations(self) -> Optional[pd.DataFrame]:
        """V8.1: Load per-cell-line hotspot mutation matrix.
        
        Returns DataFrame indexed by ModelID with binary columns per gene
        (column format: 'GENE (ENTREZ)' with values 0/1+).
        """
        if self._hotspot_df is not None:
            return self._hotspot_df
        
        hs_path = self.data_dir / 'OmicsSomaticMutationsMatrixHotspot.csv'
        if not hs_path.exists():
            logger.warning('OmicsSomaticMutationsMatrixHotspot.csv not found — subtypes disabled')
            return None
        
        logger.info('Loading hotspot mutation matrix for subtype stratification')
        df = pd.read_csv(hs_path)
        # Drop non-gene metadata columns before aggregation
        meta_cols = ['SequencingID', 'ModelConditionID', 'IsDefaultEntryForModel',
                     'IsDefaultEntryForMC']
        drop_cols = [c for c in meta_cols if c in df.columns]
        if drop_cols:
            df = df.drop(columns=drop_cols)
        if 'ModelID' in df.columns:
            # Multiple sequencing runs per cell line — take max (any mutation)
            df = df.groupby('ModelID').max()
        self._hotspot_df = df
        logger.info(f'Loaded hotspot mutations: {df.shape[0]} cell lines x {df.shape[1]} genes')
        return self._hotspot_df
    
    def _find_gene_column_in_hotspot(self, gene: str) -> Optional[str]:
        """Find the column matching 'GENE (ENTREZ)' format in hotspot matrix."""
        hs = self.load_hotspot_mutations()
        if hs is None:
            return None
        for c in hs.columns:
            if c == gene or c.startswith(gene + ' ('):
                return c
        return None
    
    def build_molecular_subtypes(self) -> Dict[str, List[str]]:
        """V8.1: Build mutation-defined subtypes for configured cancer types.
        
        For each cancer type in SUBTYPE_DRIVERS, identifies cell lines with
        hotspot mutations in clinically relevant driver genes and creates
        subtype entries like 'Non-Small Cell Lung Cancer [KRAS-mut]'.
        
        Returns dict mapping subtype name -> list of ModelIDs.
        Also populates self._subtype_cell_lines for use by get_cell_lines_for_cancer.
        """
        if self._subtype_cell_lines:
            return self._subtype_cell_lines
        
        hs = self.load_hotspot_mutations()
        if hs is None:
            return {}
        
        model_df = self._load_model_metadata()
        result: Dict[str, List[str]] = {}
        
        for cancer_type, drivers in self.SUBTYPE_DRIVERS.items():
            # Get all cell lines for this cancer type
            all_lines = model_df[
                model_df['OncotreePrimaryDisease'] == cancer_type
            ].index.tolist()
            # Intersect with lines that have mutation data
            available = [m for m in all_lines if m in hs.index]
            if len(available) < self.MIN_SUBTYPE_CELL_LINES:
                continue
            
            for gene, min_frac in drivers:
                col = self._find_gene_column_in_hotspot(gene)
                if col is None:
                    # Try damaging mutations as fallback
                    continue
                
                mutant_lines = [
                    m for m in available
                    if m in hs.index and hs.loc[m, col] > 0
                ]
                
                frac = len(mutant_lines) / len(available) if available else 0
                if frac < min_frac:
                    logger.debug(
                        f'{cancer_type} {gene}-mut: {len(mutant_lines)}/{len(available)}'
                        f' ({frac:.0%}) below threshold {min_frac:.0%}, skipping'
                    )
                    continue
                
                if len(mutant_lines) < self.MIN_SUBTYPE_CELL_LINES:
                    logger.debug(
                        f'{cancer_type} {gene}-mut: only {len(mutant_lines)} lines, '
                        f'need {self.MIN_SUBTYPE_CELL_LINES}'
                    )
                    continue
                
                subtype_name = f'{cancer_type} [{gene}-mut]'
                result[subtype_name] = mutant_lines
                logger.info(
                    f'Subtype: {subtype_name} = {len(mutant_lines)} cell lines '
                    f'({frac:.0%} of {cancer_type})'
                )
        
        # Also try loading damaging mutation matrix for genes not found via hotspot.
        # (FLT3-ITD is damaging, not always in hotspot matrix.)
        # Only load columns we need to avoid reading 226MB file fully.
        dmg_path = self.data_dir / 'OmicsSomaticMutationsMatrixDamaging.csv'
        if dmg_path.exists():
            # Collect genes still needed
            needed_genes = set()
            for cancer_type, drivers in self.SUBTYPE_DRIVERS.items():
                for gene, _ in drivers:
                    if f'{cancer_type} [{gene}-mut]' not in result:
                        needed_genes.add(gene)
            
            if needed_genes:
                # Read just header to find column names
                import csv as _csv
                with open(dmg_path) as _f:
                    header = next(_csv.reader(_f))
                cols_to_load = ['ModelID']
                gene_col_map = {}
                for gene in needed_genes:
                    for c in header:
                        if c == gene or c.startswith(gene + ' ('):
                            cols_to_load.append(c)
                            gene_col_map[gene] = c
                            break
                
                if gene_col_map:
                    dmg = pd.read_csv(dmg_path, usecols=cols_to_load)
                    # Deduplicate: multiple sequencing runs per cell line
                    dmg = dmg.groupby('ModelID').max()
                    
                    for cancer_type, drivers in self.SUBTYPE_DRIVERS.items():
                        all_lines = model_df[
                            model_df['OncotreePrimaryDisease'] == cancer_type
                        ].index.tolist()
                        available = [m for m in all_lines if m in dmg.index]
                        for gene, min_frac in drivers:
                            subtype_name = f'{cancer_type} [{gene}-mut]'
                            if subtype_name in result:
                                continue
                            flt_col = gene_col_map.get(gene)
                            if flt_col is None:
                                continue
                            mutant_lines = [
                                m for m in available
                                if dmg.at[m, flt_col] > 0
                            ]
                            if len(mutant_lines) >= self.MIN_SUBTYPE_CELL_LINES:
                                frac = len(mutant_lines) / len(available) if available else 0
                                result[subtype_name] = mutant_lines
                                logger.info(
                                    f'Subtype (damaging): {subtype_name} = {len(mutant_lines)} lines '
                                    f'({frac:.0%})'
                                )
        
        # ── CN amplification subtypes (e.g. HER2+ breast) ──────────────
        cn_path = self.data_dir / 'OmicsCNGeneWGS.csv'
        if cn_path.exists() and self.CN_SUBTYPES:
            needed_cn_genes: Dict[str, str] = {}   # gene -> column name
            for cancer_type_cn, amp_list in self.CN_SUBTYPES.items():
                for gene, _ in amp_list:
                    subtype_name = f'{cancer_type_cn} [{gene}-amp]'
                    if subtype_name not in result:
                        needed_cn_genes.setdefault(gene, None)

            if needed_cn_genes:
                # Discover correct column names from header
                import csv as _csv2
                with open(cn_path) as _f2:
                    cn_header = next(_csv2.reader(_f2))
                cols_to_load_cn = ['ModelID']
                for gene in list(needed_cn_genes.keys()):
                    for c in cn_header:
                        if c == gene or c.startswith(gene + ' ('):
                            cols_to_load_cn.append(c)
                            needed_cn_genes[gene] = c
                            break

                cn_cols_found = {g: c for g, c in needed_cn_genes.items() if c is not None}
                if cn_cols_found:
                    cn_df = pd.read_csv(cn_path, usecols=cols_to_load_cn)
                    cn_df = cn_df.groupby('ModelID').max()

                    for cancer_type_cn, amp_list in self.CN_SUBTYPES.items():
                        all_lines_cn = model_df[
                            model_df['OncotreePrimaryDisease'] == cancer_type_cn
                        ].index.tolist()
                        available_cn = [m for m in all_lines_cn if m in cn_df.index]

                        for gene, threshold in amp_list:
                            subtype_name = f'{cancer_type_cn} [{gene}-amp]'
                            if subtype_name in result:
                                continue
                            cn_col = cn_cols_found.get(gene)
                            if cn_col is None:
                                continue
                            amp_lines = [
                                m for m in available_cn
                                if cn_df.at[m, cn_col] > threshold
                            ]
                            if len(amp_lines) >= self.MIN_SUBTYPE_CELL_LINES:
                                frac = len(amp_lines) / len(available_cn) if available_cn else 0
                                result[subtype_name] = amp_lines
                                logger.info(
                                    f'Subtype (CN-amp): {subtype_name} = {len(amp_lines)} lines '
                                    f'({frac:.0%} of {cancer_type_cn})'
                                )

        self._subtype_cell_lines = result
        return result
    
    def get_cell_lines_for_cancer(self, cancer_type: str) -> List[str]:
        """Get ModelIDs for a specific cancer type.
        
        V8.1: Also handles subtype-qualified names like
        'Non-Small Cell Lung Cancer [KRAS-mut]' by returning only
        the mutation-defined subset of cell lines.
        """
        # V8.1: Check subtype overrides first
        if cancer_type in self._subtype_cell_lines:
            return self._subtype_cell_lines[cancer_type]
        
        cancer_type = normalize_cancer_type(cancer_type)
        
        # Also check after normalization
        if cancer_type in self._subtype_cell_lines:
            return self._subtype_cell_lines[cancer_type]
        
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
    
    def get_available_cancer_types(self, include_subtypes: bool = False) -> List[Tuple[str, int]]:
        """Get all available cancer types with cell line counts.
        
        Args:
            include_subtypes: If True, also include mutation-defined molecular
                subtypes (e.g., 'NSCLC [KRAS-mut]') as separate entries.
        """
        model_df = self._load_model_metadata()
        counts = model_df['OncotreePrimaryDisease'].value_counts()
        result = [(name, count) for name, count in counts.items() if pd.notna(name)]
        
        if include_subtypes:
            subtypes = self.build_molecular_subtypes()
            for subtype_name, cell_lines in subtypes.items():
                result.append((subtype_name, len(cell_lines)))
        
        return result
    
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
        for fname in ('CCLE_expression.csv', 'CCLE_RNAseq_reads.csv',
                      'OmicsExpressionProteinCodingGenesTPMLogp1.csv',
                      'OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv'):
            expr_path = self.data_dir / fname
            if expr_path.exists() and expr_path.stat().st_size > 0:
                logger.info(f"Loading expression from {fname}")
                df = self._read_csv_cached(expr_path, index_col=0)
                # 25Q3+ format has metadata columns before gene columns;
                # detect and reindex by ModelID, drop non-gene columns.
                if 'ModelID' in df.columns:
                    # Keep only default entry per model to avoid duplicate indices
                    if 'IsDefaultEntryForModel' in df.columns:
                        df = df[df['IsDefaultEntryForModel'] == 'Yes']
                    df = df.set_index('ModelID')
                    meta_cols = [c for c in df.columns
                                 if not re.match(r'^[A-Z0-9].*\(\d+\)$', c)]
                    df = df.drop(columns=meta_cols, errors='ignore')
                # Deduplicate index (keep first if still duplicated)
                if df.index.duplicated().any():
                    df = df[~df.index.duplicated(keep='first')]
                # Parse gene column names (strip Entrez IDs)
                rename = {}
                for col in df.columns:
                    m = re.match(r'^([A-Z0-9\-]+)\s*\(\d+\)$', col)
                    if m:
                        rename[col] = m.group(1)
                if rename:
                    df = df.rename(columns=rename)
                    df = df.loc[:, ~df.columns.duplicated()]
                # Ensure gene columns are numeric (chunked CSV reader may leave object dtype)
                obj_cols = df.columns[df.dtypes == object]
                if len(obj_cols) > 0:
                    df[obj_cols] = df[obj_cols].apply(pd.to_numeric, errors='coerce')
                self._expression_df = df
                logger.info(f"Expression matrix: {df.shape[0]} lines × {df.shape[1]} genes")
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

    # ── V7: Copy-number data loader ──────────────────────────────
    def load_copy_number(self) -> Optional[pd.DataFrame]:
        """Load gene-level copy-number data (rows=cell lines, cols=genes).

        Looks for OmicsCNGeneWGS.csv (25Q3+) or OmicsCNGene.csv or
        CCLE_gene_cn.csv in the depmap_data directory.
        Returns None if no CN file is found.
        """
        if self._cn_df is not None:
            return self._cn_df
        for fname in ('OmicsCNGeneWGS.csv', 'OmicsCNGene.csv', 'CCLE_gene_cn.csv'):
            cn_path = self.data_dir / fname
            if cn_path.exists() and cn_path.stat().st_size > 0:
                logger.info(f"Loading copy-number data from {fname}")
                df = self._read_csv_cached(cn_path, index_col=0)
                # Handle 25Q3+ multi-index format
                if 'ModelID' in df.columns:
                    if 'IsDefaultEntryForModel' in df.columns:
                        df = df[df['IsDefaultEntryForModel'] == 'Yes']
                    df = df.set_index('ModelID')
                    meta_cols = [c for c in df.columns
                                 if not re.match(r'^[A-Z0-9].*\(\d+\)$', c)]
                    df = df.drop(columns=meta_cols, errors='ignore')
                if df.index.duplicated().any():
                    df = df[~df.index.duplicated(keep='first')]
                # Strip Entrez IDs from column names
                rename = {}
                for col in df.columns:
                    m = re.match(r'^([A-Z0-9\-]+)\s*\(\d+\)$', col)
                    if m:
                        rename[col] = m.group(1)
                if rename:
                    df = df.rename(columns=rename)
                    df = df.loc[:, ~df.columns.duplicated()]
                obj_cols = df.columns[df.dtypes == object]
                if len(obj_cols) > 0:
                    df[obj_cols] = df[obj_cols].apply(pd.to_numeric, errors='coerce')
                self._cn_df = df
                logger.info(f"Copy-number matrix: {df.shape[0]} lines × {df.shape[1]} genes")
                return self._cn_df
        return None

    # ── V7: RNAi screen data loader ──────────────────────────────
    def load_rnai_dependencies(self) -> Optional[pd.DataFrame]:
        """Load RNAi (shRNA) gene-effect scores.

        Looks for ScreenGeneEffect.csv (DepMap 25Q3+) or
        D2_combined_gene_dep_scores.csv.  Returns a DataFrame
        (rows=cell lines, cols=genes) or None.
        """
        if self._rnai_df is not None:
            return self._rnai_df
        for fname in ('ScreenGeneEffect.csv', 'D2_combined_gene_dep_scores.csv',
                      'RNAi_merged.csv'):
            rnai_path = self.data_dir / fname
            if rnai_path.exists() and rnai_path.stat().st_size > 0:
                logger.info(f"Loading RNAi dependency data from {fname}")
                df = self._read_csv_cached(rnai_path, index_col=0)
                if df.index.duplicated().any():
                    df = df[~df.index.duplicated(keep='first')]
                rename = {}
                for col in df.columns:
                    m = re.match(r'^([A-Z0-9\-]+)\s*\(\d+\)$', col)
                    if m:
                        rename[col] = m.group(1)
                if rename:
                    df = df.rename(columns=rename)
                    df = df.loc[:, ~df.columns.duplicated()]
                obj_cols = df.columns[df.dtypes == object]
                if len(obj_cols) > 0:
                    df[obj_cols] = df[obj_cols].apply(pd.to_numeric, errors='coerce')
                self._rnai_df = df
                logger.info(f"RNAi matrix: {df.shape[0]} lines × {df.shape[1]} genes")
                return self._rnai_df
        return None

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
                    raw_df = pd.read_csv(StringIO(response.text), sep='\t')
                    
                    # The API returns both integer Entrez IDs (source, target)
                    # and gene symbols (source_genesymbol, target_genesymbol).
                    # Drop the integer columns FIRST to avoid duplicate names
                    # after renaming genesymbol columns to 'source'/'target'.
                    raw_df = raw_df.drop(columns=['source', 'target'], errors='ignore')
                    
                    # Rename to our standard format
                    raw_df = raw_df.rename(columns={
                        'source_genesymbol': 'source',
                        'target_genesymbol': 'target',
                        'is_stimulation': 'stimulation',
                        'is_inhibition': 'inhibition'
                    })
                    
                    # Determine interaction type (vectorized, not .apply)
                    conditions = [
                        raw_df.get('stimulation', pd.Series(0, index=raw_df.index)) == 1,
                        raw_df.get('inhibition', pd.Series(0, index=raw_df.index)) == 1,
                    ]
                    choices = ['activation', 'inhibition']
                    raw_df['interaction_type'] = np.select(conditions, choices, default='unknown')
                    raw_df['database'] = 'OmniPath'
                    
                    # Keep only the columns we use (saves memory with 137K rows)
                    keep_cols = ['source', 'target', 'interaction_type', 'database',
                                 'stimulation', 'inhibition']
                    keep_cols = [c for c in keep_cols if c in raw_df.columns]
                    self._network_df = raw_df[keep_cols].copy()
                    
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

    Primary source: ChEMBL 36 (1,542 gene targets with drug mechanisms,
    1,510 with cancer-specific indication data).  Falls back to the
    hand-curated constants for toxicity scores (ChEMBL doesn't carry
    tox scores) and for the 3 genes ChEMBL misses (STAT3, MYC, RB1).

    Cancer-indication-aware: ``get_druggability_score(gene, cancer_type)``
    distinguishes between *approved-for-this-cancer* (1.0) and
    *approved-for-other-cancer* (≤0.9), which prevents universally-
    druggable genes from dominating every cancer type equally.
    """

    def __init__(self):
        # ChEMBL-backed database (loads from cache in <0.5 s)
        self._chembl = None
        try:
            from alin.chembl_data import ChEMBLDrugDB
            self._chembl = ChEMBLDrugDB()
            logger.info(
                "DrugTargetDB: ChEMBL-backed (%d genes, %d cancer-indication)",
                self._chembl.gene_count,
                len([g for g in self._chembl._gene_cancer if self._chembl._gene_cancer[g]]),
            )
        except Exception as exc:
            logger.warning("ChEMBL data not available (%s); using hand-curated fallback only", exc)

        # V9: PortalCompounds.csv — DepMap-curated drug→gene target mapping
        # Provides 2,010 unique gene targets across 4,261 compounds
        self._portal_gene_to_drugs: Dict[str, List[str]] = {}
        self._portal_loaded = False
        try:
            _portal_paths = [
                Path("./data/PortalCompounds.csv"),
                Path("./depmap_data/PortalCompounds.csv"),
            ]
            for _pp in _portal_paths:
                if _pp.exists() and _pp.stat().st_size > 0:
                    import csv as _csv
                    with open(_pp, encoding="utf-8") as _f:
                        _reader = _csv.DictReader(_f)
                        for _row in _reader:
                            _targets = _row.get("GeneSymbolOfTargets", "")
                            _name = _row.get("CompoundName", "")
                            if not _targets or not _name:
                                continue
                            for _g in _targets.split(";"):
                                _g = _g.strip()
                                if _g:
                                    self._portal_gene_to_drugs.setdefault(_g, []).append(_name)
                    self._portal_loaded = True
                    logger.info("DrugTargetDB: PortalCompounds loaded (%d genes, %s)",
                                len(self._portal_gene_to_drugs), _pp.name)
                    break
        except Exception as exc:
            logger.debug("PortalCompounds load failed: %s", exc)

        # Hand-curated fallback (constants.py — 35 genes)
        self.DRUG_DB = {
            gene: {
                'drugs': drugs,
                'stage': GENE_CLINICAL_STAGE.get(gene, 'preclinical'),
                'toxicity': GENE_TOXICITY_SCORES.get(gene, 0.5),
                'toxicities': GENE_TOXICITIES.get(gene, []),
            }
            for gene, drugs in GENE_TO_DRUGS.items()
        }

    def get_druggability_score(self, gene: str,
                               cancer_type: Optional[str] = None) -> float:
        """
        Cancer-indication-aware druggability score (0–1).

        If ChEMBL data is available, uses cancer-specific phase data.
        Falls back to hand-curated constants for unknown genes.
        """
        # Try ChEMBL first (cancer-aware)
        if self._chembl is not None and self._chembl.has_gene(gene):
            return self._chembl.get_druggability_score(gene, cancer_type=cancer_type)

        # Fall back to hand-curated data
        if gene in self.DRUG_DB:
            info = self.DRUG_DB[gene]
            stage_scores = {'approved': 1.0, 'phase3': 0.8, 'phase2': 0.6, 'phase1': 0.4, 'preclinical': 0.2}
            base = stage_scores.get(info['stage'], 0.2)
            n_drugs = len(info.get('drugs', []))
            bonus = min(0.2, n_drugs * 0.05)
            return min(1.0, base + bonus)

        # V9: PortalCompounds — gene is druggable if DepMap lists compounds for it
        if self._portal_loaded and gene in self._portal_gene_to_drugs:
            n_cpds = len(self._portal_gene_to_drugs[gene])
            # PortalCompounds doesn't carry phase info, so give a moderate score
            # 0.35 base (known compound exists) + bonus for multiple compounds
            return min(0.7, 0.35 + min(0.35, n_cpds * 0.02))

        return 0.2  # Unknown genes get low score

    def has_approved_drug(self, gene: str,
                          cancer_type: Optional[str] = None) -> bool:
        """Check if gene has an approved or phase3 drug (optionally for a specific cancer)."""
        if self._chembl is not None and self._chembl.has_gene(gene):
            phase = self._chembl.get_max_phase(gene, cancer_type=cancer_type)
            return phase >= 3.0
        info = self.DRUG_DB.get(gene)
        return info is not None and info.get('stage') in ('approved', 'phase3')

    def get_toxicity_score(self, gene: str) -> float:
        """Get toxicity score (0=safe, 1=highly toxic)"""
        # Toxicity comes from hand-curated data (ChEMBL doesn't carry tox scores)
        if gene in self.DRUG_DB:
            return self.DRUG_DB[gene].get('toxicity', 0.5)
        return 0.5  # Unknown

    def get_drug_info(self, gene: str) -> Optional[DrugTarget]:
        """Get full drug target information"""
        # Try ChEMBL first for richer drug lists
        if self._chembl is not None and self._chembl.has_gene(gene):
            drugs = self._chembl.get_drugs_for_gene(gene)
            stage = self._chembl.get_clinical_stage(gene)
            tox_list = self.DRUG_DB.get(gene, {}).get('toxicities', [])
            return DrugTarget(
                gene=gene,
                available_drugs=drugs[:10],  # Cap at 10 most relevant
                clinical_stage=stage,
                known_toxicities=tox_list,
            )

        if gene not in self.DRUG_DB:
            return None

        info = self.DRUG_DB[gene]
        return DrugTarget(
            gene=gene,
            available_drugs=info.get('drugs', []),
            clinical_stage=info.get('stage', 'unknown'),
            known_toxicities=info.get('toxicities', []),
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
        Infer essential gene modules for a cancer type (vectorized).
        
        Refinements:
        1. Selectivity: only genes essential in >min_selectivity_fraction of cancer cell lines
        2. Co-essentiality clustering: genes essential together = same pathway (clustered)
        3. Expression filter: if expression data available, only count essential if expressed
        
        Uses vectorized numpy/pandas operations instead of per-gene Python loops.
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
        
        # Remove pan-essential genes and artifact genes upfront (vectorized column filter)
        non_pan_cols = [g for g in crispr_subset.columns
                        if g not in pan_essential and not _is_artifact(g)]
        crispr_subset = crispr_subset[non_pan_cols]
        
        # --- Vectorized essential mask: gene is essential if CRISPR score < threshold ---
        essential_mask = (crispr_subset < dependency_threshold)  # bool DataFrame (lines × genes)
        
        # Optional expression filter (vectorized)
        expr_df = self.depmap.load_expression()
        if expr_df is not None:
            common_lines = [cl for cl in available_lines if cl in expr_df.index]
            if common_lines:
                common_genes = [g for g in non_pan_cols if g in expr_df.columns]
                if common_genes:
                    expr_sub = expr_df.loc[common_lines, common_genes]
                    # Essential only if expressed (>= threshold); NaN → not expressed → not essential
                    expr_mask = expr_sub.fillna(0.0) >= expression_threshold
                    essential_mask.loc[common_lines, common_genes] &= expr_mask
        
        # Filter lines with >= 2 essential genes (matches original per-line filter)
        line_ess_counts = essential_mask.sum(axis=1)
        valid_lines = line_ess_counts[line_ess_counts >= 2].index
        essential_mask = essential_mask.loc[valid_lines]
        
        # Selectivity filter: genes essential in >= min_lines_essential valid lines
        gene_counts = essential_mask.sum(axis=0)
        selective_mask = gene_counts >= min_lines_essential
        selective_genes = gene_counts[selective_mask].index.tolist()
        
        if len(selective_genes) < 2:
            logger.info(f"Few selective genes ({len(selective_genes)}); falling back to consensus")
            mean_dep = crispr_subset.mean(axis=0)
            selective_genes = list(set(mean_dep[mean_dep < dependency_threshold].index))
        
        if len(selective_genes) < 2:
            return []
        
        # --- Vectorized Jaccard co-essentiality matrix ---
        # Binary matrix: (n_genes × n_valid_lines)
        ess_binary = essential_mask[selective_genes].values.astype(np.float32).T
        
        # Jaccard = |intersection| / |union|; intersection = A @ A^T
        intersection = ess_binary @ ess_binary.T
        sizes = ess_binary.sum(axis=1, keepdims=True)  # per-gene essential line count
        union = sizes + sizes.T - intersection
        union = np.maximum(union, 1)  # avoid div-by-zero
        co_essential = intersection / union
        np.fill_diagonal(co_essential, 0)
        
        # Distance matrix (1 - Jaccard similarity)
        dist = 1 - co_essential
        dist = (dist + dist.T) / 2
        np.fill_diagonal(dist, 0)
        
        # Hierarchical clustering; cut to get ~5-15 clusters
        n_genes = len(selective_genes)
        n_clusters = min(15, max(3, n_genes // 5))
        try:
            from scipy.spatial.distance import squareform
            from scipy.cluster.hierarchy import fcluster, linkage
            condensed = squareform(dist, checks=False)
            Z = linkage(condensed, method='ward')
            clusters = fcluster(Z, n_clusters, criterion='maxclust')
            cluster_to_genes = defaultdict(set)
            for gene, c in zip(selective_genes, clusters):
                cluster_to_genes[c].add(gene)

            # ── Cluster quality metric (audit recommendation #2) ──
            # Compute silhouette score to quantify how well-separated
            # the co-essentiality clusters are.  Values near +1 mean
            # tight, well-separated clusters; near 0 means overlapping;
            # negative means mis-clustered genes.
            silhouette_avg = None
            if len(set(clusters)) >= 2 and n_genes >= 4:
                try:
                    from sklearn.metrics import silhouette_score as _silhouette
                    silhouette_avg = float(_silhouette(dist, clusters, metric='precomputed'))
                    logger.info(f"Co-essentiality silhouette score: {silhouette_avg:.3f} "
                                f"({n_clusters} clusters, {n_genes} genes)")
                    if silhouette_avg < 0.1:
                        logger.warning(f"Low silhouette ({silhouette_avg:.3f}) for {cancer_type}: "
                                       f"co-essentiality clusters may not be well-separated")
                except ImportError:
                    pass  # sklearn not available — non-critical
        except Exception as e:
            logger.debug(f"Co-essentiality clustering failed ({e}), using single module")
            cluster_to_genes = {0: set(selective_genes)}
        
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
        
        # Build directed graph — two-phase approach for large OmniPath
        # networks: (1) build the full CRISPR-filtered graph, then
        # (2) extract a focused subgraph around drivers/effectors to
        # make all_simple_paths tractable (137K edges → ~10-30K edges).
        crispr_genes = set(crispr.columns)
        mask = network['source'].isin(crispr_genes) & network['target'].isin(crispr_genes)
        relevant_edges = network.loc[mask, ['source', 'target']]
        
        G_full = nx.DiGraph()
        G_full.add_edges_from(zip(relevant_edges['source'], relevant_edges['target']))
        logger.info(f"Full signaling graph: {G_full.number_of_nodes()} nodes, "
                     f"{G_full.number_of_edges()} edges (from {len(network)} total)")
        
        # Build focused subgraph: essential genes + drivers + effectors
        # + 2-hop neighborhoods (forward from drivers, reverse from
        # effectors).  This captures ALL intermediates for paths ≤ 4.
        focus_nodes = set(essential_genes) | drivers | effectors
        for d in drivers:
            if d not in G_full:
                continue
            for n1 in G_full.successors(d):
                focus_nodes.add(n1)
                for n2 in G_full.successors(n1):
                    focus_nodes.add(n2)
        for e in effectors:
            if e not in G_full:
                continue
            for n1 in G_full.predecessors(e):
                focus_nodes.add(n1)
                for n2 in G_full.predecessors(n1):
                    focus_nodes.add(n2)
        G = G_full.subgraph(focus_nodes).copy()
        del G_full  # free memory
        logger.info(f"Focused subgraph: {G.number_of_nodes()} nodes, "
                     f"{G.number_of_edges()} edges")
        
        paths = []
        seen_nodes = set()
        _MAX_PATHS_PER_PAIR = 50   # cap paths per driver→effector pair
        _MAX_TOTAL_PATHS = 500     # cap total signaling paths
        
        # Pre-compute reachable sets per driver (O(V+E) each,
        # avoids futile all_simple_paths calls for disconnected pairs).
        _reachable = {}
        for driver in drivers:
            if driver in G and G.out_degree(driver) > 0:
                _reachable[driver] = nx.descendants(G, driver)
        
        for driver in drivers:
            if driver not in _reachable:
                continue
            if len(paths) >= _MAX_TOTAL_PATHS:
                break
            
            for effector in effectors:
                if effector not in G or effector not in essential_genes:
                    continue
                if effector not in _reachable[driver]:
                    continue  # no path exists, skip immediately
                
                _pair_count = 0
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
                        _pair_count += 1
                        if _pair_count >= _MAX_PATHS_PER_PAIR:
                            break
                        if len(paths) >= _MAX_TOTAL_PATHS:
                            break
                        
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
        
        # --- Phase 1: vectorized Welch t-test for ALL testable genes ---
        # Filter genes to test (exclude pan-essential and artefacts)
        genes_to_test = [g for g in crispr.columns
                         if g not in pan_essential and not _is_artifact(g)]
        
        cancer_mat = crispr.loc[available_lines, genes_to_test].values.astype(np.float64)
        other_mat = crispr.loc[other_lines, genes_to_test].values.astype(np.float64)
        
        # Compute per-gene stats vectorized (handling NaN)
        n_c = np.sum(~np.isnan(cancer_mat), axis=0)
        n_o = np.sum(~np.isnan(other_mat), axis=0)
        
        # Means ignoring NaN
        with np.errstate(all='ignore'):
            mean_c = np.nanmean(cancer_mat, axis=0)
            mean_o = np.nanmean(other_mat, axis=0)
            var_c = np.nanvar(cancer_mat, axis=0, ddof=1)
            var_o = np.nanvar(other_mat, axis=0, ddof=1)
        
        # Filter: need >= 3 cancer and >= 10 other scores
        valid = (n_c >= 3) & (n_o >= 10)
        
        # Welch t-statistic (vectorized)
        se_c = var_c / n_c
        se_o = var_o / n_o
        se_total = np.sqrt(se_c + se_o)
        se_total = np.where(se_total > 0, se_total, np.nan)
        t_stats = (mean_c - mean_o) / se_total
        
        # Welch-Satterthwaite degrees of freedom
        num = (se_c + se_o) ** 2
        denom = se_c**2 / (n_c - 1) + se_o**2 / (n_o - 1)
        denom = np.where(denom > 0, denom, np.nan)
        df_welch = num / denom
        
        # Two-sided p-values from t-distribution (vectorized)
        from scipy.stats import t as t_dist
        p_values = 2 * t_dist.sf(np.abs(t_stats), df_welch)
        
        # Effect sizes (positive = more essential in cancer)
        effect_sizes = mean_o - mean_c
        
        # Build results for valid genes only
        gene_results = []
        genes_arr = np.array(genes_to_test)
        valid_idx = np.where(valid & np.isfinite(t_stats) & np.isfinite(p_values))[0]
        
        for i in valid_idx:
            gene_results.append({
                'gene': genes_arr[i],
                't_stat': float(t_stats[i]),
                'p_value': float(p_values[i]),
                'effect_size': float(effect_sizes[i]),
                'cancer_mean': float(mean_c[i])
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
    
    Druggability sources:
    1. Gene-level: clinical stage + drug count (DrugTargetDB)
    2. Protein-level (optional): structural, abundance, degradability, PPI
       via ProteinDruggabilityScorer
    """
    
    def __init__(self, depmap: DepMapLoader, drug_db: DrugTargetDB,
                 toxicity_cache_dir: Optional[str] = None,
                 protein_scorer=None,
                 enable_api: bool = True):
        self.depmap = depmap
        self.drug_db = drug_db
        self.toxicity_cache_dir = toxicity_cache_dir
        self.protein_scorer = protein_scorer
        self.enable_api = enable_api
        self._protein_scores = {}  # gene → ProteinDruggabilityScore (lazy cache)
        self._pan_essential = None
        self.discovery_mode = False  # set externally for discovery-mode runs
        
    def _get_pan_essential(self) -> Set[str]:
        if self._pan_essential is None:
            self._pan_essential = self.depmap.get_pan_essential_genes()
        return self._pan_essential
    
    def _get_toxicity_score(self, gene: str) -> float:
        """Get toxicity score, enhanced by OpenTargets if available."""
        base_toxicity = self.drug_db.get_toxicity_score(gene)
        if not self.enable_api:
            return max(0, min(1, base_toxicity))
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
        
        # Druggability (cancer-indication-aware via ChEMBL)
        druggability = self.drug_db.get_druggability_score(gene, cancer_type=cancer_type)
        
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
            protein_druggability_score=self._get_protein_score(gene),
            pan_essential_penalty=pan_penalty,
            base_penalty=1.0
        )
    
    # Cap on protein scoring API calls to avoid making thousands of requests
    # during MHS cost computation (1000+ candidate genes).  The top 100
    # genes get API-scored; the rest fall back to drug-level druggability.
    _PROTEIN_SCORE_API_CAP = 100
    
    def _get_protein_score(self, gene: str) -> Optional[float]:
        """Get protein-level druggability score if scorer is available."""
        if self.protein_scorer is None or not self.enable_api:
            return None
        if gene not in self._protein_scores:
            # Enforce cap: once we've made API calls for _PROTEIN_SCORE_API_CAP
            # genes, skip new API calls (return None → drug-level fallback).
            if len(self._protein_scores) >= self._PROTEIN_SCORE_API_CAP:
                return None
            try:
                result = self.protein_scorer.score_gene(gene)
                self._protein_scores[gene] = result.protein_score
            except (AttributeError, KeyError, ValueError, OSError) as exc:
                logger.debug('Protein score lookup failed for %s: %s', gene, exc)
                self._protein_scores[gene] = None
        return self._protein_scores.get(gene)

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
    
    def __init__(self, cost_function: CostFunction, *, cost_gamma: float = 0.3):
        self.cost_fn = cost_function
        self.cost_gamma = cost_gamma
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
        
        # Pre-resolve UniProt IDs for all candidate genes (batch)
        if (getattr(self.cost_fn, 'enable_api', False)
                and self.cost_fn.protein_scorer is not None
                and hasattr(self.cost_fn.protein_scorer, 'pre_resolve_genes')):
            self.cost_fn.protein_scorer.pre_resolve_genes(list(all_genes))

        # Compute costs
        # In discovery mode, gamma=0 removes the druggability reward so
        # the hitting set solver treats druggable and undruggable targets
        # equally.
        gene_costs = {}
        for gene in all_genes:
            cost_obj = self.cost_fn.compute_cost(gene, cancer_type)
            gene_costs[gene] = cost_obj.total_cost(gamma=self.cost_gamma)
        
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
        
        Uses pre-computed gene→path-index map for fast coverage lookups.
        """
        all_genes = list(gene_costs.keys())
        n_paths = len(paths)
        min_covered = int(np.ceil(min_coverage * n_paths))
        solutions = []

        # Pre-compute gene → set of path indices for O(1) coverage lookup
        gene_to_pidx: Dict[str, set] = {g: set() for g in all_genes}
        path_id_list = []
        for idx, p in enumerate(paths):
            path_id_list.append(p.path_id)
            for g in p.nodes:
                if g in gene_to_pidx:
                    gene_to_pidx[g].add(idx)

        # Pre-compute costs array aligned with gene list
        gene_cost_arr = [gene_costs[g] for g in all_genes]

        for size in range(1, min(max_size + 1, len(all_genes) + 1)):
            for idx_combo in combinations(range(len(all_genes)), size):
                # Fast coverage via set union of pre-computed path indices
                covered_set = set()
                for i in idx_combo:
                    covered_set |= gene_to_pidx[all_genes[i]]
                
                if len(covered_set) >= min_covered:
                    total_cost = sum(gene_cost_arr[i] for i in idx_combo)
                    coverage = len(covered_set) / n_paths
                    paths_covered = {path_id_list[pi] for pi in covered_set}
                    targets = frozenset(all_genes[i] for i in idx_combo)
                    
                    solutions.append(HittingSet(
                        targets=targets,
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
    - **LINCS signature complementarity** (low overlap of downregulated
      genes = independent pathways; one target's up-genes covered by
      the other's down-genes = anti-resistance).  Produces continuous
      scores per combination.
    - Pathway complementarity (hitting independent pathways)
    - Known clinical combination data
    """

    # ── LINCS complementarity sub-score weights (sum = 1.0) ──────────
    W_PATHWAY_INDEPENDENCE = 0.45   # Jaccard distance of effector gene sets
    W_ANTI_RESISTANCE = 0.35        # Feedback gene coverage
    W_Z_ANTICORRELATION = 0.20      # Pearson anti-correlation of z-score profiles
    MIN_SHARED_GENES_FOR_CORR = 20  # Minimum shared genes for z-score correlation

    # ── Synergy blending weights (known / LINCS / pathway) ───────────
    BLEND_KNOWN_LINCS_PATH = (0.40, 0.35, 0.25)  # All three sources
    BLEND_KNOWN_PATH = (0.70, 0.30)               # Known + pathway only
    BLEND_LINCS_PATH = (0.60, 0.40)               # LINCS + pathway only
    BLEND_PATH_ONLY = 0.60                         # Pathway diversity only
    
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
    
    def __init__(self, omnipath: OmniPathLoader, use_known_synergies: bool = True) -> None:
        """Initialize synergy scorer.

        Parameters
        ----------
        omnipath : OmniPathLoader
            Network loader for pathway assignments.
        use_known_synergies : bool
            Whether to incorporate curated known-synergy pairs.
        """
        self.omnipath = omnipath
        self.use_known_synergies = use_known_synergies
        self._lincs_db = None
        self._lincs_checked = False

    def _get_lincs(self) -> Optional[Any]:
        """Lazy-load the LINCS DB (thread-safe via perturbation module)."""
        if not self._lincs_checked:
            try:
                from alin.perturbation import _get_lincs_db
                self._lincs_db = _get_lincs_db()
            except Exception as exc:
                logger.debug("SynergyScorer: LINCS DB unavailable: %s", exc)
            self._lincs_checked = True
        return self._lincs_db
    
    def _lincs_complementarity(self, genes: Set[str]) -> Optional[float]:
        """
        Compute LINCS-based signature complementarity for a gene set.
        
        Three sub-scores (each 0-1, averaged):
        1. **Pathway independence** – mean (1 - Jaccard) of down_gene sets
           between all pairs.  Low overlap → independent mechanisms → synergy.
        2. **Anti-resistance** – fraction of one target's feedback genes
           (up_genes) that fall in another target's effector set (down_genes).
           Higher → combination counteracts resistance → synergy.
        3. **Z-score anti-correlation** – mean negative Pearson of
           z-score vectors between pairs.  Anti-correlated profiles
           indicate complementary mechanisms.
        
        Returns None when no LINCS data is available for *any* target.
        """
        lincs = self._get_lincs()
        if lincs is None:
            return None
        
        # Collect per-target signatures
        sigs = {}
        for g in genes:
            cs = lincs.get_consensus(g)
            if cs is not None and (cs.down_genes or cs.up_genes):
                sigs[g] = cs
        
        if len(sigs) < 2:
            return None  # need at least 2 targets with LINCS data
        
        gene_list = sorted(sigs.keys())
        n_pairs = 0
        independence_sum = 0.0
        anti_resistance_sum = 0.0
        anti_corr_sum = 0.0

        for i in range(len(gene_list)):
            for j in range(i + 1, len(gene_list)):
                a, b = sigs[gene_list[i]], sigs[gene_list[j]]
                n_pairs += 1
                
                # (1) Pathway independence via down-gene Jaccard
                down_a, down_b = set(a.down_genes), set(b.down_genes)
                union = down_a | down_b
                if union:
                    jaccard = len(down_a & down_b) / len(union)
                else:
                    jaccard = 0.0
                independence_sum += (1.0 - jaccard)
                
                # (2) Anti-resistance: A's up (resistance) covered by B's down
                up_a, up_b = set(a.up_genes), set(b.up_genes)
                # bidirectional
                ar_ab = len(up_a & down_b) / max(len(up_a), 1)
                ar_ba = len(up_b & down_a) / max(len(up_b), 1)
                anti_resistance_sum += (ar_ab + ar_ba) / 2.0
                
                # (3) Z-score anti-correlation
                shared_genes = set(a.mean_z.keys()) & set(b.mean_z.keys())
                if len(shared_genes) >= self.MIN_SHARED_GENES_FOR_CORR:
                    za = np.array([a.mean_z[g] for g in shared_genes])
                    zb = np.array([b.mean_z[g] for g in shared_genes])
                    # Pearson correlation
                    za_c = za - za.mean()
                    zb_c = zb - zb.mean()
                    denom = (np.sqrt((za_c ** 2).sum()) * np.sqrt((zb_c ** 2).sum()))
                    if denom > 0:
                        corr = float((za_c * zb_c).sum() / denom)
                    else:
                        corr = 0.0
                    # Negative correlation → complementary → higher score
                    anti_corr_sum += max(0.0, -corr)
                # else: leave at 0 (not enough shared genes to compute)

        if n_pairs == 0:
            return None
        
        independence = independence_sum / n_pairs
        anti_resistance = anti_resistance_sum / n_pairs
        anti_correlation = anti_corr_sum / n_pairs
        
        # Weighted combination (sum = 1.0)
        return round(
            self.W_PATHWAY_INDEPENDENCE * independence +
            self.W_ANTI_RESISTANCE * anti_resistance +
            self.W_Z_ANTICORRELATION * anti_correlation,
            4,
        )

    def compute_synergy_score(self, genes: Set[str], use_known_synergies: Optional[bool] = None) -> float:
        """
        Compute synergy score for a combination of genes.
        
        Blends clinical evidence (known synergies), LINCS-based signature
        complementarity, and pathway diversity to produce a continuous
        0-1 score.
        
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
        
        # LINCS signature complementarity (continuous, data-driven)
        lincs_comp = self._lincs_complementarity(genes)
        
        # Combine scores with appropriate weighting
        if pair_count > 0 and lincs_comp is not None:
            # All three sources available
            w = self.BLEND_KNOWN_LINCS_PATH
            synergy = (
                known_pair_score * w[0] +
                lincs_comp * w[1] +
                pathway_diversity * w[2]
            )
        elif pair_count > 0:
            # Known + pathway (no LINCS)
            w = self.BLEND_KNOWN_PATH
            synergy = (
                known_pair_score * w[0] +
                pathway_diversity * w[1]
            )
        elif lincs_comp is not None:
            # LINCS + pathway (no known synergies)
            w = self.BLEND_LINCS_PATH
            synergy = (
                lincs_comp * w[0] +
                pathway_diversity * w[1]
            )
        else:
            # Pathway diversity only when no other evidence
            synergy = pathway_diversity * self.BLEND_PATH_ONLY
        
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
    Estimate probability of resistance emergence.

    Sources (prioritised):
    1. **LINCS L1000 feedback genes** – genes *upregulated* when a target
       is inhibited represent empirically observed escape routes.  Z-score
       magnitude weights each gene's importance.
    2. **Curated resistance mechanisms** – literature-backed fallback for
       targets without LINCS coverage.
    3. Gene-family compensation heuristic.
    """

    # ── Named constants ──────────────────────────────────────────────
    #: Per-target scaling factor in the default-resistance formula.
    TARGET_SCALING_FACTOR: float = 0.3
    #: Base numerator for the default (no resistance data) formula.
    DEFAULT_RESISTANCE_BASE: float = 0.5
    #: Bonus subtracted from resistance prob when ≥ 2 family members covered.
    FAMILY_COVERAGE_BONUS: float = 0.2
    #: Gene-family sets used for compensation detection.
    SRC_FAMILY: frozenset = frozenset({'SRC', 'FYN', 'YES1', 'LYN', 'LCK'})
    BCL2_FAMILY: frozenset = frozenset({'BCL2', 'MCL1', 'BCL2L1'})

    # Known resistance mechanisms (literature fallback).
    # Each gene maps to a list of (resistance_gene, weight) tuples.
    # Weights reflect strength of literature evidence:
    #   1.0 = validated in clinical resistance (genomic studies)
    #   0.7 = strong preclinical evidence (in-vivo models)
    #   0.5 = moderate evidence (in-vitro or pathway logic)
    RESISTANCE_MECHANISMS: Dict[str, List[Tuple[str, float]]] = {
        # --- RTK signaling ---
        'EGFR': [('MET', 1.0), ('ERBB2', 0.7), ('KRAS', 1.0),
                 ('PIK3CA', 0.7), ('AXL', 0.5), ('FGFR1', 0.5)],
        'ERBB2': [('ERBB3', 0.7), ('PIK3CA', 1.0), ('MET', 0.5),
                  ('EGFR', 0.5), ('SRC', 0.5)],
        'MET': [('EGFR', 0.7), ('ERBB3', 0.5), ('KRAS', 0.5),
                ('AXL', 0.5)],
        'ALK': [('EGFR', 1.0), ('MET', 0.7), ('SRC', 0.7),
                ('KRAS', 0.5)],
        'FGFR1': [('RAS', 0.5), ('PIK3CA', 0.7), ('MET', 0.5),
                  ('EGFR', 0.5)],
        'FGFR2': [('FGFR1', 0.5), ('PIK3CA', 0.7), ('RAS', 0.5)],
        'RET': [('KRAS', 0.7), ('MET', 0.5), ('EGFR', 0.5)],
        # --- RAS-MAPK ---
        'BRAF': [('MAP2K1', 0.7), ('NRAS', 1.0), ('PIK3CA', 0.7),
                 ('CRAF', 0.7), ('ARAF', 0.5)],
        'KRAS': [('PIK3CA', 0.7), ('BRAF', 0.5), ('MET', 0.5),
                 ('SHP2', 0.5), ('FGFR1', 0.5)],
        'MAP2K1': [('MAP2K2', 0.7), ('BRAF', 0.7), ('PIK3CA', 0.5),
                   ('MAPK1', 0.5), ('CRAF', 0.5)],
        'MAP2K2': [('MAP2K1', 0.7), ('BRAF', 0.5)],
        # --- Cell cycle ---
        'CDK4': [('CDK6', 1.0), ('CCNE1', 0.7), ('CDK2', 0.7),
                 ('RB1', 0.5)],
        'CDK6': [('CDK4', 1.0), ('CDK2', 0.7), ('CCNE1', 0.7),
                 ('CCND1', 0.5), ('RB1', 0.5)],
        'CDK2': [('CDK4', 0.5), ('CDK6', 0.5), ('CCNE1', 0.7)],
        # --- PI3K-AKT-mTOR ---
        'PIK3CA': [('AKT1', 0.7), ('MTOR', 0.5), ('PTEN', 0.5),
                   ('RAS', 0.5)],
        'MTOR': [('PIK3CA', 0.7), ('AKT1', 0.7), ('MAPK1', 0.5),
                 ('4EBP1', 0.5)],
        'AKT1': [('PIK3CA', 0.5), ('MTOR', 0.5), ('MAPK1', 0.5)],
        # --- Apoptosis ---
        'BCL2': [('MCL1', 1.0), ('BCL2L1', 0.7)],
        'MCL1': [('BCL2', 0.7), ('BCL2L1', 0.7)],
        'BCL2L1': [('MCL1', 0.7), ('BCL2', 0.5)],
        # --- SRC family ---
        'SRC': [('FYN', 0.7), ('YES1', 0.7), ('LYN', 0.5)],
        # --- JAK-STAT ---
        'JAK1': [('JAK2', 0.7), ('STAT3', 0.5), ('STAT5A', 0.5)],
        'JAK2': [('JAK1', 0.7), ('STAT3', 0.5)],
        'STAT3': [('STAT5A', 0.7), ('NFKB1', 0.5), ('JAK1', 0.5)],
        # --- Other ---
        'PARP1': [('ATR', 0.7), ('BRCA1', 0.5), ('RAD51', 0.5)],
        'IDH1': [('IDH2', 0.7)],
        'IDH2': [('IDH1', 0.7)],
    }
    
    def __init__(self, omnipath: OmniPathLoader, depmap: DepMapLoader):
        self.omnipath = omnipath
        self.depmap = depmap
        self._lincs_db = None
        self._lincs_checked = False
        
    def _get_lincs(self) -> Optional[Any]:
        """Lazy-load the LINCS DB (thread-safe via perturbation module)."""
        if not self._lincs_checked:
            try:
                from alin.perturbation import _get_lincs_db
                self._lincs_db = _get_lincs_db()
            except Exception as exc:
                logger.debug("ResistanceProbabilityEstimator: LINCS DB unavailable: %s", exc)
            self._lincs_checked = True
        return self._lincs_db

    def estimate_resistance_probability(self, targets: Set[str], cancer_type: str) -> float:
        """
        Estimate probability of resistance for a target combination.
        
        Uses LINCS feedback genes (upregulated upon target inhibition) to
        produce continuous, target-specific resistance scores.  Falls back
        to the curated dictionary for targets not in LINCS.
        
        Lower is better (more likely to prevent resistance).
        """
        if len(targets) == 0:
            return 1.0
        
        # ── Collect resistance genes with importance weights ──────────
        # weight: 0-1 importance (from z-score magnitude or 1.0 for curated)
        resistance_weights: dict = {}  # gene → max weight
        
        lincs = self._get_lincs()
        lincs_covered_targets = 0
        
        for target in targets:
            # (a) LINCS data-driven feedback genes
            if lincs is not None:
                cs = lincs.get_consensus(target)
                if cs is not None and cs.up_genes:
                    lincs_covered_targets += 1
                    # Weight by z-score magnitude (normalised to 0-1)
                    z_vals = {g: cs.mean_z.get(g, 0) for g in cs.up_genes}
                    max_z = max(abs(v) for v in z_vals.values()) if z_vals else 1.0
                    for gene, z in z_vals.items():
                        w = abs(z) / max(max_z, 1e-6)
                        # Keep maximum weight across targets
                        if gene not in resistance_weights or w > resistance_weights[gene]:
                            resistance_weights[gene] = round(w, 4)
            
            # (b) Curated fallback (always added – may overlap with LINCS)
            if target in self.RESISTANCE_MECHANISMS:
                for gene, w in self.RESISTANCE_MECHANISMS[target]:
                    if gene not in resistance_weights or w > resistance_weights[gene]:
                        resistance_weights[gene] = w
        
        if not resistance_weights:
            # No resistance data at all → moderate default
            return max(0.0, min(1.0,
                self.DEFAULT_RESISTANCE_BASE / (1.0 + len(targets) * self.TARGET_SCALING_FACTOR)))
        
        # ── Cancer-specific modulation ───────────────────────────────
        # Scale each resistance gene's weight by its DepMap dependency in
        # this cancer type.  If a resistance gene is not essential in this
        # cancer's cell lines, it is less likely to serve as an escape
        # route.  This makes resistance scores vary per cancer type even
        # for the same target combination.
        try:
            cell_lines = self.depmap.get_cell_lines_for_cancer(cancer_type)
            if (cell_lines
                    and hasattr(self.depmap, '_crispr_df')
                    and self.depmap._crispr_df is not None):
                crispr = self.depmap._crispr_df
                usable_cls = [cl for cl in cell_lines if cl in crispr.index]
                if len(usable_cls) >= 3:
                    for gene in list(resistance_weights):
                        if gene in crispr.columns:
                            mean_dep = float(crispr.loc[usable_cls, gene].mean())
                            # mean_dep is Chronos score: more negative = more essential
                            # Map to 0-1 scale: -1.0→1.0 (very essential → full weight),
                            #  0.0→0.5 (neutral), +0.5→0.25 (non-essential → low weight)
                            expression_factor = max(0.15, min(1.0, 0.5 - mean_dep * 0.5))
                            resistance_weights[gene] = round(
                                resistance_weights[gene] * expression_factor, 4)
        except Exception:
            pass  # Fall through to unmodified weights
        
        # ── Compute weighted uncovered fraction ──────────────────────
        # Genes already targeted by the combination are "covered"
        uncovered = {g: w for g, w in resistance_weights.items() if g not in targets}
        total_weight = sum(resistance_weights.values())
        uncovered_weight = sum(uncovered.values())
        
        base_prob = uncovered_weight / (total_weight + 1e-6)
        
        # Adjust based on number of targets (more targets = harder to develop resistance)
        n_targets = len(targets)
        target_modifier = 1.0 / (1.0 + n_targets * self.TARGET_SCALING_FACTOR)
        
        # Check if we're targeting compensatory family members
        family_coverage_bonus = 0.0
        if len(targets & self.SRC_FAMILY) >= 2:
            family_coverage_bonus -= self.FAMILY_COVERAGE_BONUS
        if len(targets & self.BCL2_FAMILY) >= 2:
            family_coverage_bonus -= self.FAMILY_COVERAGE_BONUS
        
        resistance_prob = base_prob * target_modifier + family_coverage_bonus
        
        # Enforce a biologically realistic floor: no drug combination has
        # truly zero resistance probability.  Even the most comprehensive
        # multi-target regimen faces acquired resistance (~5% floor based
        # on clinical data for triple combinations in oncology).
        _RESISTANCE_FLOOR = 0.05
        return max(_RESISTANCE_FLOOR, min(1.0, resistance_prob))


# ============================================================================
# TRIPLE COMBINATION FINDER - Main Systems Biology Engine
# ============================================================================

# TripleCombination dataclass imported from core.data_structures (single source of truth)
from core.data_structures import TripleCombination


def resolve_strategy_arm(combo: TripleCombination) -> str:
    """Resolve one explicit strategy-arm label for current and legacy triples."""
    explicit_arm = str(getattr(combo, 'strategy_arm', '') or '').strip()
    if explicit_arm:
        try:
            return normalize_strategy_arm(explicit_arm)
        except ValueError:
            logger.debug("Unknown combo.strategy_arm=%s; falling back to scoring mode", explicit_arm)
    return infer_strategy_arm_from_scoring_mode(getattr(combo, 'scoring_mode', None))


class TripleCombinationFinder:
    """
    High-throughput systems biology approach to find optimal triple combinations
    
    Methodology from the PDAC paper:
    1. Identify X-nodes (network convergence points)
    2. Score synergistic interactions
    3. Minimize resistance probability
    4. Maximize pathway coverage
    5. Prioritize druggable targets (actionable) or biology (exploratory)
    """

    # ── Default scoring weights (overridden by ModeConfig when provided) ──
    W_COST: float = 0.20
    W_SYNERGY: float = 0.16
    W_RESISTANCE: float = 0.16
    W_COVERAGE: float = 0.13
    W_COMBO_TOX: float = 0.16
    W_DRUGGABLE: float = 0.09
    HUB_PENALTY_MULTIPLIER: float = 1.5
    PERTURBATION_SCALING: float = 0.12
    DRUGGABILITY_THRESHOLD: float = 0.6
    W_SELECTIVITY: float = 0.18
    W_GENOMIC: float = W_GENOMIC  # imported from alin.genomic_data
    # Biology reward weights (exploratory mode)
    W_ESSENTIALITY: float = 0.0
    W_MUTATION: float = 0.0
    W_CENTRALITY: float = 0.0
    # V7: New data-integration weights
    W_COPY_NUMBER: float = 0.0
    W_DRUG_SENSITIVITY: float = 0.0
    W_RNAI_CONCORDANCE: float = 0.0
    
    def __init__(self, depmap: DepMapLoader, omnipath: OmniPathLoader, drug_db: DrugTargetDB,
                 toxicity_cache_dir: Optional[str] = None,
                 use_known_synergies: bool = True,
                 disable_hub_penalty: bool = False,
                 protein_scorer=None,
                 enable_api: bool = True,
                 mutation_loader: Optional[TCGAMutationLoader] = None,
                 mode_config: Optional[ModeConfig] = None):
        self.cfg = mode_config or actionable_config()
        self.depmap = depmap
        self.omnipath = omnipath
        self.drug_db = drug_db
        self.mutation_loader = mutation_loader
        self.use_known_synergies = use_known_synergies
        self.disable_hub_penalty = disable_hub_penalty
        self.protein_scorer = protein_scorer
        self.network_analyzer = XNodeNetworkAnalyzer(omnipath)

        # Apply ModeConfig weights (override class defaults)
        self.W_COST = self.cfg.W_COST
        self.W_SYNERGY = self.cfg.W_SYNERGY
        self.W_RESISTANCE = self.cfg.W_RESISTANCE
        self.W_COVERAGE = self.cfg.W_COVERAGE
        self.W_COMBO_TOX = self.cfg.W_COMBO_TOX
        self.W_DRUGGABLE = self.cfg.W_DRUGGABLE
        self.W_SELECTIVITY = self.cfg.W_SELECTIVITY
        self.W_GENOMIC = self.cfg.W_GENOMIC
        self.HUB_PENALTY_MULTIPLIER = self.cfg.HUB_PENALTY_MULTIPLIER
        self.PERTURBATION_SCALING = self.cfg.PERTURBATION_SCALING
        self.DRUGGABILITY_THRESHOLD = self.cfg.DRUGGABILITY_THRESHOLD
        self.W_ESSENTIALITY = self.cfg.W_ESSENTIALITY
        self.W_MUTATION = self.cfg.W_MUTATION
        self.W_CENTRALITY = self.cfg.W_CENTRALITY
        self.W_COPY_NUMBER = self.cfg.W_COPY_NUMBER
        self.W_DRUG_SENSITIVITY = self.cfg.W_DRUG_SENSITIVITY
        self.W_RNAI_CONCORDANCE = self.cfg.W_RNAI_CONCORDANCE
        self.synergy_scorer = SynergyScorer(omnipath, use_known_synergies=use_known_synergies)
        self.resistance_estimator = ResistanceProbabilityEstimator(omnipath, depmap)
        self.cost_fn = CostFunction(depmap, drug_db, toxicity_cache_dir=toxicity_cache_dir,
                                    protein_scorer=protein_scorer, enable_api=enable_api)

        # V7: Drug sensitivity loaders (GDSC2 + PRISM)
        self._gdsc = None
        self._prism = None
        if self.W_DRUG_SENSITIVITY > 0:
            try:
                from alin.drug_sensitivity import GDSCLoader, PRISMLoader
                # Try both possible data directories
                _ds_dir = './drug_sensitivity_data' if Path('./drug_sensitivity_data').exists() else './data'
                self._gdsc = GDSCLoader(data_dir=_ds_dir)
                self._prism = PRISMLoader(data_dir=_ds_dir)
                # Load local data only — no network downloads
                _gdsc_ok = self._gdsc.download_data()
                if not _gdsc_ok:
                    self._gdsc = None
                # V9: Load primary + secondary data (secondary has IC50/AUC curves)
                _prism_ok = self._prism.download_data(download_secondary=True)
                if not _prism_ok:
                    self._prism = None
                _has_secondary = self._prism is not None and self._prism._secondary_dr is not None
                if self._gdsc or self._prism:
                    logger.info(f"V9: Drug sensitivity ready (GDSC={'ok' if self._gdsc else 'no'}, "
                                f"PRISM={'ok' if self._prism else 'no'}, "
                                f"PRISM-secondary={'ok' if _has_secondary else 'no'})")
            except Exception as exc:
                logger.debug(f"V9: Drug sensitivity init failed: {exc}")
                self._gdsc = None
                self._prism = None

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
        
        # ── Load gold standard targets for novelty scoring ─────────
        self._gold_standard_targets: List[frozenset] = []
        if self.cfg.use_novelty_bonus:
            try:
                from gold_standard import GOLD_STANDARD
                self._gold_standard_targets = [
                    entry['targets'] for entry in GOLD_STANDARD
                    if 'targets' in entry
                ]
                logger.info(f"Loaded {len(self._gold_standard_targets)} gold standard entries for novelty scoring")
            except ImportError:
                logger.debug("Gold standard not available for novelty scoring")
    def find_triple_combinations(self, 
                                  paths: List[ViabilityPath], 
                                  cancer_type: str,
                                  top_n: int = 20,
                                  min_coverage: float = 0.7,
                                  prefer_druggable: Optional[bool] = None,
                                  n_cell_lines: int = 0) -> List[TripleCombination]:
        """
        Find optimal triple combinations using systems biology scoring
        
        Args:
            paths: Viability paths to cover
            cancer_type: For computing cancer-specific costs
            top_n: Number of top combinations to return
            min_coverage: Minimum fraction of paths to cover
            prefer_druggable: Prioritize druggable targets (None=use config)
            n_cell_lines: Number of cell lines for this cancer type (for Bayesian shrinkage)
        
        Returns:
            List of TripleCombination objects sorted by combined_score
        """
        # Use config default if not explicitly overridden
        if prefer_druggable is None:
            prefer_druggable = self.cfg.prefer_druggable
        if len(paths) == 0:
            logger.warning("No viability paths provided")
            return []
        
        # Separate core (essentiality/signaling/statistical) paths from
        # perturbation-response paths.  Coverage gating uses core paths
        # only: perturbation paths represent pharmacological evidence
        # (LINCS L1000), not tumour vulnerabilities, and can number in the
        # hundreds when LINCS is loaded.  Using them in the denominator
        # makes 50 % coverage unreachable for 3-gene combos.  Perturbation
        # evidence still contributes via candidate extraction, the
        # perturbation_bonus in scoring, and the overall combined_score.
        core_paths = [p for p in paths if getattr(p, 'path_type', '') != 'perturbation_response']
        if not core_paths:
            # If ALL paths are perturbation-type, fall back to all paths
            core_paths = paths
        
        logger.info(f"Finding triple combinations for {cancer_type} "
                     f"({len(paths)} total paths, {len(core_paths)} core paths)")
        
        # Extract all candidate genes from paths (all types, including
        # perturbation, so that LINCS-backed genes enter the pool)
        all_genes = set()
        for path in paths:
            all_genes.update(path.nodes)
        
        # ── V6: Exclude hard-blocked genes from candidate pool ────
        # Translational mode excludes undruggable pan-essentials (TP53,
        # CDKN2A, etc.) to force actionable alternatives.
        if self.cfg.excluded_genes:
            _n_before_excl = len(all_genes)
            all_genes -= self.cfg.excluded_genes
            _n_excl = _n_before_excl - len(all_genes)
            if _n_excl > 0:
                logger.info(f"Excluded {_n_excl} genes from candidate pool: "
                            f"{sorted(self.cfg.excluded_genes & all_genes | self.cfg.excluded_genes)[:6]}")
        
        # Remove artifact genes from candidate pool
        _n_before_blocklist = len(all_genes)
        all_genes = {g for g in all_genes if not _is_artifact(g)}
        _n_removed = _n_before_blocklist - len(all_genes)
        if _n_removed > 0:
            logger.info(f"Artifact blocklist removed {_n_removed} genes from candidate pool")
        
        # Get top X-nodes from network analysis
        xnodes = self.network_analyzer.get_top_xnodes(n=30)
        xnode_genes = {g for g, _ in xnodes}
        
        # Get pathway bridge genes
        bridge_genes = set(self.network_analyzer.find_pathway_bridges())
        
        # Prioritize candidates: X-nodes > bridges > path genes
        priority_genes = (xnode_genes & all_genes) | (bridge_genes & all_genes)
        
        # ---- EXPANDED CANDIDATE INJECTION ----
        # Add druggable genes from viability paths (actionable mode only).
        # In exploratory mode this is disabled so undrugged genes compete
        # on equal footing.
        druggable_path_genes = set()
        if self.cfg.inject_druggable_path_genes:
            for gene in all_genes:
                if gene in self.drug_db.DRUG_DB:
                    info = self.drug_db.DRUG_DB[gene]
                    if info.get('stage') in ('approved', 'phase3', 'phase2'):
                        druggable_path_genes.add(gene)
            priority_genes |= druggable_path_genes
            logger.info(f"Injected {len(druggable_path_genes)} druggable path genes into candidates")
        # ---- END EXPANDED CANDIDATE INJECTION ----
        
        # ---- GENOMIC (TCGA MUTATION) CANDIDATE INJECTION ----
        # Two-part strategy:
        # (a) Boost genes already in the path pool that are highly mutated
        # (b) FORCE-INJECT top TCGA drivers from the FULL genome, not just
        #     path genes.  This ensures APC enters CRC candidates, BRAF
        #     enters melanoma, etc., even if the hitting-set solver
        #     didn't find them in viability paths.
        _genomic_boosted: Set[str] = set()
        _genomic_injected: Set[str] = set()
        if self.mutation_loader is not None:
            try:
                self.mutation_loader.load()
                # (a) Boost candidates already in the pool
                _genomic_boosted = compute_genomic_candidate_boost(
                    list(all_genes), cancer_type, self.mutation_loader
                )
                priority_genes |= _genomic_boosted
                # (b) Force-inject top drivers from the FULL genome
                _genomic_injected = get_driver_injection_set(
                    cancer_type, self.mutation_loader, max_inject=10
                )
                priority_genes |= _genomic_injected
                all_genes |= _genomic_injected  # also add to all_genes pool
                _total_new = len(_genomic_boosted | _genomic_injected)
                if _total_new:
                    logger.info(
                        f"Genomic injection: {len(_genomic_boosted)} boosted + "
                        f"{len(_genomic_injected)} force-injected drivers: "
                        f"{sorted(_genomic_injected)[:8]}"
                    )
            except Exception as exc:
                logger.debug("Genomic candidate injection failed: %s", exc)
        # ---- END GENOMIC CANDIDATE INJECTION ----

        # ── V6.1 FIX: Re-apply excluded-gene filter AFTER genomic injection.
        # Without this, genes like TP53 that are hard-excluded at line 2746
        # can re-enter via get_driver_injection_set (TP53 is a top TCGA
        # driver for many cancer types).
        if self.cfg.excluded_genes:
            _leaked = (all_genes | priority_genes) & self.cfg.excluded_genes
            if _leaked:
                all_genes -= self.cfg.excluded_genes
                priority_genes -= self.cfg.excluded_genes
                logger.info(f"Post-injection exclusion removed {sorted(_leaked)} from candidate pool")
        
        if len(priority_genes) < 10:
            # Add essential genes from paths
            gene_frequency = defaultdict(int)
            for path in paths:
                for gene in path.nodes:
                    gene_frequency[gene] += 1
            
            frequent_genes = sorted(gene_frequency.items(), key=lambda x: x[1], reverse=True)
            priority_genes.update(g for g, _ in frequent_genes[:30])
        
        # Filter to druggable if preferred (config-driven)
        if prefer_druggable:
            druggable = {g for g in priority_genes if self.drug_db.get_druggability_score(g, cancer_type=cancer_type) >= self.cfg.druggable_pool_threshold}
            if len(druggable) >= 6:
                candidate_genes = sorted(druggable)[:50]
            else:
                # Mix druggable and non-druggable
                non_drug = sorted(g for g in priority_genes if g not in druggable)
                candidate_genes = sorted(druggable) + non_drug[:30]
        else:
            # ── V6: Network topology candidate selection (exploratory) ──
            # In exploratory mode, use betweenness centrality ranking
            # instead of path-frequency.  This finds network bottleneck
            # vulnerabilities the hitting-set solver may miss.
            if self.cfg.use_network_topology_candidates:
                _centrality = self.network_analyzer.compute_network_centrality()
                # Rank priority genes by betweenness centrality (desc),
                # then by xnode_score (desc), then alphabetically.
                candidate_genes = sorted(
                    priority_genes,
                    key=lambda g: (
                        -_centrality.get(g, {}).get('betweenness_approx', 0),
                        -_centrality.get(g, {}).get('xnode_score', 0),
                        g,
                    ),
                )[:50]
                logger.info(f"Network topology candidate selection: top-50 by betweenness centrality")
            else:
                # Sort by path frequency (highest first) so the most relevant
                # genes always enter the candidate pool regardless of hash
                # ordering.  Tie-break alphabetically for reproducibility.
                _gene_path_count = defaultdict(int)
                for _p in paths:
                    for _g in _p.nodes:
                        if _g in priority_genes:
                            _gene_path_count[_g] += 1
                candidate_genes = sorted(
                    priority_genes,
                    key=lambda g: (-_gene_path_count.get(g, 0), g),
                )[:50]

        # ---- HARD GENOMIC FILTER ----
        # Remove genes that have no genomic footprint in this cancer type
        # UNLESS they have an approved drug.  This is stronger than the
        # soft scoring bonus: CDK6 (0.4% in melanoma) is REMOVED, not
        # just given zero bonus.
        _pre_filter_count = len(candidate_genes)
        if self.mutation_loader is not None and self.mutation_loader._loaded:
            candidate_genes = filter_genomically_irrelevant(
                candidate_genes, cancer_type, self.mutation_loader,
                drug_db=self.drug_db if self.cfg.drug_overrides_genomic_filter else None,
            )
            _removed = _pre_filter_count - len(candidate_genes)
            if _removed > 0:
                logger.info(
                    f"Genomic filter removed {_removed} irrelevant genes "
                    f"({_pre_filter_count} -> {len(candidate_genes)} candidates)"
                )
            # Safety: ensure we still have enough candidates
            if len(candidate_genes) < 6:
                # If too aggressive, relax to top priority genes
                candidate_genes = list(priority_genes)[:50]
                logger.info("Genomic filter too aggressive, reverting to full set")
        # ---- END HARD GENOMIC FILTER ----

        logger.info(f"Evaluating combinations from {len(candidate_genes)} candidate genes")
        
        # Pre-resolve UniProt IDs for the candidate pool (batch is faster)
        if (self.cost_fn.enable_api
                and self.protein_scorer is not None
                and hasattr(self.protein_scorer, 'pre_resolve_genes')):
            self.protein_scorer.pre_resolve_genes(candidate_genes)

        # Compute individual costs (gamma from config)
        gene_costs = {}
        for gene in candidate_genes:
            cost_obj = self.cost_fn.compute_cost(gene, cancer_type)
            gene_costs[gene] = cost_obj.total_cost(gamma=self.cfg.cost_gamma)
        
        # ── Pre-compute TCGA alteration frequency for each candidate ────
        # Uses combined point-mutation + CNV frequency.
        # Used to compute the genomic_bonus inside _score_combo.
        gene_mutation_freq: Dict[str, float] = {}
        if self.mutation_loader is not None:
            try:
                self.mutation_loader.load()
                for gene in candidate_genes:
                    gene_mutation_freq[gene] = get_combined_alteration_freq(
                        gene, cancer_type, self.mutation_loader
                    )
            except Exception as exc:
                logger.debug("Alteration freq pre-computation failed: %s", exc)
        
        # ── Pre-compute cancer-selectivity for each candidate gene ───
        # Selectivity = how much more essential a gene is in THIS cancer
        # vs the pan-cancer average.  Genes that are broadly essential
        # (e.g. CDK6 in most lineages) receive near-zero selectivity;
        # genes with differential dependency (e.g. BRAF in melanoma)
        # receive a high selectivity score.
        gene_selectivity: Dict[str, float] = {}
        try:
            _crispr = getattr(self.depmap, '_crispr_df', None)
            if _crispr is not None:
                _cls = self.depmap.get_cell_lines_for_cancer(cancer_type)
                _avail = [cl for cl in _cls if cl in _crispr.index]
                if len(_avail) >= 3:
                    _cancer_mean = _crispr.loc[_avail].mean(axis=0)
                    _global_mean = _crispr.mean(axis=0)
                    for gene in candidate_genes:
                        if gene in _cancer_mean.index and gene in _global_mean.index:
                            # Both are negative; more negative = more essential.
                            # selectivity = global - cancer: positive means
                            # gene is *more* essential in this cancer.
                            sel = float(_global_mean[gene]) - float(_cancer_mean[gene])
                            # Clip to [0, 1] — we only reward selectivity,
                            # never penalize genes that are less essential here.
                            gene_selectivity[gene] = max(0.0, min(1.0, sel))
        except Exception as exc:
            logger.debug("Selectivity pre-computation failed: %s", exc)
        
        # ── Pre-compute absolute essentiality for biology reward ─────
        # Unlike selectivity (differential), this is the raw mean
        # CRISPR dependency in this cancer's cell lines.  More negative
        # = more essential.  Normalised to [0, 1].
        gene_essentiality: Dict[str, float] = {}
        try:
            _crispr_ess = getattr(self.depmap, '_crispr_df', None)
            if _crispr_ess is not None:
                _cls_ess = self.depmap.get_cell_lines_for_cancer(cancer_type)
                _avail_ess = [cl for cl in _cls_ess if cl in _crispr_ess.index]
                if len(_avail_ess) >= 3:
                    _cancer_ess_mean = _crispr_ess.loc[_avail_ess].mean(axis=0)
                    for gene in candidate_genes:
                        if gene in _cancer_ess_mean.index:
                            # CRISPR effect: -1 = essential, 0 = non-essential
                            raw = float(_cancer_ess_mean[gene])
                            # Map to [0, 1]: -1 → 1.0, 0 → 0.0, positive → 0
                            gene_essentiality[gene] = max(0.0, min(1.0, -raw))
        except Exception as exc:
            logger.debug("Essentiality pre-computation failed: %s", exc)
        
        # ── V7: Pre-compute copy-number amplification bonus ──────────
        # Mean CN ratio across cancer cell lines for each candidate gene.
        # Diploid = 2.0.  Amplified genes (CN > 3) receive a bonus that
        # rewards targeting recurrently amplified oncogenes (EGFR, MYC,
        # ERBB2, CDK4 etc.).  Normalised to [0, 1].
        gene_cn_bonus: Dict[str, float] = {}
        if self.W_COPY_NUMBER > 0:
            try:
                _cn_df = self.depmap.load_copy_number()
                if _cn_df is not None:
                    _cls_cn = self.depmap.get_cell_lines_for_cancer(cancer_type)
                    _avail_cn = [cl for cl in _cls_cn if cl in _cn_df.index]
                    if len(_avail_cn) >= 3:
                        _cn_mean = _cn_df.loc[_avail_cn].mean(axis=0)
                        for gene in candidate_genes:
                            if gene in _cn_mean.index:
                                # CN ratio: 2 = diploid, >3 = amplified
                                cn_val = float(_cn_mean[gene])
                                # Bonus = (CN - 2) / 4, clipped to [0, 1]
                                # CN=2 → 0, CN=4 → 0.5, CN=6+ → 1.0
                                gene_cn_bonus[gene] = max(0.0, min(1.0, (cn_val - 2.0) / 4.0))
            except Exception as exc:
                logger.debug("V7 CN pre-computation failed: %s", exc)

        # ── V7: Pre-compute RNAi concordance bonus ───────────────────
        # Cross-validate CRISPR essentiality with orthogonal RNAi data.
        # A gene is "concordant" if both CRISPR *and* RNAi agree it is
        # essential in this cancer type.  This reduces false-positives
        # from CRISPR-specific artefacts (e.g. copy-number effects on
        # Cas9 cutting).
        gene_rnai_concordance: Dict[str, float] = {}
        if self.W_RNAI_CONCORDANCE > 0:
            try:
                _rnai_df = self.depmap.load_rnai_dependencies()
                if _rnai_df is not None:
                    _cls_rnai = self.depmap.get_cell_lines_for_cancer(cancer_type)
                    _avail_rnai = [cl for cl in _cls_rnai if cl in _rnai_df.index]
                    if len(_avail_rnai) >= 2:
                        _rnai_mean = _rnai_df.loc[_avail_rnai].mean(axis=0)
                        for gene in candidate_genes:
                            if gene in _rnai_mean.index:
                                # RNAi scores: more negative = more essential
                                _rnai_ess = max(0.0, min(1.0, -float(_rnai_mean[gene])))
                                _crispr_ess = gene_essentiality.get(gene, 0.0)
                                # Concordance = geometric mean of both signals
                                if _crispr_ess > 0 and _rnai_ess > 0:
                                    gene_rnai_concordance[gene] = (_crispr_ess * _rnai_ess) ** 0.5
                                else:
                                    gene_rnai_concordance[gene] = 0.0
            except Exception as exc:
                logger.debug("V7 RNAi concordance pre-computation failed: %s", exc)

        # ── V7: Pre-compute drug sensitivity bonus ───────────────────
        # For each candidate gene, check if any drug targeting it shows
        # actual cell-killing in GDSC2 / PRISM screens for this cancer.
        # This is stronger evidence than just "an approved drug exists".
        gene_drug_sensitivity: Dict[str, float] = {}
        if self.W_DRUG_SENSITIVITY > 0 and (self._gdsc is not None or self._prism is not None):
            # V9: Get DepMap IDs for this cancer type (for PRISM secondary filtering)
            _cancer_depmap_ids = None
            try:
                _cls_for_cancer = self.depmap.get_cell_lines_for_cancer(cancer_type)
                if _cls_for_cancer:
                    _cancer_depmap_ids = list(_cls_for_cancer)
            except Exception:
                pass

            try:
                for gene in candidate_genes:
                    _best_sens = 0.0
                    # V9: Check PRISM secondary screen FIRST (IC50/AUC, more quantitative)
                    if self._prism is not None and self._prism._secondary_dr is not None:
                        try:
                            _sec_auc = self._prism.get_secondary_auc_for_gene(
                                gene, cell_line_ids=_cancer_depmap_ids
                            )
                            if _sec_auc > 0:
                                _best_sens = max(_best_sens, _sec_auc)
                        except Exception:
                            pass
                    # Check PRISM primary screen (LFC < -0.3 = sensitive)
                    if self._prism is not None and hasattr(self._prism, '_primary_lfc'):
                        try:
                            _prism_lfc = self._prism._primary_lfc
                            if _prism_lfc is not None:
                                # Find drugs targeting this gene
                                _tinfo = self._prism._primary_treatment_info
                                if _tinfo is not None:
                                    _gene_drugs = _tinfo[
                                        _tinfo.get('target', _tinfo.get('Target', pd.Series(dtype=str)))
                                        .str.contains(gene, case=False, na=False)
                                    ]
                                    if len(_gene_drugs) > 0:
                                        _cls_p = self.depmap.get_cell_lines_for_cancer(cancer_type)
                                        _avail_p = [cl for cl in _cls_p if cl in _prism_lfc.index]
                                        if _avail_p:
                                            _drug_cols = [c for c in _gene_drugs.index if c in _prism_lfc.columns]
                                            if _drug_cols:
                                                _lfc_vals = _prism_lfc.loc[_avail_p, _drug_cols].mean().min()
                                                # LFC < -0.3 = sensitive; map to [0, 1]
                                                _best_sens = max(_best_sens, max(0.0, min(1.0, -float(_lfc_vals) / 1.0)))
                        except Exception:
                            pass
                    # Check GDSC (if data available)
                    if self._gdsc is not None and self._gdsc._ic50_data is not None:
                        try:
                            _gdsc_drugs = self._gdsc.get_drugs_for_target(gene)
                            for _dname in _gdsc_drugs[:3]:  # cap at 3
                                _prof = self._gdsc.get_drug_sensitivity(_dname, cancer_type)
                                if _prof and _prof.auc_values:
                                    _mean_auc = sum(_prof.auc_values) / len(_prof.auc_values)
                                    # AUC > 0.7 = sensitive; map to [0, 1]
                                    _best_sens = max(_best_sens, max(0.0, min(1.0, (_mean_auc - 0.3) / 0.7)))
                        except Exception:
                            pass
                    gene_drug_sensitivity[gene] = _best_sens
            except Exception as exc:
                logger.debug("V9 drug sensitivity pre-computation failed: %s", exc)

        # Pre-compute per-gene path frequencies for hub penalty
        # (uses ALL paths, including perturbation, for hub detection)
        gene_path_freqs = {}
        for gene in candidate_genes:
            gene_path_freqs[gene] = sum(
                1 for p in paths if gene in p.nodes
            ) / max(len(paths), 1)
        freq_values = sorted(gene_path_freqs.values())
        median_path_freq = freq_values[len(freq_values) // 2] if freq_values else 0.3
        
        # ── Adaptive z-score hub penalty statistics ──────────────────
        # Instead of a fixed linear penalty, compute z-scores of gene
        # path frequencies within THIS cancer type's candidate pool.
        # Genes >2σ above the mean receive a sigmoid-dampened penalty
        # that saturates for extreme outliers (diminishing returns).
        _freq_arr = np.array(list(gene_path_freqs.values()))
        _freq_mean = float(np.mean(_freq_arr)) if len(_freq_arr) > 0 else 0.3
        _freq_std = float(np.std(_freq_arr)) if len(_freq_arr) > 1 else 0.1
        _freq_std = max(_freq_std, 1e-6)  # avoid div-by-zero
        
        # Pre-compute z-scores for each candidate gene
        _gene_freq_zscore: Dict[str, float] = {}
        for gene in candidate_genes:
            _gene_freq_zscore[gene] = (gene_path_freqs.get(gene, 0) - _freq_mean) / _freq_std
        
        # Load common essential genes for additional pan-essential penalty
        _common_essentials: Set[str] = set()
        try:
            _ce_path = Path('depmap_data') / 'CRISPRInferredCommonEssentials.csv'
            if _ce_path.exists():
                with open(_ce_path) as _f:
                    for _line in _f:
                        _line = _line.strip()
                        if _line and _line != 'Essentials' and '(' in _line:
                            _gene_name = _line.split('(')[0].strip()
                            _common_essentials.add(_gene_name)
                logger.debug(f"Loaded {len(_common_essentials)} common essential genes for hub penalty")
        except Exception as _e:
            logger.debug(f"Could not load common essentials: {_e}")
        
        # Evidence-backed hub penalty exemptions for this cancer type
        cancer_exemptions = self.EVIDENCE_EXEMPTIONS.get(cancer_type, {})
        
        # ── Pre-compute LINCS cancer-relevance weights per gene ──────
        # Scores how much of a target's LINCS evidence comes from cell
        # lines matching this cancer's lineage.  Discounts pan-cancer
        # hub evidence (e.g., CDK6 from lymphoid cells in sarcoma).
        _cancer_relevance: Dict[str, float] = {}
        try:
            from alin.perturbation import _get_lincs_db
            _lincs_for_rel = _get_lincs_db()
            if _lincs_for_rel is not None and hasattr(_lincs_for_rel, 'get_cancer_relevance'):
                for gene in candidate_genes:
                    _cancer_relevance[gene] = _lincs_for_rel.get_cancer_relevance(
                        gene, cancer_type
                    )
        except Exception as exc:
            logger.debug("Cancer-relevance weighting unavailable: %s", exc)
        
        # ── Bayesian shrinkage parameters ────────────────────────────
        # For cancer types with few cell lines, shrink sub-scores
        # (synergy, resistance, coverage) toward agnostic priors.
        # Shrinkage factor α = sqrt(n) / (sqrt(n) + κ), where κ is a
        # reference strength (set so α≈0.5 at n=10, the median low-
        # power threshold).  α→1 as n→∞ (full trust in data).
        _SHRINKAGE_KAPPA = np.sqrt(10)  # reference: 50% shrinkage at 10 cell lines
        _n_cl = max(n_cell_lines, 1)
        _shrink_alpha = np.sqrt(_n_cl) / (np.sqrt(_n_cl) + _SHRINKAGE_KAPPA)
        # Agnostic (global) priors for sub-scores
        _PRIOR_SYNERGY = 0.35    # default prior synergy (moderate)
        _PRIOR_RESISTANCE = 0.40  # default prior resistance (moderate)
        _PRIOR_COVERAGE = 0.50   # default prior coverage
        if n_cell_lines > 0:
            logger.debug(f"Bayesian shrinkage: α={_shrink_alpha:.3f} (n={n_cell_lines})")
        
        # Evidence power tier for this cancer type
        if _n_cl >= 30:
            _evidence_power = 'robust'
        elif _n_cl >= 10:
            _evidence_power = 'adequate'
        elif _n_cl >= 5:
            _evidence_power = 'suggestive'
        else:
            _evidence_power = 'hypothesis'
        
        def _score_combo(combo):
            """Score a combination of any size (2 or 3 genes)."""
            combo_set = set(combo)
            
            # Calculate coverage over CORE paths only (not perturbation).
            # Perturbation paths inflate the denominator (often 80-100
            # extra LINCS-derived paths) making 50 % coverage unreachable
            # for any 3-gene combo.  Core paths (co-essentiality,
            # signaling, cancer-specific) represent actual tumour
            # vulnerabilities that must be "hit".
            covered = sum(1 for p in core_paths if any(g in combo_set for g in p.nodes))
            coverage = covered / len(core_paths)
            
            if coverage < min_coverage:
                return None
            
            # Total cost
            total_cost = sum(gene_costs.get(g, 1.0) for g in combo)
            
            # Synergy score (heuristic)
            synergy_heuristic = self.synergy_scorer.compute_synergy_score(combo_set)

            # Data-driven synergy from co-essentiality (if DepMap data available)
            # Uses cancer-type-specific cell lines so the Jaccard matrix
            # reflects actual co-dependency patterns (not all-zeros).
            synergy = synergy_heuristic
            try:
                from pharmacological_validation import CoEssentialityInteractionEstimator
                _crispr = (self.depmap._crispr_df
                           if hasattr(self.depmap, '_crispr_df')
                              and self.depmap._crispr_df is not None
                           else None)
                _cls = (self.depmap.get_cell_lines_for_cancer(cancer_type)
                        if _crispr is not None else [])
                dd = CoEssentialityInteractionEstimator.score_combination(
                    targets=tuple(sorted(combo)),
                    depmap_df=_crispr,
                    cell_lines=_cls,
                    original_synergy=synergy_heuristic,
                    original_pathway_diversity=len(set(
                        self.synergy_scorer.PATHWAY_ASSIGNMENT.get(g, g) for g in combo
                    )) / max(len(combo), 1),
                )
                # Blend: keep the continuous heuristic (LINCS + known
                # synergies + pathway diversity) and the data-driven
                # co-essentiality estimate.  Neither alone is sufficient.
                synergy = 0.5 * synergy_heuristic + 0.5 * dd.data_driven_synergy
            except ImportError:
                pass  # pharmacological_validation not installed
            except (TypeError, ValueError, KeyError) as e:
                logger.debug('Co-essentiality scoring failed for %s: %s', combo, e)
            
            # Resistance probability
            resistance = self.resistance_estimator.estimate_resistance_probability(combo_set, cancer_type)
            
            # ── V6: Resistance hard-gate (mode-agnostic) ────────────
            # Discard triples where resistance >= synergy (net-negative
            # therapeutic window).  These combos would do more harm
            # than good and are not clinically actionable.
            if self.cfg.resistance_hard_gate and resistance >= synergy:
                return None
            
            # Pathway coverage
            pathway_cov = self.network_analyzer.get_pathway_coverage(combo_set)
            
            # ── Bayesian shrinkage toward global priors ──────────────
            # For low-power cancer types (few cell lines), we don't fully
            # trust the cancer-specific synergy/resistance/coverage estimates.
            # Shrink them toward agnostic priors: score = α * observed + (1-α) * prior.
            # α ≈ 1 for well-powered types, α ≈ 0.5 for ~10 cell lines.
            if n_cell_lines > 0 and _shrink_alpha < 0.99:
                synergy = _shrink_alpha * synergy + (1 - _shrink_alpha) * _PRIOR_SYNERGY
                resistance = _shrink_alpha * resistance + (1 - _shrink_alpha) * _PRIOR_RESISTANCE
                coverage = _shrink_alpha * coverage + (1 - _shrink_alpha) * _PRIOR_COVERAGE
            
            # Count druggable targets (always use real threshold for accurate reporting;
            # the scoring formula uses W_DRUGGABLE to control mode-specific impact)
            _REPORT_DRUGGABILITY_THRESHOLD = 0.6
            druggable_count = sum(1 for g in combo if self.drug_db.get_druggability_score(g, cancer_type=cancer_type) >= _REPORT_DRUGGABILITY_THRESHOLD)
            
            # ── Continuous druggability for scoring (replaces integer count) ──
            # Mean druggability (0–1) across combo targets — smoother signal
            # than the coarse integer count.
            _target_drug_scores = [self.drug_db.get_druggability_score(g, cancer_type=cancer_type) for g in combo]
            _druggability_mean = sum(_target_drug_scores) / len(_target_drug_scores)
            
            # Mode-aware druggability tier for this combination
            # immediate: ≥2 targets at approved/phase3 (druggability ≥ 0.8)
            # partial:   ≥1 target at approved/phase3
            # research:  no targets at clinical stage
            _n_clinical = sum(1 for s in _target_drug_scores if s >= 0.8)
            if _n_clinical >= 2:
                _drug_tier = 'immediate'
            elif _n_clinical >= 1:
                _drug_tier = 'partial'
            else:
                _drug_tier = 'research'
            
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
            
            # Compute perturbation response score using LINCS multi-modal
            # evidence (concordance-weighted).
            # The perturbation_score from LINCS now incorporates:
            #   - Weighted effector coverage (concordant > single-modality)
            #   - Weighted feedback coverage
            #   - Multi-modal fraction (how many targets have >=2 modalities)
            #   - Mean cross-modal concordance
            # This is a 0-1 score where higher = better perturbation support.
            #
            # Cancer-relevance weighting: the perturbation bonus is scaled
            # by the mean cancer-relevance of the combo's targets.  If a
            # target's LINCS evidence comes primarily from cell lines of a
            # different lineage, its contribution is discounted.
            perturbation_bonus = 0.0
            try:
                from alin.perturbation import score_combination_by_perturbation
                essential_genes = set()
                for p in paths:
                    essential_genes.update(p.nodes)
                pert_result = score_combination_by_perturbation(list(combo), essential_genes)
                raw_pert = pert_result.get('perturbation_score', 0)
                # Apply cancer-relevance weight (mean across combo targets)
                if _cancer_relevance:
                    rel_weights = [_cancer_relevance.get(g, 0.5) for g in combo]
                    mean_relevance = sum(rel_weights) / len(rel_weights)
                else:
                    mean_relevance = 1.0
                perturbation_bonus = raw_pert * mean_relevance * self.PERTURBATION_SCALING
            except ImportError:
                pass
            
            # Hub gene specificity penalty (ADAPTIVE Z-SCORE)
            # Z-score-based sigmoid penalty that:
            #   - Activates only for genes >1.5σ above mean path frequency
            #   - Saturates via sigmoid (diminishing returns for extreme hubs)
            # This is a WITHIN-CANCER signal: penalizes genes that dominate
            # this cancer type's viability paths, not cross-cancer prevalence.
            # Evidence-aware: exempt genes with Tier 1 experimental evidence
            # when paired with a known synergy partner in this combination.
            hub_penalty = 0.0
            if not self.disable_hub_penalty:
                _HUB_Z_THRESHOLD = 1.5   # z-score threshold to start penalising
                _HUB_SIGMOID_SCALE = 2.0  # sigmoid steepness
                for g in combo:
                    # Check if gene is evidence-exempt in this cancer + combination
                    if g in cancer_exemptions:
                        required_partners = cancer_exemptions[g]
                        if required_partners & combo_set:
                            continue  # Skip hub penalty for evidence-backed gene
                    
                    z = _gene_freq_zscore.get(g, 0.0)
                    if z > _HUB_Z_THRESHOLD:
                        # Sigmoid-dampened penalty: saturates at ~HUB_PENALTY_MULTIPLIER
                        _sig = 1.0 / (1.0 + np.exp(-_HUB_SIGMOID_SCALE * (z - _HUB_Z_THRESHOLD)))
                        hub_penalty += _sig * self.HUB_PENALTY_MULTIPLIER
            
            # Cancer-selectivity bonus: reward combos whose targets are
            # selectively essential in this cancer type.  Pan-essential
            # hubs (CDK6, STAT3) get low selectivity; differentially
            # essential genes (e.g. BRAF in melanoma) get high selectivity.
            selectivity_bonus = 0.0
            if gene_selectivity:
                sel_scores = [gene_selectivity.get(g, 0.0) for g in combo]
                selectivity_bonus = (sum(sel_scores) / len(sel_scores)) * self.W_SELECTIVITY
            
            # Genomic relevance bonus: reward combos whose targets are
            # actually mutated in this cancer type (TCGA mc3 data).
            # Genes with high somatic mutation frequency in the matching
            # TCGA study get a strong bonus; genes with no mutations
            # (e.g. CDK6 in melanoma at 0.4%) get nothing.
            # Conditionally druggable genes (KRAS, FLT3) are penalized
            # if their specific actionable variant is absent.
            genomic_bonus = 0.0
            if gene_mutation_freq and self.mutation_loader is not None:
                genomic_bonus = compute_genomic_bonus(
                    combo, cancer_type, self.mutation_loader
                )
            
            # Combined score (lower is better)
            #
            # V8: Removed cross-cancer dominance penalties (pan-essential
            # dampener, V6.2 essentiality factor).  Per-cancer ranking
            # naturally breaks dominance without the arms race.
            # Kept: within-cancer hub penalty, selectivity bonus, clinical
            # readiness, genomic/perturbation/V7 boosts.

            # ── Biology reward signals (zero in actionable mode) ────
            essentiality_bonus = 0.0
            mutation_bonus = 0.0
            centrality_bonus = 0.0
            if self.W_ESSENTIALITY > 0 and gene_essentiality:
                ess_scores = [gene_essentiality.get(g, 0.0) for g in combo]
                essentiality_bonus = (sum(ess_scores) / len(ess_scores)) * self.W_ESSENTIALITY
            if self.W_MUTATION > 0 and gene_mutation_freq:
                mut_scores = [min(1.0, gene_mutation_freq.get(g, 0.0) * 5.0) for g in combo]
                mutation_bonus = (sum(mut_scores) / len(mut_scores)) * self.W_MUTATION
            if self.W_CENTRALITY > 0 and gene_path_freqs:
                cent_scores = [min(1.0, gene_path_freqs.get(g, 0.0) * 3.0) for g in combo]
                centrality_bonus = (sum(cent_scores) / len(cent_scores)) * self.W_CENTRALITY

            # Pan-essential penalty removed in V8 (was the dominance arms race).
            _pan_ess_penalty = 0.0

            # ── Novelty bonus (exploratory) ────────────────────────────
            # Reward triples not in the gold standard — the point of
            # exploratory mode is to find things we don't already know.
            _novelty_bonus = 0.0
            if self.cfg.use_novelty_bonus and self.cfg.novelty_bonus_weight > 0:
                _combo_frozen = frozenset(combo)
                if hasattr(self, '_gold_standard_targets'):
                    # Check if this exact combo (or subset) is in gold standard
                    _is_known = any(
                        _combo_frozen == gs_targets or _combo_frozen.issubset(gs_targets)
                        for gs_targets in self._gold_standard_targets
                    )
                    if not _is_known:
                        _novelty_bonus = self.cfg.novelty_bonus_weight
                else:
                    # No gold standard loaded — all combos are "novel"
                    _novelty_bonus = self.cfg.novelty_bonus_weight
            
            # ── Clinical readiness (translational) ─────────────────────
            # Multiplicative discount based on drug approval stage.
            # 3/3 approved → 1.0, 2/3 approved → 0.7, 1/3 → 0.4, 0/3 → 0.1
            _clinical_readiness = 0.0
            if self.cfg.use_clinical_readiness:
                _n_approved = 0
                for g in combo:
                    di = drug_info.get(g)
                    if di and di.available_drugs:
                        # Check if any approved drug (not just research)
                        if di.clinical_stage in ('approved', 'phase3'):
                            _n_approved += 1
                _readiness_map = {0: 0.1, 1: 0.4, 2: 0.7, 3: 1.0}
                _clinical_readiness = _readiness_map.get(min(_n_approved, 3), 0.1)
            
            # ── V7: Compute per-combo bonuses from pre-computed dicts ──
            _v7_cn_bonus = 0.0
            if gene_cn_bonus and self.W_COPY_NUMBER > 0:
                _cn_vals = [gene_cn_bonus.get(g, 0.0) for g in combo]
                _v7_cn_bonus = sum(_cn_vals) / len(_cn_vals) * self.W_COPY_NUMBER

            _v7_drug_sens_bonus = 0.0
            if gene_drug_sensitivity and self.W_DRUG_SENSITIVITY > 0:
                _ds_vals = [gene_drug_sensitivity.get(g, 0.0) for g in combo]
                _v7_drug_sens_bonus = sum(_ds_vals) / len(_ds_vals) * self.W_DRUG_SENSITIVITY

            _v7_rnai_bonus = 0.0
            if gene_rnai_concordance and self.W_RNAI_CONCORDANCE > 0:
                _rnai_vals = [gene_rnai_concordance.get(g, 0.0) for g in combo]
                _v7_rnai_bonus = sum(_rnai_vals) / len(_rnai_vals) * self.W_RNAI_CONCORDANCE

            # ── Mode-divergent scoring ─────────────────────────────────
            if self.cfg.scoring_mode == 'multiplicative':
                # TRANSLATIONAL: Multiplicative scoring.
                # Zero druggability → zero score.  Clinical readiness
                # multiplies the final score.
                # Lower combined_score is better, so goodness → 1 - goodness.
                _druggability_frac = _druggability_mean  # 0..1
                _goodness = (
                    synergy
                    * max(0.01, 1.0 - resistance)
                    * max(0.01, _druggability_frac)
                    * max(0.1, _clinical_readiness)
                    * max(0.01, coverage)
                    * max(0.01, 1.0 - combo_tox_score)
                )
                # Apply selectivity, genomic, perturbation as boosts
                _goodness *= (1.0 + selectivity_bonus)
                _goodness *= (1.0 + genomic_bonus * self.W_GENOMIC)
                _goodness *= (1.0 + perturbation_bonus)
                # Hub penalty as discount (within-cancer signal only)
                _goodness *= max(0.01, 1.0 - min(hub_penalty, 0.9))
                # V7: Copy-number amplification boost
                if gene_cn_bonus and self.W_COPY_NUMBER > 0:
                    _cn_vals = [gene_cn_bonus.get(g, 0.0) for g in combo]
                    _mean_cn = sum(_cn_vals) / len(_cn_vals)
                    _goodness *= (1.0 + _mean_cn * self.W_COPY_NUMBER)
                # V7: Drug sensitivity boost (experimental cell-killing)
                if gene_drug_sensitivity and self.W_DRUG_SENSITIVITY > 0:
                    _ds_vals = [gene_drug_sensitivity.get(g, 0.0) for g in combo]
                    _mean_ds = sum(_ds_vals) / len(_ds_vals)
                    _goodness *= (1.0 + _mean_ds * self.W_DRUG_SENSITIVITY)
                # V7: RNAi concordance boost
                if gene_rnai_concordance and self.W_RNAI_CONCORDANCE > 0:
                    _rnai_vals = [gene_rnai_concordance.get(g, 0.0) for g in combo]
                    _mean_rnai = sum(_rnai_vals) / len(_rnai_vals)
                    _goodness *= (1.0 + _mean_rnai * self.W_RNAI_CONCORDANCE)
                combined_score = 1.0 - min(_goodness, 1.0)  # lower is better
            else:
                # EXPLORATORY: Additive scoring.
                combined_score = (
                    total_cost * self.W_COST +
                    (1 - synergy) * self.W_SYNERGY +       # higher synergy = better
                    resistance * self.W_RESISTANCE +
                    (1 - coverage) * self.W_COVERAGE +
                    combo_tox_score * self.W_COMBO_TOX +
                    hub_penalty -                           # within-cancer hub penalty
                    _druggability_mean * 3 * self.W_DRUGGABLE -  # continuous druggability
                    perturbation_bonus -                    # LINCS perturbation bonus
                    selectivity_bonus -                     # cancer-selectivity reward
                    genomic_bonus * self.W_GENOMIC -        # TCGA mutation relevance
                    essentiality_bonus -                    # biology: essentiality
                    mutation_bonus -                        # biology: mutation burden
                    centrality_bonus -                      # biology: path centrality
                    _novelty_bonus -                        # novelty reward
                    _v7_cn_bonus -                          # V7: copy-number amplification
                    _v7_drug_sens_bonus -                   # V7: experimental drug sensitivity
                    _v7_rnai_bonus                          # V7: RNAi cross-validation
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
                resistance_implausible=(resistance < 0.06),
                druggability_tier=_drug_tier,
                evidence_power=_evidence_power,
                clinical_readiness=_clinical_readiness,
                novelty_score=_novelty_bonus,
                pan_essential_penalty=_pan_ess_penalty,
                scoring_mode=self.cfg.scoring_mode,
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

        # ── Bootstrap CI on combined scores (audit recommendation #1) ──
        # Provides uncertainty quantification on the ranking.  We
        # perturb the sub-scores by ±10% (simulating noise in synergy,
        # resistance, coverage estimates) and recompute the combined
        # score 200 times to get a 95% percentile CI.
        def _bootstrap_ci(tc, n_boot: int = 200, noise: float = 0.10):
            """Estimate 95% CI for a TripleCombination's combined_score.
            
            Noise is scaled inversely with sqrt(n_cell_lines) so that
            low-power cancer types produce wider confidence intervals.
            """
            # Adaptive noise: base ±10% for well-powered (n≥30),
            # up to ±25% for n=3
            _adaptive_noise = noise
            if n_cell_lines > 0:
                _adaptive_noise = noise * (np.sqrt(30) / max(np.sqrt(n_cell_lines), 1.0))
                _adaptive_noise = min(_adaptive_noise, 0.30)  # cap at 30%
            rng = np.random.RandomState(42)
            scores = []
            for _ in range(n_boot):
                jitter = lambda v: v * (1 + rng.uniform(-_adaptive_noise, _adaptive_noise))
                s = (
                    jitter(tc.total_cost) * self.W_COST +
                    (1 - jitter(tc.synergy_score)) * self.W_SYNERGY +
                    jitter(tc.resistance_score) * self.W_RESISTANCE +
                    (1 - jitter(tc.coverage)) * self.W_COVERAGE +
                    jitter(tc.combo_tox_score) * self.W_COMBO_TOX
                )
                scores.append(s)
            return (float(np.percentile(scores, 2.5)),
                    float(np.percentile(scores, 97.5)))

        for tc in triple_combinations[:20]:  # top 20 only (speed)
            tc.confidence_interval = _bootstrap_ci(tc)
        for tc in doublet_combinations[:10]:
            tc.confidence_interval = _bootstrap_ci(tc)

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
                 use_lineage_aware_statistical: bool = False,
                 enable_api: bool = True,
                 mode_config: Optional[ModeConfig] = None,
                 strategy_arm: Optional[str] = None,
                 structural_mode: bool = False):
        self.cfg = mode_config or actionable_config()
        self.enable_api = enable_api
        self.strategy_arm = normalize_strategy_arm(
            strategy_arm,
            structural_mode=structural_mode,
            run_mode=getattr(self.cfg.mode, 'value', self.cfg.mode),
        )
        self.structural_mode = is_structural_strategy_arm(self.strategy_arm)
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
        self.cost_fn = CostFunction(self.depmap, self.drug_db, toxicity_cache_dir=toxicity_cache_dir,
                                     enable_api=enable_api)
        self.solver = MinimalHittingSetSolver(self.cost_fn, cost_gamma=self.cfg.cost_gamma)

        # Build multi-omics protein scorer if data is available
        self._protein_scorer = None
        try:
            from alin.protein_scoring import ProteinDruggabilityScorer, GENE_TO_UNIPROT
            # In exploratory mode, use flat druggability so protein structure
            # dominates rather than clinical-stage information.
            _drug_fn = (
                (lambda g: 0.5) if self.cfg.flat_protein_druggability
                else self.drug_db.get_druggability_score
            )
            self._protein_scorer = ProteinDruggabilityScorer(
                genes=list(GENE_TO_UNIPROT.keys()),
                gene_druggability_fn=_drug_fn,
                cache_dir='./api_cache/protein',
                proteomics_dir=data_dir,
            )
            self.cost_fn.protein_scorer = self._protein_scorer
            logger.info('Multi-omics protein scorer enabled')
        except (ImportError, AttributeError, FileNotFoundError, OSError) as e:
            logger.info(f'Protein scorer not available: {e}')

        # Load TCGA mc3 mutation data for genomic-aware scoring
        self._mutation_loader: Optional[TCGAMutationLoader] = None
        try:
            _maf_path = Path(data_dir).parent / 'data' / 'mc3.v0.2.8.PUBLIC.maf.gz'
            _cache_path = Path(data_dir).parent / 'data'
            if not _maf_path.exists():
                # Also check relative to working directory
                _maf_path = Path('data/mc3.v0.2.8.PUBLIC.maf.gz')
                _cache_path = Path('data')
            if _maf_path.exists() or (_cache_path / 'mc3_mutation_summary.json.gz').exists():
                self._mutation_loader = TCGAMutationLoader(
                    maf_path=str(_maf_path), cache_dir=str(_cache_path)
                )
                self._mutation_loader.load()
                logger.info('TCGA mc3 mutation data loaded for genomic-aware scoring')
            else:
                logger.info('No TCGA mutation data found — genomic scoring disabled')
        except Exception as e:
            logger.info(f'TCGA mutation data not available: {e}')

        self.triple_finder = TripleCombinationFinder(
            self.depmap, self.omnipath, self.drug_db,
            toxicity_cache_dir=toxicity_cache_dir,
            use_known_synergies=use_known_synergies,
            disable_hub_penalty=disable_hub_penalty,
            protein_scorer=self._protein_scorer,
            enable_api=enable_api,
            mutation_loader=self._mutation_loader,
            mode_config=self.cfg,
        )
        self.validation_integrator = XNodeValidationIntegrator(validation_data_dir)

        # V10: Structural triple finder (Liaki attractor-disruption framework)
        self.structural_finder = None
        if self.structural_mode:
            try:
                from alin.structural_triples import StructuralTripleFinder
                _omni_df = self.omnipath.load_signaling_network()
                _syn_scorer = getattr(self.triple_finder, 'synergy_scorer', None)
                self.structural_finder = StructuralTripleFinder(
                    omnipath_df=_omni_df,
                    depmap_loader=self.depmap,
                    drug_db=self.drug_db,
                    synergy_scorer=_syn_scorer,
                    mode_config=self.cfg,
                    strategy_arm=self.strategy_arm,
                )
                logger.info('V10 StructuralTripleFinder enabled for strategy arm %s', self.strategy_arm)
            except Exception as _e:
                logger.warning(f'StructuralTripleFinder init failed: {_e}')
        
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
        
        # V10: Structural triple formation (Liaki framework) — runs BEFORE
        # the existing path-covering approach.  Results are prepended so that
        # structural triples rank first in the output CSV.
        structural_combinations: List[TripleCombination] = []
        if self.structural_finder is not None:
            _progress(f"Structural triples ({self.strategy_arm})", step="")
            try:
                structural_combinations = self.structural_finder.form_triples(
                    cancer_type_normalized, top_n=10
                )
                _progress(
                    f"Structural triples: {len(structural_combinations)} "
                    f"(best: {structural_combinations[0].targets if structural_combinations else 'none'})",
                    step="done"
                )
            except Exception as _se:
                logger.warning(f'structural_finder.form_triples failed: {_se}')

        # Find triple combinations using systems biology approach
        _progress("Scoring triple combinations", step="")
        _min_cov = self.cfg.default_min_coverage
        triple_combinations = self.triple_finder.find_triple_combinations(
            all_paths, cancer_type_normalized, top_n=10, min_coverage=_min_cov,
            n_cell_lines=n_cell_lines,
        )
        # Adaptive fallback: if no triples pass the configured coverage
        # threshold, retry with progressively lower thresholds.
        # For translational mode (default 50%), fallback to 40%, 30%.
        # For exploratory mode (default 30%), fallback to 20%.
        if not triple_combinations:
            _fallback_levels = [c for c in (0.40, 0.30, 0.20) if c < _min_cov]
            for _fallback_cov in _fallback_levels:
                triple_combinations = self.triple_finder.find_triple_combinations(
                    all_paths, cancer_type_normalized, top_n=10,
                    min_coverage=_fallback_cov, n_cell_lines=n_cell_lines,
                )
                if triple_combinations:
                    logger.info(
                        f"Coverage fallback to {_fallback_cov:.0%} yielded "
                        f"{len(triple_combinations)} triples for {cancer_type}"
                    )
                    break
        # Merge: structural triples prepended (they have evidence_power="adequate")
        # so they appear first in output CSV; downstream ranking can re-sort.
        structural_target_sets = {frozenset(s.targets) for s in structural_combinations}
        triple_combinations = structural_combinations + [
            tc for tc in triple_combinations
            if frozenset(tc.targets) not in structural_target_sets
        ]

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
    
    def analyze_all_cancers(self, top_n: int = 20, n_workers: int = 6,
                             include_subtypes: bool = False) -> Dict[str, CancerTypeAnalysis]:
        """Run analysis across top cancer types by cell line count.
        
        Uses ThreadPoolExecutor for parallel analysis of independent cancer types.
        Numpy/pandas vectorized ops release the GIL, so threads run truly parallel
        for the heavy computation.
        
        Args:
            top_n: Max cancer types to analyze (by cell line count)
            n_workers: Number of parallel threads (default 6; set to 1 for sequential)
            include_subtypes: V8.1 — also analyze mutation-defined molecular subtypes
        """
        
        cancer_types = self.depmap.get_available_cancer_types(
            include_subtypes=include_subtypes
        )
        logger.info(f"Found {len(cancer_types)} cancer types in DepMap")
        
        # Sort by cell line count and take top N
        cancer_types = sorted(cancer_types, key=lambda x: -x[1])[:top_n]
        
        valid_cancers = [(ct, c) for ct, c in cancer_types if pd.notna(ct)]
        n_total = len(valid_cancers)
        
        # Pre-load all shared data so threads only read cached values
        print(f"\n  Pre-loading shared data for {n_total} cancer types...", flush=True)
        t_preload = time.time()
        try:
            self.depmap.load_crispr_dependencies()
            self.depmap.load_expression()
            self.depmap.load_lineage_annotations()
            self.depmap.load_subtype_features()
            self.depmap.load_copy_number()       # V7: CN data
            self.depmap.load_rnai_dependencies()  # V7: RNAi data
            if include_subtypes:
                self.depmap.load_hotspot_mutations()  # V8.1: mutation data
            self.path_inference.omnipath.load_signaling_network(use_api=self.enable_api)
        except (FileNotFoundError, pd.errors.EmptyDataError, KeyError, ValueError, OSError) as e:
            logger.debug(f"Pre-load warning: {e}")
        print(f"  Pre-load done in {time.time() - t_preload:.1f}s", flush=True)
        
        print(f"\n  Analyzing {n_total} cancer types with {n_workers} threads:\n", flush=True)
        
        # Reduce logging verbosity during batch analysis
        original_level = logger.level
        logger.setLevel(logging.WARNING)
        
        results = {}
        t_start = time.time()
        
        if n_workers <= 1:
            # Sequential fallback
            for idx, (cancer_type, count) in enumerate(tqdm(valid_cancers, desc="Cancers", unit="type", leave=True), start=1):
                try:
                    analysis = self.analyze_cancer_type(cancer_type)
                    results[cancer_type] = analysis
                except Exception as e:
                    tqdm.write(f"  (skipped {cancer_type}: {e})")
                    continue
        else:
            # Parallel execution with ThreadPoolExecutor
            import threading
            lock = threading.Lock()
            
            def _process_cancer(cancer_type):
                return cancer_type, self.analyze_cancer_type(cancer_type)
            
            actual_workers = min(n_workers, n_total)
            with ThreadPoolExecutor(max_workers=actual_workers) as executor:
                futures = {
                    executor.submit(_process_cancer, ct): (ct, count)
                    for ct, count in valid_cancers
                }
                
                with tqdm(total=n_total, desc="Cancers", unit="type", leave=True) as pbar:
                    for future in as_completed(futures):
                        ct, count = futures[future]
                        try:
                            cancer_type, analysis = future.result()
                            with lock:
                                results[cancer_type] = analysis
                            elapsed = time.time() - t_start
                            done = len(results)
                            if done > 0:
                                eta = elapsed / done * (n_total - done)
                                pbar.set_postfix_str(f"{ct[:20]} | ETA {eta:.0f}s")
                        except Exception as e:
                            tqdm.write(f"  (skipped {ct}: {e})")
                        pbar.update(1)
        
        elapsed = time.time() - t_start
        print(f"\n  Completed {len(results)}/{n_total} cancer types in {elapsed:.1f}s "
              f"({elapsed/max(1,len(results)):.1f}s/type avg)", flush=True)
        
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
                        'lincs_supported': vr.evidence_tier.lincs_supported,
                        'lincs_targets_with_sig': vr.evidence_tier.lincs_targets_with_sig,
                        'lincs_perturbation_score': vr.evidence_tier.lincs_perturbation_score,
                        'lincs_multi_modal_targets': vr.evidence_tier.lincs_multi_modal_targets,
                        'lincs_n_modalities_per_target': vr.evidence_tier.lincs_n_modalities_per_target,
                        'lincs_mean_concordance': vr.evidence_tier.lincs_mean_concordance,
                        'lincs_compound_drugs': vr.evidence_tier.lincs_compound_drugs,
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
        ci_str = ""
        if bt.confidence_interval:
            ci_str = f"\n  95% CI: [{bt.confidence_interval[0]:.3f}, {bt.confidence_interval[1]:.3f}]"
        report += f"""
BEST TRIPLE COMBINATION: {', '.join(bt.targets)}
  Combined Score: {bt.combined_score:.3f} (lower is better){ci_str}
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


def _detect_cell_line_overlaps(results: Dict[str, 'CancerTypeAnalysis'],
                               overlap_threshold: float = 0.80
                               ) -> Dict[str, List[str]]:
    """Detect cancer types with >overlap_threshold cell line overlap.
    
    Returns a dict mapping each cancer type → list of its "duplicate"
    cancer types (those sharing >80% of cell lines).  The primary type
    (most cell lines) is kept; secondaries are flagged.
    """
    overlap_flags: Dict[str, List[str]] = {}
    cancer_list = [(ct, set(a.cell_line_ids)) for ct, a in results.items()
                   if hasattr(a, 'cell_line_ids') and a.cell_line_ids]
    
    for i, (ct_a, cls_a) in enumerate(cancer_list):
        for ct_b, cls_b in cancer_list[i + 1:]:
            if not cls_a or not cls_b:
                continue
            intersection = cls_a & cls_b
            union = cls_a | cls_b
            jaccard = len(intersection) / len(union) if union else 0
            if jaccard >= overlap_threshold:
                # Flag the smaller one as duplicate of the larger
                if len(cls_a) >= len(cls_b):
                    primary, secondary = ct_a, ct_b
                else:
                    primary, secondary = ct_b, ct_a
                overlap_flags.setdefault(secondary, []).append(primary)
                logger.info(f"Cell-line overlap: {ct_a} ∩ {ct_b} = {jaccard:.0%} "
                            f"({len(intersection)}/{len(union)}); "
                            f"flagging {secondary} as near-duplicate of {primary}")
    return overlap_flags


def export_comprehensive_findings(results: Dict[str, CancerTypeAnalysis], output_path: Path,
                                  mode_config: Optional[ModeConfig] = None):
    """Export all findings to comprehensive files"""
    
    # Detect near-duplicate cancer types (>80% cell line overlap)
    _overlap_flags = _detect_cell_line_overlaps(results)
    if _overlap_flags:
        logger.info(f"Found {len(_overlap_flags)} near-duplicate cancer types by cell line overlap")
    
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
    # V6: Mode-divergent output columns
    _is_translational = (mode_config and mode_config.mode == RunMode.ACTIONABLE)
    _is_exploratory = (mode_config and mode_config.mode == RunMode.EXPLORATORY)
    
    def _strategy_arm_for_combo(combo: TripleCombination) -> str:
        return resolve_strategy_arm(combo)

    def _build_triple_export_row(
        cancer_type: str,
        analysis: CancerTypeAnalysis,
        combo: TripleCombination,
        rank: int,
        is_top_prediction: bool,
    ) -> Dict[str, object]:
        targets = list(combo.targets)
        role_assignments = combo.role_assignments or {}
        best_combo = analysis.best_combination
        best_combo_targets = sorted(best_combo.targets) if best_combo is not None else []

        row = {
            'Cancer_Type': cancer_type,
            'Lineage': analysis.lineage,
            'Cell_Lines': analysis.n_cell_lines,
            'Evidence_Tier': (1 if analysis.n_cell_lines >= 30 else
                              2 if analysis.n_cell_lines >= 15 else
                              3 if analysis.n_cell_lines >= 10 else 4),
            'Evidence_Power': combo.evidence_power or ('robust' if analysis.n_cell_lines >= 30 else
                                                       'adequate' if analysis.n_cell_lines >= 10 else
                                                       'suggestive' if analysis.n_cell_lines >= 5 else
                                                       'hypothesis'),
            'Rank': rank,
            'Is_Top_Prediction': is_top_prediction,
            'Strategy_Arm': _strategy_arm_for_combo(combo),
            'Combination_Size': len(targets),
            'Triple_Targets': ' + '.join(targets),
            'Target_1': targets[0] if len(targets) >= 1 else '',
            'Target_2': targets[1] if len(targets) >= 2 else '',
            'Target_3': targets[2] if len(targets) >= 3 else '',
            'Role_Feeder': role_assignments.get('feeder', ''),
            'Role_Driver': role_assignments.get('driver', ''),
            'Role_Escape': role_assignments.get('escape', ''),
            'Combined_Score': f"{combo.combined_score:.3f}",
            'Score_CI_Lower': f"{combo.confidence_interval[0]:.3f}" if combo.confidence_interval else '',
            'Score_CI_Upper': f"{combo.confidence_interval[1]:.3f}" if combo.confidence_interval else '',
            'Synergy_Score': f"{combo.synergy_score:.2f}",
            'Resistance_Score': f"{combo.resistance_score:.2f}",
            'Resistance_Implausible': combo.resistance_implausible,
            'Path_Coverage': f"{combo.coverage:.1%}",
            'Scoring_Mode': combo.scoring_mode,
            'Best_Combo_Size': len(best_combo_targets),
            'Best_Combo_1': best_combo_targets[0] if len(best_combo_targets) >= 1 else '',
            'Best_Combo_2': best_combo_targets[1] if len(best_combo_targets) >= 2 else '',
            'Best_Combo_3': best_combo_targets[2] if len(best_combo_targets) >= 3 else '',
            'Best_Combo_Score': f"{best_combo.combined_score:.3f}" if best_combo is not None else '',
        }

        if _is_translational:
            for i, tgt in enumerate(targets):
                di = drug_db.get_drug_info(tgt)
                drug_name = di.available_drugs[0] if di and di.available_drugs else 'N/A'
                stage = di.clinical_stage if di else 'unknown'
                row[f'Drug_{i+1}'] = drug_name
                row[f'Drug_{i+1}_Stage'] = stage
            row['Druggable_Count'] = combo.druggable_count
            row['Druggability_Tier'] = combo.druggability_tier
            row['Clinical_Readiness'] = f"{combo.clinical_readiness:.2f}"
            row['Combo_Tox_Score'] = f"{combo.combo_tox_score:.2f}"
            _ddi_warnings = []
            if combo.combo_tox_details:
                for pair_key, pair_info in combo.combo_tox_details.items():
                    if isinstance(pair_info, dict) and pair_info.get('ddi_risk'):
                        _ddi_warnings.append(str(pair_key))
            row['DDI_Warnings'] = '; '.join(_ddi_warnings) if _ddi_warnings else 'None'
            row['Therapeutic_Window'] = f"{combo.synergy_score - combo.resistance_score:.2f}"
            row['Pathways_Covered'] = sum(1 for c in combo.pathway_coverage.values() if c > 0)
        else:
            row['Druggable_Count'] = combo.druggable_count
            for i, tgt in enumerate(targets):
                di = drug_db.get_drug_info(tgt)
                row[f'Drug_{i+1}'] = (di.available_drugs[0]
                                      if di and di.available_drugs else 'N/A')
            row['Novelty_Score'] = f"{combo.novelty_score:.2f}"
            row['Pan_Essential_Penalty'] = f"{combo.pan_essential_penalty:.3f}"
            row['Therapeutic_Window'] = f"{combo.synergy_score - combo.resistance_score:.2f}"
            row['Pathways_Covered'] = sum(1 for c in combo.pathway_coverage.values() if c > 0)
            _top_pathways = sorted(
                ((pw, cov) for pw, cov in combo.pathway_coverage.items() if cov > 0),
                key=lambda x: x[1], reverse=True
            )[:3]
            row['Top_Pathways'] = '; '.join(f"{pw}({cov:.0%})" for pw, cov in _top_pathways)

        return row

    triple_rows = []
    ranked_triple_rows = []
    for cancer_type, analysis in results.items():
        if not analysis.best_triple:
            continue

        triple_rows.append(
            _build_triple_export_row(
                cancer_type,
                analysis,
                analysis.best_triple,
                rank=1,
                is_top_prediction=True,
            )
        )

        for rank, combo in enumerate(analysis.triple_combinations, start=1):
            ranked_triple_rows.append(
                _build_triple_export_row(
                    cancer_type,
                    analysis,
                    combo,
                    rank=rank,
                    is_top_prediction=(rank == 1),
                )
            )
    
    if triple_rows:
        df_triples = pd.DataFrame(triple_rows)
        if _is_translational:
            # Translational: rank by clinical readiness * synergy (descending)
            df_triples['_sort_key'] = df_triples['Clinical_Readiness'].astype(float) * df_triples['Synergy_Score'].astype(float)
            df_triples = df_triples.sort_values('_sort_key', ascending=False).drop(columns=['_sort_key'])
        else:
            # Exploratory: rank by synergy desc, resistance asc
            df_triples = df_triples.sort_values(['Synergy_Score', 'Resistance_Score'], ascending=[False, True])
        df_triples.to_csv(output_path / "triple_combinations.csv", index=False)

    if ranked_triple_rows:
        df_ranked_triples = pd.DataFrame(ranked_triple_rows)
        df_ranked_triples['_rank_sort'] = pd.to_numeric(df_ranked_triples['Rank'], errors='coerce').fillna(float('inf'))
        df_ranked_triples = df_ranked_triples.sort_values(
            ['Cancer_Type', '_rank_sort'],
            ascending=[True, True],
        ).drop(columns=['_rank_sort'])
        df_ranked_triples.to_csv(output_path / "ranked_triple_combinations.csv", index=False)
    
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
        'mode': mode_config.mode.value if mode_config else 'actionable',
        'mode_config': mode_config.to_dict() if mode_config else None,
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
                'strategy_arm': resolve_strategy_arm(bt),
                'role_assignments': dict(bt.role_assignments or {}),
                'combined_score': bt.combined_score,
                'confidence_interval': list(bt.confidence_interval) if bt.confidence_interval else None,
                'synergy_score': bt.synergy_score,
                'resistance_score': bt.resistance_score,
                'resistance_implausible': bt.resistance_implausible,
                'path_coverage': bt.coverage,
                'druggable_count': bt.druggable_count,
                'druggability_tier': bt.druggability_tier,
                'evidence_power': bt.evidence_power,
                'pathways_covered': {k: v for k, v in bt.pathway_coverage.items() if v > 0}
            }

        # ── Evidence tier (audit recommendation #3) ──────────────────
        # Flag cancer types by statistical power based on cell line count.
        # Tier 1: ≥30 cell lines — robust statistics
        # Tier 2: 15–29 — adequate power for most tests
        # Tier 3: 10–14 — marginal; some tests underpowered
        # Tier 4: <10  — low power; results should be treated as
        #                 hypothesis-generating only
        n_cl = analysis.n_cell_lines
        if n_cl >= 30:
            evidence_tier = 1
            tier_label = 'robust'
        elif n_cl >= 15:
            evidence_tier = 2
            tier_label = 'adequate'
        elif n_cl >= 10:
            evidence_tier = 3
            tier_label = 'marginal'
        else:
            evidence_tier = 4
            tier_label = 'low_power'

        analysis_dict = {
            'cancer_type': analysis.cancer_type,
            'lineage': analysis.lineage,
            'n_cell_lines': analysis.n_cell_lines,
            'evidence_tier': evidence_tier,
            'evidence_tier_label': tier_label,
            'driver_mutations': analysis.driver_mutations,
            'recommended_combination': analysis.recommended_combination,
            'combination_cost': analysis.minimal_hitting_sets[0].total_cost if analysis.minimal_hitting_sets else None,
            'path_coverage': analysis.minimal_hitting_sets[0].coverage if analysis.minimal_hitting_sets else None,
            'n_viability_paths': len(analysis.viability_paths),
            'top_essential_genes': dict(list(analysis.essential_genes.items())[:10]),
            'best_triple_combination': triple_info,
            'n_triple_alternatives': len(analysis.triple_combinations),
            'near_duplicate_of': _overlap_flags.get(cancer_type, []),
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
    print(f"  4. triple_combinations.csv         - Top-ranked triple per cancer")
    print(f"  5. ranked_triple_combinations.csv  - Explicit ranked triple predictions")
    print(f"  6. triple_target_frequency.csv     - Most common targets in triples")
    print(f"  7. all_findings.json               - Complete analysis data")
    print(f"  8. pan_cancer_summary.csv          - Summary table")
    print(f"  9. [CancerType]_report.txt         - Individual reports")

    # 7. Unresolved gene report (protein scoring gaps → wet-lab priorities)
    try:
        from alin.protein_scoring import get_unresolved_genes
        unresolved = get_unresolved_genes()
        if unresolved:
            unresolved_sorted = sorted(unresolved)
            with open(output_path / "unresolved_genes_wetlab_gaps.txt", 'w') as f:
                f.write("# Genes with no reviewed human UniProt (Swiss-Prot) entry\n")
                f.write("# These genes received fallback protein druggability scores.\n")
                f.write("# Wet-lab characterization (structure, abundance, degradability)\n")
                f.write("# would improve scoring accuracy for these targets.\n")
                f.write(f"# Total: {len(unresolved_sorted)} genes\n\n")
                for g in unresolved_sorted:
                    f.write(f"{g}\n")
            print(f"  9. unresolved_genes_wetlab_gaps.txt - {len(unresolved_sorted)} genes needing wet-lab data")
    except (ImportError, OSError, KeyError) as exc:
        logger.debug('Unresolved-genes export skipped: %s', exc)
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
            'Rank': 1,
            'Strategy_Arm': resolve_strategy_arm(bt),
            'Scoring_Mode': bt.scoring_mode,
            'Target 1': bt.targets[0],
            'Target 2': bt.targets[1],
            'Target 3': bt.targets[2],
            'Role Feeder': (bt.role_assignments or {}).get('feeder', ''),
            'Role Driver': (bt.role_assignments or {}).get('driver', ''),
            'Role Escape': (bt.role_assignments or {}).get('escape', ''),
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
  python pan_cancer_xnode.py --all-cancers --triples --discovery-mode  # Biology-first (no druggability bias)
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
    parser.add_argument('--workers', type=int, default=6,
                        help='Number of parallel threads for pan-cancer analysis (default: 6)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--mode', type=str, default='actionable',
                        choices=['actionable', 'exploratory'],
                        help='Run mode: actionable (translational, drug-focused) or '
                             'exploratory (biology-first, undrugged targets welcome)')
    parser.add_argument('--subtypes', action='store_true',
                        help='V8.1: Enable mutation-defined molecular subtype stratification '
                             '(e.g., KRAS-mut NSCLC, BRAF-mut CRC analyzed separately)')
    parser.add_argument('--strategy-arm', type=str, default=None,
                        choices=SUPPORTED_STRATEGY_ARMS,
                        help='Explicit comparison arm to run: default, liaki_role, or '
                             'liaki_pdac_template. When omitted, actionable mode defaults '
                             'to liaki_role and exploratory mode defaults to default. '
                             'Structural arms enable the Liaki-style upstream/driver/escape '
                             'finder.')
    parser.add_argument('--structural', action='store_true',
                        help='Legacy alias for --strategy-arm liaki_role. Enables structural '
                             '(upstream, driver, escape) triple formation based on the Liaki '
                             'attractor-disruption framework.')
    
    args = parser.parse_args()
    
    # Build mode config
    from alin.run_modes import get_config
    mode_cfg = get_config(args.mode)
    
    # Pin global random seed for reproducibility
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    
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
    # Apply output suffix from mode config (e.g. results_exploratory/)
    _output_dir = args.output
    if mode_cfg.output_suffix:
        base = args.output.rstrip('/').rstrip('\\')
        if not base.endswith(mode_cfg.output_suffix):
            _output_dir = base + mode_cfg.output_suffix
    analyzer = PanCancerXNodeAnalyzer(
        data_dir=args.data_dir,
        validation_data_dir=args.validation_dir,
        enable_api=not args.no_api,
        mode_config=mode_cfg,
        strategy_arm=args.strategy_arm,
        structural_mode=getattr(args, 'structural', False),
    )
    logger.info(f"Run mode: {mode_cfg.mode.value} (W_DRUGGABLE={mode_cfg.W_DRUGGABLE}, "
                f"W_ESSENTIALITY={mode_cfg.W_ESSENTIALITY}, "
                f"prefer_druggable={mode_cfg.prefer_druggable}, "
                f"strategy_arm={analyzer.strategy_arm})")
    
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
    
    _use_subtypes = getattr(args, 'subtypes', False)

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
        if _use_subtypes:
            analyzer.depmap.load_hotspot_mutations()
            analyzer.depmap.build_molecular_subtypes()
        logger.info(f"Running single-cancer analysis: {args.cancer_type}")
        analysis = analyzer.analyze_cancer_type(args.cancer_type)
        
        report = generate_cancer_report(analysis)
        print(report)
        
        # Save results
        output_path = Path(_output_dir)
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
        if _use_subtypes:
            print("  V8.1: Molecular subtype stratification ENABLED\n", flush=True)
        logger.info(f"Running pan-cancer analysis (max {args.top_n} cancer types, {args.workers} workers)")
        results = analyzer.analyze_all_cancers(
            top_n=args.top_n, n_workers=args.workers,
            include_subtypes=_use_subtypes,
        )
        
        # Save results
        output_path = Path(_output_dir)
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
        export_comprehensive_findings(results, output_path, mode_config=mode_cfg)
        
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
