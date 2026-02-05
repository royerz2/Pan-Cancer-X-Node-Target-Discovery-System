#!/usr/bin/env python3
"""
Drug Sensitivity Data Integration Module
==========================================
Integrates CellMiner (NCI-60), GDSC, and PRISM drug sensitivity data
to validate predicted drug combinations.

Data Sources:
- CellMiner (NCI-60): https://discover.nci.nih.gov/cellminer/
- GDSC (Sanger): https://www.cancerrxgene.org/
- PRISM (Broad): https://depmap.org/portal/prism/

Features:
- Drug sensitivity correlation with gene dependency
- Biomarker discovery for drug response
- Combination synergy prediction from single-agent data
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from pathlib import Path
from scipy import stats
from scipy.spatial.distance import correlation
import logging
import requests
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
logger = logging.getLogger(__name__)

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class DrugSensitivityProfile:
    """Drug sensitivity across cell lines"""
    drug_name: str
    target_genes: List[str]
    cell_lines: List[str]
    ic50_values: List[float]  # -log10(IC50) typically
    auc_values: List[float]   # Area under dose-response curve
    source: str               # GDSC, PRISM, CellMiner
    
@dataclass
class GeneDrugCorrelation:
    """Correlation between gene dependency and drug sensitivity"""
    gene: str
    drug: str
    correlation: float
    p_value: float
    n_samples: int
    interpretation: str  # 'sensitive' if negative correlation
    
@dataclass
class CombinationPrediction:
    """Predicted combination synergy from single-agent data"""
    drug_a: str
    drug_b: str
    drug_c: Optional[str]
    target_a: str
    target_b: str
    target_c: Optional[str]
    bliss_independence: float  # Expected effect if independent
    predicted_synergy: float   # Deviation from Bliss
    confidence: float
    evidence: str

@dataclass  
class DrugSensitivityValidation:
    """Complete drug sensitivity validation result"""
    targets: List[str]
    drugs: List[str]
    cancer_type: str
    
    # Single agent data
    drug_profiles: List[DrugSensitivityProfile] = field(default_factory=list)
    
    # Gene-drug correlations
    correlations: List[GeneDrugCorrelation] = field(default_factory=list)
    
    # Combination predictions
    synergy_prediction: Optional[CombinationPrediction] = None
    
    # Biomarkers
    response_biomarkers: Dict[str, List[str]] = field(default_factory=dict)
    
    # Scores
    validation_score: float = 0.0
    confidence: str = "unknown"

# ============================================================================
# GDSC DATA LOADER
# ============================================================================

class GDSCLoader:
    """
    Loader for Genomics of Drug Sensitivity in Cancer (GDSC) data
    https://www.cancerrxgene.org/
    """
    
    # Common drug-target mappings
    DRUG_TARGETS = {
        'Erlotinib': ['EGFR'],
        'Gefitinib': ['EGFR'],
        'Osimertinib': ['EGFR'],
        'Lapatinib': ['EGFR', 'ERBB2'],
        'Afatinib': ['EGFR', 'ERBB2', 'ERBB4'],
        'Vemurafenib': ['BRAF'],
        'Dabrafenib': ['BRAF'],
        'Encorafenib': ['BRAF'],
        'Trametinib': ['MAP2K1', 'MAP2K2'],
        'Cobimetinib': ['MAP2K1'],
        'Binimetinib': ['MAP2K1'],
        'Palbociclib': ['CDK4', 'CDK6'],
        'Ribociclib': ['CDK4', 'CDK6'],
        'Abemaciclib': ['CDK4', 'CDK6'],
        'Sotorasib': ['KRAS'],
        'Adagrasib': ['KRAS'],
        'Alpelisib': ['PIK3CA'],
        'Everolimus': ['MTOR'],
        'Temsirolimus': ['MTOR'],
        'Crizotinib': ['ALK', 'MET', 'ROS1'],
        'Alectinib': ['ALK'],
        'Lorlatinib': ['ALK', 'ROS1'],
        'Capmatinib': ['MET'],
        'Tepotinib': ['MET'],
        'Ruxolitinib': ['JAK1', 'JAK2'],
        'Tofacitinib': ['JAK1', 'JAK3'],
        'Venetoclax': ['BCL2'],
        'Navitoclax': ['BCL2', 'BCL2L1', 'BCL2L2'],
        'Dasatinib': ['SRC', 'ABL1', 'FYN', 'LCK'],
        'Bosutinib': ['SRC', 'ABL1'],
        'Imatinib': ['ABL1', 'KIT', 'PDGFRA'],
        'Sorafenib': ['RAF1', 'BRAF', 'VEGFR2', 'KIT'],
        'Regorafenib': ['RAF1', 'VEGFR2', 'KIT'],
        'Sunitinib': ['VEGFR2', 'KIT', 'PDGFRA'],
        'Pazopanib': ['VEGFR1', 'VEGFR2', 'VEGFR3'],
        'Olaparib': ['PARP1', 'PARP2'],
        'Niraparib': ['PARP1', 'PARP2'],
        'Talazoparib': ['PARP1', 'PARP2'],
    }
    
    # Reverse mapping: target to drugs
    TARGET_TO_DRUGS = {}
    for drug, targets in DRUG_TARGETS.items():
        for target in targets:
            if target not in TARGET_TO_DRUGS:
                TARGET_TO_DRUGS[target] = []
            TARGET_TO_DRUGS[target].append(drug)
    
    def __init__(self, data_dir: str = "./drug_sensitivity_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True, parents=True)
        
        self._ic50_data = None
        self._cell_info = None
        self._drug_info = None
        self._gdsc_warned = False
        
    def download_data(self) -> bool:
        """
        Download GDSC data files
        
        Note: In practice, download from https://www.cancerrxgene.org/downloads
        Files needed:
        - GDSC2_fitted_dose_response.xlsx or .csv
        - Cell_Lines_Details.xlsx
        """
        ic50_file = self.data_dir / "GDSC2_IC50.csv"
        
        if ic50_file.exists():
            logger.info("Loading existing GDSC data...")
            self._ic50_data = pd.read_csv(ic50_file)
            return True
        
        if not self._gdsc_warned:
            self._gdsc_warned = True
            logger.info("GDSC data not found. Using PRISM. Download GDSC from: https://www.cancerrxgene.org/downloads")
        return False
    
    def get_drug_sensitivity(self, drug_name: str, cancer_type: str = None) -> Optional[DrugSensitivityProfile]:
        """Get sensitivity profile for a drug"""
        if self._ic50_data is None:
            if not self.download_data():
                return None
        
        # Filter by drug
        drug_data = self._ic50_data[
            self._ic50_data['DRUG_NAME'].str.lower() == drug_name.lower()
        ]
        
        if drug_data.empty:
            return None
        
        # Filter by cancer type if specified
        if cancer_type and 'CANCER_TYPE' in drug_data.columns:
            drug_data = drug_data[
                drug_data['CANCER_TYPE'].str.contains(cancer_type, case=False, na=False)
            ]
        
        # Get targets
        targets = self.DRUG_TARGETS.get(drug_name, [])
        
        return DrugSensitivityProfile(
            drug_name=drug_name,
            target_genes=targets,
            cell_lines=drug_data['CELL_LINE_NAME'].tolist() if 'CELL_LINE_NAME' in drug_data else [],
            ic50_values=drug_data['LN_IC50'].tolist() if 'LN_IC50' in drug_data else [],
            auc_values=drug_data['AUC'].tolist() if 'AUC' in drug_data else [],
            source='GDSC'
        )
    
    def get_drugs_for_target(self, target: str) -> List[str]:
        """Get drugs that target a specific gene"""
        return self.TARGET_TO_DRUGS.get(target, [])


# ============================================================================
# PRISM DATA LOADER
# ============================================================================
#
# PRISM Repurposing data from DepMap/Broad Institute
# https://depmap.org/repurposing/
#
# Primary screen: 4,518 compounds x 578 cell lines (LFC values)
# Secondary screen: 1,448 compounds x 489 cell lines (IC50, AUC dose-response)
#
# Figshare download URLs (Corsello et al. 2020):
# - primary-screen-replicate-collapsed-logfold-change.csv (39 MB)
# - primary-screen-replicate-collapsed-treatment-info.csv (1 MB)
# - secondary-screen-dose-response-curve-parameters.csv (252 MB)
# ============================================================================

PRISM_FIGSHARE_URLS = {
    'primary_lfc': 'https://ndownloader.figshare.com/files/20237709',
    'primary_treatment_info': 'https://ndownloader.figshare.com/files/20237715',
    'secondary_dose_response': 'https://ndownloader.figshare.com/files/20237739',
}

# Drug name aliases for matching PRISM compound names
PRISM_DRUG_ALIASES = {
    'sotorasib': ['sotorasib', 'amg 510', 'amg510', 'luma'],
    'adagrasib': ['adagrasib', 'mrtx849', 'krazati'],
    'palbociclib': ['palbociclib', 'ibrance', 'pd-0332991'],
    'ribociclib': ['ribociclib', 'kisqali', 'lee011'],
    'erlotinib': ['erlotinib', 'tarceva', 'osl-774'],
    'vemurafenib': ['vemurafenib', 'zelboraf', 'plx4032'],
    'trametinib': ['trametinib', 'mekinist', 'gsk1120212'],
    'dabrafenib': ['dabrafenib', 'tafinlar', 'gsb2118436'],
    'osimertinib': ['osimertinib', 'tagrisso', 'azd9291'],
    'capmatinib': ['capmatinib', 'tabrecta', 'inc280'],
    'napabucasin': ['napabucasin', 'bbi608', 'bb608'],
    'dinaciclib': ['dinaciclib', 'sch727965'],
}


class PRISMLoader:
    """
    Loader for PRISM drug sensitivity data (Broad Institute)
    https://depmap.org/portal/prism/ | https://depmap.org/repurposing/
    
    Supports:
    - Primary screen: LFC values (negative = sensitive)
    - Secondary screen: IC50, AUC dose-response parameters
    - Auto-download from Figshare
    - Drug name matching via treatment metadata
    """
    
    def __init__(self, data_dir: str = "./drug_sensitivity_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True, parents=True)
        
        self._primary_lfc = None      # Rows=cell lines, Cols=column_name
        self._primary_treatment_info = None  # column_name -> name, target, etc.
        self._secondary_dr = None     # Long format: name, depmap_id, auc, ic50
        self._drug_to_columns = {}   # drug_name -> [column_names]
        
    def download_data(self, download_secondary: bool = False) -> bool:
        """
        Load or download PRISM data.
        
        Accepts both PRISM naming and simplified prism_*.csv.
        Set download_secondary=True to fetch dose-response data (252 MB).
        """
        # Try standard PRISM filenames first
        primary_lfc = self.data_dir / "primary-screen-replicate-collapsed-logfold-change.csv"
        primary_info = self.data_dir / "primary-screen-replicate-collapsed-treatment-info.csv"
        secondary_dr = self.data_dir / "secondary-screen-dose-response-curve-parameters.csv"
        
        # Fallback to simplified names
        if not primary_lfc.exists():
            primary_lfc = self.data_dir / "prism_primary.csv"
        if not primary_info.exists():
            primary_info = self.data_dir / "prism_treatment_info.csv"
        if not secondary_dr.exists():
            secondary_dr = self.data_dir / "prism_secondary_dr.csv"
        
        # Load primary LFC
        if primary_lfc.exists():
            logger.info("Loading PRISM primary screen (LFC)...")
            self._primary_lfc = pd.read_csv(primary_lfc, index_col=0)
            
            # Load treatment info for drug name mapping
            if primary_info.exists():
                self._primary_treatment_info = pd.read_csv(primary_info)
                self._build_drug_column_map()
            else:
                # Use column names as drug names if no treatment info
                self._drug_to_columns = {c: [c] for c in self._primary_lfc.columns}
            return True
        
        # Auto-download from Figshare
        logger.info("PRISM data not found. Attempting download from Figshare...")
        try:
            self._download_file(PRISM_FIGSHARE_URLS['primary_lfc'], primary_lfc)
            self._download_file(PRISM_FIGSHARE_URLS['primary_treatment_info'], primary_info)
            
            self._primary_lfc = pd.read_csv(primary_lfc, index_col=0)
            self._primary_treatment_info = pd.read_csv(primary_info)
            self._build_drug_column_map()
            return True
        except Exception as e:
            logger.warning(f"PRISM download failed: {e}")
            logger.info("Manual download: https://depmap.org/repurposing/")
            logger.info("  - primary-screen-replicate-collapsed-logfold-change.csv")
            logger.info("  - primary-screen-replicate-collapsed-treatment-info.csv")
            return False
    
    def _download_file(self, url: str, dest: Path):
        """Download file from URL with progress"""
        import urllib.request
        dest.parent.mkdir(exist_ok=True, parents=True)
        urllib.request.urlretrieve(url, dest)
        logger.info(f"Downloaded {dest.name}")
    
    def _build_drug_column_map(self):
        """Build drug name -> column_name mapping from treatment info"""
        if self._primary_treatment_info is None:
            return
        
        self._drug_to_columns = {}
        for _, row in self._primary_treatment_info.iterrows():
            col = row.get('column_name', row.get('column_name', ''))
            name = str(row.get('name', '')).strip()
            if pd.isna(name) or not name or name.lower() in ('nan', ''):
                continue
            name_lower = name.lower()
            if name_lower not in self._drug_to_columns:
                self._drug_to_columns[name_lower] = []
            self._drug_to_columns[name_lower].append(col)
        
        # Also map by first word (e.g., "Palbociclib" from "Palbociclib (Selleck)")
        for name_lower, cols in list(self._drug_to_columns.items()):
            first_word = name_lower.split()[0] if name_lower else ''
            if first_word and first_word not in self._drug_to_columns:
                self._drug_to_columns[first_word] = cols
    
    def _match_drug_to_columns(self, drug_name: str) -> List[str]:
        """Find PRISM columns matching drug name"""
        drug_lower = drug_name.lower()
        
        # Direct match
        if drug_lower in self._drug_to_columns:
            return self._drug_to_columns[drug_lower]
        
        # Alias match
        aliases = PRISM_DRUG_ALIASES.get(drug_lower, [drug_lower])
        for alias in aliases:
            for key, cols in self._drug_to_columns.items():
                if alias in key or key in alias:
                    return cols
        
        # Substring match
        for key, cols in self._drug_to_columns.items():
            if drug_lower in key or key in drug_lower:
                return cols
        
        return []
    
    def get_drug_sensitivity(self, drug_name: str, cancer_type: str = None) -> Optional[DrugSensitivityProfile]:
        """
        Get drug sensitivity profile from PRISM.
        
        Primary screen: Uses LFC (log2 fold-change). More negative = more sensitive.
        Secondary screen: Uses AUC/IC50 when available (better for correlation).
        """
        if self._primary_lfc is None:
            if not self.download_data():
                return None
        
        # Find matching columns
        matching_cols = self._match_drug_to_columns(drug_name)
        if not matching_cols:
            # Fallback: search in raw column names
            matching_cols = [c for c in self._primary_lfc.columns 
                            if drug_name.lower() in str(c).lower()]
        
        if not matching_cols:
            logger.debug(f"PRISM: No match for drug '{drug_name}'")
            return None
        
        # Use first matching column (or average if multiple doses)
        col = matching_cols[0]
        if col not in self._primary_lfc.columns:
            return None
        
        series = self._primary_lfc[col].dropna()
        cell_lines = series.index.tolist()
        
        # LFC: negative = cell death = sensitive. Use -LFC as "sensitivity" for consistency
        lfc_values = series.values.tolist()
        # For correlation with DepMap: more negative LFC = more sensitive
        # Store as-is (negative = sensitive) - DepMap dependency is also negative for essential
        ic50_values = lfc_values  # Using LFC as proxy; lower = more sensitive
        
        # Filter by cancer type if cell line metadata available
        if cancer_type:
            cell_info_path = self.data_dir / "primary-screen-cell-line-info.csv"
            if not cell_info_path.exists():
                cell_info_path = self.data_dir / "prism_cell_info.csv"
            if cell_info_path.exists():
                cell_info = pd.read_csv(cell_info_path)
                if 'primary_tissue' in cell_info.columns or 'secondary_tissue' in cell_info.columns:
                    tissue_col = 'primary_tissue' if 'primary_tissue' in cell_info.columns else 'secondary_tissue'
                    cancer_cells = cell_info[
                        cell_info[tissue_col].str.contains(cancer_type.split()[0], case=False, na=False)
                    ]['row_name'].tolist()
                    if cancer_cells:
                        mask = [c in cancer_cells for c in cell_lines]
                        cell_lines = [c for c, m in zip(cell_lines, mask) if m]
                        ic50_values = [v for v, m in zip(ic50_values, mask) if m]
        
        # Get target from treatment info
        targets = []
        if self._primary_treatment_info is not None:
            row = self._primary_treatment_info[
                self._primary_treatment_info['column_name'] == col
            ].iloc[0]
            target = row.get('target', '')
            if pd.notna(target) and target:
                targets = [str(t).strip() for t in str(target).split(';')]
        
        return DrugSensitivityProfile(
            drug_name=drug_name,
            target_genes=targets,
            cell_lines=cell_lines,
            ic50_values=ic50_values,
            auc_values=[],  # Primary screen has no AUC
            source='PRISM'
        )
    
    def get_drug_sensitivity_secondary(self, drug_name: str, cancer_type: str = None) -> Optional[DrugSensitivityProfile]:
        """
        Get drug sensitivity from PRISM secondary screen (IC50, AUC).
        Better for correlation with DepMap - requires secondary data file.
        """
        secondary_file = self.data_dir / "secondary-screen-dose-response-curve-parameters.csv"
        if not secondary_file.exists():
            secondary_file = self.data_dir / "prism_secondary_dr.csv"
        
        if not secondary_file.exists():
            logger.info("PRISM secondary data not found. Download from depmap.org/repurposing/")
            return None
        
        try:
            dr = pd.read_csv(secondary_file)
        except Exception as e:
            logger.warning(f"PRISM secondary load failed: {e}")
            return None
        
        # Match drug by name
        name_col = 'name' if 'name' in dr.columns else 'Name'
        dr_drug = dr[dr[name_col].str.lower().str.contains(drug_name.lower(), na=False)]
        
        if dr_drug.empty:
            return None
        
        # Get cell line IDs and AUC/IC50
        id_col = 'depmap_id' if 'depmap_id' in dr.columns else 'DepMap_ID'
        cell_col = 'ccle_name' if 'ccle_name' in dr.columns else 'CCLE_Name'
        auc_col = 'auc' if 'auc' in dr.columns else 'AUC'
        ic50_col = 'ic50' if 'ic50' in dr.columns else 'IC50'
        
        cell_lines = dr_drug[cell_col].tolist() if cell_col in dr_drug else dr_drug[id_col].tolist()
        auc_values = dr_drug[auc_col].tolist() if auc_col in dr_drug else []
        ic50_values = dr_drug[ic50_col].tolist() if ic50_col in dr_drug else []
        
        # Target from metadata
        target_col = 'target' if 'target' in dr_drug.columns else 'Target'
        targets = []
        if target_col in dr_drug and dr_drug[target_col].notna().any():
            targets = [str(t) for t in dr_drug[target_col].dropna().iloc[0].split(';')]
        
        return DrugSensitivityProfile(
            drug_name=drug_name,
            target_genes=targets,
            cell_lines=cell_lines,
            ic50_values=ic50_values,
            auc_values=auc_values,
            source='PRISM_secondary'
        )
    
    def list_available_drugs(self, limit: int = 50) -> List[str]:
        """List drugs available in PRISM (for debugging)"""
        if self._primary_lfc is None and not self.download_data():
            return []
        
        if self._drug_to_columns:
            return sorted(self._drug_to_columns.keys())[:limit]
        return list(self._primary_lfc.columns)[:limit]


# ============================================================================
# DRUG SENSITIVITY VALIDATOR
# ============================================================================

class DrugSensitivityValidator:
    """
    Validates drug combinations using drug sensitivity data
    
    Approaches:
    1. Single-agent sensitivity correlation with target dependency
    2. Synthetic lethality prediction from sensitivity patterns
    3. Bliss independence model for combination prediction
    """
    
    def __init__(self, data_dir: str = "./drug_sensitivity_data", 
                 depmap_data: pd.DataFrame = None):
        self.gdsc = GDSCLoader(data_dir)
        self.prism = PRISMLoader(data_dir)
        self.depmap = depmap_data  # Gene dependency data
        
    def correlate_dependency_sensitivity(self, 
                                         gene: str, 
                                         drug: str,
                                         min_samples: int = 20) -> Optional[GeneDrugCorrelation]:
        """
        Correlate gene dependency with drug sensitivity
        
        Hypothesis: If gene X is essential and drug Y targets X,
        cell lines dependent on X should be sensitive to Y.
        """
        if self.depmap is None:
            logger.warning("DepMap data not provided for correlation analysis")
            return None
        
        # Get drug sensitivity (GDSC first, then PRISM primary, then PRISM secondary)
        drug_sens = self.gdsc.get_drug_sensitivity(drug)
        if drug_sens is None:
            drug_sens = self.prism.get_drug_sensitivity_secondary(drug)
        if drug_sens is None:
            drug_sens = self.prism.get_drug_sensitivity(drug)
        if drug_sens is None:
            return None
        ic50_values = drug_sens.ic50_values
        cell_lines = drug_sens.cell_lines
        
        # Get gene dependency
        if gene not in self.depmap.columns:
            return None
        
        # Match cell lines
        common_cells = list(set(cell_lines) & set(self.depmap.index))
        
        if len(common_cells) < min_samples:
            return None
        
        # Calculate correlation
        dep_values = self.depmap.loc[common_cells, gene].values
        sens_values = [ic50_values[cell_lines.index(c)] for c in common_cells]
        
        corr, pval = stats.pearsonr(dep_values, sens_values)
        
        # Interpretation
        if corr < -0.3 and pval < 0.05:
            interp = "Cell lines dependent on gene are sensitive to drug"
        elif corr > 0.3 and pval < 0.05:
            interp = "Unexpected positive correlation - requires investigation"
        else:
            interp = "No significant correlation"
        
        return GeneDrugCorrelation(
            gene=gene,
            drug=drug,
            correlation=corr,
            p_value=pval,
            n_samples=len(common_cells),
            interpretation=interp
        )
    
    def predict_combination_synergy(self,
                                    targets: List[str],
                                    drugs: List[str]) -> Optional[CombinationPrediction]:
        """
        Predict combination synergy using Bliss independence model
        
        Bliss Independence: E_AB = E_A + E_B - E_A*E_B
        If observed > expected: synergy
        If observed < expected: antagonism
        """
        if len(targets) < 2 or len(drugs) < 2:
            return None
        
        # Get sensitivity profiles (GDSC first, then PRISM)
        profiles = []
        for drug in drugs[:3]:  # Limit to triple
            profile = self.gdsc.get_drug_sensitivity(drug)
            if profile is None:
                profile = self.prism.get_drug_sensitivity_secondary(drug)
            if profile is None:
                profile = self.prism.get_drug_sensitivity(drug)
            if profile:
                profiles.append(profile)
        
        if len(profiles) < 2:
            return None
        
        # Find common cell lines
        common_cells = set(profiles[0].cell_lines)
        for p in profiles[1:]:
            common_cells &= set(p.cell_lines)
        
        common_cells = list(common_cells)[:50]  # Limit for computation
        
        if len(common_cells) < 10:
            return None
        
        # Calculate Bliss independence
        # Convert IC50 to effect (0-1 scale)
        effects = []
        for p in profiles:
            ic50_dict = dict(zip(p.cell_lines, p.ic50_values))
            cell_ic50 = [ic50_dict.get(c, np.nan) for c in common_cells]
            # Normalize to 0-1 (higher = more sensitive)
            cell_ic50 = np.array(cell_ic50)
            effect = 1 / (1 + np.exp(cell_ic50))  # Sigmoid transform
            effects.append(effect)
        
        # Bliss independence for double/triple
        if len(effects) == 2:
            bliss = effects[0] + effects[1] - effects[0] * effects[1]
        else:  # Triple
            bliss = (effects[0] + effects[1] + effects[2] 
                    - effects[0]*effects[1] 
                    - effects[0]*effects[2] 
                    - effects[1]*effects[2]
                    + effects[0]*effects[1]*effects[2])
        
        bliss_mean = np.nanmean(bliss)
        
        # Estimate synergy (positive = synergistic)
        # This is a simplified prediction - real synergy requires actual combo data
        synergy_estimate = 0.0
        
        # Heuristic: targets in same pathway often show synergy
        pathway_overlap = self._estimate_pathway_overlap(targets)
        synergy_estimate += 0.1 * pathway_overlap
        
        # Confidence based on data availability
        confidence = min(1.0, len(common_cells) / 30)
        
        return CombinationPrediction(
            drug_a=drugs[0],
            drug_b=drugs[1],
            drug_c=drugs[2] if len(drugs) > 2 else None,
            target_a=targets[0],
            target_b=targets[1],
            target_c=targets[2] if len(targets) > 2 else None,
            bliss_independence=bliss_mean,
            predicted_synergy=synergy_estimate,
            confidence=confidence,
            evidence=f"Based on {len(common_cells)} cell lines, pathway overlap: {pathway_overlap:.2f}"
        )
    
    def _estimate_pathway_overlap(self, targets: List[str]) -> float:
        """Estimate if targets are in related pathways"""
        # Simplified pathway assignment
        pathways = {
            'KRAS': 'RAS-MAPK', 'BRAF': 'RAS-MAPK', 'MAP2K1': 'RAS-MAPK', 'MAP2K2': 'RAS-MAPK',
            'PIK3CA': 'PI3K-AKT', 'AKT1': 'PI3K-AKT', 'MTOR': 'PI3K-AKT', 'PTEN': 'PI3K-AKT',
            'EGFR': 'RTK', 'ERBB2': 'RTK', 'MET': 'RTK', 'FGFR1': 'RTK',
            'CDK4': 'Cell Cycle', 'CDK6': 'Cell Cycle', 'CDK2': 'Cell Cycle', 'CCND1': 'Cell Cycle',
            'STAT3': 'JAK-STAT', 'JAK1': 'JAK-STAT', 'JAK2': 'JAK-STAT',
            'SRC': 'SFK', 'FYN': 'SFK', 'LCK': 'SFK', 'YES1': 'SFK',
            'BCL2': 'Apoptosis', 'MCL1': 'Apoptosis', 'BCL2L1': 'Apoptosis',
        }
        
        target_pathways = [pathways.get(t, 'Unknown') for t in targets]
        
        # Calculate pairwise overlap
        n_same = sum(1 for i, p1 in enumerate(target_pathways) 
                    for p2 in target_pathways[i+1:] 
                    if p1 == p2 and p1 != 'Unknown')
        
        n_pairs = len(targets) * (len(targets) - 1) / 2
        return n_same / n_pairs if n_pairs > 0 else 0.0
    
    def find_response_biomarkers(self, 
                                 drug: str, 
                                 n_top: int = 10) -> List[Tuple[str, float, str]]:
        """
        Find genetic biomarkers that predict drug response
        
        Returns:
            List of (gene, correlation, direction) tuples
        """
        if self.depmap is None:
            return []
        
        drug_sens = self.gdsc.get_drug_sensitivity(drug)
        if drug_sens is None:
            drug_sens = self.prism.get_drug_sensitivity_secondary(drug)
        if drug_sens is None:
            drug_sens = self.prism.get_drug_sensitivity(drug)
        if drug_sens is None:
            return []
        
        # Correlate each gene with drug sensitivity
        biomarkers = []
        
        for gene in self.depmap.columns[:500]:  # Limit for speed
            corr_result = self.correlate_dependency_sensitivity(gene, drug, min_samples=15)
            if corr_result and abs(corr_result.correlation) > 0.2:
                direction = 'sensitivity' if corr_result.correlation < 0 else 'resistance'
                biomarkers.append((gene, corr_result.correlation, direction))
        
        # Sort by absolute correlation
        biomarkers.sort(key=lambda x: abs(x[1]), reverse=True)
        return biomarkers[:n_top]
    
    def validate_combination(self, 
                            targets: List[str], 
                            drugs: List[str],
                            cancer_type: str,
                            skip_biomarkers: bool = False) -> DrugSensitivityValidation:
        """
        Complete drug sensitivity validation for a combination
        """
        validation = DrugSensitivityValidation(
            targets=targets,
            drugs=drugs,
            cancer_type=cancer_type
        )
        
        # Get drug profiles (GDSC first, then PRISM)
        for drug in drugs:
            profile = self.gdsc.get_drug_sensitivity(drug, cancer_type)
            if profile is None:
                profile = self.prism.get_drug_sensitivity_secondary(drug, cancer_type)
            if profile is None:
                profile = self.prism.get_drug_sensitivity(drug, cancer_type)
            if profile:
                validation.drug_profiles.append(profile)
        
        # Calculate gene-drug correlations
        for gene, drug in zip(targets, drugs):
            corr = self.correlate_dependency_sensitivity(gene, drug)
            if corr:
                validation.correlations.append(corr)
        
        # Predict synergy
        validation.synergy_prediction = self.predict_combination_synergy(targets, drugs)
        
        # Find biomarkers for each drug (skip for faster batch runs)
        if not skip_biomarkers:
            for drug in drugs:
                biomarkers = self.find_response_biomarkers(drug, n_top=5)
                validation.response_biomarkers[drug] = [b[0] for b in biomarkers]
        
        # Calculate validation score
        scores = []
        
        # Score from drug profiles (data availability)
        if validation.drug_profiles:
            scores.append(min(1.0, len(validation.drug_profiles) / len(drugs)))
        
        # Score from correlations
        if validation.correlations:
            avg_corr = np.mean([abs(c.correlation) for c in validation.correlations])
            scores.append(avg_corr * 2)  # Scale up
        
        # Score from synergy prediction
        if validation.synergy_prediction:
            scores.append(validation.synergy_prediction.confidence)
        
        validation.validation_score = np.mean(scores) if scores else 0.0
        
        # Confidence level
        if validation.validation_score >= 0.6:
            validation.confidence = "HIGH"
        elif validation.validation_score >= 0.3:
            validation.confidence = "MEDIUM"
        else:
            validation.confidence = "LOW"
        
        return validation


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_sensitivity_report(validation: DrugSensitivityValidation) -> str:
    """Generate detailed drug sensitivity validation report"""
    
    report = f"""
{'='*80}
DRUG SENSITIVITY VALIDATION REPORT
{'='*80}
Targets: {' + '.join(validation.targets)}
Drugs: {' + '.join(validation.drugs)}
Cancer Type: {validation.cancer_type}
Validation Score: {validation.validation_score:.3f}
Confidence: {validation.confidence}
{'='*80}

DRUG PROFILES AVAILABLE:
{'-'*80}
"""
    
    for profile in validation.drug_profiles:
        report += f"""
{profile.drug_name} ({profile.source})
  Targets: {', '.join(profile.target_genes)}
  Cell lines tested: {len(profile.cell_lines)}
  IC50 range: {min(profile.ic50_values):.2f} to {max(profile.ic50_values):.2f}
"""
    
    report += f"""
{'='*80}
GENE-DRUG CORRELATIONS:
{'-'*80}
"""
    
    for corr in validation.correlations:
        report += f"""
{corr.gene} <-> {corr.drug}
  Correlation: {corr.correlation:.3f} (p={corr.p_value:.4f})
  Samples: {corr.n_samples}
  Interpretation: {corr.interpretation}
"""
    
    if validation.synergy_prediction:
        sp = validation.synergy_prediction
        report += f"""
{'='*80}
COMBINATION SYNERGY PREDICTION:
{'-'*80}
Drugs: {sp.drug_a} + {sp.drug_b}{f' + {sp.drug_c}' if sp.drug_c else ''}
Bliss Independence: {sp.bliss_independence:.3f}
Predicted Synergy: {sp.predicted_synergy:.3f}
Confidence: {sp.confidence:.3f}
Evidence: {sp.evidence}
"""
    
    report += f"""
{'='*80}
RESPONSE BIOMARKERS:
{'-'*80}
"""
    
    for drug, biomarkers in validation.response_biomarkers.items():
        report += f"{drug}: {', '.join(biomarkers)}\n"
    
    return report


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Drug sensitivity validation")
    parser.add_argument('--targets', nargs='+', default=['KRAS', 'CDK6', 'STAT3'])
    parser.add_argument('--drugs', nargs='+', default=['sotorasib', 'palbociclib', 'napabucasin'])
    parser.add_argument('--cancer', type=str, default='Pancreatic Adenocarcinoma')
    parser.add_argument('--data-dir', type=str, default='./drug_sensitivity_data')
    parser.add_argument('--download-prism', action='store_true', 
                        help='Download PRISM primary screen data from Figshare (~40 MB)')
    parser.add_argument('--list-prism', action='store_true',
                        help='List drugs available in PRISM')
    
    args = parser.parse_args()
    
    if args.download_prism:
        print("\nDownloading PRISM data from Figshare...")
        prism = PRISMLoader(args.data_dir)
        if prism.download_data():
            print("✓ PRISM primary screen downloaded successfully")
            print("  Files: primary-screen-replicate-collapsed-*.csv")
        else:
            print("Download failed. Manual: https://depmap.org/repurposing/")
        exit(0)
    
    if args.list_prism:
        print("\nLoading PRISM drug list...")
        prism = PRISMLoader(args.data_dir)
        if prism.download_data():
            drugs = prism.list_available_drugs(limit=100)
            print(f"Found {len(drugs)} drugs. Sample: {drugs[:20]}")
        else:
            print("PRISM data not found. Run with --download-prism first.")
        exit(0)
    
    print(f"\nDrug Sensitivity Validation")
    print(f"Targets: {args.targets}")
    print(f"Drugs: {args.drugs}")
    print(f"Cancer: {args.cancer}\n")
    
    validator = DrugSensitivityValidator(data_dir=args.data_dir)
    result = validator.validate_combination(args.targets, args.drugs, args.cancer)
    
    report = generate_sensitivity_report(result)
    print(report)
