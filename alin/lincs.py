#!/usr/bin/env python3
"""
LINCS L1000 Integration Module
===============================
Data-driven perturbation signatures from the LINCS L1000 Phase II dataset
(Subramanian et al. 2017, Cell).  Replaces the 13 hand-curated signatures
in ``alin.perturbation`` with genome-scale measured profiles covering:

- **CRISPR knockout/overexpression** (``trt_xpr``, ~142 k signatures)
- **shRNA knockdown** (``trt_sh``, ~238 k signatures)
- **Compound treatment** (``trt_cp``, ~720 k signatures)

The module reads GCTX Level 5 (z-score) files and their metadata, builds
an in-memory or on-disk index keyed by gene symbol, and exposes an API
that is drop-in compatible with ``alin.perturbation``.

Dependencies
------------
- ``h5py >=3.0`` — HDF5 reader for GCTX format
- ``cmapPy >=4.0`` (optional) — official CMap parser, used when available;
  falls back to direct h5py reads otherwise.

Data files (expected in ``lincs_data/`` by default)
---------------------------------------------------
::

    lincs_data/
    ├── geneinfo_beta.txt            # Gene metadata (landmark + inferred)
    ├── cellinfo_beta.txt            # Cell-line metadata
    ├── siginfo_beta.txt             # Signature metadata
    ├── compoundinfo_beta.txt        # Compound metadata
    ├── level5_beta_trt_xpr_*.gctx  # CRISPR signatures
    ├── level5_beta_trt_sh_*.gctx   # shRNA signatures  (optional)
    └── level5_beta_trt_cp_*.gctx   # Compound signatures (optional)

Usage
-----
::

    from alin.lincs import LINCSSignatureDB

    db = LINCSSignatureDB("lincs_data")
    sig = db.get_perturbation_signature("EGFR")  # PerturbationSignature
    responders = db.get_perturbation_responders("KRAS")
"""

from __future__ import annotations

import glob
import logging
import os
import pickle
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    FrozenSet,
    List,
    Optional,
    Set,
    Tuple,
)

import numpy as np
import pandas as pd

from alin.perturbation import PerturbationSignature

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional heavy imports — fail gracefully when not installed
# ---------------------------------------------------------------------------

_HAS_H5PY = False
_HAS_CMAPPY = False

try:
    import h5py

    _HAS_H5PY = True
except ImportError:
    pass

try:
    from cmapPy.pandasGEXpress import parse as cmap_parse

    _HAS_CMAPPY = True
except ImportError:
    pass


# ============================================================================
# Constants
# ============================================================================

# Default z-score thresholds for calling a gene as differentially expressed.
# Level 5 LINCS data are moderated z-scores; |z| >= 2 is conventional.
DEFAULT_UP_THRESHOLD: float = 2.0
DEFAULT_DOWN_THRESHOLD: float = -2.0

# Minimum number of replicate signatures required to trust a consensus.
MIN_REPLICATE_SIGNATURES: int = 2

# Maximum number of signatures to aggregate per gene (to cap memory).
MAX_SIGNATURES_PER_GENE: int = 500

# Perturbation types mapped to LINCS pert_type codes
PERT_TYPE_MAP = {
    "trt_xpr": "knockout",  # CRISPR knockout / overexpression
    "trt_sh": "knockdown",  # shRNA knockdown
    "trt_cp": "compound",   # compound treatment
}

# Index filename
INDEX_FILENAME = "lincs_index.pkl"

# ------------ Confidence computation parameters -------------------------
#: Number of replicates at which replicate bonus saturates.
CONFIDENCE_REP_SATURATION: float = 20.0
#: Number of distinct cell lines at which cell-line bonus saturates.
CONFIDENCE_CL_SATURATION: float = 5.0
#: Perturbation-type weights for confidence scoring.
CONFIDENCE_TYPE_WEIGHTS = {"knockout": 1.0, "knockdown": 0.85, "compound": 0.7}
#: Default type weight when perturbation type is not in the map.
CONFIDENCE_TYPE_DEFAULT: float = 0.6
#: Weights for combining replicate, cell-line, and type scores.
CONFIDENCE_COMPONENT_WEIGHTS: Tuple[float, float, float] = (0.4, 0.3, 0.3)
#: Global hard cap on confidence scores.
CONFIDENCE_CAP: float = 0.98
#: Max concordance boost added to confidence for multi-modal targets.
CONCORDANCE_BOOST_FACTOR: float = 0.15

# ------------ Perturbation composite score weights ----------------------
#: Weights for ``score_combination_by_perturbation`` composite (sum = 1.0).
PERT_W_EFFECTOR: float = 0.35
PERT_W_FEEDBACK: float = 0.25
PERT_W_MULTIMODAL: float = 0.20
PERT_W_CONCORDANCE: float = 0.20
#: Discount applied to single-modality genes in weighted coverage.
SINGLE_MODALITY_DISCOUNT: float = 0.5

# ------------ GCTX loading parameters -----------------------------------
#: Number of signatures loaded per chunk to control memory usage.
GCTX_CHUNK_SIZE: int = 500


# ============================================================================
# LINCS cell line → cancer lineage mapping
# ============================================================================
# Core LINCS L1000 cell lines mapped to OncotreeLineage values.  Used to
# compute cancer-specific relevance of LINCS signatures.  When a target's
# consensus signature was measured primarily in cell lines from a different
# lineage, the evidence is discounted for the current cancer type.
#
# Lineage values match DepMap ``OncotreeLineage`` (e.g. "Lung", "Breast").
# Cell line names use LINCS ``cell_iname`` format (uppercase, no dashes).

LINCS_CELL_LINEAGE: Dict[str, str] = {
    # Lung
    "A549": "Lung", "HCC515": "Lung", "NCI-H596": "Lung",
    "NCI-H1299": "Lung", "NCI-H460": "Lung", "NCI-H1975": "Lung",
    "NCI-H2228": "Lung", "NCI-H1650": "Lung", "NCI-H522": "Lung",
    "NCI-H23": "Lung", "NCI-H358": "Lung", "PC9": "Lung",
    "NCIH716": "Lung", "EKVX": "Lung",
    # Breast
    "MCF7": "Breast", "BT20": "Breast", "MDAMB231": "Breast",
    "T47D": "Breast", "HS578T": "Breast", "SKBR3": "Breast",
    "MCF10A": "Breast", "HCC1954": "Breast", "ZR751": "Breast",
    "CAL51": "Breast", "BT474": "Breast", "AU565": "Breast",
    "MDAMB468": "Breast", "HCC1143": "Breast",
    # Skin (Melanoma)
    "A375": "Skin", "SKMEL5": "Skin", "A2058": "Skin",
    "WM115": "Skin", "COLO679": "Skin",
    # Bowel (Colorectal)
    "HT29": "Bowel", "SW620": "Bowel", "LOVO": "Bowel",
    "HCT116": "Bowel", "SW480": "Bowel", "COLO205": "Bowel",
    "SW948": "Bowel", "RKO": "Bowel", "DLD1": "Bowel",
    # Kidney
    "HA1E": "Kidney", "CAKI1": "Kidney", "786O": "Kidney",
    "A498": "Kidney", "ACHN": "Kidney", "OSRC2": "Kidney",
    # Prostate
    "PC3": "Prostate", "VCAP": "Prostate", "LNCAP": "Prostate",
    "DU145": "Prostate", "22RV1": "Prostate",
    # CNS/Brain
    "NPC": "CNS/Brain", "U251MG": "CNS/Brain", "SF268": "CNS/Brain",
    "SNB75": "CNS/Brain", "U87MG": "CNS/Brain", "T98G": "CNS/Brain",
    "LN229": "CNS/Brain",
    # Ovary
    "SKOV3": "Ovary/Fallopian Tube", "ES2": "Ovary/Fallopian Tube",
    "OVCAR8": "Ovary/Fallopian Tube", "IGROV1": "Ovary/Fallopian Tube",
    "CAOV3": "Ovary/Fallopian Tube",
    # Liver
    "HEPG2": "Liver", "HUH7": "Liver", "SKHEP1": "Liver",
    # Blood / Lymphoid
    "JURKAT": "Lymphoid", "K562": "Myeloid", "THP1": "Myeloid",
    "U937": "Myeloid", "HL60": "Myeloid", "MOLT4": "Lymphoid",
    "NALM6": "Lymphoid", "KASUMI1": "Myeloid", "NB4": "Myeloid",
    "OCI-LY3": "Lymphoid", "OCILY19": "Lymphoid", "RAJI": "Lymphoid",
    "SUDHL4": "Lymphoid",
    # Pancreas
    "PANC1": "Pancreas", "YAPC": "Pancreas", "ASPC1": "Pancreas",
    "BXPC3": "Pancreas",
    # Bone
    "U2OS": "Bone", "SAOS2": "Bone",
    # Uterus
    "HELA": "Cervix", "ISHIKAWA": "Uterus",
    # Esophagus/Stomach
    "KYSE30": "Esophagus/Stomach", "AGS": "Esophagus/Stomach",
    "OE19": "Esophagus/Stomach",
    # Bladder
    "T24": "Bladder/Urinary Tract", "RT4": "Bladder/Urinary Tract",
    # Head & Neck
    "CAL27": "Head and Neck", "SCC25": "Head and Neck",
    # Soft Tissue
    "HT1080": "Soft Tissue",
    # Thyroid
    "TT": "Thyroid", "CAL62": "Thyroid",
    # Non-cancer (fibroblast etc.) — mapped to special value
    # Note: NPC intentionally mapped to CNS/Brain above (neural progenitor cell),
    # not re-mapped here. NEU/ASC/PHH etc. are non-cancer controls.
    "NEU": "Normal", "ASC": "Normal", "PHH": "Normal",
    "SKL": "Normal", "HEK293T": "Normal",
    "HEK293": "Normal", "RWPE1": "Normal",
    "IMR90": "Normal", "WI38": "Normal",
}

# OncotreePrimaryDisease → OncotreeLineage mapping (partial, for cancer types
# that the pipeline analyses).  Used to look up the lineage for a given cancer
# type when computing LINCS cancer-relevance weights.
CANCER_TYPE_TO_LINEAGE: Dict[str, str] = {
    "Non-Small Cell Lung Cancer": "Lung",
    "Small Cell Lung Cancer": "Lung",
    "Lung Neuroendocrine Tumor": "Lung",
    "Breast Cancer": "Breast",
    "Breast Invasive Ductal Carcinoma": "Breast",
    "Invasive Breast Carcinoma": "Breast",
    "Colorectal Adenocarcinoma": "Bowel",
    "Colon Adenocarcinoma": "Bowel",
    "Colon/Rectal Cancer": "Bowel",
    "Colorectal Cancer": "Bowel",
    "Melanoma": "Skin",
    "Cutaneous Melanoma": "Skin",
    "Renal Cell Carcinoma": "Kidney",
    "Kidney Cancer": "Kidney",
    "Prostate Cancer": "Prostate",
    "Prostate Adenocarcinoma": "Prostate",
    "Glioblastoma": "CNS/Brain",
    "Glioma": "CNS/Brain",
    "Low-Grade Glioma": "CNS/Brain",
    "Diffuse Glioma": "CNS/Brain",
    "Neuroblastoma": "CNS/Brain",
    "Ovarian Cancer": "Ovary/Fallopian Tube",
    "High-Grade Serous Ovarian Cancer": "Ovary/Fallopian Tube",
    "Ovarian Epithelial Tumor": "Ovary/Fallopian Tube",
    "Hepatocellular Carcinoma": "Liver",
    "Liver Cancer": "Liver",
    "B-Lymphoblastic Leukemia/Lymphoma": "Lymphoid",
    "T-Lymphoblastic Leukemia/Lymphoma": "Lymphoid",
    "Acute Myeloid Leukemia": "Myeloid",
    "Chronic Myelogenous Leukemia": "Myeloid",
    "Diffuse Large B-Cell Lymphoma": "Lymphoid",
    "Non-Hodgkin Lymphoma": "Lymphoid",
    "Multiple Myeloma": "Lymphoid",
    "Leukemia": "Myeloid",
    "Pancreatic Adenocarcinoma": "Pancreas",
    "Pancreatic Cancer": "Pancreas",
    "Osteosarcoma": "Bone",
    "Ewing Sarcoma": "Bone",
    "Bone Cancer": "Bone",
    "Cervical Cancer": "Cervix",
    "Endometrial Cancer": "Uterus",
    "Uterine Carcinosarcoma": "Uterus",
    "Esophageal Cancer": "Esophagus/Stomach",
    "Esophagogastric Adenocarcinoma": "Esophagus/Stomach",
    "Gastric Cancer": "Esophagus/Stomach",
    "Stomach Adenocarcinoma": "Esophagus/Stomach",
    "Bladder Cancer": "Bladder/Urinary Tract",
    "Bladder Urothelial Carcinoma": "Bladder/Urinary Tract",
    "Head and Neck Squamous Cell Carcinoma": "Head and Neck",
    "Rhabdomyosarcoma": "Soft Tissue",
    "Soft Tissue Sarcoma": "Soft Tissue",
    "Thyroid Cancer": "Thyroid",
    "Anaplastic Thyroid Cancer": "Thyroid",
    "Non-Cancerous": "Normal",
}


# ============================================================================
# Gene info loader
# ============================================================================


def load_gene_info(lincs_dir: str) -> pd.DataFrame:
    """Load ``geneinfo_beta.txt`` and return a DataFrame indexed by gene_id."""
    path = os.path.join(lincs_dir, "geneinfo_beta.txt")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Gene info not found: {path}")
    df = pd.read_csv(path, sep="\t", dtype={"gene_id": int})
    df.set_index("gene_id", inplace=True)
    return df


def load_sig_info(lincs_dir: str) -> pd.DataFrame:
    """Load ``siginfo_beta.txt`` and return a DataFrame indexed by sig_id.

    Uses pyarrow engine to avoid Python 3.14 C-parser OOM issues with
    large TSV files.  Falls back to chunked C-parser if pyarrow unavailable.
    """
    path = os.path.join(lincs_dir, "siginfo_beta.txt")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Sig info not found: {path}")

    # Only load columns we actually need (saves memory with 1.2M rows)
    usecols = [
        "sig_id", "pert_type", "pert_iname", "cmap_name", "cell_iname",
        "is_exemplar_sig",
    ]

    try:
        df = pd.read_csv(
            path, sep="\t", engine="pyarrow",
            usecols=lambda c: c in usecols,
        )
    except Exception:
        # Fallback: chunked C parser
        chunks = []
        for chunk in pd.read_csv(
            path, sep="\t", low_memory=True, chunksize=200_000,
            usecols=lambda c: c in usecols,
        ):
            chunks.append(chunk)
        df = pd.concat(chunks, ignore_index=True)

    if "sig_id" in df.columns:
        df.set_index("sig_id", inplace=True)
    return df


def load_cell_info(lincs_dir: str) -> pd.DataFrame:
    """Load ``cellinfo_beta.txt``.

    Currently unused by the pipeline (cell-line lineage is provided via
    the ``LINCS_CELL_LINEAGE`` dictionary), but retained for interactive
    exploration and future lineage-enrichment analyses.
    """
    path = os.path.join(lincs_dir, "cellinfo_beta.txt")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Cell info not found: {path}")
    return pd.read_csv(path, sep="\t")


def load_compound_info(lincs_dir: str) -> pd.DataFrame:
    """Load ``compoundinfo_beta.txt`` with compound→target gene mapping.

    The ``target`` column contains pipe-delimited gene symbols
    (e.g. ``"AKT1|AKT2|MTOR"``).  Only rows with non-empty targets
    are useful for mapping compounds back to gene-level signatures.
    """
    path = os.path.join(lincs_dir, "compoundinfo_beta.txt")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Compound info not found: {path}")
    df = pd.read_csv(path, sep="\t", low_memory=False)
    return df


def build_compound_to_gene_map(lincs_dir: str) -> Dict[str, List[str]]:
    """Build mapping: compound_name → list of target gene symbols.

    Reads compoundinfo_beta.txt and returns a dict keyed by cmap_name
    (matching the "cmap_name" column in siginfo) with values being
    lists of HGNC gene symbols parsed from the pipe-delimited ``target``
    column.

    Returns
    -------
    dict
        ``{"vorinostat": ["HDAC1", "HDAC2", ...], ...}``
    """
    try:
        df = load_compound_info(lincs_dir)
    except FileNotFoundError:
        logger.warning("compoundinfo_beta.txt not found — no compound→gene mapping")
        return {}

    # Identify the name and target columns
    name_col = "cmap_name" if "cmap_name" in df.columns else "pert_iname"
    target_col = "target" if "target" in df.columns else None
    if target_col is None:
        logger.warning("No 'target' column in compoundinfo; cannot map compounds→genes")
        return {}

    mapping: Dict[str, List[str]] = {}
    moa_mapping: Dict[str, str] = {}
    moa_col = "moa" if "moa" in df.columns else None

    for _, row in df.iterrows():
        name = str(row.get(name_col, "")).strip()
        targets_raw = str(row.get(target_col, "")).strip()
        if not name or name == "-666" or not targets_raw or targets_raw in ("nan", "-666", ""):
            continue
        # Parse pipe-delimited target genes → upper-case HGNC symbols
        genes = [g.strip().upper() for g in targets_raw.split("|") if g.strip()]
        if genes:
            mapping[name] = genes
        if moa_col and pd.notna(row.get(moa_col)):
            moa_mapping[name] = str(row[moa_col])

    logger.info(
        "Compound->gene mapping: %d compounds -> %d unique gene targets",
        len(mapping),
        len(set(g for gs in mapping.values() for g in gs)),
    )
    return mapping


# ============================================================================
# GCTX reader (h5py fallback when cmapPy unavailable)
# ============================================================================


def _read_gctx_h5py(
    gctx_path: str,
    rid: Optional[List[int]] = None,
    cid: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[int], List[str]]:
    """
    Read a GCTX file using h5py directly.

    Parameters
    ----------
    gctx_path : str
        Path to ``.gctx`` file.
    rid : list of int, optional
        Row indices (gene_ids) to read.  None = all rows.
    cid : list of str, optional
        Column ids (sig_ids) to read.  None = all columns.

    Returns
    -------
    data : np.ndarray, shape (n_genes, n_sigs)
    row_ids : list of int
    col_ids : list of str
    """
    if not _HAS_H5PY:
        raise ImportError("h5py is required to read GCTX files")

    with h5py.File(gctx_path, "r") as f:
        # Row / col metadata
        all_rids = f["0/META/ROW/id"][:]
        all_cids = f["0/META/COL/id"][:]

        # Decode bytes → str/int
        if all_rids.dtype.kind == "S":
            all_rids = np.array([int(x) for x in all_rids.astype(str)])
        if all_cids.dtype.kind == "S":
            all_cids = [x.decode() if isinstance(x, bytes) else str(x) for x in all_cids]
        else:
            all_cids = [str(x) for x in all_cids]

        mat = f["0/DATA/0/matrix"]

        if rid is None and cid is None:
            data = mat[:]
            return data, all_rids.tolist(), all_cids
        else:
            # Build index masks
            if rid is not None:
                rid_set = set(rid)
                row_mask = np.array([x in rid_set for x in all_rids])
            else:
                row_mask = np.ones(len(all_rids), dtype=bool)

            if cid is not None:
                cid_set = set(cid)
                col_mask = np.array([x in cid_set for x in all_cids])
            else:
                col_mask = np.ones(len(all_cids), dtype=bool)

            # Read subset using fancy indexing — avoid loading entire matrix.
            # GCTX stores (cols, rows) transposed: mat shape = (n_sigs, n_genes)
            col_idx = np.where(col_mask)[0]
            row_idx = np.where(row_mask)[0]

            is_transposed = mat.shape[0] == len(all_cids)

            if is_transposed:
                # mat shape: (n_sigs, n_genes) → select sigs first, genes second
                col_idx_sorted = np.sort(col_idx)
                # Single-axis h5py read, then numpy column subset
                data = mat[col_idx_sorted]       # (n_sigs_kept, all_genes)
                data = data[:, row_idx]           # (n_sigs_kept, n_genes_kept)
                data = data.T                     # → (n_genes, n_sigs)
            else:
                # mat shape: (n_genes, n_sigs)
                row_idx_sorted = np.sort(row_idx)
                data = mat[row_idx_sorted]        # (n_genes_kept, all_sigs)
                data = data[:, col_idx]           # (n_genes_kept, n_sigs_kept)

            out_rids = all_rids[row_mask].tolist()
            out_cids = [c for c, m in zip(all_cids, col_mask) if m]
            return data, out_rids, out_cids


def read_gctx(
    gctx_path: str,
    rid: Optional[List[int]] = None,
    cid: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Read a GCTX file and return a DataFrame (genes × signatures).

    Uses cmapPy when available, otherwise falls back to raw h5py.
    """
    if _HAS_CMAPPY:
        gctoo = cmap_parse.parse(gctx_path, rid=rid, cid=cid)
        return gctoo.data_df
    else:
        data, row_ids, col_ids = _read_gctx_h5py(gctx_path, rid=rid, cid=cid)
        return pd.DataFrame(data, index=row_ids, columns=col_ids)


# ============================================================================
# Consensus signature computation
# ============================================================================


@dataclass
class ModalitySignature:
    """Per-modality signature data for one perturbagen."""
    pert_type: str          # knockout / knockdown / compound
    n_signatures: int
    mean_z: Dict[str, float]
    up_genes: FrozenSet[str] = field(default_factory=frozenset)
    down_genes: FrozenSet[str] = field(default_factory=frozenset)
    cell_lines: FrozenSet[str] = field(default_factory=frozenset)
    confidence: float = 0.0
    compound_names: FrozenSet[str] = field(default_factory=frozenset)
    moa: Optional[str] = None


@dataclass
class ConsensusSignature:
    """Aggregated signature for one perturbagen across replicates/cell lines.

    Now supports multi-modal data: separate per-modality signatures
    (knockout, knockdown, compound) plus cross-modal concordance.
    """

    target_gene: str
    pert_type: str  # primary / highest-confidence modality
    n_signatures: int
    mean_z: Dict[str, float]  # gene_symbol → mean z-score across replicates
    up_genes: FrozenSet[str] = field(default_factory=frozenset)
    down_genes: FrozenSet[str] = field(default_factory=frozenset)
    cell_lines: FrozenSet[str] = field(default_factory=frozenset)
    confidence: float = 0.0

    # ---- Multi-modal fields (added for trt_xpr + trt_sh + trt_cp) ----
    modalities: Dict[str, 'ModalitySignature'] = field(default_factory=dict)
    cross_modal_concordance: float = 0.0  # 0-1; how well modalities agree
    compound_names: FrozenSet[str] = field(default_factory=frozenset)
    moa: Optional[str] = None

    # ---- Concordance-weighted gene classifications ----
    # Genes confirmed in ≥2 modalities (high confidence, not off-target)
    concordant_up_genes: FrozenSet[str] = field(default_factory=frozenset)
    concordant_down_genes: FrozenSet[str] = field(default_factory=frozenset)
    # Genes appearing in only one modality (may be off-target or context-specific)
    single_modality_up_genes: FrozenSet[str] = field(default_factory=frozenset)
    single_modality_down_genes: FrozenSet[str] = field(default_factory=frozenset)

    # Backward-compat: v1 pickles lack new fields; supply defaults
    _V2_DEFAULTS: Dict[str, Any] = field(
        default=None, init=False, repr=False, compare=False
    )

    def __getattr__(self, name: str):
        """Return sensible defaults for fields absent in v1 pickled objects."""
        _defaults = {
            'modalities': {},
            'cross_modal_concordance': 0.0,
            'compound_names': frozenset(),
            'moa': None,
            'concordant_up_genes': frozenset(),
            'concordant_down_genes': frozenset(),
            'single_modality_up_genes': frozenset(),
            'single_modality_down_genes': frozenset(),
        }
        if name in _defaults:
            return _defaults[name]
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    @property
    def n_modalities(self) -> int:
        mods = self.__dict__.get('modalities', {})
        return len(mods) if mods else 1

    @property
    def has_genetic(self) -> bool:
        """Has knockout or knockdown evidence."""
        mods = self.__dict__.get('modalities', {})
        if not mods:
            return self.pert_type in ('knockout', 'knockdown')
        return any(m in mods for m in ('knockout', 'knockdown'))

    @property
    def has_compound(self) -> bool:
        """Has compound treatment evidence."""
        mods = self.__dict__.get('modalities', {})
        if not mods:
            return self.pert_type == 'compound'
        return 'compound' in mods

    @property
    def modality_types(self) -> List[str]:
        mods = self.__dict__.get('modalities', {})
        if mods:
            return sorted(mods.keys())
        return [self.pert_type]

    def to_perturbation_signature(self) -> PerturbationSignature:
        """Convert to the legacy ``PerturbationSignature`` format.

        When multi-modal concordance data is available, the signature
        is split: **concordant genes** (confirmed in >=2 modalities) go
        into expression_decreased/increased with full confidence;
        **single-modality genes** are included but with a discounted
        confidence to reflect uncertainty about off-target artefacts.

        The ``confidence`` field is boosted for multi-modal targets with
        high concordance, reflecting greater mechanistic certainty.
        """
        # Compute effective confidence: boost for concordant multi-modal
        eff_confidence = self.confidence
        if self.n_modalities >= 2 and self.cross_modal_concordance > 0:
            # Up to +15% confidence boost for high concordance
            concordance_boost = self.cross_modal_concordance * CONCORDANCE_BOOST_FACTOR
            eff_confidence = min(CONFIDENCE_CAP, self.confidence + concordance_boost)

        return PerturbationSignature(
            target=self.target_gene,
            perturbation_type=self.pert_type,
            phospho_decreased=set(),  # L1000 measures mRNA, not phospho
            phospho_increased=set(),
            expression_decreased=set(self.down_genes),
            expression_increased=set(self.up_genes),
            confidence=round(eff_confidence, 3),
            source=f"LINCS_L1000_{self.pert_type}"
                   f"{'_multimodal' if self.n_modalities >= 2 else ''}",
            pmid="28678552",  # Subramanian et al. 2017
        )

    def get_concordant_genes(self) -> Dict[str, Set[str]]:
        """Return concordance-stratified gene sets.

        Returns a dict with keys:
        - ``concordant_up``: Genes up-regulated in >=2 modalities (high confidence)
        - ``concordant_down``: Genes down-regulated in >=2 modalities
        - ``single_up``: Genes up only in 1 modality (may be off-target)
        - ``single_down``: Genes down only in 1 modality
        - ``all_up``: Full union (backward compatible)
        - ``all_down``: Full union (backward compatible)
        """
        return {
            'concordant_up': set(self.concordant_up_genes),
            'concordant_down': set(self.concordant_down_genes),
            'single_up': set(self.single_modality_up_genes),
            'single_down': set(self.single_modality_down_genes),
            'all_up': set(self.up_genes),
            'all_down': set(self.down_genes),
        }


def compute_consensus(
    z_matrix: pd.DataFrame,
    gene_id_to_symbol: Dict[int, str],
    up_threshold: float = DEFAULT_UP_THRESHOLD,
    down_threshold: float = DEFAULT_DOWN_THRESHOLD,
    min_freq: float = 0.5,
) -> Tuple[Dict[str, float], FrozenSet[str], FrozenSet[str]]:
    """
    Compute consensus up/down gene sets from a z-score matrix.

    Parameters
    ----------
    z_matrix : DataFrame
        Genes (rows, indexed by gene_id) × signatures (columns).
    gene_id_to_symbol : dict
        Mapping of gene_id → gene_symbol.
    up_threshold, down_threshold : float
        Z-score thresholds.
    min_freq : float
        Minimum fraction of signatures in which a gene must be up/down
        to be included in consensus.

    Returns
    -------
    mean_z : dict
        gene_symbol → mean z-score
    up_genes : frozenset
    down_genes : frozenset
    """
    n_sigs = z_matrix.shape[1]
    if n_sigs == 0:
        return {}, frozenset(), frozenset()

    mean_z_arr = z_matrix.mean(axis=1)
    up_freq = (z_matrix >= up_threshold).sum(axis=1) / n_sigs
    down_freq = (z_matrix <= down_threshold).sum(axis=1) / n_sigs

    mean_z: Dict[str, float] = {}
    up_genes: Set[str] = set()
    down_genes: Set[str] = set()

    for gid in z_matrix.index:
        symbol = gene_id_to_symbol.get(int(gid))
        if symbol is None:
            continue
        z = float(mean_z_arr.loc[gid])
        mean_z[symbol] = round(z, 4)
        if up_freq.loc[gid] >= min_freq and z >= up_threshold * 0.5:
            up_genes.add(symbol)
        if down_freq.loc[gid] >= min_freq and z <= down_threshold * 0.5:
            down_genes.add(symbol)

    return mean_z, frozenset(up_genes), frozenset(down_genes)


def _compute_confidence(
    n_sigs: int,
    n_cell_lines: int,
    pert_type: str,
) -> float:
    """
    Heuristic confidence score [0, 1] for a consensus signature.

    Rewards: more replicates, more cell lines, CRISPR > shRNA > compound.
    """
    # Replicate bonus: saturates around 20 replicates
    rep_score = min(n_sigs / CONFIDENCE_REP_SATURATION, 1.0)

    # Cell-line diversity: saturates around 5 cell lines
    cl_score = min(n_cell_lines / CONFIDENCE_CL_SATURATION, 1.0)

    # Perturbation-type weight
    type_weight = CONFIDENCE_TYPE_WEIGHTS.get(
        pert_type, CONFIDENCE_TYPE_DEFAULT
    )

    # Combined: geometric-ish mean
    w_rep, w_cl, w_type = CONFIDENCE_COMPONENT_WEIGHTS
    raw = (w_rep * rep_score + w_cl * cl_score + w_type * type_weight)
    return round(min(raw, CONFIDENCE_CAP), 3)


# ============================================================================
# LINCSSignatureDB — main class
# ============================================================================


class LINCSSignatureDB:
    """
    In-memory database of LINCS L1000 perturbation signatures.

    Lazily loads GCTX files, builds consensus signatures per pert gene,
    and caches the result for fast lookup.

    Parameters
    ----------
    lincs_dir : str
        Directory containing GCTX + metadata files.
    pert_types : list of str, optional
        Which perturbation types to load.  Default: ``["trt_xpr"]``.
    up_threshold : float
        Z-score threshold for calling a gene upregulated.
    down_threshold : float
        Z-score threshold for calling a gene downregulated.
    landmark_only : bool
        If True, restrict to the 978 landmark genes (faster, smaller).
    """

    def __init__(
        self,
        lincs_dir: str,
        pert_types: Optional[List[str]] = None,
        up_threshold: float = DEFAULT_UP_THRESHOLD,
        down_threshold: float = DEFAULT_DOWN_THRESHOLD,
        landmark_only: bool = False,
    ):
        self.lincs_dir = lincs_dir
        self.pert_types = pert_types or ["trt_xpr"]
        self.up_threshold = up_threshold
        self.down_threshold = down_threshold
        self.landmark_only = landmark_only

        # State
        self._gene_info: Optional[pd.DataFrame] = None
        self._sig_info: Optional[pd.DataFrame] = None
        self._gene_id_to_symbol: Dict[int, str] = {}
        self._symbol_to_gene_id: Dict[str, int] = {}
        self._consensus: Dict[str, ConsensusSignature] = {}
        self._loaded = False

        # Compound→gene mapping (populated on demand for trt_cp)
        self._compound_to_genes: Optional[Dict[str, List[str]]] = None
        self._compound_moa: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Metadata loaders
    # ------------------------------------------------------------------

    def _ensure_gene_info(self) -> None:
        if self._gene_info is not None:
            return
        self._gene_info = load_gene_info(self.lincs_dir)
        # Build mappings
        for gid, row in self._gene_info.iterrows():
            symbol = row.get("gene_symbol", "")
            if pd.notna(symbol) and symbol != "" and symbol != "-666":
                self._gene_id_to_symbol[int(gid)] = str(symbol)
                self._symbol_to_gene_id[str(symbol)] = int(gid)
        logger.info(
            "Loaded gene info: %d genes (%d landmark)",
            len(self._gene_id_to_symbol),
            (self._gene_info.get("feature_space") == "landmark").sum()
            if "feature_space" in self._gene_info.columns
            else "?",
        )

    def _ensure_sig_info(self) -> None:
        if self._sig_info is not None:
            return
        self._sig_info = load_sig_info(self.lincs_dir)
        logger.info("Loaded sig info: %d signatures", len(self._sig_info))

    def _ensure_compound_info(self) -> None:
        """Load compound→gene target mapping from compoundinfo_beta.txt."""
        if self._compound_to_genes is not None:
            return
        self._compound_to_genes = build_compound_to_gene_map(self.lincs_dir)

        # Also load MOA info
        try:
            df = load_compound_info(self.lincs_dir)
            name_col = "cmap_name" if "cmap_name" in df.columns else "pert_iname"
            moa_col = "moa" if "moa" in df.columns else None
            if moa_col:
                for _, row in df.iterrows():
                    name = str(row.get(name_col, "")).strip()
                    moa = row.get(moa_col)
                    if name and pd.notna(moa) and str(moa).strip():
                        self._compound_moa[name] = str(moa).strip()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # GCTX discovery
    # ------------------------------------------------------------------

    def _find_gctx(self, pert_type: str) -> Optional[str]:
        """Find the GCTX file for a given perturbation type."""
        pattern = os.path.join(self.lincs_dir, f"level5_beta_{pert_type}_*.gctx")
        matches = glob.glob(pattern)
        if not matches:
            return None
        # Take the largest (most complete) if multiple
        return max(matches, key=os.path.getsize)

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def _index_path(self) -> str:
        return os.path.join(self.lincs_dir, INDEX_FILENAME)

    def _load_cached_index(self, allow_stale: bool = False) -> bool:
        """Try to load a pre-built index from disk.  Returns True on success.

        Parameters
        ----------
        allow_stale : bool
            If True, accept a v1 (single-modal) cache even when multiple
            pert_types are requested.  This avoids triggering a multi-GB
            rebuild in contexts that merely *query* the index (e.g.
            evidence tiering).
        """
        idx_path = self._index_path()
        if not os.path.isfile(idx_path):
            return False
        try:
            with open(idx_path, "rb") as f:
                data = pickle.load(f)
            if not isinstance(data, dict) or "consensus" not in data:
                return False
            # version 2+ has multi-modal data; version 1 is single-modal
            cached_version = data.get("version", 1)
            if cached_version < 2 and len(self.pert_types) > 1:
                if not allow_stale:
                    logger.info(
                        "Cached index is v%d (single-modal); rebuilding for multi-modal",
                        cached_version,
                    )
                    return False
                logger.info(
                    "Cached index is v%d (single-modal); using stale cache "
                    "(allow_stale=True, %d targets)",
                    cached_version,
                    len(data["consensus"]),
                )
            self._consensus = data["consensus"]
            self._loaded = True
            logger.info(
                "Loaded LINCS index from cache: %d target genes", len(self._consensus)
            )
            return True
        except Exception as exc:
            logger.warning("Failed to load LINCS index cache: %s", exc)
            return False

    def _save_index(self) -> None:
        """Persist the built index to disk for fast reload."""
        idx_path = self._index_path()
        try:
            with open(idx_path, "wb") as f:
                pickle.dump({"consensus": self._consensus, "version": 2}, f)
            logger.info("Saved LINCS index (v2) to %s", idx_path)
        except Exception as exc:
            logger.warning("Failed to save LINCS index: %s", exc)

    # ------------------------------------------------------------------
    # Core: build consensus signatures from GCTX
    # ------------------------------------------------------------------

    def build_index(
        self, force: bool = False, cache_only: bool = False
    ) -> None:
        """
        Build or load the consensus signature index.

        1. Tries to load a cached pickle index.
        2. If not found (or ``force=True``), reads the GCTX files,
           groups signatures by perturbed gene, computes consensus,
           and caches the result.

        Parameters
        ----------
        force : bool
            Rebuild even if a cached index exists.
        cache_only : bool
            If True, only load from cache (accepting stale v1 indexes).
            Never trigger a full GCTX rebuild.  Useful for callers that
            need a quick answer (evidence tiering, tests).
        """
        if self._loaded and not force:
            return

        if not force and self._load_cached_index(allow_stale=cache_only):
            return

        if cache_only:
            # Prefer returning with no data over a multi-GB rebuild
            logger.info(
                "cache_only=True but no usable cached index; "
                "LINCS DB will have 0 targets"
            )
            self._loaded = True
            return

        if not _HAS_H5PY:
            raise ImportError(
                "h5py is required to build the LINCS index.  "
                "Install it: pip install h5py>=3.0"
            )

        self._ensure_gene_info()
        self._ensure_sig_info()

        # Filter to landmark genes if requested
        if self.landmark_only and "feature_space" in self._gene_info.columns:
            landmark_ids = set(
                self._gene_info[
                    self._gene_info["feature_space"] == "landmark"
                ].index.tolist()
            )
        else:
            landmark_ids = None

        # Load compound info if we need trt_cp
        if "trt_cp" in self.pert_types:
            self._ensure_compound_info()

        for pert_type in self.pert_types:
            gctx_path = self._find_gctx(pert_type)
            if gctx_path is None:
                logger.warning("No GCTX found for pert_type=%s", pert_type)
                continue

            logger.info("Processing %s: %s", pert_type, gctx_path)
            self._process_gctx(gctx_path, pert_type, landmark_ids)

        # Compute cross-modal concordance for all targets that have modalities
        self._compute_all_cross_modal_concordances()

        self._loaded = True
        self._save_index()

        logger.info(
            "LINCS index built: %d target genes with consensus signatures",
            len(self._consensus),
        )

    @staticmethod
    def _load_subset_gctx(
        gctx_path: str,
        needed_cids: Optional[Set[str]] = None,
        needed_rids: Optional[Set[int]] = None,
    ) -> Tuple[np.ndarray, List[int], List[str]]:
        """Load a **subset** of a GCTX matrix into memory.

        Only reads the columns (sig_ids) in *needed_cids* and the rows
        (gene_ids) in *needed_rids*.  This avoids loading the full
        multi-GB matrix when only a fraction of columns are required.

        Returns (data, row_ids, col_ids) where data has shape
        (n_kept_genes, n_kept_sigs) as float32.
        """
        import h5py as _h5

        logger.info("  Opening GCTX: %s", gctx_path)
        with _h5.File(gctx_path, "r") as f:
            all_rids_raw = f["0/META/ROW/id"][:]
            all_cids_raw = f["0/META/COL/id"][:]

            # Decode metadata
            if all_rids_raw.dtype.kind == "S":
                all_rids = np.array([int(x) for x in all_rids_raw.astype(str)])
            else:
                all_rids = all_rids_raw.astype(int)
            if all_cids_raw.dtype.kind == "S":
                all_cids = [
                    x.decode() if isinstance(x, bytes) else str(x)
                    for x in all_cids_raw
                ]
            else:
                all_cids = [str(x) for x in all_cids_raw]

            # Build column mask
            if needed_cids is not None:
                col_mask = np.array([c in needed_cids for c in all_cids])
            else:
                col_mask = np.ones(len(all_cids), dtype=bool)

            # Build row mask
            if needed_rids is not None:
                row_mask = np.array([r in needed_rids for r in all_rids])
            else:
                row_mask = np.ones(len(all_rids), dtype=bool)

            col_idx = np.where(col_mask)[0]
            row_idx = np.where(row_mask)[0]

            mat = f["0/DATA/0/matrix"]
            is_transposed = mat.shape[0] == len(all_cids)

            # Determine output size
            n_out_rows = int(row_mask.sum())  # genes
            n_out_cols = int(col_mask.sum())  # sigs

            # Pre-allocate output array (genes × sigs)
            data = np.empty((n_out_rows, n_out_cols), dtype=np.float32)

            if is_transposed:
                # mat shape: (n_sigs, n_genes)
                # Read in chunks along the sig axis to cap memory.
                # IMPORTANT: h5py does NOT support chained fancy indexing
                # (mat[rows][:, cols]) on compressed datasets — it raises
                # OSError.  Instead we read full-row slices then subset
                # columns in-memory (numpy), which is reliable.
                sig_indices = np.where(col_mask)[0]
                gene_indices = np.sort(np.where(row_mask)[0])
                chunk_size = GCTX_CHUNK_SIZE     # sigs per chunk (conservative for memory)
                n_written = 0
                logger.info(
                    "  Reading %d/%d sigs x %d/%d genes in chunks of %d (transposed)",
                    len(sig_indices), len(all_cids),
                    len(gene_indices), len(all_rids),
                    chunk_size,
                )
                for start in range(0, len(sig_indices), chunk_size):
                    batch = sig_indices[start:start + chunk_size]
                    sorted_batch = np.sort(batch)
                    # Step 1: read full rows (all genes) for this batch of sigs.
                    # h5py handles single-axis fancy indexing fine.
                    chunk_full = mat[sorted_batch]          # (batch, all_genes)
                    # Step 2: subset columns in numpy (fast, in-memory)
                    chunk_data = chunk_full[:, gene_indices]  # (batch, kept_genes)
                    data[:, start:start + len(batch)] = chunk_data.T.astype(
                        np.float32, copy=False
                    )
                    n_written += len(batch)
                    if (start // chunk_size) % 20 == 0:
                        logger.info("    chunk %d/%d", n_written, len(sig_indices))
                    del chunk_full, chunk_data  # free immediately
            else:
                # mat shape: (n_genes, n_sigs)
                sorted_row_idx = np.sort(row_idx)
                logger.info(
                    "  Reading %d/%d genes x %d/%d signatures",
                    len(sorted_row_idx), len(all_rids),
                    len(col_idx), len(all_cids),
                )
                # Same pattern: single-axis fancy index, then numpy subset
                chunk = mat[sorted_row_idx]              # (n_kept_genes, all_sigs)
                chunk = chunk[:, col_idx]                # (n_kept_genes, n_kept_sigs)
                data = chunk.astype(np.float32, copy=False)

            kept_rids = all_rids[row_mask].tolist()
            kept_cids = [c for c, m in zip(all_cids, col_mask) if m]

        logger.info(
            "  Loaded subset: %d genes × %d signatures (%.1f MB)",
            data.shape[0],
            data.shape[1],
            data.nbytes / 1e6,
        )
        return data, kept_rids, kept_cids

    def _process_gctx(
        self,
        gctx_path: str,
        pert_type: str,
        landmark_ids: Optional[Set[int]] = None,
    ) -> None:
        """
        Process one GCTX file: group signatures by target gene,
        compute consensus, and store in ``self._consensus`` with
        per-modality tracking.

        For **trt_cp** (compound), signatures are re-grouped by their
        *gene targets* (via ``compoundinfo_beta.txt``) rather than by
        compound name, so they integrate with the gene-centric index.
        """
        mapped_type = PERT_TYPE_MAP.get(pert_type, pert_type)
        is_compound = pert_type == "trt_cp"

        # ── Filter signatures for this pert_type ─────────────────────
        sig_df = self._sig_info
        mask = (
            sig_df["pert_type"] == pert_type
            if "pert_type" in sig_df.columns
            else pd.Series(True, index=sig_df.index)
        )

        # Quality filter: is_exemplar_sig if available
        if "is_exemplar_sig" in sig_df.columns:
            mask = mask & (sig_df["is_exemplar_sig"] == 1)

        filtered_sigs = sig_df[mask]
        logger.info(
            "  %s: %d quality-filtered signatures (of %d total)",
            pert_type,
            len(filtered_sigs),
            len(sig_df[sig_df["pert_type"] == pert_type])
            if "pert_type" in sig_df.columns
            else len(sig_df),
        )

        # ── Determine grouping column ────────────────────────────────
        pert_col = (
            "pert_iname" if "pert_iname" in filtered_sigs.columns else "cmap_name"
        )
        if pert_col not in filtered_sigs.columns:
            logger.warning("  Cannot find perturbagen name column; skipping")
            return

        # ── For compounds: build gene-level groups via compound info ─
        if is_compound and self._compound_to_genes:
            gene_groups = self._group_compound_sigs_by_gene(
                filtered_sigs, pert_col
            )
            logger.info(
                "  trt_cp: mapped %d compound groups → %d gene-level groups",
                filtered_sigs[pert_col].nunique(),
                len(gene_groups),
            )
        else:
            # For trt_xpr / trt_sh: group directly by gene name
            gene_groups = {}
            for pert_name, group_df in filtered_sigs.groupby(pert_col):
                pn = str(pert_name).strip().upper()
                if not pn or pn == "-666":
                    continue
                gene_groups.setdefault(pn, {
                    "sig_ids": [],
                    "group_df": None,
                    "compound_names": set(),
                })
                gene_groups[pn]["sig_ids"].extend(group_df.index.tolist())
                # Merge group_df rows
                if gene_groups[pn]["group_df"] is None:
                    gene_groups[pn]["group_df"] = group_df
                else:
                    gene_groups[pn]["group_df"] = pd.concat(
                        [gene_groups[pn]["group_df"], group_df]
                    )

        # ── Collect all needed sig_ids upfront ───────────────────────
        all_needed_sigs: Set[str] = set()
        for gene_sym, info in gene_groups.items():
            sids = info["sig_ids"]
            if len(sids) < MIN_REPLICATE_SIGNATURES:
                continue
            if len(sids) > MAX_SIGNATURES_PER_GENE:
                sids = sids[:MAX_SIGNATURES_PER_GENE]
                info["sig_ids"] = sids
            all_needed_sigs.update(sids)

        if not all_needed_sigs:
            logger.info("  No valid signature groups for %s; skipping", pert_type)
            return

        logger.info("  Need %d unique sig_ids from GCTX", len(all_needed_sigs))

        # ── Load only the needed subset ──────────────────────────────
        full_data, row_ids, col_ids = self._load_subset_gctx(
            gctx_path,
            needed_cids=all_needed_sigs,
            needed_rids=landmark_ids,
        )
        cid_to_idx = {c: i for i, c in enumerate(col_ids)}

        # ── Iterate gene groups using in-memory slicing ──────────────
        n_processed = 0
        n_targets = 0
        n_groups = len(gene_groups)

        for i, (gene_symbol, info) in enumerate(gene_groups.items()):
            sig_ids = info["sig_ids"]
            if len(sig_ids) < MIN_REPLICATE_SIGNATURES:
                continue

            # Slice columns from the in-memory matrix
            col_indices = [cid_to_idx[s] for s in sig_ids if s in cid_to_idx]
            if len(col_indices) < MIN_REPLICATE_SIGNATURES:
                continue

            z_arr = full_data[:, col_indices]  # (genes, n_sigs)
            valid_sids = [s for s in sig_ids if s in cid_to_idx]
            z_df = pd.DataFrame(z_arr, index=row_ids, columns=valid_sids)

            if z_df.empty:
                continue

            mean_z, up_genes, down_genes = compute_consensus(
                z_df,
                self._gene_id_to_symbol,
                up_threshold=self.up_threshold,
                down_threshold=self.down_threshold,
            )

            if not up_genes and not down_genes:
                continue

            # Cell line diversity
            group_df = info.get("group_df")
            cell_lines: FrozenSet[str] = frozenset()
            if group_df is not None and "cell_iname" in group_df.columns:
                cell_lines = frozenset(
                    group_df["cell_iname"].dropna().unique().tolist()
                )

            confidence = _compute_confidence(
                n_sigs=len(col_indices),
                n_cell_lines=len(cell_lines),
                pert_type=mapped_type,
            )

            compound_names = frozenset(info.get("compound_names", set()))
            moa = None
            if compound_names and self._compound_moa:
                moas = {
                    self._compound_moa[c]
                    for c in compound_names
                    if c in self._compound_moa
                }
                if moas:
                    moa = "; ".join(sorted(moas))

            # Build ModalitySignature for this modality
            modality_sig = ModalitySignature(
                pert_type=mapped_type,
                n_signatures=len(col_indices),
                mean_z=mean_z,
                up_genes=up_genes,
                down_genes=down_genes,
                cell_lines=cell_lines,
                confidence=confidence,
                compound_names=compound_names,
                moa=moa,
            )

            # ── Merge into consensus ─────────────────────────────────
            if gene_symbol in self._consensus:
                existing = self._consensus[gene_symbol]

                # Add this modality to the existing entry
                new_modalities = dict(existing.modalities)
                new_modalities[mapped_type] = modality_sig

                # Re-compute aggregate: highest-conf modality is primary
                best_mod = max(new_modalities.values(), key=lambda m: m.confidence)
                all_up = existing.up_genes | up_genes
                all_down = existing.down_genes | down_genes
                all_cells = existing.cell_lines | cell_lines
                total_sigs = existing.n_signatures + len(col_indices)

                # Merged mean_z: average across modalities for shared genes,
                # keep modality-specific otherwise
                merged_z = dict(existing.mean_z)
                for g, z in mean_z.items():
                    if g in merged_z:
                        merged_z[g] = round((merged_z[g] + z) / 2, 4)
                    else:
                        merged_z[g] = z

                # Aggregate compound names
                all_compounds = existing.compound_names | compound_names

                self._consensus[gene_symbol] = ConsensusSignature(
                    target_gene=gene_symbol,
                    pert_type=best_mod.pert_type,
                    n_signatures=total_sigs,
                    mean_z=merged_z,
                    up_genes=all_up,
                    down_genes=all_down,
                    cell_lines=all_cells,
                    confidence=best_mod.confidence,
                    modalities=new_modalities,
                    compound_names=all_compounds,
                    moa=moa if moa else existing.moa,
                )
            else:
                self._consensus[gene_symbol] = ConsensusSignature(
                    target_gene=gene_symbol,
                    pert_type=mapped_type,
                    n_signatures=len(col_indices),
                    mean_z=mean_z,
                    up_genes=up_genes,
                    down_genes=down_genes,
                    cell_lines=cell_lines,
                    confidence=confidence,
                    modalities={mapped_type: modality_sig},
                    compound_names=compound_names,
                    moa=moa,
                )
                n_targets += 1

            n_processed += len(col_indices)

            if (i + 1) % 200 == 0:
                logger.info(
                    "  Progress: %d/%d groups, %d targets so far",
                    i + 1, n_groups, n_targets,
                )

        # Free the large matrix
        del full_data

        logger.info(
            "  Processed %d signatures -> %d new target genes with consensus "
            "(modality: %s)",
            n_processed,
            n_targets,
            mapped_type,
        )

    def _group_compound_sigs_by_gene(
        self,
        filtered_sigs: pd.DataFrame,
        pert_col: str,
    ) -> Dict[str, Dict]:
        """Re-group compound signatures by their *target gene* rather than
        by compound name.

        Uses ``self._compound_to_genes`` to map each compound to one or
        more gene targets.  Signatures for compounds that target the same
        gene are pooled together, providing a pharmacological perturbation
        consensus for that gene.

        Returns
        -------
        dict
            ``{gene_symbol: {"sig_ids": [...], "group_df": DataFrame,
                             "compound_names": set}}``
        """
        gene_groups: Dict[str, Dict] = {}
        n_unmapped = 0

        for cpd_name, group_df in filtered_sigs.groupby(pert_col):
            cpd_name_str = str(cpd_name).strip()
            if not cpd_name_str or cpd_name_str == "-666":
                continue

            # Look up target genes for this compound
            target_genes = self._compound_to_genes.get(cpd_name_str)
            if not target_genes:
                n_unmapped += 1
                continue

            sig_ids = group_df.index.tolist()
            if len(sig_ids) < 1:
                continue

            # Assign this compound's signatures to each of its gene targets
            for gene in target_genes:
                gene = gene.upper()
                if gene not in gene_groups:
                    gene_groups[gene] = {
                        "sig_ids": [],
                        "group_df": None,
                        "compound_names": set(),
                    }
                gene_groups[gene]["sig_ids"].extend(sig_ids)
                gene_groups[gene]["compound_names"].add(cpd_name_str)
                if gene_groups[gene]["group_df"] is None:
                    gene_groups[gene]["group_df"] = group_df
                else:
                    gene_groups[gene]["group_df"] = pd.concat(
                        [gene_groups[gene]["group_df"], group_df]
                    )

        if n_unmapped:
            logger.info(
                "  trt_cp: %d compounds had no target gene annotation (skipped)",
                n_unmapped,
            )

        return gene_groups

    # ------------------------------------------------------------------
    # Cross-modal concordance
    # ------------------------------------------------------------------

    def _compute_all_cross_modal_concordances(self) -> None:
        """Compute cross-modal concordance and classify genes for every
        multi-modal target.

        For each target with >=2 modalities:
        1. Measure Jaccard similarity of up/down gene sets across all
           modality pairs.
        2. Classify genes into **concordant** (confirmed in >=2 modalities)
           versus **single-modality** (only in one — may be off-target or
           context-specific).
        3. Store classifications on the ConsensusSignature.

        This prevents naive union from inflating the gene sets with
        compound off-target artefacts.
        """
        n_multi = 0
        for gene, cs in self._consensus.items():
            if len(cs.modalities) < 2:
                # Single-modality: all genes are "single-modality" by definition
                self._consensus[gene] = ConsensusSignature(
                    target_gene=cs.target_gene,
                    pert_type=cs.pert_type,
                    n_signatures=cs.n_signatures,
                    mean_z=cs.mean_z,
                    up_genes=cs.up_genes,
                    down_genes=cs.down_genes,
                    cell_lines=cs.cell_lines,
                    confidence=cs.confidence,
                    modalities=cs.modalities,
                    cross_modal_concordance=0.0,
                    compound_names=cs.compound_names,
                    moa=cs.moa,
                    concordant_up_genes=frozenset(),
                    concordant_down_genes=frozenset(),
                    single_modality_up_genes=cs.up_genes,
                    single_modality_down_genes=cs.down_genes,
                )
                continue

            n_multi += 1
            modalities = list(cs.modalities.values())

            # ── Jaccard concordance across pairs ────────────────────
            up_jaccard_sum = 0.0
            down_jaccard_sum = 0.0
            n_pairs = 0

            for a_idx in range(len(modalities)):
                for b_idx in range(a_idx + 1, len(modalities)):
                    a = modalities[a_idx]
                    b = modalities[b_idx]

                    up_union = a.up_genes | b.up_genes
                    if up_union:
                        up_jaccard_sum += len(a.up_genes & b.up_genes) / len(up_union)

                    down_union = a.down_genes | b.down_genes
                    if down_union:
                        down_jaccard_sum += len(a.down_genes & b.down_genes) / len(down_union)

                    n_pairs += 1

            concordance = (
                (up_jaccard_sum + down_jaccard_sum) / (2 * n_pairs)
                if n_pairs > 0
                else 0.0
            )

            # ── Classify genes by cross-modal support ───────────────
            # Count how many modalities each gene appears in
            up_counts: Dict[str, int] = {}
            down_counts: Dict[str, int] = {}
            for mod in modalities:
                for g in mod.up_genes:
                    up_counts[g] = up_counts.get(g, 0) + 1
                for g in mod.down_genes:
                    down_counts[g] = down_counts.get(g, 0) + 1

            # Concordant = in >=2 modalities; single = in exactly 1
            concordant_up = frozenset(g for g, c in up_counts.items() if c >= 2)
            concordant_down = frozenset(g for g, c in down_counts.items() if c >= 2)
            single_up = frozenset(g for g, c in up_counts.items() if c == 1)
            single_down = frozenset(g for g, c in down_counts.items() if c == 1)

            self._consensus[gene] = ConsensusSignature(
                target_gene=cs.target_gene,
                pert_type=cs.pert_type,
                n_signatures=cs.n_signatures,
                mean_z=cs.mean_z,
                up_genes=cs.up_genes,
                down_genes=cs.down_genes,
                cell_lines=cs.cell_lines,
                confidence=cs.confidence,
                modalities=cs.modalities,
                cross_modal_concordance=round(concordance, 4),
                compound_names=cs.compound_names,
                moa=cs.moa,
                concordant_up_genes=concordant_up,
                concordant_down_genes=concordant_down,
                single_modality_up_genes=single_up,
                single_modality_down_genes=single_down,
            )

        logger.info(
            "Cross-modal concordance computed for %d multi-modal targets", n_multi
        )

    # ------------------------------------------------------------------
    # Public API (compatible with alin.perturbation)
    # ------------------------------------------------------------------

    def ensure_loaded(self, cache_only: bool = False) -> None:
        """Load or build the index if not already done."""
        if not self._loaded:
            self.build_index(cache_only=cache_only)

    @property
    def available_targets(self) -> List[str]:
        """Gene symbols for which LINCS consensus signatures are available."""
        self.ensure_loaded()
        return sorted(self._consensus.keys())

    @property
    def n_targets(self) -> int:
        """Number of targets with consensus signatures."""
        self.ensure_loaded()
        return len(self._consensus)

    def get_consensus(self, gene: str) -> Optional[ConsensusSignature]:
        """Get the consensus signature for a target gene, or None."""
        self.ensure_loaded()
        return self._consensus.get(gene.upper())

    def get_perturbation_signature(
        self, target: str
    ) -> Optional[PerturbationSignature]:
        """
        Get a ``PerturbationSignature`` for *target*.

        Returns a LINCS-derived signature if available, otherwise ``None``.
        This is a drop-in replacement for
        ``alin.perturbation.get_perturbation_signature()``.
        """
        cs = self.get_consensus(target)
        if cs is None:
            return None
        return cs.to_perturbation_signature()

    def get_perturbation_responders(self, target: str) -> Set[str]:
        """All genes that respond to perturbation of *target*."""
        cs = self.get_consensus(target)
        if cs is None:
            return set()
        return set(cs.up_genes) | set(cs.down_genes)

    def get_direct_effectors(self, target: str) -> Set[str]:
        """Genes downregulated when *target* is perturbed (pathway members)."""
        cs = self.get_consensus(target)
        if cs is None:
            return set()
        return set(cs.down_genes)

    def get_feedback_genes(self, target: str) -> Set[str]:
        """Genes upregulated when *target* is perturbed (resistance/feedback)."""
        cs = self.get_consensus(target)
        if cs is None:
            return set()
        return set(cs.up_genes)

    def get_cancer_relevance(
        self, target: str, cancer_type: str, lineage: Optional[str] = None
    ) -> float:
        """
        Compute how relevant a target's LINCS evidence is for *cancer_type*.

        Returns a value in [0.3, 1.0]:
        - 1.0 if a majority of the consensus cell lines match the cancer lineage
        - 0.3 minimum floor so non-matching evidence is discounted but not ignored

        Parameters
        ----------
        target : str
            Gene symbol.
        cancer_type : str
            OncotreePrimaryDisease string (e.g. "Non-Small Cell Lung Cancer").
        lineage : str, optional
            OncotreeLineage (e.g. "Lung").  If not provided, looked up from
            ``CANCER_TYPE_TO_LINEAGE``.
        """
        cs = self.get_consensus(target)
        if cs is None or not cs.cell_lines:
            return 0.5  # no data → neutral weight

        if lineage is None:
            lineage = CANCER_TYPE_TO_LINEAGE.get(cancer_type)
        if lineage is None:
            return 0.5  # unknown cancer type → neutral

        # Count cell lines matching the target lineage
        n_match = 0
        n_mapped = 0
        for cl in cs.cell_lines:
            cl_upper = cl.upper().replace("-", "").replace(" ", "")
            # Try exact match first, then uppercase
            cl_lineage = LINCS_CELL_LINEAGE.get(cl) or LINCS_CELL_LINEAGE.get(cl_upper)
            if cl_lineage is not None:
                n_mapped += 1
                if cl_lineage == lineage:
                    n_match += 1

        if n_mapped == 0:
            return 0.5  # all unmapped → neutral

        relevance = n_match / n_mapped
        # Floor at 0.3: even non-matching evidence has some value
        # (pathway biology is partially conserved across lineages)
        return max(0.3, round(relevance, 3))

    def get_top_responders(
        self, target: str, n: int = 50, direction: str = "both"
    ) -> List[Tuple[str, float]]:
        """
        Return the top *n* responding genes ranked by |mean z-score|.

        Parameters
        ----------
        direction : str
            ``"up"``, ``"down"``, or ``"both"`` (default).
        """
        cs = self.get_consensus(target)
        if cs is None:
            return []

        items = list(cs.mean_z.items())
        if direction == "up":
            items = [(g, z) for g, z in items if z > 0]
            items.sort(key=lambda x: -x[1])
        elif direction == "down":
            items = [(g, z) for g, z in items if z < 0]
            items.sort(key=lambda x: x[1])
        else:
            items.sort(key=lambda x: -abs(x[1]))

        return items[:n]

    def score_combination_by_perturbation(
        self,
        targets: List[str],
        essential_genes: Set[str],
    ) -> Dict[str, Any]:
        """
        Score a combination using concordance-weighted LINCS evidence.

        Concordant genes (confirmed in >=2 modalities) are weighted at 1.0;
        single-modality genes are weighted at 0.5 to discount potential
        off-target artefacts.  Multi-modal coverage and mean concordance
        are also factored in.

        Returns a dict compatible with
        ``alin.perturbation.score_combination_by_perturbation()``.
        """
        all_effectors: Set[str] = set()
        concordant_effectors: Set[str] = set()
        all_feedback: Set[str] = set()
        concordant_feedback: Set[str] = set()
        total_responders: Set[str] = set()
        n_multi_modal = 0
        concordance_sum = 0.0

        for target in targets:
            cs = self.get_consensus(target)
            if cs is not None:
                all_effectors.update(cs.down_genes)
                all_feedback.update(cs.up_genes)
                total_responders.update(cs.up_genes)
                total_responders.update(cs.down_genes)

                # Track concordant (high-confidence) genes
                concordant_effectors.update(cs.concordant_down_genes)
                concordant_feedback.update(cs.concordant_up_genes)

                if cs.n_modalities >= 2:
                    n_multi_modal += 1
                    concordance_sum += cs.cross_modal_concordance

        feedback_targeted = set(targets) & all_feedback
        essential_covered = essential_genes & total_responders

        # Concordance-weighted scoring:
        # - Concordant genes count fully (1.0)
        # - Single-modality genes count at 0.5
        concordant_ess = essential_genes & (concordant_effectors | concordant_feedback)
        single_ess = essential_covered - concordant_ess
        weighted_coverage = (
            (len(concordant_ess) * 1.0 + len(single_ess) * SINGLE_MODALITY_DISCOUNT)
            / max(len(essential_genes), 1)
        )

        # Same for feedback: concordant feedback genes are more credible
        concordant_fb_targeted = set(targets) & concordant_feedback
        single_fb_targeted = feedback_targeted - concordant_fb_targeted
        weighted_fb = (
            (len(concordant_fb_targeted) * 1.0 + len(single_fb_targeted) * SINGLE_MODALITY_DISCOUNT)
            / max(len(all_feedback), 1)
        ) if all_feedback else 0.0

        # Multi-modal bonus
        multi_modal_frac = n_multi_modal / max(len(targets), 1)
        mean_concordance = (
            concordance_sum / n_multi_modal if n_multi_modal > 0 else 0.0
        )

        feedback_coverage = len(feedback_targeted) / max(len(all_feedback), 1)
        effector_coverage = len(essential_covered) / max(len(essential_genes), 1)

        # Composite perturbation score (sum = 1.00):
        # Weights reflect importance of each evidence type for
        # predicting gold-standard pair-overlap (calibration_results/).
        # Sensitivity analysis (scripts/weight_sensitivity_test.py, ±20%,
        # N=500): Spearman ρ=0.997.  See validation_results/weight_sensitivity.json.
        #   Weighted effector coverage: 0.35 (core signal)
        #   Weighted feedback coverage: 0.25 (resistance anticipation)
        #   Multi-modal fraction: 0.20 (cross-method validation)
        #   Mean concordance: 0.20 (modality agreement)
        perturbation_score = (
            PERT_W_EFFECTOR * weighted_coverage
            + PERT_W_FEEDBACK * weighted_fb
            + PERT_W_MULTIMODAL * multi_modal_frac
            + PERT_W_CONCORDANCE * mean_concordance
        )

        return {
            "feedback_coverage": round(feedback_coverage, 3),
            "effector_coverage": round(effector_coverage, 3),
            "weighted_effector_coverage": round(weighted_coverage, 3),
            "weighted_feedback_coverage": round(weighted_fb, 3),
            "multi_modal_fraction": round(multi_modal_frac, 3),
            "mean_concordance": round(mean_concordance, 3),
            "feedback_genes_targeted": feedback_targeted,
            "essential_effectors": essential_covered,
            "concordant_effectors": concordant_ess,
            "resistance_genes_untargeted": all_feedback - set(targets),
            "perturbation_score": round(perturbation_score, 3),
        }

    def build_perturbation_response_paths(
        self,
        essential_genes: Set[str],
        targets: Optional[List[str]] = None,
        min_overlap: int = 2,
    ) -> List[Tuple[str, Set[str], float]]:
        """
        Build viability paths — drop-in for
        ``alin.perturbation.build_perturbation_response_paths()``.
        """
        self.ensure_loaded()
        if targets is None:
            targets = list(self._consensus.keys())

        paths = []
        for target in targets:
            cs = self.get_consensus(target)
            if cs is None:
                continue

            responders = set(cs.up_genes) | set(cs.down_genes)
            essential_responders = essential_genes & responders

            if len(essential_responders) >= min_overlap:
                direct = essential_responders & set(cs.down_genes)
                n_direct = len(direct)
                confidence = cs.confidence * (
                    0.5 + 0.5 * n_direct / max(len(essential_responders), 1)
                )
                paths.append(
                    (target, essential_responders | {target}, round(confidence, 2))
                )

        return paths

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Return a summary dict for diagnostics / logging."""
        self.ensure_loaded()
        if not self._consensus:
            return {"n_targets": 0, "source": "LINCS_L1000"}

        confs = [c.confidence for c in self._consensus.values()]
        n_sigs = [c.n_signatures for c in self._consensus.values()]
        up_sizes = [len(c.up_genes) for c in self._consensus.values()]
        down_sizes = [len(c.down_genes) for c in self._consensus.values()]

        # Multi-modal statistics
        n_multi_modal = sum(
            1 for c in self._consensus.values() if len(c.modalities) >= 2
        )
        n_with_compound = sum(
            1 for c in self._consensus.values() if c.has_compound
        )
        n_with_genetic = sum(
            1 for c in self._consensus.values() if c.has_genetic
        )
        concordances = [
            c.cross_modal_concordance
            for c in self._consensus.values()
            if c.cross_modal_concordance > 0
        ]

        # Modality breakdown
        modality_counts: Dict[str, int] = {}
        for cs in self._consensus.values():
            for m in cs.modality_types:
                modality_counts[m] = modality_counts.get(m, 0) + 1

        result = {
            "n_targets": len(self._consensus),
            "median_confidence": round(float(np.median(confs)), 3),
            "median_n_signatures": int(np.median(n_sigs)),
            "median_up_genes": int(np.median(up_sizes)),
            "median_down_genes": int(np.median(down_sizes)),
            "pert_types": list(set(c.pert_type for c in self._consensus.values())),
            "modality_counts": modality_counts,
            "n_multi_modal": n_multi_modal,
            "n_with_genetic_evidence": n_with_genetic,
            "n_with_compound_evidence": n_with_compound,
            "source": "LINCS_L1000",
        }
        if concordances:
            result["median_cross_modal_concordance"] = round(
                float(np.median(concordances)), 4
            )
        return result

    # ------------------------------------------------------------------
    # Multi-modal public API
    # ------------------------------------------------------------------

    def get_cross_modal_concordance(self, gene: str) -> Optional[Dict[str, Any]]:
        """Return cross-modal concordance details for a target gene.

        Returns
        -------
        dict or None
            ``{"concordance": float, "n_modalities": int,
               "modalities": [...], "has_genetic": bool,
               "has_compound": bool, "compound_names": [...]}``
        """
        cs = self.get_consensus(gene)
        if cs is None:
            return None
        return {
            "concordance": cs.cross_modal_concordance,
            "n_modalities": cs.n_modalities,
            "modalities": cs.modality_types,
            "has_genetic": cs.has_genetic,
            "has_compound": cs.has_compound,
            "compound_names": sorted(cs.compound_names) if cs.compound_names else [],
            "moa": cs.moa,
            "confidence": cs.confidence,
            "n_signatures": cs.n_signatures,
        }

    def get_compound_evidence(self, gene: str) -> Optional[Dict[str, Any]]:
        """Return compound-specific evidence for a target gene.

        Returns data from trt_cp modality: which drugs target this gene,
        their mechanism of action, and the pharmacological signature.
        """
        cs = self.get_consensus(gene)
        if cs is None or not cs.has_compound:
            return None

        cpd_mod = cs.modalities.get("compound")
        if cpd_mod is None:
            return None

        return {
            "target_gene": gene,
            "n_compounds": len(cpd_mod.compound_names),
            "compound_names": sorted(cpd_mod.compound_names),
            "moa": cpd_mod.moa,
            "n_signatures": cpd_mod.n_signatures,
            "n_up_genes": len(cpd_mod.up_genes),
            "n_down_genes": len(cpd_mod.down_genes),
            "confidence": cpd_mod.confidence,
            "cell_lines": sorted(cpd_mod.cell_lines),
        }

    def get_multi_modal_targets(self, min_modalities: int = 2) -> List[str]:
        """Return genes with evidence from at least ``min_modalities`` modalities."""
        self.ensure_loaded()
        return sorted(
            g for g, cs in self._consensus.items()
            if cs.n_modalities >= min_modalities
        )


# ============================================================================
# Module-level convenience (singleton pattern)
# ============================================================================

_DEFAULT_DB: Optional[LINCSSignatureDB] = None


def get_default_db(
    lincs_dir: str = "lincs_data",
    pert_types: Optional[List[str]] = None,
    cache_only: bool = False,
) -> Optional[LINCSSignatureDB]:
    """
    Get or create the module-level LINCS database singleton.

    By default loads all three modalities (trt_xpr, trt_sh, trt_cp)
    when available.  Returns None if the ``lincs_dir`` does not exist.

    Parameters
    ----------
    cache_only : bool
        If True, instruct the DB to never trigger a full GCTX rebuild.
        It will load from a cached pickle (accepting stale v1 indexes)
        or return an empty DB.  This is critical for callers inside
        tight loops (evidence tiering) where a 50 GB rebuild would hang.
    """
    global _DEFAULT_DB
    if _DEFAULT_DB is not None:
        return _DEFAULT_DB
    if not os.path.isdir(lincs_dir):
        return None
    # Default to all 3 pert types for multi-modal integration
    if pert_types is None:
        pert_types = ["trt_xpr", "trt_sh", "trt_cp"]
    db = LINCSSignatureDB(lincs_dir, pert_types=pert_types)
    if cache_only:
        db.ensure_loaded(cache_only=True)
    _DEFAULT_DB = db
    return _DEFAULT_DB


def lincs_available(lincs_dir: str = "lincs_data") -> bool:
    """Check if LINCS data is present (geneinfo + at least one GCTX)."""
    if not os.path.isdir(lincs_dir):
        return False
    gene_info = os.path.join(lincs_dir, "geneinfo_beta.txt")
    if not os.path.isfile(gene_info):
        return False
    # Check for at least one GCTX
    return bool(glob.glob(os.path.join(lincs_dir, "level5_beta_*.gctx")))
