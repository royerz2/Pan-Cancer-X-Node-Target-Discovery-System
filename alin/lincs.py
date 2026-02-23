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
import json
import logging
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
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
    """Load ``siginfo_beta.txt`` and return a DataFrame indexed by sig_id."""
    path = os.path.join(lincs_dir, "siginfo_beta.txt")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Sig info not found: {path}")
    df = pd.read_csv(path, sep="\t", low_memory=False)
    if "sig_id" in df.columns:
        df.set_index("sig_id", inplace=True)
    return df


def load_cell_info(lincs_dir: str) -> pd.DataFrame:
    """Load ``cellinfo_beta.txt``."""
    path = os.path.join(lincs_dir, "cellinfo_beta.txt")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Cell info not found: {path}")
    return pd.read_csv(path, sep="\t")


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

            # Read subset — GCTX stores (cols, rows) transposed
            data = mat[:]
            data = data[np.ix_(col_mask, row_mask)].T if data.shape[0] == len(all_cids) else data[np.ix_(row_mask, col_mask)]

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
class ConsensusSignature:
    """Aggregated signature for one perturbagen across replicates/cell lines."""

    target_gene: str
    pert_type: str  # knockout / knockdown / compound
    n_signatures: int
    mean_z: Dict[str, float]  # gene_symbol → mean z-score across replicates
    up_genes: FrozenSet[str] = field(default_factory=frozenset)
    down_genes: FrozenSet[str] = field(default_factory=frozenset)
    cell_lines: FrozenSet[str] = field(default_factory=frozenset)
    confidence: float = 0.0

    def to_perturbation_signature(self) -> PerturbationSignature:
        """Convert to the legacy ``PerturbationSignature`` format."""
        return PerturbationSignature(
            target=self.target_gene,
            perturbation_type=self.pert_type,
            phospho_decreased=set(),  # L1000 measures mRNA, not phospho
            phospho_increased=set(),
            expression_decreased=set(self.down_genes),
            expression_increased=set(self.up_genes),
            confidence=self.confidence,
            source=f"LINCS_L1000_{self.pert_type}",
            pmid="28678552",  # Subramanian et al. 2017
        )


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
    rep_score = min(n_sigs / 20.0, 1.0)

    # Cell-line diversity: saturates around 5 cell lines
    cl_score = min(n_cell_lines / 5.0, 1.0)

    # Perturbation-type weight
    type_weight = {"knockout": 1.0, "knockdown": 0.85, "compound": 0.7}.get(
        pert_type, 0.6
    )

    # Combined: geometric-ish mean
    raw = (0.4 * rep_score + 0.3 * cl_score + 0.3 * type_weight)
    return round(min(raw, 0.98), 3)


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

    def _load_cached_index(self) -> bool:
        """Try to load a pre-built index from disk.  Returns True on success."""
        idx_path = self._index_path()
        if not os.path.isfile(idx_path):
            return False
        try:
            with open(idx_path, "rb") as f:
                data = pickle.load(f)
            if not isinstance(data, dict) or "consensus" not in data:
                return False
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
                pickle.dump({"consensus": self._consensus, "version": 1}, f)
            logger.info("Saved LINCS index to %s", idx_path)
        except Exception as exc:
            logger.warning("Failed to save LINCS index: %s", exc)

    # ------------------------------------------------------------------
    # Core: build consensus signatures from GCTX
    # ------------------------------------------------------------------

    def build_index(self, force: bool = False) -> None:
        """
        Build or load the consensus signature index.

        1. Tries to load a cached pickle index.
        2. If not found (or ``force=True``), reads the GCTX files,
           groups signatures by perturbed gene, computes consensus,
           and caches the result.
        """
        if self._loaded and not force:
            return

        if not force and self._load_cached_index():
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

        for pert_type in self.pert_types:
            gctx_path = self._find_gctx(pert_type)
            if gctx_path is None:
                logger.warning("No GCTX found for pert_type=%s", pert_type)
                continue

            logger.info("Processing %s: %s", pert_type, gctx_path)
            self._process_gctx(gctx_path, pert_type, landmark_ids)

        self._loaded = True
        self._save_index()

        logger.info(
            "LINCS index built: %d target genes with consensus signatures",
            len(self._consensus),
        )

    def _process_gctx(
        self,
        gctx_path: str,
        pert_type: str,
        landmark_ids: Optional[Set[int]] = None,
    ) -> None:
        """
        Process one GCTX file: group signatures by pert_iname (target gene),
        compute consensus, and store in ``self._consensus``.
        """
        mapped_type = PERT_TYPE_MAP.get(pert_type, pert_type)

        # Get sig_ids that match this pert_type with quality filter
        sig_df = self._sig_info
        mask = sig_df["pert_type"] == pert_type if "pert_type" in sig_df.columns else pd.Series(True, index=sig_df.index)

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

        # Group by perturbed gene/compound
        pert_col = "pert_iname" if "pert_iname" in filtered_sigs.columns else "cmap_name"
        if pert_col not in filtered_sigs.columns:
            logger.warning("  Cannot find perturbagen name column; skipping")
            return

        grouped = filtered_sigs.groupby(pert_col)

        # Read the full GCTX matrix in chunks to control memory.
        # For very large files, we read only the columns (sig_ids) we need.

        # Build per-gene signature groups
        n_processed = 0
        n_targets = 0

        for pert_name, group_df in grouped:
            pert_name = str(pert_name).strip()
            if not pert_name or pert_name == "-666":
                continue

            sig_ids = group_df.index.tolist()
            if len(sig_ids) < MIN_REPLICATE_SIGNATURES:
                continue

            # Cap signatures to avoid memory explosion
            if len(sig_ids) > MAX_SIGNATURES_PER_GENE:
                sig_ids = sig_ids[:MAX_SIGNATURES_PER_GENE]

            try:
                z_df = read_gctx(gctx_path, cid=sig_ids)
            except Exception as exc:
                logger.debug("  Could not read sigs for %s: %s", pert_name, exc)
                continue

            # Filter to landmark genes if requested
            if landmark_ids is not None:
                valid_rows = [r for r in z_df.index if int(r) in landmark_ids]
                z_df = z_df.loc[valid_rows]

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
            cell_lines = set()
            if "cell_iname" in group_df.columns:
                cell_lines = frozenset(
                    group_df["cell_iname"].dropna().unique().tolist()
                )

            confidence = _compute_confidence(
                n_sigs=len(sig_ids),
                n_cell_lines=len(cell_lines),
                pert_type=mapped_type,
            )

            # Normalise gene symbol: LINCS pert_iname may differ from
            # HGNC (e.g., lowercase, alias).  Try to match.
            gene_symbol = pert_name.upper()

            # Store (or merge with existing if another pert_type already added)
            if gene_symbol in self._consensus:
                existing = self._consensus[gene_symbol]
                # Merge: take the higher-confidence version
                if confidence > existing.confidence:
                    self._consensus[gene_symbol] = ConsensusSignature(
                        target_gene=gene_symbol,
                        pert_type=mapped_type,
                        n_signatures=len(sig_ids) + existing.n_signatures,
                        mean_z={**existing.mean_z, **mean_z},
                        up_genes=existing.up_genes | up_genes,
                        down_genes=existing.down_genes | down_genes,
                        cell_lines=existing.cell_lines | cell_lines,
                        confidence=confidence,
                    )
            else:
                self._consensus[gene_symbol] = ConsensusSignature(
                    target_gene=gene_symbol,
                    pert_type=mapped_type,
                    n_signatures=len(sig_ids),
                    mean_z=mean_z,
                    up_genes=up_genes,
                    down_genes=down_genes,
                    cell_lines=cell_lines,
                    confidence=confidence,
                )
                n_targets += 1

            n_processed += len(sig_ids)

        logger.info(
            "  Processed %d signatures → %d target genes with consensus",
            n_processed,
            n_targets,
        )

    # ------------------------------------------------------------------
    # Public API (compatible with alin.perturbation)
    # ------------------------------------------------------------------

    def ensure_loaded(self) -> None:
        """Load or build the index if not already done."""
        if not self._loaded:
            self.build_index()

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
        Score a combination — drop-in for
        ``alin.perturbation.score_combination_by_perturbation()``.
        """
        all_effectors: Set[str] = set()
        all_feedback: Set[str] = set()
        total_responders: Set[str] = set()

        for target in targets:
            cs = self.get_consensus(target)
            if cs is not None:
                all_effectors.update(cs.down_genes)
                all_feedback.update(cs.up_genes)
                total_responders.update(cs.up_genes)
                total_responders.update(cs.down_genes)

        feedback_targeted = set(targets) & all_feedback
        essential_covered = essential_genes & total_responders

        feedback_coverage = len(feedback_targeted) / max(len(all_feedback), 1)
        effector_coverage = len(essential_covered) / max(len(essential_genes), 1)

        return {
            "feedback_coverage": round(feedback_coverage, 3),
            "effector_coverage": round(effector_coverage, 3),
            "feedback_genes_targeted": feedback_targeted,
            "essential_effectors": essential_covered,
            "resistance_genes_untargeted": all_feedback - set(targets),
            "perturbation_score": round(
                0.6 * effector_coverage + 0.4 * feedback_coverage, 3
            ),
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

        return {
            "n_targets": len(self._consensus),
            "median_confidence": round(float(np.median(confs)), 3),
            "median_n_signatures": int(np.median(n_sigs)),
            "median_up_genes": int(np.median(up_sizes)),
            "median_down_genes": int(np.median(down_sizes)),
            "pert_types": list(set(c.pert_type for c in self._consensus.values())),
            "source": "LINCS_L1000",
        }


# ============================================================================
# Module-level convenience (singleton pattern)
# ============================================================================

_DEFAULT_DB: Optional[LINCSSignatureDB] = None


def get_default_db(
    lincs_dir: str = "lincs_data",
    pert_types: Optional[List[str]] = None,
) -> Optional[LINCSSignatureDB]:
    """
    Get or create the module-level LINCS database singleton.

    Returns None if the ``lincs_dir`` does not exist.
    """
    global _DEFAULT_DB
    if _DEFAULT_DB is not None:
        return _DEFAULT_DB
    if not os.path.isdir(lincs_dir):
        return None
    _DEFAULT_DB = LINCSSignatureDB(lincs_dir, pert_types=pert_types)
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
