"""
Structural Triple Finder — Liaki Framework
===========================================
Implements the attractor-disruption triple discovery framework from
Liaki et al. (Nature 2025): (upstream_feeder, driver, escape_route).

The three legs have specific biological roles that CANNOT be swapped:

  upstream_feeder  ─► driver  ─► downstream effectors
                                        │
                              (when blocked)
                                        │
                            feedback RTKs reactivate
                                        │
                              escape_route activated

The triple is structurally enforced — not a scoring artifact.

Architecture: Filter by biology, Score by therapy
-------------------------------------------------
Biological validity is enforced by FILTERING stages (expression, alteration
status, CRISPR dependency, cancer specificity, OmniPath connectivity).
Therapeutic ranking uses a SIMPLE score (essentiality, druggability, evidence).
These concerns are kept separate — no double-counting.

Algorithm
---------
Stage 1 — Driver:
    Cancer-specific CRISPR dependency (mean CERES < -0.3 for oncogenes),
    minus pan-essentials, filtered to OmniPath.  Somatically altered drivers
    (hotspot mutations, CN amplification) are preferred over collateral
    dependencies.

Stage 2 — Upstream feeder:
    OmniPath in_act[driver] (1-hop direct activators with stimulation=True).
    Two-stage filter: (a) CRISPR-dependent in cancer type, (b) cancer-specific
    z-score.  Eliminates network hubs without functional feeder activity.

Stage 3 — Escape route:
    From the curated PerturbationSignature for the driver:
      phospho_increased → 1-hop OmniPath out_act extension.
    Orthogonality filter: escape  ∩ upstream = ∅

Stage 4 — Triple formation:
    Expression gate (all three genes must be expressed in cancer type).
    Druggability gate on all three roles.
    Structural uniqueness filter (frozenset dedup).

Stage 5 — Ranking:
    Composite structural score (essentiality 40%, druggability 35%,
    escape confidence 25%).  Optional SynergyScorer overlay.
"""

from __future__ import annotations

import collections
import logging
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd

from core.data_structures import TripleCombination
from alin.perturbation import get_perturbation_signature, PERTURBATION_SIGNATURES

if TYPE_CHECKING:
    from pan_cancer_xnode import DrugTargetDB, DepMapLoader, SynergyScorer

logger = logging.getLogger(__name__)


PDAC_TEMPLATE_FEEDER_PRIORS: Dict[str, float] = {
    'EGFR': 1.00,
    'ERBB2': 0.95,
    'ERBB3': 0.90,
    'MET': 0.90,
    'FGFR1': 0.85,
    'FGFR2': 0.80,
    'IGF1R': 0.75,
    'AXL': 0.75,
    'PDGFRA': 0.70,
    'PDGFRB': 0.70,
    'SRC': 0.65,
    'JAK1': 0.60,
    'JAK2': 0.60,
    'KIT': 0.55,
    'PTPN11': 0.55,
    'GRB2': 0.45,
    'SOS1': 0.45,
}

PDAC_TEMPLATE_DRIVER_PRIORS: Dict[str, float] = {
    'KRAS': 1.00,
    'BRAF': 0.90,
    'RAF1': 0.88,
    'NRAS': 0.85,
    'MAP2K1': 0.85,
    'MAP2K2': 0.82,
    'MAPK1': 0.80,
    'MAPK3': 0.76,
    'PIK3CA': 0.75,
    'AKT1': 0.72,
    'AKT2': 0.66,
    'MTOR': 0.64,
}

PDAC_TEMPLATE_ESCAPE_PRIORS: Dict[str, float] = {
    'STAT3': 1.00,
    'JAK1': 0.92,
    'JAK2': 0.92,
    'SRC': 0.85,
    'YAP1': 0.82,
    'AXL': 0.80,
    'TYK2': 0.76,
    'ERBB2': 0.68,
    'ERBB3': 0.68,
    'FGFR1': 0.66,
    'MET': 0.66,
    'BCL2L1': 0.60,
    'BCL2': 0.56,
    'CTNNB1': 0.54,
}


@dataclass
class StructuralTriple:
    """Intermediate representation before building TripleCombination."""
    upstream: str           # Stage 2 gene (feeds into driver)
    driver: str             # Stage 1 gene (primary oncogenic dependency)
    escape: str             # Stage 3 gene (activated when driver blocked)
    driver_dep_score: float # CERES score (more negative = more essential)
    driver_alteration_freq: float = 0.0  # fraction of cancer-type cell lines with mutation/CN amp
    upstream_dep_score: float = 0.0
    escape_dep_score: float = 0.0
    upstream_in_network: bool = True
    escape_confidence: float = 0.8  # from PerturbationSignature.confidence
    upstream_specificity: float = 0.0
    escape_specificity: float = 0.0
    druggability_up: float = 0.0
    druggability_driver: float = 0.0
    druggability_esc: float = 0.0

    @property
    def genes(self) -> FrozenSet[str]:
        return frozenset({self.upstream, self.driver, self.escape})

    @property
    def structural_score(self) -> float:
        """
        Lower = better (consistent with TripleCombination.combined_score convention).

        Four components reflecting therapeutic priority:
          1. Driver essentiality (25%) — cancer dependency on this driver (CRISPR)
          2. Alteration evidence (20%) — oncogene addiction signal (mutation/CN)
          3. Mean druggability  (30%) — targetability of all three genes
          4. Escape confidence  (25%) — evidence quality for resistance mechanism

        Alteration evidence distinguishes true oncogene addictions (BRAF V600E,
        KRAS G12C, FLT3-ITD) from collateral dependencies (CCND1, MTOR) that
        are essential but not somatically altered.  Without this, CRISPR
        dependency alone rewards pan-essential genes over cancer-type drivers.
        """
        ess_driver = min(abs(self.driver_dep_score) / 2.0, 1.0)
        alt_evidence = min(self.driver_alteration_freq, 1.0)
        drug_mean = (self.druggability_up + self.druggability_driver + self.druggability_esc) / 3.0
        raw = (
            ess_driver * 0.25
            + alt_evidence * 0.20
            + drug_mean * 0.30
            + self.escape_confidence * 0.25
        )
        return round(1.0 - raw, 4)


@dataclass(frozen=True)
class StructuralArmConfig:
    """Configuration for one structural comparison arm."""

    arm_name: str
    scoring_label: str
    prioritize_role_templates: bool = False
    priority_weight: float = 0.0
    feeder_weight: float = 0.0
    driver_weight: float = 0.0
    escape_weight: float = 0.0
    feeder_priors: Dict[str, float] = field(default_factory=dict)
    driver_priors: Dict[str, float] = field(default_factory=dict)
    escape_priors: Dict[str, float] = field(default_factory=dict)


def get_structural_arm_config(strategy_arm: str) -> StructuralArmConfig:
    """Return discovery/ranking settings for one explicit structural arm."""
    normalized = (strategy_arm or 'liaki_role').strip().lower()
    if normalized == 'liaki_role':
        return StructuralArmConfig(
            arm_name='liaki_role',
            scoring_label='structural:liaki_role',
        )
    if normalized == 'liaki_pdac_template':
        return StructuralArmConfig(
            arm_name='liaki_pdac_template',
            scoring_label='structural:liaki_pdac_template',
            prioritize_role_templates=True,
            priority_weight=0.25,
            feeder_weight=0.30,
            driver_weight=0.35,
            escape_weight=0.35,
            feeder_priors=PDAC_TEMPLATE_FEEDER_PRIORS,
            driver_priors=PDAC_TEMPLATE_DRIVER_PRIORS,
            escape_priors=PDAC_TEMPLATE_ESCAPE_PRIORS,
        )
    raise ValueError(f"Unknown structural strategy arm: {strategy_arm}")


class StructuralTripleFinder:
    """
    Discovers (upstream_feeder, driver, escape_route) triples for a given
    cancer type using directed OmniPath signaling + DepMap CRISPR +
    curated perturbation signatures.

    Parameters
    ----------
    omnipath_df : pd.DataFrame
        OmniPath signaling network with columns: source, target, stimulation, inhibition.
    depmap_loader : DepMapLoader
        DepMap data loader (provides CRISPR gene effect matrix and cell-line metadata).
    drug_db : DrugTargetDB
        Drug-target database for druggability scoring.
    synergy_scorer : SynergyScorer, optional
        If provided, synergy scores are computed after structural formation and used
        as a secondary ranking criterion (does NOT override structural filtering).
    mode_config : ModeConfig, optional
        Pipeline mode configuration.  Controls druggability threshold.
    """

    # Genes with complex names (contain '_') are multi-protein complexes —
    # always exclude from structural roles.
    COMPLEX_SEP = '_'

    # Fraction cutoff for pan-essentiality: genes essential in more than this
    # fraction of all cell lines are excluded from all structural roles.
    # Must match the main pipeline threshold (>90%) to avoid over-filtering
    # bona-fide oncogenic drivers such as KRAS (essential in ~70-80% of lines
    # due to activating mutations across multiple tumour types).
    PAN_ESSENTIAL_FRAC     = 0.90
    ONCOGENIC_THRESHOLD   = -0.30  # relaxed CERES for known oncogenes with curated phospho_increased

    # CRISPR score cutoff for cancer-specific driver identification.
    DRIVER_DEP_THRESHOLD = -0.5

    # Minimum expression level (log2(TPM+1)) for a gene to be considered
    # as a structural triple member.  TPM > 1 is a standard expression
    # threshold for "expressed in this tissue" (Eisenberg & Levanon 2013).
    EXPRESSION_THRESHOLD = 1.0

    # Minimum fraction of cancer-type cell lines with a somatic alteration
    # (hotspot mutation or CN amplification) for a driver to be considered
    # "somatically altered" — these represent true oncogene addictions.
    ALTERATION_MIN_FRAC = 0.05

    # Upstream feeders confirm the biological circuit but are not always
    # direct drug targets.  A relaxed druggability gate prevents filtering
    # out real pathway members (e.g. NRAS upstream of BRAF) that are
    # important for structural validity even if not yet fully druggable.
    FEEDER_DRUG_MIN = 0.3

    def __init__(
        self,
        omnipath_df: pd.DataFrame,
        depmap_loader,
        drug_db,
        synergy_scorer=None,
        mode_config=None,
        strategy_arm: str = 'liaki_role',
    ):
        self.depmap = depmap_loader
        self.drug_db = drug_db
        self.synergy_scorer = synergy_scorer
        self.cfg = mode_config
        self.arm_config = get_structural_arm_config(strategy_arm)
        self.strategy_arm = self.arm_config.arm_name
        self._pan_essential: Optional[Set[str]] = None
        self._global_crispr_stats: Optional[Tuple[pd.Series, pd.Series]] = None

        # Build in_act / out_act adjacency from directed+signed edges
        self.in_act: Dict[str, Set[str]]  = collections.defaultdict(set)
        self.out_act: Dict[str, Set[str]] = collections.defaultdict(set)
        self._build_adjacency(omnipath_df)

        logger.info(
            f"StructuralTripleFinder[{self.strategy_arm}]: OmniPath adjacency built — "
            f"{sum(len(v) for v in self.in_act.values())} activation edges, "
            f"{len(self.in_act)} target genes"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_adjacency(self, G: pd.DataFrame) -> None:
        """Build `in_act` and `out_act` from OmniPath edge table."""
        for _, row in G.iterrows():
            s = row["source"]
            t = row["target"]
            # Skip protein complexes (joined by underscore in OmniPath notation)
            if self.COMPLEX_SEP in s or self.COMPLEX_SEP in t:
                continue
            if row.get("stimulation", False):
                self.out_act[s].add(t)
                self.in_act[t].add(s)

    def _get_pan_essential(self) -> Set[str]:
        """Genes essential in >70% of all cell lines — excluded from all roles."""
        if self._pan_essential is not None:
            return self._pan_essential
        crispr = self.depmap.load_crispr_dependencies()
        n_lines = len(crispr)
        if n_lines == 0:
            self._pan_essential = set()
            return self._pan_essential
        frac = (crispr < -0.5).sum(axis=0) / n_lines
        self._pan_essential = set(frac[frac > self.PAN_ESSENTIAL_FRAC].index)
        logger.info(f"Pan-essential genes: {len(self._pan_essential)}")
        return self._pan_essential

    def _get_global_crispr_stats(self) -> Tuple[pd.Series, pd.Series]:
        """Return (global_mean, global_std) per gene across ALL cell lines.

        Used to compute cancer-type-specific z-scores so that broadly essential
        genes like CCND1 don't automatically outrank cancer-specific oncogenes
        like BRAF (BRAF-mutant melanoma) or FLT3 (AML).
        """
        if self._global_crispr_stats is not None:
            return self._global_crispr_stats
        crispr = self.depmap.load_crispr_dependencies()
        g_mean = crispr.mean(axis=0)
        g_std  = crispr.std(axis=0).clip(lower=0.01)
        self._global_crispr_stats = (g_mean, g_std)
        logger.info("Global CRISPR stats computed.")
        return self._global_crispr_stats

    def _is_druggable(self, gene: str, threshold: float) -> bool:
        try:
            return self.drug_db.get_druggability_score(gene) >= threshold
        except Exception:
            return False

    def _druggability(self, gene: str) -> float:
        try:
            return float(self.drug_db.get_druggability_score(gene))
        except Exception:
            return 0.0

    def _role_prior(self, role: str, gene: str) -> float:
        role_map = {
            'feeder': self.arm_config.feeder_priors,
            'driver': self.arm_config.driver_priors,
            'escape': self.arm_config.escape_priors,
        }
        return float(role_map.get(role, {}).get(gene, 0.0))

    def _has_role_prior(self, role: str, gene: str) -> bool:
        return self._role_prior(role, gene) > 0.0

    def _template_priority_score(self, st: StructuralTriple) -> float:
        if not self.arm_config.prioritize_role_templates:
            return 0.0
        weighted = (
            self.arm_config.feeder_weight * self._role_prior('feeder', st.upstream)
            + self.arm_config.driver_weight * self._role_prior('driver', st.driver)
            + self.arm_config.escape_weight * self._role_prior('escape', st.escape)
        )
        return max(0.0, min(1.0, weighted))

    def _rank_score(self, st: StructuralTriple) -> float:
        base_score = st.structural_score
        priority_score = self._template_priority_score(st)
        if priority_score <= 0.0 or self.arm_config.priority_weight <= 0.0:
            return base_score
        adjusted = base_score * (1.0 - self.arm_config.priority_weight * priority_score)
        return round(max(0.0, adjusted), 4)

    def _get_cancer_expression(self, cancer_type: str) -> Optional[pd.Series]:
        """Mean log2(TPM+1) expression per gene for cancer-type cell lines."""
        expr = self.depmap.load_expression()
        if expr is None:
            return None
        cell_lines = self.depmap.get_cell_lines_for_cancer(cancer_type)
        available = [cl for cl in cell_lines if cl in expr.index]
        if not available:
            return None
        return expr.loc[available].mean(axis=0)

    def _get_alteration_freq(self, gene: str, cancer_type: str) -> float:
        """Fraction of cancer-type cell lines with hotspot mutation or CN amp."""
        cell_lines = self.depmap.get_cell_lines_for_cancer(cancer_type)
        if not cell_lines:
            return 0.0
        max_freq = 0.0
        # Hotspot mutations
        hs = self.depmap.load_hotspot_mutations()
        if hs is not None:
            col = None
            for c in hs.columns:
                if c == gene or c.startswith(gene + ' ('):
                    col = c
                    break
            if col is not None:
                avail = [cl for cl in cell_lines if cl in hs.index]
                if avail:
                    max_freq = max(max_freq, float((hs.loc[avail, col] > 0).mean()))
        # Copy-number amplification
        cn = self.depmap.load_copy_number()
        if cn is not None and gene in cn.columns:
            avail = [cl for cl in cell_lines if cl in cn.index]
            if avail:
                max_freq = max(max_freq, float((cn.loc[avail, gene] > 3.0).mean()))
        return max_freq

    # ------------------------------------------------------------------
    # Stage 1: Driver identification
    # ------------------------------------------------------------------

    def find_drivers(
        self, cancer_type: str, top_n: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Return top-N cancer-specific driver genes ranked by CRISPR dependency.

        Filters applied
        ~~~~~~~~~~~~~~~
        1. CERES score < -0.5 in mean over cancer-type cell lines
        2. Not pan-essential (>70% of all lines)
        3. Present in OmniPath network (must be a network node)

        Returns
        -------
        List of (gene_symbol, mean_CERES_score) sorted ascending (most essential first).
        """
        crispr = self.depmap.load_crispr_dependencies()
        cell_lines = self.depmap.get_cell_lines_for_cancer(cancer_type)
        available = [cl for cl in cell_lines if cl in crispr.index]

        if not available:
            logger.warning(f"StructuralTripleFinder: no CRISPR cell lines for '{cancer_type}'")
            return []

        pan_ess = self._get_pan_essential()
        mean_dep = crispr.loc[available].mean(axis=0)

        # Keep only genes that are:
        # (a) more essential than threshold
        # (b) not pan-essential
        # (c) present in OmniPath
        network_genes = set(self.in_act.keys()) | set(self.out_act.keys())

        # Helper: gene has curated phospho_increased (needed for escape-route discovery).
        # Checking the curated dict directly rather than get_perturbation_signature()
        # to avoid LINCS entries for housekeeping genes (ribosomal proteins etc.)
        # that have no phospho_increased data but would otherwise pass the filter.
        def _has_curated_phospho(g: str) -> bool:
            sig = PERTURBATION_SIGNATURES.get(g)
            return bool(sig and sig.phospho_increased)

        # PRIMARY: Relaxed CERES threshold (-0.3) for known oncogenes with curated
        # phospho_increased.  This catches cancer driver genes like BRAF, EGFR, FLT3
        # that are oncogenic but not universally essential across a heterogeneous
        # cancer-type cell line panel (so don't reach the strict -0.5 threshold).
        #
        # Sorting is by cancer-type-specific z-score = (cancer_mean - global_mean)
        # / global_std, not by raw CERES.  This ensures BRAF in melanoma or KRAS in
        # PDAC (anomalously essential in that cancer type) wins over CCND1 (broadly
        # essential pan-cancer, so high CERES but low specificity z-score).
        g_mean, g_std = self._get_global_crispr_stats()

        raw_cands = (
            mean_dep[mean_dep < self.ONCOGENIC_THRESHOLD]
            .pipe(lambda s: s[~s.index.isin(pan_ess)])
            .pipe(lambda s: s[s.index.isin(network_genes)])
            .pipe(lambda s: s[s.index.map(_has_curated_phospho)])
        )

        if not raw_cands.empty:
            # Compute differential z-score for each candidate
            z_scores = (raw_cands - g_mean.reindex(raw_cands.index)) / g_std.reindex(raw_cands.index).fillna(0.01)

            # Prefer somatically altered drivers (oncogene addiction) over
            # collateral dependencies.  Altered drivers (hotspot mutations or
            # CN amplification in ≥5% of cancer-type cell lines) sort first;
            # within each group, sort by cancer-specificity z-score.
            alt_freq = pd.Series(
                {g: self._get_alteration_freq(g, cancer_type) for g in raw_cands.index},
                dtype=float,
            )
            is_altered = alt_freq > self.ALTERATION_MIN_FRAC
            sort_df = pd.DataFrame({
                'not_altered': (~is_altered).astype(int),
                'z': z_scores.values,
            }, index=raw_cands.index)
            oncogenic_candidates = raw_cands.loc[sort_df.sort_values(['not_altered', 'z']).index]
            n_alt = int(is_altered.sum())
            if n_alt:
                logger.debug(
                    f"{n_alt}/{len(raw_cands)} driver candidates somatically "
                    f"altered in {cancer_type}"
                )
        else:
            oncogenic_candidates = raw_cands

        if not oncogenic_candidates.empty:
            logger.debug(
                f"{len(oncogenic_candidates)} oncogenic driver candidates (CERES < "
                f"{self.ONCOGENIC_THRESHOLD}, curated phospho, z-sorted) for {cancer_type}"
            )
            result = list(oncogenic_candidates.head(top_n).items())
        else:
            # Strict fallback: nothing qualifies as an oncogenic driver with phospho-sig.
            # Still try to find anything below the strict threshold in the network.
            strict_candidates = (
                mean_dep[mean_dep < self.DRIVER_DEP_THRESHOLD]
                .pipe(lambda s: s[~s.index.isin(pan_ess)])
                .pipe(lambda s: s[s.index.isin(network_genes)])
                .sort_values(ascending=True)
            )
            if strict_candidates.empty:
                logger.warning(f"No CRISPR driver candidates found for '{cancer_type}'")
                result = []
            else:
                phospho_in_strict = strict_candidates[
                    strict_candidates.index.map(_has_curated_phospho)
                ]
                if not phospho_in_strict.empty:
                    result = list(phospho_in_strict.head(top_n).items())
                else:
                    logger.debug(
                        f"No curated-phospho drivers for {cancer_type}; "
                        "structural triples unlikely"
                    )
                    result = list(strict_candidates.head(top_n).items())

        logger.info(
            f"Drivers for {cancer_type}: "
            + ", ".join(f"{g}={v:.2f}" for g, v in result)
        )
        return result

    # ------------------------------------------------------------------
    # Stage 2: Upstream feeder identification
    # ------------------------------------------------------------------

    def find_upstream_feeders(
        self,
        driver: str,
        ct_mean_dep: Optional[pd.Series] = None,
    ) -> Set[str]:
        """
        Genes that directly activate the driver via OmniPath stimulation edges,
        filtered to those with actual dependency in the cancer type.

        First-principles filter: a feeder must have mean CRISPR score < 0 in the
        cancer type's cell lines, meaning knocking it out measurably impairs
        those cells.  This eliminates network hub genes (e.g. ABL1) that are
        graph-connected to many drivers but aren't biologically active feeders
        in a given cancer.  Falls back to the full in_act set if no candidate
        passes (preserves structural triple formation for data-sparse types).
        """
        candidates = set(self.in_act[driver])  # already complex-filtered

        if ct_mean_dep is None or candidates.isdisjoint(ct_mean_dep.index):
            return candidates

        # First gate: keep only feeders with measurable dependency in this cancer type.
        active = {
            g for g in candidates
            if g in ct_mean_dep.index and ct_mean_dep[g] < 0.0
        }
        if not active:
            return candidates

        # Second gate: prefer cancer-specific feeders over pan-cancer hubs.
        # z = (cancer_mean - global_mean) / global_std (more negative = more specific)
        try:
            g_mean, g_std = self._get_global_crispr_stats()
            z_map: Dict[str, float] = {}
            selective: Set[str] = set()
            for g in active:
                if g not in g_mean.index:
                    continue
                sd = max(float(g_std.get(g, 0.01)), 0.01)
                z = (float(ct_mean_dep[g]) - float(g_mean[g])) / sd
                z_map[g] = z
                if z < -0.25:
                    selective.add(g)
            if selective:
                return selective
            # No feeder meets the specificity threshold.  Instead of returning
            # all active feeders (which lets network hubs dominate), pick the
            # most cancer-specific among them — the ones closest to being
            # genuinely selective, even if they don't reach z < -0.25.
            if z_map:
                top_feeders = sorted(z_map, key=z_map.get)[:5]
                return set(top_feeders)
        except Exception:
            pass

        # Fallback: dependency-active feeders (only if z-score computation failed).
        return active

    # ------------------------------------------------------------------
    # Stage 3: Escape route identification
    # ------------------------------------------------------------------

    def find_escape_routes(self, driver: str) -> Tuple[Set[str], Set[str]]:
        """
        Genes that become activated when the driver is inhibited.

        Returns (direct, extended):
          - direct: genes from perturbation signature phospho_increased
            (experimental evidence — feedback RTKs reactivated upon inhibition)
          - extended: 1-hop OmniPath activation targets of direct genes
            (computational predictions — require corroborating dependency evidence)
        """
        sig = get_perturbation_signature(driver)
        if sig is None:
            logger.debug(f"No perturbation signature for {driver}; escape routes empty")
            return set(), set()

        # Direct feedback RTKs that reactivate upon driver inhibition
        direct = {
            g for g in sig.phospho_increased if self.COMPLEX_SEP not in g
        }
        direct.discard(driver)

        # 1-hop OmniPath extension from feedback RTKs
        extended: Set[str] = set()
        for rtk in direct:
            for downstream in self.out_act.get(rtk, set()):
                if self.COMPLEX_SEP not in downstream:
                    extended.add(downstream)
        extended -= direct
        extended.discard(driver)

        return direct, extended

    # ------------------------------------------------------------------
    # Stage 4 + 5: Triple formation and ranking
    # ------------------------------------------------------------------

    def form_triples(
        self,
        cancer_type: str,
        top_n: int = 20,
        druggability_threshold: Optional[float] = None,
        max_drivers: int = 4,
    ) -> List[TripleCombination]:
        """
        Main entry point: build and rank structural (upstream, driver, escape) triples.

        Parameters
        ----------
        cancer_type : str
            OncotreePrimaryDisease string.
        top_n : int
            Maximum number of triples to return.
        druggability_threshold : float, optional
            Minimum druggability score for a gene to be included.
            Defaults to `cfg.DRUGGABILITY_THRESHOLD` if set, else 0.3.
        max_drivers : int
            Number of top drivers to consider.

        Returns
        -------
        List[TripleCombination] sorted by combined_score ascending (lower = better).
        """
        threshold = druggability_threshold
        if threshold is None:
            threshold = getattr(self.cfg, "DRUGGABILITY_THRESHOLD", 0.3) if self.cfg else 0.3

        drivers = self.find_drivers(cancer_type, top_n=max_drivers)
        if not drivers:
            logger.warning(f"No structural drivers found for {cancer_type}")
            return []

        if self.arm_config.prioritize_role_templates:
            preferred_drivers = [item for item in drivers if self._has_role_prior('driver', item[0])]
            if preferred_drivers:
                drivers = preferred_drivers

        # Compute cancer-type mean CRISPR once — used for upstream feeder filtering
        crispr = self.depmap.load_crispr_dependencies()
        cell_lines = self.depmap.get_cell_lines_for_cancer(cancer_type)
        available = [cl for cl in cell_lines if cl in crispr.index]
        ct_mean_dep: Optional[pd.Series] = (
            crispr.loc[available].mean(axis=0) if available else None
        )
        g_mean: Optional[pd.Series] = None
        g_std: Optional[pd.Series] = None
        if ct_mean_dep is not None:
            try:
                g_mean, g_std = self._get_global_crispr_stats()
            except Exception:
                g_mean, g_std = None, None

        # Expression gate: genes must be expressed in the cancer type.
        # Optional — if expression data isn't available, skip this gate.
        ct_expr: Optional[pd.Series] = self._get_cancer_expression(cancer_type)
        if ct_expr is not None:
            logger.debug(f"Expression gate active for {cancer_type}")

        def _specificity_score(gene: str) -> float:
            if ct_mean_dep is None or g_mean is None or g_std is None:
                return 0.0
            if gene not in ct_mean_dep.index or gene not in g_mean.index:
                return 0.0
            sd = max(float(g_std.get(gene, 0.01)), 0.01)
            z = (float(ct_mean_dep[gene]) - float(g_mean[gene])) / sd
            return max(0.0, min(1.0, -z / 2.5))

        structural: List[StructuralTriple] = []
        seen: Set[FrozenSet[str]] = set()

        # Dependency threshold for 1-hop extension escape genes:
        # they need corroborating CRISPR evidence in this cancer type.
        ESCAPE_EXT_DEP_THR = -0.1

        for driver, dep_score in drivers:
            upstream_all = self.find_upstream_feeders(driver, ct_mean_dep)
            direct_esc, extended_esc = self.find_escape_routes(driver)

            # 1-hop extensions require cancer-type dependency evidence
            if ct_mean_dep is not None:
                extended_esc = {
                    g for g in extended_esc
                    if g in ct_mean_dep.index
                    and float(ct_mean_dep[g]) < ESCAPE_EXT_DEP_THR
                }

            escape_all = direct_esc | extended_esc

            # Orthogonality: escape must be outside the upstream-driver axis
            escape_all -= upstream_all
            escape_all.discard(driver)

            # Escape confidence from perturbation signature
            sig = get_perturbation_signature(driver)
            esc_conf = sig.confidence if sig else 0.7

            # Alteration frequency for this driver in the cancer type
            driver_alt = self._get_alteration_freq(driver, cancer_type)

            # Druggability gate
            # Driver and escape must meet the mode's strict threshold (direct
            # drug targets).  Upstream feeders use a relaxed threshold:
            # they confirm the biological circuit but need not be tier-1
            # drug targets themselves.
            feeder_threshold = max(self.FEEDER_DRUG_MIN, threshold * 0.5)
            druggable_up  = {g for g in upstream_all if self._is_druggable(g, feeder_threshold)}
            druggable_esc = {g for g in escape_all   if self._is_druggable(g, threshold)}

            # Expression gate: all targets must be expressed in cancer type
            if ct_expr is not None:
                expr_thr = self.EXPRESSION_THRESHOLD
                druggable_up  = {g for g in druggable_up  if g in ct_expr.index and ct_expr[g] > expr_thr}
                druggable_esc = {g for g in druggable_esc if g in ct_expr.index and ct_expr[g] > expr_thr}
                if driver not in ct_expr.index or ct_expr[driver] <= expr_thr:
                    logger.debug(f"Driver {driver} not expressed in {cancer_type}")
                    continue

            if self.arm_config.prioritize_role_templates:
                preferred_up = {g for g in druggable_up if self._has_role_prior('feeder', g)}
                if preferred_up:
                    druggable_up = preferred_up
                preferred_esc = {g for g in druggable_esc if self._has_role_prior('escape', g)}
                if preferred_esc:
                    druggable_esc = preferred_esc

            # Feeder fallback for constitutively active drivers (KRAS G12C,
            # BRAF V600E, etc.): when no upstream feeder survives all gates,
            # pick the single most druggable expressed OmniPath activator.
            # This preserves triple formation for self-activating oncogenes
            # that don't biologically require an upstream feeder.
            if not druggable_up and upstream_all:
                fallback = upstream_all
                if ct_expr is not None:
                    fallback = {g for g in fallback
                                if g in ct_expr.index and ct_expr[g] > expr_thr}
                if fallback:
                    best = max(fallback, key=lambda g: self._druggability(g))
                    druggable_up = {best}
                    logger.debug(
                        f"Feeder fallback for {driver}: using {best} "
                        f"(drug={self._druggability(best):.2f})"
                    )

            if not druggable_up or not druggable_esc:
                logger.debug(
                    f"No druggable upstream ({len(druggable_up)}) or escape "
                    f"({len(druggable_esc)}) for driver={driver} — skipping"
                )
                continue

            logger.info(
                f"Driver={driver} (dep={dep_score:.2f}): "
                f"upstream={len(druggable_up)}, escape={len(druggable_esc)}"
            )

            for up_gene in sorted(druggable_up):
                for esc_gene in sorted(druggable_esc):
                    if up_gene == esc_gene:
                        continue
                    key = frozenset({up_gene, driver, esc_gene})
                    if key in seen:
                        continue
                    seen.add(key)

                    st = StructuralTriple(
                        upstream=up_gene,
                        driver=driver,
                        escape=esc_gene,
                        driver_dep_score=dep_score,
                        driver_alteration_freq=driver_alt,
                        upstream_dep_score=float(ct_mean_dep[up_gene]) if ct_mean_dep is not None and up_gene in ct_mean_dep.index else 0.0,
                        escape_dep_score=float(ct_mean_dep[esc_gene]) if ct_mean_dep is not None and esc_gene in ct_mean_dep.index else 0.0,
                        escape_confidence=esc_conf,
                        upstream_specificity=_specificity_score(up_gene),
                        escape_specificity=_specificity_score(esc_gene),
                        druggability_up=self._druggability(up_gene),
                        druggability_driver=self._druggability(driver),
                        druggability_esc=self._druggability(esc_gene),
                    )
                    structural.append(st)

        if not structural:
            logger.warning(
                f"StructuralTripleFinder[{self.strategy_arm}]: no triples formed for {cancer_type}"
            )
            return []

        # Sort structural triples by structural_score
        structural.sort(key=self._rank_score)

        # Convert top candidates to TripleCombination objects
        top_structural = structural[:min(top_n * 3, len(structural))]  # extra headroom for synergy re-ranking
        result = [self._to_triple_combination(st, cancer_type) for st in top_structural]

        # Apply synergy scorer overlay if available (re-ranks within structural set)
        if self.synergy_scorer is not None:
            result = self._apply_synergy_overlay(result)

        result.sort(key=lambda tc: tc.combined_score)
        logger.info(
            f"StructuralTripleFinder[{self.strategy_arm}]: {len(result)} triples for {cancer_type} "
            f"(best: {result[0].targets if result else 'none'})"
        )
        return result[:top_n]

    def _to_triple_combination(
        self, st: StructuralTriple, cancer_type: str
    ) -> TripleCombination:
        """Convert a StructuralTriple to a TripleCombination."""
        targets = tuple(sorted(st.genes))
        drug_info = {}
        druggable_count = 0
        total_cost = 0.0

        for gene in targets:
            score = self._druggability(gene)
            if score >= 0.3:
                druggable_count += 1
            total_cost += max(0.0, 1.0 - score)
            # Minimal drug_info — full drug lookup happens downstream
            drug_info[gene] = None

        # Synergy score: default from structural overlap
        # (upstream → driver edge, driver → escape via feedback = two known edges)
        synergy_score = 0.50  # base; may be overwritten by synergy_scorer overlay

        # Resistance score: lower escape confidence → higher resistance probability
        resistance_score = 1.0 - st.escape_confidence

        # Coverage: 3 structural roles = complete path coverage
        coverage = 1.0

        # Pathway coverage
        pathway_coverage = {
            "upstream_blocker":   1.0,
            "driver_inhibition":  1.0,
            "escape_blockade":    1.0,
        }

        # Combined score stays structural-first; template priors only re-order within biologically valid triples.
        combined_score = self._rank_score(st)

        # Druggability tier
        if druggable_count == 3:
            tier = "immediate"
        elif druggable_count == 2:
            tier = "partial"
        else:
            tier = "research"

        return TripleCombination(
            targets=targets,
            total_cost=total_cost,
            synergy_score=synergy_score,
            resistance_score=resistance_score,
            pathway_coverage=pathway_coverage,
            coverage=coverage,
            druggable_count=druggable_count,
            combined_score=combined_score,
            drug_info=drug_info,
            druggability_tier=tier,
            evidence_power="adequate",  # structural evidence is hypothesis-level
            scoring_mode=self.arm_config.scoring_label,
            strategy_arm=self.arm_config.arm_name,
            role_assignments={
                'feeder': st.upstream,
                'driver': st.driver,
                'escape': st.escape,
            },
        )

    def _apply_synergy_overlay(
        self, triples: List[TripleCombination]
    ) -> List[TripleCombination]:
        """
        Re-weight combined_score using synergy scorer output.
        Synergy acts as a 30% modifier — never dominates structural score.
        """
        updated = []
        for tc in triples:
            try:
                syn = self.synergy_scorer.compute_synergy(tc.targets)
                # Blend: 70% structural + 30% synergy overlay (inverted: higher syn → lower score)
                new_score = tc.combined_score * 0.70 + (1.0 - syn) * 0.30
                updated.append(TripleCombination(
                    targets=tc.targets,
                    total_cost=tc.total_cost,
                    synergy_score=syn,
                    resistance_score=tc.resistance_score,
                    pathway_coverage=tc.pathway_coverage,
                    coverage=tc.coverage,
                    druggable_count=tc.druggable_count,
                    combined_score=round(new_score, 4),
                    drug_info=tc.drug_info,
                    druggability_tier=tc.druggability_tier,
                    evidence_power=tc.evidence_power,
                    scoring_mode=f"{self.arm_config.scoring_label}+synergy",
                    strategy_arm=tc.strategy_arm,
                    role_assignments=dict(tc.role_assignments),
                ))
            except Exception as exc:
                logger.debug(f"Synergy overlay failed for {tc.targets}: {exc}")
                updated.append(tc)
        return updated
