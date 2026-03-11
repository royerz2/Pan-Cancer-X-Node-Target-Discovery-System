#!/usr/bin/env python3
"""
ChEMBL 36 Drug-Target Integration
==================================

Extracts drug-target-indication data from ChEMBL 36 SQLite and
provides cancer-indication-aware druggability scoring.

Replaces the 35-gene hand-curated GENE_TO_DRUGS / GENE_CLINICAL_STAGE
dicts in constants.py with data covering ~1,800 drug targets and
~1,200 cancer-indication-specific gene-drug mappings.

Cache architecture:
  - First run queries ChEMBL SQLite (~29 GB) and writes a compact JSON
    cache (~2 MB) at data/chembl_cache.json.gz.
  - Subsequent runs load from cache in <0.5 s.

Usage:
    from alin.chembl_data import ChEMBLDrugDB
    db = ChEMBLDrugDB()          # auto-loads or builds cache
    db.get_druggability_score('EGFR', cancer_type='Non-Small Cell Lung Cancer')
    db.get_drugs_for_gene('BRAF')
    db.get_max_phase('KRAS', cancer_type='Colorectal Adenocarcinoma')
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import re
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent
_DATA_DIR = _THIS_DIR.parent / "data"
_CHEMBL_DB = _DATA_DIR / "chembl_36" / "chembl_36_sqlite" / "chembl_36.db"
_CACHE_PATH = _DATA_DIR / "chembl_cache.json.gz"

# ---------------------------------------------------------------------------
# Cancer type → ChEMBL EFO / MeSH term mapping
# ---------------------------------------------------------------------------
# Maps OncoTree full names used by ALIN to sets of EFO / MeSH keywords that
# will match drug_indication.efo_term or drug_indication.mesh_heading.

CANCER_TO_CHEMBL_TERMS: Dict[str, List[str]] = {
    # Major solid tumors
    "Invasive Breast Carcinoma": ["breast cancer", "breast neoplasm", "breast carcinoma"],
    "Breast Invasive Carcinoma": ["breast cancer", "breast neoplasm", "breast carcinoma"],
    "Breast Ductal Carcinoma In Situ": ["breast cancer", "breast neoplasm"],
    "Non-Small Cell Lung Cancer": ["non-small cell lung", "nsclc", "lung adenocarcinoma", "lung squamous"],
    "Lung Adenocarcinoma": ["non-small cell lung", "lung adenocarcinoma", "lung cancer"],
    "Lung Squamous Cell Carcinoma": ["non-small cell lung", "lung squamous", "lung cancer"],
    "Small Cell Lung Cancer": ["small cell lung"],
    "Colorectal Adenocarcinoma": ["colorectal cancer", "colon cancer", "colorectal neoplasm", "colonic neoplasm", "rectal neoplasm"],
    "Colon Adenocarcinoma": ["colorectal cancer", "colon cancer", "colonic neoplasm"],
    "Rectal Adenocarcinoma": ["colorectal cancer", "rectal neoplasm"],
    "Pancreatic Adenocarcinoma": ["pancreatic cancer", "pancreatic carcinoma", "pancreatic neoplasm", "pancreatic adenocarcinoma"],
    "Melanoma": ["melanoma", "cutaneous melanoma"],
    "Cutaneous Melanoma": ["melanoma", "cutaneous melanoma"],
    "Mucosal Melanoma of the Vulva/Vagina": ["melanoma"],
    "Uveal Melanoma": ["uveal melanoma", "melanoma"],
    "Hepatocellular Carcinoma": ["hepatocellular carcinoma", "liver cancer", "liver neoplasm"],
    "Ovarian Epithelial Tumor": ["ovarian cancer", "ovarian carcinoma", "ovarian neoplasm"],
    "High-Grade Serous Ovarian Cancer": ["ovarian cancer", "ovarian carcinoma"],
    "Prostate Adenocarcinoma": ["prostate cancer", "prostate adenocarcinoma", "prostatic neoplasm"],
    "Glioblastoma": ["glioblastoma", "glioma", "brain neoplasm"],
    "Diffuse Glioma": ["glioma", "glioblastoma", "brain neoplasm"],
    "Stomach Adenocarcinoma": ["gastric cancer", "stomach neoplasm", "stomach cancer"],
    "Esophagogastric Adenocarcinoma": ["gastric cancer", "esophageal neoplasm", "stomach neoplasm"],
    "Esophageal Adenocarcinoma": ["esophageal neoplasm", "esophageal cancer"],
    "Bladder Urothelial Carcinoma": ["bladder cancer", "urinary bladder", "urothelial"],
    "Renal Cell Carcinoma": ["renal cell carcinoma", "kidney cancer", "kidney neoplasm"],
    "Renal Clear Cell Carcinoma": ["renal cell carcinoma", "kidney cancer"],
    "Head and Neck Squamous Cell Carcinoma": ["head and neck", "squamous cell carcinoma of head"],
    "Thyroid Cancer": ["thyroid cancer", "thyroid neoplasm", "thyroid carcinoma"],
    "Well-Differentiated Thyroid Cancer": ["thyroid cancer", "thyroid neoplasm"],
    "Anaplastic Thyroid Cancer": ["thyroid cancer", "anaplastic thyroid"],
    "Endometrial Carcinoma": ["endometrial cancer", "endometrial neoplasm", "uterine cancer"],
    "Cervical Squamous Cell Carcinoma": ["cervical cancer", "uterine cervical neoplasm"],
    "Cholangiocarcinoma": ["cholangiocarcinoma", "bile duct cancer"],
    "Pleural Mesothelioma": ["mesothelioma"],
    "Adrenocortical Carcinoma": ["adrenal cortex neoplasm", "adrenocortical carcinoma"],
    "Pheochromocytoma": ["pheochromocytoma"],
    "Neuroblastoma": ["neuroblastoma"],
    "Retinoblastoma": ["retinoblastoma"],
    "Hepatoblastoma": ["hepatoblastoma"],
    "Wilms Tumor": ["wilms tumor", "nephroblastoma"],
    "Testicular Cancer": ["testicular cancer", "testicular neoplasm"],
    # Hematologic malignancies
    "Acute Myeloid Leukemia": ["acute myeloid leukemia", "aml"],
    "Acute Lymphoblastic Leukemia": ["acute lymphoblastic leukemia", "all"],
    "Chronic Myeloid Leukemia": ["chronic myeloid leukemia", "cml"],
    "Chronic Lymphocytic Leukemia": ["chronic lymphocytic leukemia", "cll"],
    "Mature B-Cell Neoplasms": ["b-cell lymphoma", "non-hodgkins lymphoma", "diffuse large b-cell"],
    "T-Lymphoblastic Leukemia/Lymphoma": ["t-cell lymphoma", "lymphoblastic leukemia"],
    "Plasma Cell Myeloma": ["multiple myeloma", "plasma cell myeloma"],
    "Hodgkin Lymphoma": ["hodgkin lymphoma"],
    "Myeloproliferative Neoplasms": ["myeloproliferative", "polycythemia", "myelofibrosis"],
    "Myelodysplastic Syndromes": ["myelodysplastic"],
    # Sarcomas
    "Soft Tissue Sarcoma": ["sarcoma", "soft tissue sarcoma"],
    "Osteosarcoma": ["osteosarcoma"],
    "Ewing Sarcoma": ["ewing sarcoma"],
    "Rhabdomyosarcoma": ["rhabdomyosarcoma"],
    "Liposarcoma": ["liposarcoma", "sarcoma"],
    "Gastrointestinal Stromal Tumor": ["gastrointestinal stromal", "gist"],
    "Synovial Sarcoma": ["sarcoma", "synovial sarcoma"],
    "Leiomyosarcoma": ["leiomyosarcoma", "sarcoma"],
    # Generic fallbacks
    "Sarcoma, NOS": ["sarcoma"],
    "Fibrosarcoma": ["sarcoma", "fibrosarcoma"],
    "Non-Cancerous": [],
}

# For cancer types not in the mapping, try to extract keywords
_CANCER_KEYWORD_RE = re.compile(
    r"(melanoma|carcinoma|leukemia|lymphoma|sarcoma|myeloma|"
    r"glioma|glioblastoma|neuroblastoma|mesothelioma|"
    r"breast|lung|colon|rectal|pancrea|liver|kidney|ovarian|"
    r"prostate|bladder|thyroid|brain|stomach|gastric|esophag|"
    r"cervical|endometri|cholang|hepato)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Cache builder
# ---------------------------------------------------------------------------

def build_chembl_cache(db_path: Path = _CHEMBL_DB,
                       cache_path: Path = _CACHE_PATH) -> Dict[str, Any]:
    """
    Query ChEMBL 36 SQLite and build a compact JSON cache.

    Returns dict with keys:
      gene_drugs   : {GENE: [{name, max_phase, mechanism, action_type}, ...]}
      gene_cancer  : {GENE: {efo_term_lower: max_phase_for_ind, ...}}

    The cache is written gzipped to cache_path (~2 MB).
    """
    if not db_path.exists():
        raise FileNotFoundError(
            f"ChEMBL database not found at {db_path}. "
            "Run: tar -xzf data/chembl_36_sqlite.tar.gz -C data/"
        )

    logger.info("Building ChEMBL cache from %s ...", db_path)
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    # -------------------------------------------------------------------
    # 1. Gene → drugs (via drug_mechanism + target_components + gene symbols)
    #    Filter to human targets only (tax_id = 9606)
    # -------------------------------------------------------------------
    cur.execute("""
        SELECT UPPER(cs.component_synonym) AS gene,
               md.pref_name               AS drug_name,
               md.max_phase               AS max_phase,
               dm.mechanism_of_action     AS mechanism,
               dm.action_type             AS action_type
        FROM drug_mechanism dm
        JOIN molecule_dictionary md ON dm.molregno = md.molregno
        JOIN target_components  tc ON dm.tid      = tc.tid
        JOIN component_synonyms cs ON tc.component_id = cs.component_id
        JOIN component_sequences cseq ON tc.component_id = cseq.component_id
        WHERE cs.syn_type = 'GENE_SYMBOL'
          AND cseq.tax_id = 9606
          AND md.pref_name IS NOT NULL
        ORDER BY gene, max_phase DESC
    """)

    gene_drugs: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    seen_gene_drug: set = set()

    for gene, drug_name, max_phase, mechanism, action_type in cur.fetchall():
        key = (gene, drug_name)
        if key in seen_gene_drug:
            continue
        seen_gene_drug.add(key)
        gene_drugs[gene].append({
            "name": drug_name,
            "max_phase": float(max_phase) if max_phase is not None else 0.0,
            "mechanism": mechanism or "",
            "action_type": action_type or "",
        })

    logger.info("  gene_drugs: %d genes, %d drug entries",
                len(gene_drugs), sum(len(v) for v in gene_drugs.values()))

    # -------------------------------------------------------------------
    # 2. Gene → cancer indication → max phase
    #    Join drug_mechanism → drug_indication via molregno
    # -------------------------------------------------------------------
    cur.execute("""
        SELECT UPPER(cs.component_synonym) AS gene,
               LOWER(COALESCE(di.efo_term, di.mesh_heading)) AS indication,
               MAX(di.max_phase_for_ind) AS max_phase
        FROM drug_mechanism dm
        JOIN target_components  tc ON dm.tid = tc.tid
        JOIN component_synonyms cs ON tc.component_id = cs.component_id
        JOIN component_sequences cseq ON tc.component_id = cseq.component_id
        JOIN drug_indication    di ON dm.molregno = di.molregno
        WHERE cs.syn_type = 'GENE_SYMBOL'
          AND cseq.tax_id = 9606
          AND di.max_phase_for_ind IS NOT NULL
          AND di.max_phase_for_ind > 0
        GROUP BY gene, indication
        ORDER BY gene, max_phase DESC
    """)

    gene_cancer: Dict[str, Dict[str, float]] = defaultdict(dict)
    for gene, indication, max_phase in cur.fetchall():
        if indication:
            gene_cancer[gene][indication] = float(max_phase)

    logger.info("  gene_cancer: %d genes with indication data",
                len(gene_cancer))

    conn.close()

    # -------------------------------------------------------------------
    # 3. Write cache
    # -------------------------------------------------------------------
    cache = {
        "gene_drugs": dict(gene_drugs),
        "gene_cancer": {g: dict(v) for g, v in gene_cancer.items()},
        "version": "chembl_36",
    }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(str(cache_path), "wt", encoding="utf-8") as f:
        json.dump(cache, f, separators=(",", ":"))

    size_mb = os.path.getsize(str(cache_path)) / 1e6
    logger.info("  Cache written to %s (%.1f MB)", cache_path, size_mb)

    return cache


# ---------------------------------------------------------------------------
# ChEMBL Drug Database
# ---------------------------------------------------------------------------

class ChEMBLDrugDB:
    """
    Cancer-indication-aware drug-target database backed by ChEMBL 36.

    Key improvement over the old 35-gene hand-curated dict:
      - ~1,800 gene targets with known drug mechanisms
      - ~1,200 of those have cancer-specific indication data
      - Druggability scoring is cancer-type-aware: a gene approved for
        THIS cancer gets score 1.0, while approved for a DIFFERENT cancer
        gets 0.85, etc.
    """

    # Phase → score mapping
    _PHASE_SCORE = {4.0: 1.0, 3.0: 0.75, 2.0: 0.55, 1.0: 0.35, 0.5: 0.25}

    def __init__(self,
                 db_path: Optional[Path] = None,
                 cache_path: Optional[Path] = None,
                 auto_build: bool = True):
        self._db_path = Path(db_path) if db_path else _CHEMBL_DB
        self._cache_path = Path(cache_path) if cache_path else _CACHE_PATH
        self._gene_drugs: Dict[str, List[Dict[str, Any]]] = {}
        self._gene_cancer: Dict[str, Dict[str, float]] = {}
        self._indication_cache: Dict[str, List[str]] = {}  # cancer_type → [efo_terms]

        if self._cache_path.exists():
            self._load_cache()
        elif auto_build and self._db_path.exists():
            cache = build_chembl_cache(self._db_path, self._cache_path)
            self._gene_drugs = cache["gene_drugs"]
            self._gene_cancer = cache["gene_cancer"]
        else:
            logger.warning(
                "ChEMBL data not available (no cache at %s, no DB at %s). "
                "Falling back to empty database.",
                self._cache_path, self._db_path,
            )

        logger.info(
            "ChEMBLDrugDB loaded: %d gene targets, %d with cancer indications",
            len(self._gene_drugs), len(self._gene_cancer),
        )

    # ------------------------------------------------------------------
    # Cache I/O
    # ------------------------------------------------------------------

    def _load_cache(self) -> None:
        with gzip.open(str(self._cache_path), "rt", encoding="utf-8") as f:
            cache = json.load(f)
        self._gene_drugs = cache.get("gene_drugs", {})
        self._gene_cancer = cache.get("gene_cancer", {})
        logger.info("Loaded ChEMBL cache: %d genes, version=%s",
                     len(self._gene_drugs), cache.get("version", "?"))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def has_gene(self, gene: str) -> bool:
        """Check if gene has any known drug mechanism in ChEMBL."""
        return gene.upper() in self._gene_drugs

    def get_drugs_for_gene(self, gene: str) -> List[str]:
        """Return list of drug names targeting this gene."""
        entries = self._gene_drugs.get(gene.upper(), [])
        return [e["name"] for e in entries]

    def get_max_phase(self, gene: str, cancer_type: Optional[str] = None) -> float:
        """
        Get maximum clinical phase for drugs targeting this gene.

        If cancer_type is given, returns the max phase specifically for
        that cancer indication.  Falls back to global max_phase if no
        cancer-specific data exists.

        Returns:
            Phase as float (4.0=approved, 3.0=phase3, ..., 0.0=none)
        """
        gene_u = gene.upper()
        entries = self._gene_drugs.get(gene_u, [])
        if not entries:
            return 0.0

        # Cancer-specific phase
        if cancer_type:
            cancer_phase = self._get_cancer_phase(gene_u, cancer_type)
            if cancer_phase > 0:
                return cancer_phase

        # Global max phase
        return max((e["max_phase"] for e in entries), default=0.0)

    def get_druggability_score(self, gene: str,
                               cancer_type: Optional[str] = None) -> float:
        """
        Cancer-indication-aware druggability score (0.0–1.0).

        Scoring logic:
          1. If gene has drugs approved/in trials FOR THIS cancer type:
             Use cancer-specific max phase → score.
          2. If gene has drugs for OTHER cancers but not this one:
             Use global max phase with a penalty (× 0.85).
          3. If gene has no drug mechanisms at all:
             Return 0.1 (slightly above zero — still potentially
             discoverable via screening).

        This replaces the old flat scoring that gave EGFR 1.0 for ALL
        cancers regardless of whether EGFR inhibitors are indicated.
        """
        gene_u = gene.upper()
        entries = self._gene_drugs.get(gene_u, [])

        if not entries:
            return 0.1  # Unknown target

        global_max = max((e["max_phase"] for e in entries), default=0.0)

        if cancer_type:
            cancer_phase = self._get_cancer_phase(gene_u, cancer_type)
            if cancer_phase > 0:
                # Drug exists for THIS cancer
                base = self._phase_to_score(cancer_phase)
                # Bonus for multiple drugs
                n_drugs = len(entries)
                bonus = min(0.15, n_drugs * 0.02)
                return min(1.0, base + bonus)
            elif global_max > 0:
                # Drug exists but NOT for this cancer — penalized
                base = self._phase_to_score(global_max)
                n_drugs = len(entries)
                bonus = min(0.1, n_drugs * 0.015)
                return min(0.9, (base + bonus) * 0.85)

        # No cancer context or no cancer-indication data
        base = self._phase_to_score(global_max)
        n_drugs = len(entries)
        bonus = min(0.1, n_drugs * 0.015)
        return min(0.95, base + bonus)

    def get_drug_count(self, gene: str) -> int:
        """Number of distinct drugs targeting this gene."""
        return len(self._gene_drugs.get(gene.upper(), []))

    def get_clinical_stage(self, gene: str,
                           cancer_type: Optional[str] = None) -> str:
        """
        Return clinical stage label for this gene (optionally cancer-specific).
        Returns: 'approved', 'phase3', 'phase2', 'phase1', 'preclinical'
        """
        phase = self.get_max_phase(gene, cancer_type)
        if phase >= 4.0:
            return "approved"
        elif phase >= 3.0:
            return "phase3"
        elif phase >= 2.0:
            return "phase2"
        elif phase >= 1.0:
            return "phase1"
        elif phase > 0:
            return "preclinical"
        return "preclinical"

    def get_mechanism(self, gene: str) -> str:
        """Return the most common mechanism of action for this gene's drugs."""
        entries = self._gene_drugs.get(gene.upper(), [])
        if not entries:
            return ""
        # Pick the mechanism from the highest-phase drug
        best = max(entries, key=lambda e: e["max_phase"])
        return best.get("mechanism", "")

    def get_action_type(self, gene: str) -> str:
        """Return the most common action type (INHIBITOR, AGONIST, etc.)."""
        entries = self._gene_drugs.get(gene.upper(), [])
        if not entries:
            return ""
        best = max(entries, key=lambda e: e["max_phase"])
        return best.get("action_type", "")

    def get_cancer_indications(self, gene: str) -> Dict[str, float]:
        """Return dict of {indication_term: max_phase} for this gene."""
        return dict(self._gene_cancer.get(gene.upper(), {}))

    @property
    def all_genes(self) -> Set[str]:
        """Set of all gene symbols with known drug mechanisms."""
        return set(self._gene_drugs.keys())

    @property
    def gene_count(self) -> int:
        return len(self._gene_drugs)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _phase_to_score(self, phase: float) -> float:
        """Convert numeric phase to 0–1 score."""
        if phase >= 4.0:
            return 1.0
        elif phase >= 3.0:
            return 0.75
        elif phase >= 2.0:
            return 0.55
        elif phase >= 1.0:
            return 0.35
        elif phase > 0:
            return 0.25
        return 0.1

    def _get_cancer_phase(self, gene: str, cancer_type: str) -> float:
        """
        Look up max phase for gene + cancer_type by matching ChEMBL
        indication terms against our cancer type mapping.
        """
        indications = self._gene_cancer.get(gene, {})
        if not indications:
            return 0.0

        search_terms = self._get_search_terms(cancer_type)
        best_phase = 0.0

        for term in search_terms:
            for ind_key, phase in indications.items():
                if term in ind_key:
                    best_phase = max(best_phase, phase)

        return best_phase

    def _get_search_terms(self, cancer_type: str) -> List[str]:
        """Get ChEMBL search terms for a cancer type (cached)."""
        if cancer_type in self._indication_cache:
            return self._indication_cache[cancer_type]

        # Try exact mapping first
        terms = CANCER_TO_CHEMBL_TERMS.get(cancer_type, [])

        if not terms:
            # Try to extract keywords from the cancer type name
            lower = cancer_type.lower()
            matches = _CANCER_KEYWORD_RE.findall(lower)
            terms = [m.lower() for m in matches]

            # Also add the full name lowered for partial matching
            if lower:
                terms.append(lower)

        # Always lowercase for matching
        terms = [t.lower() for t in terms]
        self._indication_cache[cancer_type] = terms
        return terms


# ---------------------------------------------------------------------------
# Module-level convenience: singleton instance
# ---------------------------------------------------------------------------

_INSTANCE: Optional[ChEMBLDrugDB] = None


def get_chembl_db() -> ChEMBLDrugDB:
    """Get or create the singleton ChEMBLDrugDB instance."""
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = ChEMBLDrugDB()
    return _INSTANCE
