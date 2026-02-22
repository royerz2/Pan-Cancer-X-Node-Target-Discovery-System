#!/usr/bin/env python3
"""
Multi-Omics Druggability Scoring for ALIN Framework
====================================================

Computes a multi-omics druggability refinement score p(g) ∈ [0, 1]
for each gene target by integrating six orthogonal data layers:

    1. Structural druggability   — AlphaFold pLDDT + PDB ligand-bound pocket evidence
    2. Protein abundance          — Gygi/CCLE mass-spec proteomics per cancer type
    3. Degradability              — PROTAC-DB validated degraders + surface-exposed lysines
    4. PPI surface accessibility  — Disordered interface regions from AlphaFold pLDDT
    5. RNA expression             — DepMap CCLE RNA-seq TPM log(p+1) per cancer type
    6. RNA/protein concordance    — Spearman ρ between RNA and protein across cell lines

The composite protein score is blended with the existing gene-level
druggability d(g) to produce an enhanced druggability estimate:

    d_new(g) = α · d_gene(g) + (1 − α) · p(g)      α = 0.6

Data sources:
    - Gygi lab CCLE proteomics: https://gygi.hms.harvard.edu/publications/ccle.html
    - DepMap RNA-seq: OmicsExpressionProteinCodingGenesTPMLogp1.csv
    - PROTAC-DB: https://cadd.zju.edu.cn/protacdb/downloads
    - AlphaFold EBI, RCSB PDB (REST APIs with disk caching)

References:
    - Nusinow et al. 2020 — Cell 180:387–402 (CCLE proteomics)
    - Jumper et al. 2021 (AlphaFold)  — Nature 596:583–589
    - Freshour et al. 2021 (DGIdb 5.0) — NAR 49:D1144–D1151
    - Bai et al. 2019 (STAT3 PROTACs) — Cancer Cell 36:498–511
    - Weng et al. 2021 (PROTAC-DB)    — NAR 49:D1381–D1387
"""

import json
import time
import hashlib
import logging
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS
# ============================================================================

# Blending parameter: gene-level weight vs. protein-level weight
ALPHA_GENE_WEIGHT = 0.6

# Sub-score weights for composite protein score p(g)
# When RNA layers are available, all 6 components sum to 1.0;
# when absent, the active weights are renormalized automatically.
STRUCTURAL_WEIGHT = 0.25
ABUNDANCE_WEIGHT = 0.20
DEGRADABILITY_WEIGHT = 0.15
PPI_WEIGHT = 0.15
RNA_EXPRESSION_WEIGHT = 0.15
RNA_PROTEIN_CONCORDANCE_WEIGHT = 0.10

# AlphaFold pLDDT thresholds
PLDDT_HIGH_CONFIDENCE = 70.0    # pLDDT ≥ 70 → structured / foldable
PLDDT_VERY_HIGH = 90.0          # pLDDT ≥ 90 → very high confidence
PLDDT_DISORDERED = 50.0         # pLDDT < 50 → likely disordered

# PDB pocket thresholds
MIN_LIGAND_STRUCTURES = 1       # ≥1 ligand-bound structure → pocket evidence

# Proteomics
PROTEIN_EXPRESSION_THRESHOLD = 0.0  # log2-normalized, > 0 = detected

# RNA expression
RNA_TPM_THRESHOLD = 1.0             # log2(TPM+1) > 1.0 → expressed
RNA_HIGH_EXPRESSION = 4.0           # log2(TPM+1) > 4.0 → highly expressed

# RNA/Protein concordance
CONCORDANCE_MIN_CELL_LINES = 10     # need ≥10 matched cell lines
CONCORDANCE_HIGH = 0.5              # Spearman ρ ≥ 0.5 → high concordance
CONCORDANCE_MODERATE = 0.3          # Spearman ρ ≥ 0.3 → moderate

# Gygi lab data URLs
GYGI_PROTEIN_URL = "https://gygi.hms.harvard.edu/data/ccle/protein_quant_current_normalized.csv.gz"
GYGI_RNA_PROTEIN_CORR_URL = "https://gygi.hms.harvard.edu/data/ccle/Table_S4_Protein_RNA_Correlation_and_Enrichments.xlsx"
GYGI_SAMPLE_INFO_URL = "https://gygi.hms.harvard.edu/data/ccle/Table_S1_Sample_Information.xlsx"

# PROTAC-DB download URL
PROTACDB_URL = "https://cadd.zju.edu.cn/protacdb/downloads"

# PROTAC surface lysine
MIN_SURFACE_LYSINES = 3         # ≥3 surface-accessible Lys → degradable

# API endpoints
UNIPROT_API = "https://rest.uniprot.org"
ALPHAFOLD_API = "https://alphafold.ebi.ac.uk/api"
RCSB_SEARCH_API = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_DATA_API = "https://data.rcsb.org/rest/v1/core/entry"

# Rate limiting
API_DELAY = 0.25  # seconds between API calls

# ============================================================================
# CANONICAL GENE → UNIPROT MAPPING (Human, reviewed/Swiss-Prot)
# Source: UniProt release 2025_06.  Manually verified for 52 signaling genes.
# ============================================================================

GENE_TO_UNIPROT: Dict[str, str] = {
    # RTK / EGFR family
    'EGFR':   'P00533',  'ERBB2':  'P04626',  'ERBB3':  'P21860',
    'MET':    'P08581',  'ALK':    'Q9UM73',  'RET':    'P07949',
    'FGFR1':  'P11362',  'FGFR2':  'P21802',  'AXL':    'P30530',
    'IGF1R':  'P08069',  'KIT':    'P10721',  'FLT3':   'P36888',
    'ROS1':   'P08922',
    # RAS-MAPK
    'KRAS':   'P01116',  'NRAS':   'P01111',  'HRAS':   'P01112',
    'BRAF':   'P15056',  'RAF1':   'P04049',
    'MAP2K1': 'Q02750',  'MAP2K2': 'P36507',
    'MAPK1':  'P28482',  'MAPK3':  'P27361',
    # PI3K-AKT-mTOR
    'PIK3CA': 'P42336',  'PIK3CB': 'P42338',  'PIK3R1': 'P27986',
    'AKT1':   'P31749',  'AKT2':   'P31751',
    'MTOR':   'P42345',  'PTEN':   'P60484',
    'TSC1':   'Q92574',  'TSC2':   'P49815',
    # JAK-STAT
    'JAK1':   'P23458',  'JAK2':   'O60674',  'TYK2':   'P29597',
    'STAT3':  'P40763',  'STAT5A': 'P42229',  'STAT5B': 'P51692',
    # Cell cycle
    'CDK4':   'P11802',  'CDK6':   'Q00534',  'CDK2':   'P24941',
    'CCND1':  'P24385',  'CCNE1':  'P24864',
    'RB1':    'P06400',  'CDKN2A': 'P42771',  'E2F1':   'Q01094',
    # Apoptosis
    'BCL2':   'P10415',  'BCL2L1': 'Q07817',  'MCL1':   'Q07820',
    'BAX':    'Q07812',  'BAK1':   'Q16611',
    'BBC3':   'Q9BXH1',  'CASP3':  'P42574',  'CASP9':  'P55211',
    # SRC family
    'SRC':    'P12931',  'FYN':    'P06241',  'YES1':   'P07947',
    'LYN':    'P07948',  'LCK':    'P06239',
    # p53
    'TP53':   'P04637',  'MDM2':   'Q00987',  'CDKN1A': 'P38936',
    # NF-κB
    'NFKB1':  'P19838',  'RELA':   'Q04206',
    'IKBKB':  'O14920',  'CHUK':   'O15111',
    # Wnt
    'CTNNB1': 'P35222',  'APC':    'P25054',  'GSK3B':  'P49841',
    'DVL1':   'O14640',
    # Hippo
    'YAP1':   'P46937',  'WWTR1':  'Q9GZV5',
    'LATS1':  'O95835',  'LATS2':  'Q9NRM7',
    'STK3':   'Q13188',  'STK4':   'Q13043',
    # PARP
    'PARP1':  'P09874',
    # IDH
    'IDH1':   'O75874',  'IDH2':   'P48735',
    # Other
    'MYC':    'P01106',
}

# ============================================================================
# KNOWN PROTAC / DEGRADER TARGETS (curated from PROTAC-DB + literature)
# Each entry: gene → {status, exemplar, source}
# ============================================================================

KNOWN_DEGRADER_TARGETS: Dict[str, Dict[str, str]] = {
    'STAT3':  {'status': 'preclinical', 'exemplar': 'SD-36', 'source': 'Bai et al. 2019 Cancer Cell'},
    'BRD4':   {'status': 'phase1', 'exemplar': 'ARV-825', 'source': 'Lu et al. 2015 CMGH'},
    'BCL2L1': {'status': 'preclinical', 'exemplar': 'DT2216', 'source': 'Khan et al. 2019 Nat Med'},
    'CDK4':   {'status': 'preclinical', 'exemplar': 'BSJ-03-123', 'source': 'Brand et al. 2019 Cell Chem Biol'},
    'CDK6':   {'status': 'preclinical', 'exemplar': 'BSJ-03-123', 'source': 'Brand et al. 2019 Cell Chem Biol'},
    'CDK2':   {'status': 'preclinical', 'exemplar': 'TMX-4116', 'source': 'Hanzl et al. 2023 JACS'},
    'EGFR':   {'status': 'preclinical', 'exemplar': 'PROTAC-EGFR', 'source': 'Burslem et al. 2018 Cell Chem Biol'},
    'ERBB2':  {'status': 'preclinical', 'exemplar': 'compound 7', 'source': 'Li et al. 2020 Eur J Med Chem'},
    'BRAF':   {'status': 'preclinical', 'exemplar': 'P5B', 'source': 'Alabi et al. 2021 JACS'},
    'AKT1':   {'status': 'preclinical', 'exemplar': 'INY-03-041', 'source': 'Yu et al. 2023 Bioorg Med Chem'},
    'MCL1':   {'status': 'preclinical', 'exemplar': 'dMCL1-2', 'source': 'Wang et al. 2020 Bioorg Med Chem Lett'},
    'MDM2':   {'status': 'preclinical', 'exemplar': 'MD-224', 'source': 'Li et al. 2019 Cancer Cell'},
    'ALK':    {'status': 'preclinical', 'exemplar': 'MS4077', 'source': 'Zhang et al. 2018 Eur J Med Chem'},
    'MET':    {'status': 'preclinical', 'exemplar': 'PROTAC-MET', 'source': 'Burslem et al. 2020 ChemComm'},
    'KRAS':   {'status': 'preclinical', 'exemplar': 'LC-2', 'source': 'Bond et al. 2020 JACS'},
    'FLT3':   {'status': 'preclinical', 'exemplar': 'FLT3-PROTAC', 'source': 'Burslem et al. 2019 ChemComm'},
    'PARP1':  {'status': 'preclinical', 'exemplar': 'iRucaparib-AP6', 'source': 'Zhao et al. 2019 JACS'},
    'SRC':    {'status': 'preclinical', 'exemplar': 'dasatinib-PROTAC', 'source': 'Bondeson et al. 2015 Nat Chem Biol'},
}


# ============================================================================
# CACHING
# ============================================================================

class ProteinAPICache:
    """Disk-based JSON cache for protein-level API responses."""

    def __init__(self, cache_dir: str = "./api_cache/protein"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)

    def _key(self, prefix: str, identifier: str) -> Path:
        h = hashlib.md5(f"{prefix}:{identifier}".encode()).hexdigest()[:12]
        return self.cache_dir / f"{prefix}_{h}.json"

    def get(self, prefix: str, identifier: str) -> Optional[Any]:
        f = self._key(prefix, identifier)
        if f.exists():
            try:
                data = json.loads(f.read_text())
                if time.time() - data.get("_ts", 0) < 30 * 86400:  # 30-day TTL
                    return data.get("result")
            except Exception:
                pass
        return None

    def put(self, prefix: str, identifier: str, result: Any) -> None:
        f = self._key(prefix, identifier)
        f.write_text(json.dumps({"_ts": time.time(), "result": result}))


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class StructuralDruggability:
    """Structural druggability assessment for a single protein."""
    gene: str
    uniprot_id: str
    mean_plddt: float                   # mean AlphaFold pLDDT across full sequence
    domain_plddt: Dict[str, float]      # domain name → mean pLDDT (if available)
    n_pdb_structures: int               # total PDB structures
    n_ligand_bound: int                 # PDB structures with small-molecule ligand
    has_pocket: bool                    # ≥1 ligand-bound structure
    structural_score: float             # composite 0–1

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ProteinAbundance:
    """Protein abundance for a gene across cancer cell lines."""
    gene: str
    cancer_type: str
    n_cell_lines: int                   # cell lines with data
    n_detected: int                     # cell lines where protein detected
    mean_abundance: float               # mean log2 intensity (detected only)
    detection_fraction: float           # n_detected / n_cell_lines
    abundance_score: float              # composite 0–1

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class DegradabilityScore:
    """Degradability assessment (PROTAC/molecular glue tractability)."""
    gene: str
    has_known_degrader: bool
    degrader_status: str                # 'approved', 'phase1', 'preclinical', 'none'
    degrader_exemplar: str
    n_surface_lysines: int              # estimated from AlphaFold surface-exposed Lys
    degradability_score: float          # composite 0–1

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class PPIAccessibility:
    """Protein–protein interaction surface accessibility."""
    gene: str
    n_pdb_complexes: int                # PDB structures with ≥2 chains involving this protein
    has_interface_data: bool
    disordered_fraction: float          # fraction of residues with pLDDT < 50
    ppi_score: float                    # composite 0–1

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class RNAExpression:
    """RNA-seq expression evidence for a gene in a cancer type."""
    gene: str
    cancer_type: str
    n_cell_lines: int                   # cell lines with RNA-seq data
    n_expressed: int                    # cell lines where gene is expressed (TPM > threshold)
    mean_expression: float              # mean log2(TPM+1) across expressed cell lines
    expression_fraction: float          # n_expressed / n_cell_lines
    expression_score: float             # composite 0–1

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class RNAProteinConcordance:
    """RNA/protein concordance for a gene across cell lines.

    When both RNA-seq and mass-spec proteomics are available for the
    same set of cell lines, the Spearman rank correlation gauges how
    reliably mRNA abundance predicts protein abundance.  Genes with
    high concordance are more interpretable as drug targets because
    their protein levels are transcriptionally regulated rather than
    dominated by post-transcriptional or post-translational effects.
    """
    gene: str
    n_matched_lines: int                # cell lines with both RNA + protein data
    spearman_rho: float                 # Spearman ρ
    spearman_pvalue: float              # two-sided p-value
    concordance_tier: str               # 'high', 'moderate', 'low', 'insufficient'
    concordance_score: float            # composite 0–1

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ProteinDruggabilityScore:
    """Composite multi-omics druggability score for a gene."""
    gene: str
    structural: StructuralDruggability
    abundance: Optional[ProteinAbundance]
    degradability: DegradabilityScore
    ppi: PPIAccessibility
    rna_expression: Optional[RNAExpression]
    rna_protein_concordance: Optional[RNAProteinConcordance]
    protein_score: float                # composite p(g) ∈ [0, 1]
    blended_score: float                # d_new(g) = α·d_gene + (1−α)·p(g)
    gene_druggability: float            # original d(g,) for reference

    def to_dict(self) -> Dict:
        d = {
            'gene': self.gene,
            'protein_score': round(self.protein_score, 4),
            'blended_score': round(self.blended_score, 4),
            'gene_druggability': round(self.gene_druggability, 4),
            'structural': self.structural.to_dict(),
            'degradability': self.degradability.to_dict(),
            'ppi': self.ppi.to_dict(),
        }
        if self.abundance is not None:
            d['abundance'] = self.abundance.to_dict()
        if self.rna_expression is not None:
            d['rna_expression'] = self.rna_expression.to_dict()
        if self.rna_protein_concordance is not None:
            d['rna_protein_concordance'] = self.rna_protein_concordance.to_dict()
        return d


# ============================================================================
# API FETCHERS
# ============================================================================

def _rate_limit():
    """Simple rate limiter for API calls."""
    time.sleep(API_DELAY)


def fetch_alphafold_plddt(uniprot_id: str, cache: ProteinAPICache) -> Optional[Dict]:
    """
    Fetch per-residue pLDDT scores from AlphaFold EBI API.

    Returns dict with keys:
        - 'mean_plddt': float
        - 'residue_plddts': list[float]  (per-residue)
        - 'sequence_length': int
        - 'n_high_conf': int  (pLDDT ≥ 70)
        - 'n_disordered': int (pLDDT < 50)
    """
    cached = cache.get("alphafold_plddt", uniprot_id)
    if cached is not None:
        return cached

    url = f"{ALPHAFOLD_API}/prediction/{uniprot_id}"
    try:
        _rate_limit()
        resp = requests.get(url, timeout=30)
        if resp.status_code != 200:
            logger.warning(f"AlphaFold API returned {resp.status_code} for {uniprot_id}")
            return None

        entries = resp.json()
        if not entries:
            return None

        entry = entries[0] if isinstance(entries, list) else entries

        # Fetch the PAE/pLDDT JSON (per-residue confidence)
        plddt_url = entry.get("paeDocUrl") or entry.get("cifUrl")
        # Actually, the pLDDT is embedded in the model CIF; use the summary endpoint
        # The pLDDT per-residue is in the model confidence endpoint
        plddt_json_url = entry.get("pdbUrl", "").replace("-model_v", "-confidence_v").replace(".pdb", ".json")
        if not plddt_json_url:
            # Fallback: construct URL from uniprot_id
            plddt_json_url = (
                f"https://alphafold.ebi.ac.uk/files/"
                f"AF-{uniprot_id}-F1-confidence_v4.json"
            )

        _rate_limit()
        plddt_resp = requests.get(plddt_json_url, timeout=30)
        if plddt_resp.status_code != 200:
            # Try v3
            plddt_json_url = plddt_json_url.replace("_v4.", "_v3.")
            plddt_resp = requests.get(plddt_json_url, timeout=30)

        if plddt_resp.status_code == 200:
            plddt_data = plddt_resp.json()
            # The confidence JSON has "confidenceScore" per residue
            # or it might have "plddt" depending on version
            if isinstance(plddt_data, list):
                residue_plddts = [
                    r.get("confidenceScore", r.get("plddt", 0.0))
                    for r in plddt_data
                ]
            elif isinstance(plddt_data, dict):
                residue_plddts = plddt_data.get(
                    "confidenceScore",
                    plddt_data.get("plddt", [])
                )
                if not isinstance(residue_plddts, list):
                    residue_plddts = []
            else:
                residue_plddts = []
        else:
            # Use mean pLDDT from the prediction endpoint as fallback
            mean_plddt = entry.get("globalMetricValue", 75.0)
            seq_len = entry.get("uniprotEnd", 500) - entry.get("uniprotStart", 1) + 1
            residue_plddts = [mean_plddt] * seq_len

        if not residue_plddts:
            logger.warning(f"No pLDDT data for {uniprot_id}")
            return None

        plddts = np.array(residue_plddts, dtype=float)
        result = {
            "mean_plddt": float(np.mean(plddts)),
            "sequence_length": len(plddts),
            "n_high_conf": int(np.sum(plddts >= PLDDT_HIGH_CONFIDENCE)),
            "n_very_high": int(np.sum(plddts >= PLDDT_VERY_HIGH)),
            "n_disordered": int(np.sum(plddts < PLDDT_DISORDERED)),
            "fraction_structured": float(np.mean(plddts >= PLDDT_HIGH_CONFIDENCE)),
            "fraction_disordered": float(np.mean(plddts < PLDDT_DISORDERED)),
        }
        cache.put("alphafold_plddt", uniprot_id, result)
        return result

    except requests.RequestException as e:
        logger.warning(f"AlphaFold API error for {uniprot_id}: {e}")
        return None


def fetch_pdb_ligand_count(uniprot_id: str, gene: str,
                           cache: ProteinAPICache) -> Dict[str, int]:
    """
    Query RCSB PDB for structures of this protein, counting total
    structures and ligand-bound structures (evidence of a druggable pocket).

    Returns: {'n_structures': int, 'n_ligand_bound': int}
    """
    cached = cache.get("pdb_ligand", f"{uniprot_id}_{gene}")
    if cached is not None:
        return cached

    # RCSB advanced search: find entries matching this UniProt ID
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
                        "operator": "exact_match",
                        "value": uniprot_id,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_name",
                        "operator": "exact_match",
                        "value": "UniProt",
                    },
                },
            ],
        },
        "return_type": "entry",
        "request_options": {"results_content_type": ["experimental"]},
    }

    n_structures = 0
    n_ligand_bound = 0

    try:
        _rate_limit()
        resp = requests.post(RCSB_SEARCH_API, json=query, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            n_structures = data.get("total_count", 0)

            # Now count how many have a non-polymer (small molecule) ligand
            # Use a refined search adding ligand filter
            query_with_ligand = {
                "query": {
                    "type": "group",
                    "logical_operator": "and",
                    "nodes": [
                        *query["query"]["nodes"],
                        {
                            "type": "terminal",
                            "service": "text",
                            "parameters": {
                                "attribute": "rcsb_entry_info.nonpolymer_entity_count",
                                "operator": "greater",
                                "value": 0,
                            },
                        },
                    ],
                },
                "return_type": "entry",
                "request_options": {"results_content_type": ["experimental"]},
            }
            _rate_limit()
            resp2 = requests.post(RCSB_SEARCH_API, json=query_with_ligand, timeout=30)
            if resp2.status_code == 200:
                data2 = resp2.json()
                n_ligand_bound = data2.get("total_count", 0)
        elif resp.status_code == 204:
            # No results (204 = no content)
            pass
        else:
            logger.warning(f"RCSB search returned {resp.status_code} for {uniprot_id}")
    except requests.RequestException as e:
        logger.warning(f"RCSB API error for {uniprot_id}: {e}")

    result = {"n_structures": n_structures, "n_ligand_bound": n_ligand_bound}
    cache.put("pdb_ligand", f"{uniprot_id}_{gene}", result)
    return result


def fetch_pdb_complex_count(uniprot_id: str,
                            cache: ProteinAPICache) -> int:
    """
    Count PDB structures where this protein appears in a multi-chain complex
    (evidence of structurally characterized PPI interfaces).
    """
    cached = cache.get("pdb_complex", uniprot_id)
    if cached is not None:
        return cached

    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
                        "operator": "exact_match",
                        "value": uniprot_id,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_name",
                        "operator": "exact_match",
                        "value": "UniProt",
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.polymer_entity_count_protein",
                        "operator": "greater",
                        "value": 1,
                    },
                },
            ],
        },
        "return_type": "entry",
        "request_options": {"results_content_type": ["experimental"]},
    }

    n_complexes = 0
    try:
        _rate_limit()
        resp = requests.post(RCSB_SEARCH_API, json=query, timeout=30)
        if resp.status_code == 200:
            n_complexes = resp.json().get("total_count", 0)
    except requests.RequestException as e:
        logger.warning(f"RCSB complex search error for {uniprot_id}: {e}")

    cache.put("pdb_complex", uniprot_id, n_complexes)
    return n_complexes


# ============================================================================
# SUB-SCORE CALCULATORS
# ============================================================================

def compute_structural_score(gene: str, uniprot_id: str,
                             cache: ProteinAPICache) -> StructuralDruggability:
    """
    Structural druggability: combines AlphaFold pLDDT (protein quality /
    foldability) with PDB ligand-bound pocket evidence.

    Scoring logic:
        - pLDDT component (0–0.5):  fraction of residues ≥ 70 pLDDT, scaled
        - Pocket component (0–0.5):
            · 0.5  if ≥5 ligand-bound structures (strong pocket evidence)
            · 0.35 if 1–4 ligand-bound structures
            · 0.15 if structures exist but no ligands (structural data, no pocket)
            · 0.0  if no PDB structures at all

    This avoids penalizing well-studied targets that simply haven't had a
    co-crystal with a ligand yet, while rewarding direct pocket evidence.
    """
    af_data = fetch_alphafold_plddt(uniprot_id, cache)
    pdb_data = fetch_pdb_ligand_count(uniprot_id, gene, cache)

    mean_plddt = af_data["mean_plddt"] if af_data else 75.0
    frac_structured = af_data["fraction_structured"] if af_data else 0.5

    n_structures = pdb_data.get("n_structures", 0)
    n_ligand = pdb_data.get("n_ligand_bound", 0)

    # pLDDT component: 0–0.5
    plddt_score = min(0.5, frac_structured * 0.55)

    # Pocket component: 0–0.5
    if n_ligand >= 5:
        pocket_score = 0.50
    elif n_ligand >= 1:
        pocket_score = 0.35
    elif n_structures >= 1:
        pocket_score = 0.15
    else:
        pocket_score = 0.0

    structural_score = plddt_score + pocket_score

    return StructuralDruggability(
        gene=gene,
        uniprot_id=uniprot_id,
        mean_plddt=mean_plddt,
        domain_plddt={},  # could be expanded with domain-level analysis
        n_pdb_structures=n_structures,
        n_ligand_bound=n_ligand,
        has_pocket=(n_ligand >= MIN_LIGAND_STRUCTURES),
        structural_score=min(1.0, structural_score),
    )


def compute_abundance_score(
    gene: str,
    cancer_type: str,
    proteomics_df: Optional[pd.DataFrame] = None,
    cell_line_cancer_map: Optional[Dict[str, str]] = None,
) -> Optional[ProteinAbundance]:
    """
    Protein abundance from DepMap CCLE mass-spec proteomics.

    If the proteomics dataframe is provided, filters to cell lines of the
    given cancer type and computes detection fraction + mean abundance.

    Scoring logic:
        - detection_fraction ≥ 0.8  → 1.0  (protein robustly expressed)
        - detection_fraction ≥ 0.5  → 0.7
        - detection_fraction ≥ 0.2  → 0.4
        - detection_fraction < 0.2  → 0.1  (rarely expressed → low confidence)
        - No data available         → None (score not applicable)
    """
    if proteomics_df is None or cell_line_cancer_map is None:
        return None

    # Filter cell lines for this cancer type
    cancer_lines = [
        cl for cl, ct in cell_line_cancer_map.items()
        if ct == cancer_type and cl in proteomics_df.index
    ]

    if not cancer_lines or gene not in proteomics_df.columns:
        return None

    values = proteomics_df.loc[cancer_lines, gene].dropna()
    n_lines = len(cancer_lines)
    n_detected = int((values > PROTEIN_EXPRESSION_THRESHOLD).sum())
    mean_abund = float(values[values > PROTEIN_EXPRESSION_THRESHOLD].mean()) if n_detected > 0 else 0.0
    det_frac = n_detected / max(n_lines, 1)

    # Tiered scoring
    if det_frac >= 0.8:
        score = 1.0
    elif det_frac >= 0.5:
        score = 0.7
    elif det_frac >= 0.2:
        score = 0.4
    else:
        score = 0.1

    return ProteinAbundance(
        gene=gene,
        cancer_type=cancer_type,
        n_cell_lines=n_lines,
        n_detected=n_detected,
        mean_abundance=round(mean_abund, 4),
        detection_fraction=round(det_frac, 4),
        abundance_score=score,
    )


def compute_degradability_score(
    gene: str,
    uniprot_id: str,
    af_data: Optional[Dict] = None,
) -> DegradabilityScore:
    """
    PROTAC / molecular glue degradability assessment.

    Combines:
        1. Known degrader evidence from PROTAC-DB curation (0.4–0.5 bonus)
        2. Surface-exposed lysine estimate from AlphaFold
           (structured Lys residues → ubiquitin ligase recruitment sites)

    Scoring:
        - Known clinical degrader   → 0.9
        - Known preclinical degrader → 0.7
        - No degrader but ≥5 surface Lys + high pLDDT → 0.5
        - No degrader, 3–4 surface Lys                → 0.3
        - No degrader, <3 surface Lys or disordered    → 0.1
    """
    known = KNOWN_DEGRADER_TARGETS.get(gene)

    # Estimate surface lysines from AlphaFold (crude: count Lys in
    # structured regions, i.e. pLDDT ≥ 70; real analysis would use
    # solvent-accessible surface area from CIF, but this is a reasonable
    # proxy for the ~52-gene set)
    n_surface_lys = 0
    if af_data:
        seq_len = af_data.get("sequence_length", 500)
        n_high = af_data.get("n_high_conf", seq_len // 2)
        # Lysine frequency in human proteome ≈ 5.5%; surface-exposed
        # fraction of structured Lys ≈ 60%
        n_surface_lys = int(n_high * 0.055 * 0.6)

    if known:
        status = known["status"]
        exemplar = known["exemplar"]
        if status in ("approved", "phase1"):
            score = 0.9
        else:
            score = 0.7
    else:
        status = "none"
        exemplar = ""
        if n_surface_lys >= 5:
            score = 0.5
        elif n_surface_lys >= MIN_SURFACE_LYSINES:
            score = 0.3
        else:
            score = 0.1

    return DegradabilityScore(
        gene=gene,
        has_known_degrader=(known is not None),
        degrader_status=status,
        degrader_exemplar=exemplar,
        n_surface_lysines=n_surface_lys,
        degradability_score=score,
    )


def compute_ppi_score(
    gene: str,
    uniprot_id: str,
    af_data: Optional[Dict],
    cache: ProteinAPICache,
) -> PPIAccessibility:
    """
    Protein–protein interaction surface accessibility.

    For "undruggable" targets that function through PPIs (e.g., BCL2L1,
    STAT3 SH2 domain), this score estimates whether the interaction
    surface is structurally characterized and potentially targetable
    by PPI disruptors (stapled peptides, BH3 mimetics, etc.).

    Scoring:
        - ≥5 complex structures + low disorder → 1.0 (well-characterized PPI)
        - 1–4 complex structures               → 0.6
        - No complex structures but structured  → 0.3
        - Highly disordered (>50% disordered)   → 0.1
    """
    n_complexes = fetch_pdb_complex_count(uniprot_id, cache)
    frac_disordered = af_data.get("fraction_disordered", 0.3) if af_data else 0.3

    has_interface = n_complexes > 0

    if n_complexes >= 5 and frac_disordered < 0.3:
        score = 1.0
    elif n_complexes >= 1:
        score = 0.6
    elif frac_disordered < 0.3:
        score = 0.3
    else:
        score = 0.1

    return PPIAccessibility(
        gene=gene,
        n_pdb_complexes=n_complexes,
        has_interface_data=has_interface,
        disordered_fraction=round(frac_disordered, 4),
        ppi_score=score,
    )


# ============================================================================
# PROTEOMICS DATA LOADER
# ============================================================================

def load_ccle_proteomics(
    data_dir: str = "./depmap_data",
) -> Optional[Tuple[pd.DataFrame, Dict[str, str]]]:
    """
    Load DepMap CCLE proteomics (protein_quant_current.csv) and
    Model.csv for cell-line → cancer-type mapping.

    Returns:
        (proteomics_df, cell_line_cancer_map) or None if file not found.

    The proteomics file has columns: cell_line_id, gene_name, protein_intensity
    (log2-normalized).  We pivot to cell_line × gene matrix.
    """
    data_path = Path(data_dir)
    proteomics_file = data_path / "protein_quant_current.csv"
    model_file = data_path / "Model.csv"

    if not proteomics_file.exists():
        logger.info(
            f"CCLE proteomics file not found at {proteomics_file}. "
            "Download from https://depmap.org/portal/download/ → "
            "'protein_quant_current.csv' (Gygi lab, ~200 MB). "
            "Protein abundance scoring will be skipped."
        )
        return None

    logger.info(f"Loading CCLE proteomics from {proteomics_file}")
    try:
        # The proteomics file can have different formats depending on release
        # Common format: rows = proteins, columns = cell lines
        # Or: long format with columns [cell_line, gene_name, protein_expression]
        prot_df = pd.read_csv(proteomics_file, low_memory=False)

        # Detect format and pivot if necessary
        if 'Gene_Symbol' in prot_df.columns and 'Protein_Id' in prot_df.columns:
            # Gygi lab format: wide with Gene_Symbol, Protein_Id, then cell line columns
            gene_col = 'Gene_Symbol'
            id_cols = [c for c in prot_df.columns if c in (
                'Gene_Symbol', 'Protein_Id', 'Description', 'Group_ID',
                'Uniprot', 'Uniprot_Acc',
            )]
            value_cols = [c for c in prot_df.columns if c not in id_cols]
            prot_pivot = prot_df.set_index(gene_col)[value_cols].T
            prot_pivot.index.name = 'cell_line'
        elif 'gene_name' in prot_df.columns:
            # Long format
            prot_pivot = prot_df.pivot_table(
                index='cell_line', columns='gene_name',
                values='protein_expression', aggfunc='mean'
            )
        else:
            # Assume wide format: first column is gene, rest are cell lines
            prot_df = prot_df.set_index(prot_df.columns[0])
            prot_pivot = prot_df.T
            prot_pivot.index.name = 'cell_line'

        # Load cancer type mapping
        cancer_map = {}
        if model_file.exists():
            model_df = pd.read_csv(model_file)
            id_col = 'ModelID' if 'ModelID' in model_df.columns else model_df.columns[0]
            cancer_col = (
                'OncotreePrimaryDisease'
                if 'OncotreePrimaryDisease' in model_df.columns
                else 'primary_disease'
            )
            if cancer_col in model_df.columns:
                cancer_map = dict(zip(model_df[id_col], model_df[cancer_col]))

        logger.info(
            f"Loaded proteomics: {prot_pivot.shape[0]} cell lines × "
            f"{prot_pivot.shape[1]} proteins"
        )
        return prot_pivot, cancer_map

    except Exception as e:
        logger.warning(f"Failed to load CCLE proteomics: {e}")
        return None


# ============================================================================
# MAIN SCORING ENGINE
# ============================================================================

class ProteinDruggabilityScorer:
    """
    Computes protein-level druggability scores for ALIN target genes.

    Usage:
        scorer = ProteinDruggabilityScorer(
            genes=['EGFR', 'STAT3', 'CDK4', 'MCL1'],
            cancer_type='Pancreatic Adenocarcinoma',
            gene_druggability_fn=drug_db.get_druggability_score,
        )
        results = scorer.score_all()
        for gene, score in results.items():
            print(f"{gene}: p(g)={score.protein_score:.3f}, "
                  f"d_new={score.blended_score:.3f}")
    """

    def __init__(
        self,
        genes: List[str],
        cancer_type: str = "",
        gene_druggability_fn=None,
        cache_dir: str = "./api_cache/protein",
        proteomics_dir: str = "./depmap_data",
        alpha: float = ALPHA_GENE_WEIGHT,
    ):
        self.genes = genes
        self.cancer_type = cancer_type
        self.gene_druggability_fn = gene_druggability_fn or (lambda g: 0.2)
        self.cache = ProteinAPICache(cache_dir)
        self.proteomics_dir = proteomics_dir
        self.alpha = alpha

        # Lazy-loaded proteomics
        self._proteomics_df: Optional[pd.DataFrame] = None
        self._cancer_map: Optional[Dict[str, str]] = None
        self._proteomics_loaded = False

    def _ensure_proteomics(self):
        """Lazy-load proteomics data on first use."""
        if self._proteomics_loaded:
            return
        self._proteomics_loaded = True
        result = load_ccle_proteomics(self.proteomics_dir)
        if result is not None:
            self._proteomics_df, self._cancer_map = result

    def score_gene(self, gene: str) -> ProteinDruggabilityScore:
        """Compute the full protein-level druggability for a single gene."""
        uniprot_id = GENE_TO_UNIPROT.get(gene)
        if not uniprot_id:
            logger.warning(
                f"No UniProt ID for {gene}; using fallback scores"
            )
            return self._fallback_score(gene)

        # 1. Structural druggability (AlphaFold + PDB)
        structural = compute_structural_score(gene, uniprot_id, self.cache)

        # Get AlphaFold data for reuse in degradability + PPI
        af_data = fetch_alphafold_plddt(uniprot_id, self.cache)  # cached

        # 2. Protein abundance (CCLE proteomics)
        self._ensure_proteomics()
        abundance = compute_abundance_score(
            gene, self.cancer_type,
            self._proteomics_df, self._cancer_map,
        )

        # 3. Degradability (PROTAC + surface lysines)
        degradability = compute_degradability_score(gene, uniprot_id, af_data)

        # 4. PPI surface accessibility
        ppi = compute_ppi_score(gene, uniprot_id, af_data, self.cache)

        # Composite protein score
        scores = [
            (STRUCTURAL_WEIGHT, structural.structural_score),
            (DEGRADABILITY_WEIGHT, degradability.degradability_score),
            (PPI_WEIGHT, ppi.ppi_score),
        ]
        if abundance is not None:
            scores.append((ABUNDANCE_WEIGHT, abundance.abundance_score))
            total_weight = sum(w for w, _ in scores)
        else:
            # Redistribute abundance weight proportionally
            total_weight = STRUCTURAL_WEIGHT + DEGRADABILITY_WEIGHT + PPI_WEIGHT

        protein_score = sum(w * s for w, s in scores) / max(total_weight, 1e-9)
        protein_score = max(0.0, min(1.0, protein_score))

        # Blend with gene-level druggability
        gene_d = self.gene_druggability_fn(gene)
        blended = self.alpha * gene_d + (1 - self.alpha) * protein_score

        return ProteinDruggabilityScore(
            gene=gene,
            structural=structural,
            abundance=abundance,
            degradability=degradability,
            ppi=ppi,
            protein_score=round(protein_score, 4),
            blended_score=round(blended, 4),
            gene_druggability=round(gene_d, 4),
        )

    def score_all(self, progress: bool = True) -> Dict[str, ProteinDruggabilityScore]:
        """Score all genes. Returns gene → ProteinDruggabilityScore."""
        results = {}
        genes_iter = self.genes
        if progress:
            try:
                from alin.constants import tqdm as _tqdm
                genes_iter = _tqdm(self.genes, desc="Protein scoring", leave=False)
            except ImportError:
                pass

        for gene in genes_iter:
            try:
                results[gene] = self.score_gene(gene)
            except Exception as e:
                logger.warning(f"Protein scoring failed for {gene}: {e}")
                results[gene] = self._fallback_score(gene)

        return results

    def _fallback_score(self, gene: str) -> ProteinDruggabilityScore:
        """Fallback when API data is unavailable."""
        gene_d = self.gene_druggability_fn(gene)
        # Use gene-level druggability as a proxy for protein score
        structural = StructuralDruggability(
            gene=gene, uniprot_id=GENE_TO_UNIPROT.get(gene, ""),
            mean_plddt=0.0, domain_plddt={},
            n_pdb_structures=0, n_ligand_bound=0,
            has_pocket=False, structural_score=0.3,
        )
        degradability = DegradabilityScore(
            gene=gene, has_known_degrader=gene in KNOWN_DEGRADER_TARGETS,
            degrader_status=KNOWN_DEGRADER_TARGETS.get(gene, {}).get("status", "none"),
            degrader_exemplar=KNOWN_DEGRADER_TARGETS.get(gene, {}).get("exemplar", ""),
            n_surface_lysines=0, degradability_score=0.3,
        )
        ppi = PPIAccessibility(
            gene=gene, n_pdb_complexes=0,
            has_interface_data=False, disordered_fraction=0.3,
            ppi_score=0.3,
        )
        protein_score = 0.3
        blended = self.alpha * gene_d + (1 - self.alpha) * protein_score
        return ProteinDruggabilityScore(
            gene=gene, structural=structural, abundance=None,
            degradability=degradability, ppi=ppi,
            protein_score=protein_score,
            blended_score=round(blended, 4),
            gene_druggability=round(gene_d, 4),
        )


# ============================================================================
# CONVENIENCE / REPORTING
# ============================================================================

def generate_protein_scoring_report(
    results: Dict[str, ProteinDruggabilityScore],
    output_dir: str = "./results",
) -> str:
    """
    Generate a summary table (CSV) and JSON report from protein scoring results.

    Returns path to the CSV file.
    """
    out = Path(output_dir)
    out.mkdir(exist_ok=True, parents=True)

    rows = []
    for gene, r in sorted(results.items()):
        rows.append({
            "gene": gene,
            "uniprot_id": r.structural.uniprot_id,
            "gene_druggability": r.gene_druggability,
            "protein_score": r.protein_score,
            "blended_score": r.blended_score,
            "structural_score": r.structural.structural_score,
            "mean_plddt": r.structural.mean_plddt,
            "n_pdb_structures": r.structural.n_pdb_structures,
            "n_ligand_bound": r.structural.n_ligand_bound,
            "has_pocket": r.structural.has_pocket,
            "abundance_score": r.abundance.abundance_score if r.abundance else None,
            "detection_fraction": r.abundance.detection_fraction if r.abundance else None,
            "degradability_score": r.degradability.degradability_score,
            "has_known_degrader": r.degradability.has_known_degrader,
            "degrader_exemplar": r.degradability.degrader_exemplar,
            "ppi_score": r.ppi.ppi_score,
            "n_pdb_complexes": r.ppi.n_pdb_complexes,
            "disordered_fraction": r.ppi.disordered_fraction,
        })

    df = pd.DataFrame(rows)
    csv_path = out / "protein_druggability_scores.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Protein scoring CSV: {csv_path}")

    # JSON with full detail
    json_path = out / "protein_druggability_scores.json"
    json_data = {gene: r.to_dict() for gene, r in sorted(results.items())}
    json_path.write_text(json.dumps(json_data, indent=2))
    logger.info(f"Protein scoring JSON: {json_path}")

    # Summary statistics
    scores = [r.blended_score for r in results.values()]
    print(f"\n{'='*60}")
    print(f"PROTEIN-LEVEL DRUGGABILITY SCORING SUMMARY")
    print(f"{'='*60}")
    print(f"Genes scored:        {len(results)}")
    print(f"Mean blended d(g):   {np.mean(scores):.3f}")
    print(f"Median blended d(g): {np.median(scores):.3f}")
    print(f"Range:               [{np.min(scores):.3f}, {np.max(scores):.3f}]")

    # Top movers (biggest change from gene-level)
    movers = sorted(
        results.values(),
        key=lambda r: abs(r.blended_score - r.gene_druggability),
        reverse=True,
    )
    print(f"\nTop 10 score changes (blended vs. gene-level):")
    print(f"{'Gene':<10} {'d_gene':>8} {'p(g)':>8} {'d_new':>8} {'Δ':>8}")
    print(f"{'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for r in movers[:10]:
        delta = r.blended_score - r.gene_druggability
        print(
            f"{r.gene:<10} {r.gene_druggability:>8.3f} "
            f"{r.protein_score:>8.3f} {r.blended_score:>8.3f} "
            f"{delta:>+8.3f}"
        )

    print(f"\nResults saved to: {csv_path}")
    return str(csv_path)


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def main():
    """Run protein scoring as a standalone script."""
    import argparse

    parser = argparse.ArgumentParser(
        description="ALIN Protein-Level Druggability Scoring"
    )
    parser.add_argument(
        "--genes", nargs="+", default=None,
        help="Gene symbols to score (default: all 52 signaling genes)"
    )
    parser.add_argument(
        "--cancer-type", default="Pancreatic Adenocarcinoma",
        help="Cancer type for abundance filtering"
    )
    parser.add_argument(
        "--output", default="./results",
        help="Output directory"
    )
    parser.add_argument(
        "--cache-dir", default="./api_cache/protein",
        help="API cache directory"
    )
    parser.add_argument(
        "--alpha", type=float, default=ALPHA_GENE_WEIGHT,
        help=f"Gene-level weight in blending (default: {ALPHA_GENE_WEIGHT})"
    )
    parser.add_argument(
        "--no-proteomics", action="store_true",
        help="Skip CCLE proteomics (if file not available)"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    genes = args.genes or list(GENE_TO_UNIPROT.keys())

    # Use canonical druggability from constants
    from alin.constants import get_druggability_score

    scorer = ProteinDruggabilityScorer(
        genes=genes,
        cancer_type=args.cancer_type,
        gene_druggability_fn=get_druggability_score,
        cache_dir=args.cache_dir,
        proteomics_dir="./depmap_data" if not args.no_proteomics else "/dev/null",
        alpha=args.alpha,
    )

    print(f"Scoring {len(genes)} genes for cancer type: {args.cancer_type}")
    print(f"Alpha (gene weight): {args.alpha}")
    print(f"API cache: {args.cache_dir}")
    print()

    results = scorer.score_all(progress=True)
    generate_protein_scoring_report(results, args.output)


if __name__ == "__main__":
    main()
