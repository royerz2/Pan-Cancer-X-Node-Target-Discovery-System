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
API_DELAY = 0.10  # seconds between API calls (AlphaFold/PDB handle this rate)

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
# DYNAMIC UNIPROT RESOLVER
# ============================================================================

# Runtime cache: gene → uniprot_id (or None if truly unmappable)
_DYNAMIC_UNIPROT_CACHE: Dict[str, Optional[str]] = {}
# Track genes that could not be resolved for reporting
_UNRESOLVED_GENES: Set[str] = set()

# Proteomics-file-derived Gene→UniProt mapping (12,000+ entries)
_PROTEOMICS_UNIPROT_MAP: Dict[str, str] = {}
_PROTEOMICS_UNIPROT_LOADED = False


def _load_proteomics_uniprot_map(
    data_dir: str = "./depmap_data",
) -> Dict[str, str]:
    """Extract Gene_Symbol → Uniprot_Acc from the Gygi proteomics file.

    The Gygi lab file (protein_quant_current_normalized.csv) has both
    ``Gene_Symbol`` and ``Uniprot_Acc`` columns, providing 12,000+
    high-quality gene→UniProt mappings from mass-spec identification.
    This is inserted as resolution tier 1.5 (between the static dict
    and the runtime/API cache) to cover the 96.7% of MHS-candidate
    genes that were previously falling back to flat 0.3 scores.
    """
    global _PROTEOMICS_UNIPROT_MAP, _PROTEOMICS_UNIPROT_LOADED
    if _PROTEOMICS_UNIPROT_LOADED:
        return _PROTEOMICS_UNIPROT_MAP

    _PROTEOMICS_UNIPROT_LOADED = True
    data_path = Path(data_dir)
    candidates = [
        "protein_quant_current_normalized.csv.gz",
        "protein_quant_current_normalized.csv",
        "protein_quant_current.csv.gz",
        "protein_quant_current.csv",
    ]
    for name in candidates:
        p = data_path / name
        if p.exists():
            try:
                # Only read Gene_Symbol + Uniprot_Acc — no need to load the full matrix
                df = pd.read_csv(p, usecols=['Gene_Symbol', 'Uniprot_Acc'],
                                 low_memory=False)
                df = df.dropna(subset=['Gene_Symbol', 'Uniprot_Acc'])
                # Strip isoform suffix (e.g. P21675-9 → P21675) so that
                # AlphaFold / PDB API lookups use canonical accessions
                df['Uniprot_Acc'] = df['Uniprot_Acc'].str.replace(
                    r'-\d+$', '', regex=True
                )
                # Keep first entry per gene (most abundant isoform)
                df = df.drop_duplicates(subset='Gene_Symbol', keep='first')
                mapping = dict(zip(df['Gene_Symbol'], df['Uniprot_Acc']))
                _PROTEOMICS_UNIPROT_MAP = mapping
                logger.info(
                    f"Loaded {len(mapping)} Gene→UniProt mappings from "
                    f"proteomics file ({p.name})"
                )
                return mapping
            except Exception as exc:
                logger.debug(f"Could not extract UniProt map from {p}: {exc}")
                break

    return _PROTEOMICS_UNIPROT_MAP


def resolve_uniprot_id(
    gene: str,
    cache: Optional[ProteinAPICache] = None,
    data_dir: str = "./depmap_data",
) -> Optional[str]:
    """Resolve a gene symbol to a reviewed human UniProt (Swiss-Prot) ID.

    Resolution order:
      1. Static ``GENE_TO_UNIPROT`` dictionary (79 signaling genes).
      1.5. Proteomics-file ``Uniprot_Acc`` column (12,000+ genes from
           Gygi lab mass-spec identification).
      2. In-memory runtime cache (``_DYNAMIC_UNIPROT_CACHE``).
      3. Disk cache via ``ProteinAPICache`` (30-day TTL).
      4. UniProt REST API query (``gene_exact`` + ``organism_id:9606`` +
         ``reviewed:true``).  Caches result on success.

    Returns the primary accession (e.g. ``'P00533'``) or ``None``
    if the gene cannot be mapped.
    """
    # 1. Static dict — fastest path
    if gene in GENE_TO_UNIPROT:
        return GENE_TO_UNIPROT[gene]

    # 1.5. Proteomics-file-derived mapping (12,000+ genes)
    prot_map = _load_proteomics_uniprot_map(data_dir)
    if gene in prot_map:
        uid = prot_map[gene]
        _DYNAMIC_UNIPROT_CACHE[gene] = uid
        return uid

    # 2. Runtime memory cache
    if gene in _DYNAMIC_UNIPROT_CACHE:
        return _DYNAMIC_UNIPROT_CACHE[gene]

    # 3. Disk cache
    if cache is not None:
        cached = cache.get("uniprot_resolve", gene)
        if cached is not None:
            uid = cached if cached != "__NONE__" else None
            _DYNAMIC_UNIPROT_CACHE[gene] = uid
            return uid

    # 4. UniProt REST API
    try:
        url = (
            f"{UNIPROT_API}/uniprotkb/search"
            f"?query=gene_exact:{gene}"
            f"+AND+organism_id:9606"
            f"+AND+reviewed:true"
            f"&format=json&size=1"
            f"&fields=accession,gene_names"
        )
        time.sleep(API_DELAY)
        resp = requests.get(url, timeout=15)
        if resp.ok:
            data = resp.json()
            results = data.get("results", [])
            if results:
                accession = results[0].get("primaryAccession")
                if accession:
                    _DYNAMIC_UNIPROT_CACHE[gene] = accession
                    if cache is not None:
                        cache.put("uniprot_resolve", gene, accession)
                    logger.debug(
                        f"Resolved {gene} → {accession} via UniProt API"
                    )
                    return accession
        # No result — gene doesn't have a reviewed human Swiss-Prot entry
        _DYNAMIC_UNIPROT_CACHE[gene] = None
        _UNRESOLVED_GENES.add(gene)
        if cache is not None:
            cache.put("uniprot_resolve", gene, "__NONE__")
        logger.debug(f"No Swiss-Prot entry for {gene} (organism 9606)")
        return None
    except Exception as exc:
        logger.debug(f"UniProt API error for {gene}: {exc}")
        # Don't cache failures — try again next time
        return None


def get_unresolved_genes() -> Set[str]:
    """Return the set of genes that could not be resolved to UniProt IDs.

    Useful for wet-lab gap reporting.
    """
    return _UNRESOLVED_GENES.copy()


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
    confidence_weight: float = 1.0      # sample-size confidence ∈ (0, 1]
    rna_imputed: bool = False           # True if RNA was used to rescue sparse proteomics

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
        # Return None for cached negative results (prior 404s)
        if isinstance(cached, dict) and cached.get("_miss"):
            return None
        return cached

    url = f"{ALPHAFOLD_API}/prediction/{uniprot_id}"
    try:
        _rate_limit()
        resp = requests.get(url, timeout=30)
        if resp.status_code != 200:
            logger.warning(f"AlphaFold API returned {resp.status_code} for {uniprot_id}")
            # Cache negative result to avoid repeated 404 retries
            cache.put("alphafold_plddt", uniprot_id, {"_miss": True})
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


# Minimum cell lines for full-confidence proteomics abundance scoring
ABUNDANCE_MIN_FULL_CONFIDENCE = 10


def _sigmoid_abundance(det_frac: float, mean_abund: float) -> float:
    """Continuous abundance score from detection fraction + intensity.

    Uses a shifted logistic sigmoid so the function is smooth and
    differentiable, eliminating the discontinuities of the old 4-tier
    system (which jumped at 0.2, 0.5, 0.8).

    .. math::

        s_{\\text{det}} = \\frac{1}{1 + e^{-10(f - 0.4)}}

    where *f* = detection_fraction.  The midpoint 0.4 and steepness 10
    are chosen so that f=0.2 → ~0.12, f=0.5 → ~0.73, f=0.8 → ~0.98,
    closely matching the old tiers but without jumps.

    An intensity bonus of up to +0.1 is added when the mean detected
    abundance is high (>1.0 in log2-normalised units), rewarding genes
    that are not just detected but abundant.
    """
    s_det = 1.0 / (1.0 + np.exp(-10.0 * (det_frac - 0.4)))
    # Intensity bonus: saturates at mean_abund ≈ 2.0
    intensity_bonus = 0.1 * min(1.0, max(0.0, mean_abund) / 2.0)
    return float(min(1.0, s_det + intensity_bonus))


def compute_abundance_score(
    gene: str,
    cancer_type: str,
    proteomics_df: Optional[pd.DataFrame] = None,
    cell_line_cancer_map: Optional[Dict[str, str]] = None,
    rna_expression: Optional['RNAExpression'] = None,
    concordance: Optional['RNAProteinConcordance'] = None,
) -> Optional[ProteinAbundance]:
    """
    Protein abundance from DepMap CCLE mass-spec proteomics.

    **Improvements over v1 (4-tier scoring):**

    1. *Continuous sigmoid* replaces 4 discrete tiers, eliminating score
       discontinuities at threshold boundaries.

    2. *Confidence weighting* based on sample size — when fewer than
       ``ABUNDANCE_MIN_FULL_CONFIDENCE`` cell lines have proteomics,
       the score is tagged with a reduced ``confidence_weight`` so the
       composite engine can downweight it.

    3. *RNA imputation* — when protein detection is sparse (< 5 lines)
       but RNA expression is high and concordance is ≥ moderate, the
       RNA evidence is used as a Bayesian prior, partway rescuing genes
       like FYN that are clearly transcribed but missed by TMT-MS.

       .. math::

           s_{\\text{rescued}} = \\alpha_{\\text{conf}} \\cdot s_{\\text{prot}}
                                + (1 - \\alpha_{\\text{conf}}) \\cdot s_{\\text{rna}} \\cdot w_c

       where  :math:`\\alpha_{\\text{conf}} = \\min(1, n_{\\text{detected}} / 5)`,
       :math:`s_{\\text{rna}}` is the RNA expression score, and
       :math:`w_c` is 0.8 for high concordance, 0.5 for moderate.
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

    # --- Continuous sigmoid scoring ---
    raw_score = _sigmoid_abundance(det_frac, mean_abund)

    # --- Confidence weight from sample size ---
    confidence = min(1.0, n_detected / ABUNDANCE_MIN_FULL_CONFIDENCE)

    # --- RNA imputation for sparse proteomics ---
    rna_imputed = False
    if n_detected < 5 and rna_expression is not None and concordance is not None:
        if (rna_expression.expression_score >= 0.6
                and concordance.concordance_tier in ('high', 'moderate')):
            # Concordance weighting: high → 0.8 trust, moderate → 0.5
            w_c = 0.8 if concordance.concordance_tier == 'high' else 0.5
            alpha_conf = min(1.0, n_detected / 5.0)
            rna_prior = rna_expression.expression_score * w_c
            raw_score = alpha_conf * raw_score + (1.0 - alpha_conf) * rna_prior
            rna_imputed = True
            confidence = max(confidence, 0.5)  # RNA rescue gives at least 50% confidence

    return ProteinAbundance(
        gene=gene,
        cancer_type=cancer_type,
        n_cell_lines=n_lines,
        n_detected=n_detected,
        mean_abundance=round(mean_abund, 4),
        detection_fraction=round(det_frac, 4),
        abundance_score=round(min(1.0, raw_score), 4),
        confidence_weight=round(confidence, 4),
        rna_imputed=rna_imputed,
    )


def compute_degradability_score(
    gene: str,
    uniprot_id: str,
    af_data: Optional[Dict] = None,
    degrader_targets: Optional[Dict[str, Dict[str, str]]] = None,
) -> DegradabilityScore:
    """
    PROTAC / molecular glue degradability assessment.

    Combines:
        1. Known degrader evidence from PROTAC-DB curation (0.4–0.5 bonus)
        2. Surface-exposed lysine estimate from AlphaFold
           (structured Lys residues → ubiquitin ligase recruitment sites)

    Parameters
    ----------
    degrader_targets : dict, optional
        Gene → {"status", "exemplar"} mapping.  When *None*, falls back
        to the built-in ``KNOWN_DEGRADER_TARGETS`` dictionary.

    Scoring:
        - Known clinical degrader   → 0.9
        - Known preclinical degrader → 0.7
        - No degrader but ≥5 surface Lys + high pLDDT → 0.5
        - No degrader, 3–4 surface Lys                → 0.3
        - No degrader, <3 surface Lys or disordered    → 0.1
    """
    targets = degrader_targets if degrader_targets is not None else KNOWN_DEGRADER_TARGETS
    known = targets.get(gene)

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
# DATA LOADERS — Proteomics, RNA-seq, Gygi lab, PROTAC-DB
# ============================================================================

def _load_cancer_map(data_dir: str = "./depmap_data") -> Dict[str, str]:
    """
    Load cell-line → OncotreePrimaryDisease mapping from Model.csv.

    Returns a dict keyed by **both** ModelID (ACH-00xxxx) and CCLEName
    (e.g. 'NIHOVCAR3_OVARY') so that either DepMap or Gygi cell-line
    identifiers can be looked up.
    """
    model_file = Path(data_dir) / "Model.csv"
    if not model_file.exists():
        return {}
    model_df = pd.read_csv(model_file)
    cancer_col = None
    for c in ('OncotreePrimaryDisease', 'primary_disease', 'OncotreeLineage'):
        if c in model_df.columns:
            cancer_col = c
            break
    if cancer_col is None:
        return {}

    cmap: Dict[str, str] = {}
    id_col = 'ModelID' if 'ModelID' in model_df.columns else model_df.columns[0]
    for _, row in model_df.iterrows():
        disease = row.get(cancer_col)
        if pd.isna(disease):
            continue
        disease = str(disease)
        # Key by ModelID (ACH-xxxxx)
        mid = row.get(id_col)
        if pd.notna(mid):
            cmap[str(mid)] = disease
        # Key by CCLEName (e.g. NIHOVCAR3_OVARY)
        ccle = row.get('CCLEName')
        if pd.notna(ccle):
            cmap[str(ccle)] = disease
        # Key by StrippedCellLineName (e.g. NIHOVCAR3)
        stripped = row.get('StrippedCellLineName')
        if pd.notna(stripped):
            cmap[str(stripped)] = disease

    return cmap


def load_ccle_proteomics(
    data_dir: str = "./depmap_data",
) -> Optional[Tuple[pd.DataFrame, Dict[str, str]]]:
    """
    Load CCLE proteomics data.

    Searches for (in priority order):
        1. protein_quant_current_normalized.csv.gz  (Gygi lab download)
        2. protein_quant_current_normalized.csv
        3. protein_quant_current.csv.gz
        4. protein_quant_current.csv

    The Gygi lab file (Nusinow et al. 2020) is wide format:
        First few columns = Gene_Symbol, Protein_Id, Description, etc.
        Remaining columns = cell line names → log2 normalized protein intensity

    Returns:
        (proteomics_df [cell_line × gene], cell_line_cancer_map) or None.
    """
    data_path = Path(data_dir)
    candidates = [
        "protein_quant_current_normalized.csv.gz",
        "protein_quant_current_normalized.csv",
        "protein_quant_current.csv.gz",
        "protein_quant_current.csv",
    ]
    proteomics_file = None
    for name in candidates:
        p = data_path / name
        if p.exists():
            proteomics_file = p
            break

    if proteomics_file is None:
        logger.info(
            "CCLE proteomics file not found. Download from:\n"
            f"  {GYGI_PROTEIN_URL}\n"
            f"  Place in {data_path}/\n"
            "  Protein abundance scoring will be skipped."
        )
        return None

    logger.info(f"Loading CCLE proteomics from {proteomics_file}")
    try:
        prot_df = pd.read_csv(proteomics_file, low_memory=False)

        # Detect Gygi lab format
        meta_cols = {
            c for c in prot_df.columns
            if c in (
                'Gene_Symbol', 'Protein_Id', 'Description', 'Group_ID',
                'Uniprot', 'Uniprot_Acc', 'Protein_Id_2', 'Num_Peptides',
            )
            or c.endswith('_Peptides')  # peptide-count columns
        }
        if 'Gene_Symbol' in prot_df.columns:
            import re as _re
            gene_col = 'Gene_Symbol'
            value_cols = [c for c in prot_df.columns if c not in meta_cols]
            # Gygi lab: rows = proteins, value columns = cell lines
            prot_pivot = prot_df.drop_duplicates(subset=gene_col).set_index(gene_col)[value_cols].T
            prot_pivot.index.name = 'cell_line'
            # Convert to numeric (some entries might be strings)
            prot_pivot = prot_pivot.apply(pd.to_numeric, errors='coerce')
            # Gygi column names: CELLNAME_TISSUE_TenPxNN → CELLNAME_TISSUE
            # Strip the _TenPxNN plex suffix so names match CCLEName
            _plex_re = _re.compile(r'_TenPx\d+$', _re.IGNORECASE)
            prot_pivot.index = pd.Index(
                [_plex_re.sub('', idx) for idx in prot_pivot.index],
                name='cell_line',
            )
            # If multiple plex replicates exist per cell line, average them
            if prot_pivot.index.duplicated().any():
                prot_pivot = prot_pivot.groupby(level=0).mean()
        elif 'gene_name' in prot_df.columns:
            prot_pivot = prot_df.pivot_table(
                index='cell_line', columns='gene_name',
                values='protein_expression', aggfunc='mean'
            )
        else:
            prot_df = prot_df.set_index(prot_df.columns[0])
            prot_pivot = prot_df.T
            prot_pivot.index.name = 'cell_line'

        cancer_map = _load_cancer_map(data_dir)

        logger.info(
            f"Loaded proteomics: {prot_pivot.shape[0]} cell lines × "
            f"{prot_pivot.shape[1]} proteins"
        )
        return prot_pivot, cancer_map

    except Exception as e:
        logger.warning(f"Failed to load CCLE proteomics: {e}")
        return None


def load_ccle_rnaseq(
    data_dir: str = "./depmap_data",
) -> Optional[Tuple[pd.DataFrame, Dict[str, str]]]:
    """
    Load DepMap CCLE RNA-seq expression matrix.

    Searches for (in priority order):
        1. OmicsExpressionProteinCodingGenesTPMLogp1.csv
        2. CCLE_expression.csv
        3. CCLE_RNAseq_reads.csv

    All are wide format: rows = cell lines (ModelID), cols = gene symbols
    (possibly with Entrez ID in parentheses).  Values = log2(TPM+1).

    Returns:
        (expression_df [cell_line × gene], cell_line_cancer_map) or None.
    """
    data_path = Path(data_dir)
    candidates = [
        "OmicsExpressionProteinCodingGenesTPMLogp1.csv",
        "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv",
        "CCLE_expression.csv",
        "CCLE_RNAseq_reads.csv",
    ]
    expr_file = None
    for name in candidates:
        p = data_path / name
        if p.exists():
            expr_file = p
            break

    if expr_file is None:
        logger.info(
            "CCLE RNA-seq file not found. Download from DepMap portal:\n"
            "  https://depmap.org/portal/download/ →\n"
            "  OmicsExpressionProteinCodingGenesTPMLogp1.csv\n"
            f"  Place in {data_path}/\n"
            "  RNA expression scoring will be skipped."
        )
        return None

    logger.info(f"Loading CCLE RNA-seq from {expr_file}")
    try:
        expr_df = pd.read_csv(expr_file, low_memory=False)

        # Newer DepMap format has metadata columns before gene columns:
        #   (index), SequencingID, ModelID, IsDefaultEntryForModel, ...
        # Detect and handle this format
        meta_cols = {
            'SequencingID', 'ModelID', 'IsDefaultEntryForModel',
            'ModelConditionID', 'IsDefaultEntryForMC',
        }
        has_meta = bool(meta_cols & set(expr_df.columns))

        if has_meta and 'ModelID' in expr_df.columns:
            # Set ModelID as index and drop metadata
            expr_df = expr_df.set_index('ModelID')
            drop_cols = [c for c in meta_cols if c in expr_df.columns]
            expr_df = expr_df.drop(columns=drop_cols, errors='ignore')
            # Drop any unnamed index column left over
            drop_unnamed = [c for c in expr_df.columns
                           if c.startswith('Unnamed') or c.isdigit()]
            expr_df = expr_df.drop(columns=drop_unnamed, errors='ignore')
        else:
            # Legacy format: first column is cell line ID
            if expr_df.columns[0] in ('', 'Unnamed: 0') or expr_df.iloc[:, 0].str.startswith('ACH-').all():
                expr_df = expr_df.set_index(expr_df.columns[0])
            # else already has cell line as index from index_col=0

        # Strip Entrez IDs from column names: "EGFR (1956)" → "EGFR"
        import re
        cleaned = {}
        for col in expr_df.columns:
            m = re.match(r'^([A-Za-z0-9_.\-]+)\s*\(', str(col))
            if m:
                cleaned[col] = m.group(1)
            else:
                cleaned[col] = str(col).split(' ')[0]
        expr_df = expr_df.rename(columns=cleaned)

        # Convert to numeric
        expr_df = expr_df.apply(pd.to_numeric, errors='coerce')

        # Deduplicate columns (keep first)
        expr_df = expr_df.loc[:, ~expr_df.columns.duplicated()]

        cancer_map = _load_cancer_map(data_dir)

        logger.info(
            f"Loaded RNA-seq: {expr_df.shape[0]} cell lines × "
            f"{expr_df.shape[1]} genes"
        )
        return expr_df, cancer_map

    except Exception as e:
        logger.warning(f"Failed to load CCLE RNA-seq: {e}")
        return None


def load_protacdb(
    data_dir: str = "./depmap_data",
) -> Dict[str, Dict[str, str]]:
    """
    Parse PROTAC-DB download to expand degrader target dictionary.

    Looks for 'protac_data.csv' or 'protacdb.csv' in data_dir.
    Expected columns (from PROTAC-DB v2 CSV):
        - Target_Name or Target: gene name
        - PROTAC_Name or Name: compound name
        - Clinical_Phase or Phase: approval status
        - E3_Ligase: recruited E3 ligase
        - DC50_nM: half-degradation concentration

    Returns a dict like KNOWN_DEGRADER_TARGETS: gene → {status, exemplar, source}.
    Merges with the built-in curation — database entries take precedence only
    if they have a more advanced clinical stage.
    """
    data_path = Path(data_dir)
    candidates = ["protac_data.csv", "protacdb.csv", "protac_db.csv", "PROTAC_data.csv"]
    protac_file = None
    for name in candidates:
        p = data_path / name
        if p.exists():
            protac_file = p
            break

    # Start with built-in curated set
    merged = dict(KNOWN_DEGRADER_TARGETS)

    if protac_file is None:
        logger.info(
            "PROTAC-DB CSV not found. Download from:\n"
            f"  {PROTACDB_URL}\n"
            f"  Place as {data_path / 'protac_data.csv'}\n"
            "  Using built-in curated set only."
        )
        return merged

    logger.info(f"Loading PROTAC-DB from {protac_file}")
    try:
        df = pd.read_csv(protac_file, low_memory=False)

        # Normalize column names
        col_map = {}
        for c in df.columns:
            cl = c.lower().strip()
            if 'target' in cl and 'name' in cl:
                col_map[c] = 'target'
            elif cl == 'target':
                col_map[c] = 'target'
            elif 'protac' in cl and 'name' in cl:
                col_map[c] = 'protac_name'
            elif cl == 'name':
                col_map[c] = 'protac_name'
            elif 'phase' in cl or 'clinical' in cl:
                col_map[c] = 'phase'
            elif 'dc50' in cl:
                col_map[c] = 'dc50'
        df = df.rename(columns=col_map)

        if 'target' not in df.columns:
            logger.warning("PROTAC-DB CSV missing target column — skipping")
            return merged

        stage_rank = {'approved': 4, 'phase3': 3, 'phase2': 2, 'phase1': 1, 'preclinical': 0}

        # Regex to strip mutation/variant suffixes (e.g. "EGFR E19D" → "EGFR")
        import re
        _mut_re = re.compile(r'^([A-Z][A-Z0-9]+)[\s_]+([\w/]+)$')

        for _, row in df.iterrows():
            gene_raw = str(row.get('target', '')).strip().upper()
            if not gene_raw or gene_raw == 'NAN':
                continue

            # Collect both raw name and base gene (first word) for fusions/mutations
            gene_variants = {gene_raw}
            m = _mut_re.match(gene_raw)
            if m:
                gene_variants.add(m.group(1))   # base gene, e.g. EGFR
            # Handle fusions like "BCR-ABL" — register both partners
            if '-' in gene_raw and len(gene_raw.split('-')) == 2:
                for part in gene_raw.split('-'):
                    if len(part) >= 2:
                        gene_variants.add(part)

            phase_raw = str(row.get('phase', 'preclinical')).lower().strip()
            if 'approv' in phase_raw:
                status = 'approved'
            elif 'phase' in phase_raw and ('3' in phase_raw or 'iii' in phase_raw):
                status = 'phase3'
            elif 'phase' in phase_raw and ('2' in phase_raw or 'ii' in phase_raw):
                status = 'phase2'
            elif 'phase' in phase_raw and ('1' in phase_raw or phase_raw.endswith('i')):
                status = 'phase1'
            else:
                status = 'preclinical'

            exemplar = str(row.get('protac_name', '')).strip()
            if not exemplar or exemplar.lower() == 'nan':
                exemplar = f'PROTAC compound (DB #{row.get("Compound ID", "?")})'
            source = 'PROTAC-DB'

            for gene in gene_variants:
                # Only upgrade if new entry has higher clinical stage
                if gene in merged:
                    existing_rank = stage_rank.get(merged[gene]['status'], 0)
                    new_rank = stage_rank.get(status, 0)
                    if new_rank <= existing_rank:
                        continue

                merged[gene] = {'status': status, 'exemplar': exemplar, 'source': source}

        logger.info(f"PROTAC-DB: {len(merged)} degrader targets (built-in + DB)")
        return merged

    except Exception as e:
        logger.warning(f"Failed to parse PROTAC-DB: {e}")
        return merged


# ============================================================================
# GYGI LAB DATA LOADERS — Correlation, Replicates, Mutations
# ============================================================================

def load_gygi_correlations(
    data_dir: str = "./depmap_data",
) -> Optional[Dict[str, Dict[str, float]]]:
    """Load pre-computed RNA/protein correlations from Gygi Table S4.

    The Gygi lab (Nusinow et al. 2020) computed per-gene Spearman and
    Pearson correlations between mRNA (RNA-seq) and protein (TMT mass-
    spec) abundance across 375 CCLE cell lines.  These are lab-grade
    values computed on the full matched dataset — far more reliable than
    the per-cancer-type correlations we compute on <30 cell lines.

    Returns:
        gene → {pearson, spearman} or None if file not found.
    """
    data_path = Path(data_dir)
    fpath = data_path / "Table_S4_Protein_RNA_Correlation_and_Enrichments.xlsx"
    if not fpath.exists():
        logger.info("Table S4 (Gygi correlations) not found — skipping")
        return None

    try:
        df = pd.read_excel(fpath, sheet_name='Protein RNA Correlation')
        # Detect gene symbol column
        gene_col = None
        for c in df.columns:
            if 'gene' in c.lower() and 'symbol' in c.lower():
                gene_col = c
                break
        if gene_col is None:
            # Fallback: first column is often gene symbol
            gene_col = df.columns[0]

        # Detect Spearman and Pearson columns
        spearman_col = None
        pearson_col = None
        for c in df.columns:
            cl = c.lower()
            if 'spearman' in cl:
                spearman_col = c
            elif 'pearson' in cl:
                pearson_col = c

        if spearman_col is None and pearson_col is None:
            logger.warning("Table S4: no correlation columns found")
            return None

        result: Dict[str, Dict[str, float]] = {}
        for _, row in df.iterrows():
            gene = str(row.get(gene_col, '')).strip()
            if not gene or gene == 'nan':
                continue
            entry: Dict[str, float] = {}
            if spearman_col is not None and pd.notna(row.get(spearman_col)):
                entry['spearman'] = float(row[spearman_col])
            if pearson_col is not None and pd.notna(row.get(pearson_col)):
                entry['pearson'] = float(row[pearson_col])
            if entry:
                result[gene] = entry

        logger.info(
            f"Loaded Gygi correlations for {len(result)} genes from Table S4"
        )
        return result

    except Exception as e:
        logger.warning(f"Failed to load Table S4 correlations: {e}")
        return None


def load_gygi_replicate_cv(
    data_dir: str = "./depmap_data",
) -> Optional[Dict[str, float]]:
    """Load per-gene replicate CV from Gygi Table S3 biological replicates.

    The Gygi lab measured 18 cell lines in triplicate (3 independent
    biological replicates).  The coefficient of variation (CV) across
    replicates gauges measurement reproducibility: low CV → high-
    confidence protein quantification; high CV → noisy measurement.

    We compute mean CV across all 18 cell lines for each gene, then
    convert to a confidence weight: ``w = 1 / (1 + CV)``.

    Returns:
        gene → mean_cv (across cell lines) or None if file not found.
    """
    data_path = Path(data_dir)
    fpath = data_path / "Table_S3_Biological_Replicates_Protein_Quant_Normalized.xlsx"
    if not fpath.exists():
        logger.info("Table S3 (biological replicates) not found — skipping")
        return None

    try:
        # Try named sheet first, fall back to first sheet
        try:
            df = pd.read_excel(fpath, sheet_name='Replicates Expression')
        except (ValueError, KeyError):
            df = pd.read_excel(fpath, sheet_name=0)

        # Detect gene symbol column
        gene_col = None
        for c in df.columns:
            cl = c.lower()
            if 'gene' in cl and ('symbol' in cl or 'name' in cl):
                gene_col = c
                break
        if gene_col is None:
            if 'Gene_Symbol' in df.columns:
                gene_col = 'Gene_Symbol'
            else:
                gene_col = df.columns[0]

        # Identify replicate columns.
        # Gygi Table S3 format: CELLNAME_TISSUE_TenPxNN-RN
        # e.g. "MDAMB468_BREAST_TenPx01-R3", "HEP3B217_LIVER_TenPx02-R1"
        # Group by cell_line_name (everything before the -RN suffix)
        import re
        # Match columns ending in -R1, -R2, -R3 (Gygi format)
        # or _R1, _R2, _R3 (generic format)
        rep_pattern = re.compile(r'^(.+?)[-_](R|Rep)(\d+)$', re.IGNORECASE)
        # Also skip peptide count columns (*_Peptides)
        peptide_pattern = re.compile(r'_Peptides$', re.IGNORECASE)

        # Group columns by cell line base name
        cell_line_reps: Dict[str, Dict[int, str]] = {}
        for c in df.columns:
            if peptide_pattern.search(c):
                continue
            m = rep_pattern.match(c)
            if m:
                base = m.group(1)
                rep_num = int(m.group(3))
                cell_line_reps.setdefault(base, {})[rep_num] = c

        if not cell_line_reps:
            # Fallback: any numeric column is a measurement
            logger.warning(
                "Table S3: no replicate columns detected — trying all numeric"
            )
            return None

        # For each gene, compute CV across replicates for each cell line,
        # then average across cell lines
        result: Dict[str, float] = {}
        for _, row in df.iterrows():
            gene = str(row.get(gene_col, '')).strip()
            if not gene or gene == 'nan':
                continue

            cvs = []
            for base, reps in cell_line_reps.items():
                if len(reps) < 2:
                    continue
                vals = []
                for rep_num, col_name in sorted(reps.items()):
                    v = row.get(col_name)
                    if pd.notna(v):
                        vals.append(float(v))
                if len(vals) >= 2:
                    mean_val = np.mean(vals)
                    std_val = np.std(vals, ddof=1)
                    if abs(mean_val) > 1e-9:
                        cvs.append(std_val / abs(mean_val))

            if cvs:
                result[gene] = float(np.mean(cvs))

        logger.info(
            f"Loaded replicate CV for {len(result)} genes from Table S3 "
            f"(median CV = {np.median(list(result.values())):.3f})"
        )
        return result

    except Exception as e:
        logger.warning(f"Failed to load Table S3 replicates: {e}")
        return None


def load_gygi_raw_replicate_cv(
    data_dir: str = "./depmap_data",
) -> Optional[Dict[str, float]]:
    """Load per-gene replicate CV from non-normalized biological replicate data.

    The non-normalized CSV contains raw (pre-normalization) protein
    intensities for 18 cell lines measured in triplicate across two
    ten-plexes.  Unlike the normalized Table S3, these CVs reflect
    genuine measurement variability without normalization artifacts
    (median raw CV ≈ 0.44 vs. normalized CV ≈ 1.20).

    Bridge samples (inter-plex reference standards) are excluded from
    per-gene CV computation but provide an independent quality check.

    Returns:
        gene → mean_raw_cv (across cell lines) or None if file not found.
    """
    import re as _re

    data_path = Path(data_dir)
    fpath = data_path / "ccle_biological_replicates_nonnormalized.csv"
    if not fpath.exists():
        logger.info("Non-normalized replicate CSV not found — skipping")
        return None

    try:
        df = pd.read_csv(fpath)

        # Identify data columns (exclude metadata and peptide count cols)
        data_cols = [
            c for c in df.columns
            if not c.startswith(('Protein.Id', 'Gene.Symbol',
                                 'Description', 'TenPx'))
        ]

        # Group columns by cell line (exclude bridge samples)
        cl_groups: Dict[str, list] = {}
        for c in data_cols:
            m = _re.match(r'(.+?)\.(TenPx\d+)\.(R\d+)', c)
            if m and m.group(1) != 'bridge':
                cl_groups.setdefault(m.group(1), []).append(c)

        if not cl_groups:
            logger.warning("Non-normalized CSV: no cell-line replicate columns")
            return None

        # Vectorised CV computation per cell line
        genes = df['Gene.Symbol'].values
        n_genes = len(df)
        cv_sums = np.zeros(n_genes)
        cv_counts = np.zeros(n_genes, dtype=int)

        for _cl, cols in cl_groups.items():
            vals = df[cols].values.astype(float)
            with np.errstate(invalid='ignore'):
                means = np.nanmean(vals, axis=1)
                stds = np.nanstd(vals, axis=1, ddof=1)
                n_valid = np.sum(~np.isnan(vals), axis=1)
                cvs = stds / means
                mask = (n_valid >= 2) & (means > 1e-9) & np.isfinite(cvs)
            cv_sums[mask] += cvs[mask]
            cv_counts[mask] += 1

        result: Dict[str, float] = {}
        for i in range(n_genes):
            g = genes[i]
            if cv_counts[i] > 0 and isinstance(g, str) and g != 'nan':
                result[g] = float(cv_sums[i] / cv_counts[i])

        if result:
            logger.info(
                f"Loaded raw replicate CV for {len(result)} genes from "
                f"non-normalized CSV (median CV = "
                f"{np.median(list(result.values())):.3f})"
            )
        return result if result else None

    except Exception as e:
        logger.warning(f"Failed to load non-normalized replicates: {e}")
        return None


def load_gygi_mutations(
    data_dir: str = "./depmap_data",
) -> Optional[Dict[str, List[Dict[str, Any]]]]:
    """Load mutation-protein associations from Gygi Table S7.

    The Gygi lab tested whether recurrent mutations in each gene
    associate with changes in protein abundance (via linear regression
    across CCLE lines).  Significant associations (FDR < 0.1) indicate
    that a mutation → protein-level effect exists, which is critical
    for PROTAC/degrader target validation: you want to degrade the
    *mutant* protein specifically.

    Returns:
        gene → [{'mutant_gene', 'coefficient', 'pvalue', 'fdr', 'lfdr'}]
        or None if file not found.
    """
    data_path = Path(data_dir)
    fpath = data_path / "Table_S7_Mutation_Associations.xlsx"
    if not fpath.exists():
        logger.info("Table S7 (mutation associations) not found — skipping")
        return None

    try:
        # Try named sheet first, fall back to first sheet
        try:
            df = pd.read_excel(fpath, sheet_name='Mutation Protein Associations')
        except (ValueError, KeyError):
            df = pd.read_excel(fpath, sheet_name=0)

        # Detect columns — Gygi Table S7 format:
        # 'Mutant Gene', 'Protein_Id', 'Gene Symbol',
        # 'Coefficient Estimate', 'Coefficient Std Error',
        # 'Coefficient t-Statistic', 'P-Value', 'FDR'
        gene_col = None
        mutant_col = None
        coeff_col = None
        pval_col = None
        fdr_col = None
        lfdr_col = None

        for c in df.columns:
            cl = c.lower().strip()
            if cl in ('gene symbol', 'gene_symbol'):
                if gene_col is None:
                    gene_col = c
            elif 'mutant' in cl and 'gene' in cl:
                mutant_col = c
            elif 'coefficient' in cl and ('estimate' in cl or cl in ('coefficient', 'coeff', 'beta')):
                coeff_col = c
            elif cl in ('coefficient', 'coeff', 'beta'):
                coeff_col = c
            elif cl in ('p-value', 'p_value', 'pvalue', 'p.value'):
                pval_col = c
            elif cl == 'fdr':
                fdr_col = c
            elif cl == 'lfdr':
                lfdr_col = c

        # Fallback column detection
        if gene_col is None:
            for c in df.columns:
                if 'gene' in c.lower() and 'mutant' not in c.lower():
                    gene_col = c
                    break
        if gene_col is None:
            gene_col = df.columns[0]

        result: Dict[str, List[Dict[str, Any]]] = {}
        for _, row in df.iterrows():
            gene = str(row.get(gene_col, '')).strip()
            if not gene or gene == 'nan':
                continue

            entry: Dict[str, Any] = {}
            if mutant_col:
                entry['mutant_gene'] = str(row.get(mutant_col, '')).strip()
            if coeff_col and pd.notna(row.get(coeff_col)):
                entry['coefficient'] = float(row[coeff_col])
            if pval_col and pd.notna(row.get(pval_col)):
                entry['pvalue'] = float(row[pval_col])
            if fdr_col and pd.notna(row.get(fdr_col)):
                entry['fdr'] = float(row[fdr_col])
            if lfdr_col and pd.notna(row.get(lfdr_col)):
                entry['lfdr'] = float(row[lfdr_col])

            result.setdefault(gene, []).append(entry)

        logger.info(
            f"Loaded mutation associations for {len(result)} genes "
            f"from Table S7"
        )
        return result

    except Exception as e:
        logger.warning(f"Failed to load Table S7 mutations: {e}")
        return None


# ============================================================================
# RNA EXPRESSION SCORING
# ============================================================================

def compute_rna_expression_score(
    gene: str,
    cancer_type: str,
    expression_df: Optional[pd.DataFrame] = None,
    cell_line_cancer_map: Optional[Dict[str, str]] = None,
) -> Optional[RNAExpression]:
    """
    RNA-seq expression evidence from DepMap CCLE.

    Rationale: a gene is only a viable drug target if it is expressed
    at the mRNA level in the cancer type of interest.  High expression
    correlates with functional relevance and druggability (the protein
    is present and pharmacologically accessible).

    Scoring logic (log2(TPM+1) values):
        - expression_fraction ≥ 0.8 AND mean > 4.0 → 1.0  (highly expressed)
        - expression_fraction ≥ 0.8                 → 0.85
        - expression_fraction ≥ 0.5                 → 0.6
        - expression_fraction ≥ 0.2                 → 0.3
        - expression_fraction < 0.2                 → 0.1  (rarely expressed)
        - No data                                   → None
    """
    if expression_df is None or cell_line_cancer_map is None:
        return None

    cancer_lines = [
        cl for cl, ct in cell_line_cancer_map.items()
        if ct == cancer_type and cl in expression_df.index
    ]

    if not cancer_lines or gene not in expression_df.columns:
        return None

    values = expression_df.loc[cancer_lines, gene].dropna()
    n_lines = len(cancer_lines)
    n_expressed = int((values > RNA_TPM_THRESHOLD).sum())
    mean_expr = float(values[values > RNA_TPM_THRESHOLD].mean()) if n_expressed > 0 else 0.0
    expr_frac = n_expressed / max(n_lines, 1)

    if expr_frac >= 0.8 and mean_expr >= RNA_HIGH_EXPRESSION:
        score = 1.0
    elif expr_frac >= 0.8:
        score = 0.85
    elif expr_frac >= 0.5:
        score = 0.6
    elif expr_frac >= 0.2:
        score = 0.3
    else:
        score = 0.1

    return RNAExpression(
        gene=gene,
        cancer_type=cancer_type,
        n_cell_lines=n_lines,
        n_expressed=n_expressed,
        mean_expression=round(mean_expr, 4),
        expression_fraction=round(expr_frac, 4),
        expression_score=score,
    )


# ============================================================================
# RNA / PROTEIN CONCORDANCE SCORING
# ============================================================================

def _build_ach_to_ccle_map(data_dir: str = "./depmap_data") -> Dict[str, str]:
    """
    Build ModelID (ACH-xxx) → CCLEName mapping from Model.csv.

    Needed because RNA-seq uses ACH-IDs while Gygi proteomics uses
    CCLEName (e.g. MDAMB468_BREAST) as cell-line identifiers.
    """
    model_file = Path(data_dir) / "Model.csv"
    if not model_file.exists():
        return {}
    model_df = pd.read_csv(model_file)
    if 'ModelID' not in model_df.columns or 'CCLEName' not in model_df.columns:
        return {}
    mapping: Dict[str, str] = {}
    for _, row in model_df.iterrows():
        mid = row.get('ModelID')
        ccle = row.get('CCLEName')
        if pd.notna(mid) and pd.notna(ccle):
            mapping[str(mid)] = str(ccle)
    return mapping


def compute_rna_protein_concordance(
    gene: str,
    expression_df: Optional[pd.DataFrame] = None,
    proteomics_df: Optional[pd.DataFrame] = None,
    id_map: Optional[Dict[str, str]] = None,
) -> Optional[RNAProteinConcordance]:
    """
    Spearman correlation between RNA-seq TPM and mass-spec protein
    abundance across matched cell lines.

    The Gygi lab (Nusinow et al. 2020) showed the average per-protein
    RNA/protein Spearman ρ ≈ 0.5, with wide per-gene variation.  Genes
    with high concordance (ρ ≥ 0.5) are membrane / extracellular targets
    whose protein levels are transcriptionally regulated — ideal for
    druggability because mRNA silencing (siRNA) phenotypes reliably
    predict protein-level effects.  Low-concordance genes (ρ < 0.3) are
    dominated by post-transcriptional regulation (protein complexes,
    degradation), which adds uncertainty.

    Parameters
    ----------
    id_map : dict, optional
        Mapping from expression_df index (e.g. ACH-IDs) to
        proteomics_df index (e.g. CCLEName).  If the two DataFrames
        use different cell-line ID schemes, this is required for
        matching.

    Scoring:
        - ρ ≥ 0.5  (high)        → 1.0
        - 0.3 ≤ ρ < 0.5 (moderate) → 0.6
        - ρ < 0.3  (low)         → 0.3
        - <10 matched cell lines  → None (insufficient data)
    """
    from scipy.stats import spearmanr

    if expression_df is None or proteomics_df is None:
        return None

    if gene not in expression_df.columns or gene not in proteomics_df.columns:
        return None

    # When IDs are different (ACH vs CCLEName), map expression index → proteomics index
    if id_map:
        # Re-index expression to proteomics namespace for matching
        mapped_expr = expression_df.rename(index=id_map)
        # Drop duplicate indices (multiple ACH-IDs → same CCLEName)
        mapped_expr = mapped_expr[~mapped_expr.index.duplicated(keep='first')]
        common = mapped_expr.index.intersection(proteomics_df.index)
        if gene not in mapped_expr.columns:
            return None
        rna_vals = mapped_expr.loc[common, gene].dropna()
    else:
        common = expression_df.index.intersection(proteomics_df.index)
        rna_vals = expression_df.loc[common, gene].dropna()

    prot_vals = proteomics_df.loc[common, gene].dropna()
    # Ensure both arrays have same set of cell lines
    matched = rna_vals.index.intersection(prot_vals.index)
    # Deduplicate matched (safety)
    matched = matched.drop_duplicates()

    if len(matched) < CONCORDANCE_MIN_CELL_LINES:
        return RNAProteinConcordance(
            gene=gene,
            n_matched_lines=len(matched),
            spearman_rho=0.0,
            spearman_pvalue=1.0,
            concordance_tier='insufficient',
            concordance_score=0.0,
        )

    rho, pval = spearmanr(rna_vals.loc[matched], prot_vals.loc[matched])

    if rho >= CONCORDANCE_HIGH:
        tier = 'high'
        score = 1.0
    elif rho >= CONCORDANCE_MODERATE:
        tier = 'moderate'
        score = 0.6
    else:
        tier = 'low'
        score = 0.3

    return RNAProteinConcordance(
        gene=gene,
        n_matched_lines=len(matched),
        spearman_rho=round(float(rho), 4),
        spearman_pvalue=float(pval),
        concordance_tier=tier,
        concordance_score=score,
    )


# ============================================================================
# ADAPTIVE BLENDING
# ============================================================================

# Discordance threshold: |gene - protein| > this → switch to geometric mean
DISCORDANCE_THRESHOLD = 0.4

# Low-protein veto threshold: protein_score below this triggers a penalty
LOW_PROTEIN_VETO = 0.25


def _adaptive_blend(
    gene_d: float,
    protein_score: float,
    base_alpha: float = ALPHA_GENE_WEIGHT,
) -> float:
    """Adaptive blending of gene-level and protein-level druggability.

    **Three improvements over fixed arithmetic blending:**

    1. *Discordance penalty* — when gene and protein scores disagree by
       more than ``DISCORDANCE_THRESHOLD`` (0.4), the function switches
       from arithmetic mean to geometric mean for that gene.  This
       ensures that a protein score of 0.1 (e.g. ROS1 in PDAC) can
       actually pull down a gene score of 1.0, producing ~0.40 instead
       of the old 0.64.

    2. *Low-protein veto* — when ``protein_score < LOW_PROTEIN_VETO``
       (0.25), the gene-level score is capped at 0.5.  This prevents
       the system from confidently recommending a target whose protein
       is barely detectable / structurally intractable.

    3. *Adaptive alpha* — protein_score quality adjusts the weight.
       When protein evidence is strong (> 0.7), protein weight increases
       slightly (alpha down by 0.08); when protein evidence is weak
       (< 0.35), gene-level weight dominates (alpha up by 0.05).

    Returns blended score ∈ [0, 1].
    """
    discordance = abs(gene_d - protein_score)

    # Adaptive alpha: shift weight toward whichever has stronger signal
    if protein_score > 0.7:
        alpha = base_alpha - 0.08   # trusts protein more → 0.52
    elif protein_score < 0.35:
        alpha = base_alpha + 0.05   # trusts gene more → 0.65
    else:
        alpha = base_alpha          # default 0.60

    alpha = max(0.3, min(0.75, alpha))

    if discordance > DISCORDANCE_THRESHOLD:
        # Geometric mean — naturally penalises low outlier
        # Avoid log(0): floor both at 0.01
        g_safe = max(gene_d, 0.01)
        p_safe = max(protein_score, 0.01)
        blended = (g_safe ** alpha) * (p_safe ** (1.0 - alpha))
    else:
        # Standard arithmetic mean
        blended = alpha * gene_d + (1.0 - alpha) * protein_score

    # Low-protein veto: cap the blended score
    if protein_score < LOW_PROTEIN_VETO:
        blended = min(blended, 0.5)

    return max(0.0, min(1.0, blended))


# ============================================================================
# WEIGHT CALIBRATION & SENSITIVITY ANALYSIS
# ============================================================================

def calibrate_layer_weights(
    results: Dict[str, 'ProteinDruggabilityScore'],
    n_bootstrap: int = 200,
    perturbation: float = 0.30,
    rng_seed: int = 42,
) -> Dict[str, Any]:
    """Data-driven weight calibration and robustness analysis.

    Uses gene-level druggability as an independent reference label:
    genes with approved drugs (gene_d ≥ 0.8) constitute the positive
    set; genes with gene_d ≤ 0.3 are negatives.  The protein composite
    should separate these two groups.

    **Method:**

    1. *Grid search* over weight combinations (resolution 0.05) to find
       weights maximising the AUC between positive and negative sets.
    2. *Bootstrap robustness* — perturb each weight by ±``perturbation``
       (default 30%) and recompute per-gene rankings.  Report the
       Spearman rank-stability (mean ρ between perturbed and nominal
       rankings).

    Parameters
    ----------
    results : dict
        Gene → ProteinDruggabilityScore from a completed scoring run.
    n_bootstrap : int
        Number of random weight perturbation trials.
    perturbation : float
        Maximum fractional perturbation on each weight (e.g. 0.3 = ±30%).
    rng_seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        'nominal_weights': current weights
        'optimized_weights': best weights from grid search
        'nominal_auc': AUC with current weights
        'optimized_auc': AUC with best weights
        'rank_stability_mean_rho': mean Spearman ρ across bootstrap
        'rank_stability_std_rho': std of Spearman ρ
        'weight_sensitivity': per-layer sensitivity (∂AUC/∂w_i)
    """
    from scipy.stats import spearmanr

    genes = sorted(results.keys())
    if len(genes) < 10:
        return {'error': 'Too few genes for calibration'}

    # Extract per-layer scores into matrix
    layer_names = ['structural', 'degradability', 'ppi', 'abundance',
                   'rna_expression', 'concordance']
    nominal_w = np.array([STRUCTURAL_WEIGHT, DEGRADABILITY_WEIGHT, PPI_WEIGHT,
                          ABUNDANCE_WEIGHT, RNA_EXPRESSION_WEIGHT,
                          RNA_PROTEIN_CONCORDANCE_WEIGHT])

    # Build score matrix: genes × layers
    score_matrix = np.zeros((len(genes), 6))
    available_mask = np.ones((len(genes), 6), dtype=bool)

    for i, g in enumerate(genes):
        r = results[g]
        score_matrix[i, 0] = r.structural.structural_score
        score_matrix[i, 1] = r.degradability.degradability_score
        score_matrix[i, 2] = r.ppi.ppi_score
        if r.abundance is not None:
            score_matrix[i, 3] = r.abundance.abundance_score
        else:
            available_mask[i, 3] = False
        if r.rna_expression is not None:
            score_matrix[i, 4] = r.rna_expression.expression_score
        else:
            available_mask[i, 4] = False
        if r.rna_protein_concordance is not None and r.rna_protein_concordance.concordance_tier != 'insufficient':
            score_matrix[i, 5] = r.rna_protein_concordance.concordance_score
        else:
            available_mask[i, 5] = False

    # Reference labels
    gene_d = np.array([results[g].gene_druggability for g in genes])
    positives = gene_d >= 0.8
    negatives = gene_d <= 0.3
    if positives.sum() < 3 or negatives.sum() < 3:
        return {'error': 'Too few positive/negative genes for calibration'}

    def _compute_composite(weights):
        """Compute protein composite with given weights."""
        w = np.array(weights)
        effective_w = w * available_mask
        totals = effective_w.sum(axis=1, keepdims=True)
        totals = np.maximum(totals, 1e-9)
        return (effective_w * score_matrix).sum(axis=1) / totals.ravel()

    def _auc_score(composite):
        """Simple AUC: fraction of (pos, neg) pairs correctly ordered."""
        pos_scores = composite[positives]
        neg_scores = composite[negatives]
        correct = 0
        total = len(pos_scores) * len(neg_scores)
        if total == 0:
            return 0.5
        for p in pos_scores:
            correct += (p > neg_scores).sum() + 0.5 * (p == neg_scores).sum()
        return correct / total

    # Nominal AUC
    nominal_composite = _compute_composite(nominal_w)
    nominal_auc = _auc_score(nominal_composite)

    # Grid search (resolution 0.05, constrained: each weight ∈ [0.05, 0.40])
    best_auc = nominal_auc
    best_w = nominal_w.copy()
    rng = np.random.default_rng(rng_seed)

    # Directed grid: try 5000 random weight vectors
    for _ in range(5000):
        w_trial = rng.uniform(0.05, 0.40, size=6)
        w_trial /= w_trial.sum()  # normalise to 1
        c = _compute_composite(w_trial)
        a = _auc_score(c)
        if a > best_auc:
            best_auc = a
            best_w = w_trial.copy()

    # Bootstrap robustness: perturb nominal weights
    nominal_ranks = np.argsort(np.argsort(-nominal_composite))
    rho_values = []
    for _ in range(n_bootstrap):
        perturb = 1.0 + rng.uniform(-perturbation, perturbation, size=6)
        w_perturbed = nominal_w * perturb
        w_perturbed /= w_perturbed.sum()
        c_perturbed = _compute_composite(w_perturbed)
        perturbed_ranks = np.argsort(np.argsort(-c_perturbed))
        rho, _ = spearmanr(nominal_ranks, perturbed_ranks)
        rho_values.append(rho)

    # Per-layer sensitivity: change each weight by ±10% and measure ΔAUC
    sensitivity = {}
    for j, name in enumerate(layer_names):
        w_up = nominal_w.copy()
        w_up[j] *= 1.1
        w_up /= w_up.sum()
        w_down = nominal_w.copy()
        w_down[j] *= 0.9
        w_down /= w_down.sum()
        auc_up = _auc_score(_compute_composite(w_up))
        auc_down = _auc_score(_compute_composite(w_down))
        sensitivity[name] = round((auc_up - auc_down) / 0.2, 4)  # ∂AUC/∂(10%Δw)

    return {
        'nominal_weights': {n: round(float(w), 4) for n, w in zip(layer_names, nominal_w)},
        'optimized_weights': {n: round(float(w), 4) for n, w in zip(layer_names, best_w)},
        'nominal_auc': round(nominal_auc, 4),
        'optimized_auc': round(best_auc, 4),
        'rank_stability_mean_rho': round(float(np.mean(rho_values)), 4),
        'rank_stability_std_rho': round(float(np.std(rho_values)), 4),
        'weight_sensitivity': sensitivity,
        'n_positive_genes': int(positives.sum()),
        'n_negative_genes': int(negatives.sum()),
    }


# ============================================================================
# MAIN SCORING ENGINE
# ============================================================================

class ProteinDruggabilityScorer:
    """
    Computes multi-omics druggability scores for ALIN target genes.

    Integrates 6 data layers: structural, protein abundance, degradability,
    PPI accessibility, RNA expression, and RNA/protein concordance.

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

        # Lazy-loaded data
        self._proteomics_df: Optional[pd.DataFrame] = None
        self._cancer_map: Optional[Dict[str, str]] = None
        self._proteomics_loaded = False
        self._expression_df: Optional[pd.DataFrame] = None
        self._expr_cancer_map: Optional[Dict[str, str]] = None
        self._expression_loaded = False
        self._degrader_targets: Optional[Dict[str, Dict[str, str]]] = None
        self._id_map: Optional[Dict[str, str]] = None  # ACH→CCLEName

        # Gygi lab extended data (lazy-loaded)
        self._gygi_correlations: Optional[Dict[str, Dict[str, float]]] = None
        self._gygi_correlations_loaded = False
        self._gygi_replicate_cv: Optional[Dict[str, float]] = None
        self._gygi_replicate_cv_loaded = False
        self._gygi_raw_replicate_cv: Optional[Dict[str, float]] = None
        self._gygi_raw_replicate_cv_loaded = False
        self._gygi_mutations: Optional[Dict[str, List[Dict[str, Any]]]] = None
        self._gygi_mutations_loaded = False

    def _ensure_proteomics(self):
        """Lazy-load proteomics data on first use."""
        if self._proteomics_loaded:
            return
        self._proteomics_loaded = True
        result = load_ccle_proteomics(self.proteomics_dir)
        if result is not None:
            self._proteomics_df, self._cancer_map = result

    def _ensure_expression(self):
        """Lazy-load RNA-seq data on first use."""
        if self._expression_loaded:
            return
        self._expression_loaded = True
        result = load_ccle_rnaseq(self.proteomics_dir)
        if result is not None:
            self._expression_df, self._expr_cancer_map = result

    def _ensure_degrader_targets(self):
        """Lazy-load PROTAC-DB on first use."""
        if self._degrader_targets is not None:
            return
        self._degrader_targets = load_protacdb(self.proteomics_dir)

    def _ensure_gygi_correlations(self):
        """Lazy-load Gygi lab pre-computed RNA/protein correlations."""
        if self._gygi_correlations_loaded:
            return
        self._gygi_correlations_loaded = True
        self._gygi_correlations = load_gygi_correlations(self.proteomics_dir)

    def _ensure_gygi_replicate_cv(self):
        """Lazy-load Gygi lab biological replicate CVs (both normalized and raw)."""
        if self._gygi_replicate_cv_loaded:
            return
        self._gygi_replicate_cv_loaded = True
        self._gygi_replicate_cv = load_gygi_replicate_cv(self.proteomics_dir)
        # Also load raw (non-normalized) CVs — preferred when available
        if not self._gygi_raw_replicate_cv_loaded:
            self._gygi_raw_replicate_cv_loaded = True
            self._gygi_raw_replicate_cv = load_gygi_raw_replicate_cv(
                self.proteomics_dir
            )

    def _ensure_gygi_mutations(self):
        """Lazy-load Gygi lab mutation associations."""
        if self._gygi_mutations_loaded:
            return
        self._gygi_mutations_loaded = True
        self._gygi_mutations = load_gygi_mutations(self.proteomics_dir)

    def pre_resolve_genes(self, genes: List[str]) -> Dict[str, Optional[str]]:
        """Batch-resolve UniProt IDs for a list of genes.

        This is more efficient than resolving one-at-a-time inside
        ``score_gene`` because it reuses the cache lookup path and
        batches API calls.  Call this before scoring a large gene pool.

        Returns mapping of gene → UniProt accession (or None).
        """
        resolved = {}
        to_query = []
        for g in genes:
            uid = GENE_TO_UNIPROT.get(g) or _DYNAMIC_UNIPROT_CACHE.get(g)
            if uid is not None:
                resolved[g] = uid
            elif g in _DYNAMIC_UNIPROT_CACHE:
                # Explicitly cached as None (unmappable)
                resolved[g] = None
            else:
                # Check disk cache
                cached = self.cache.get("uniprot_resolve", g) if self.cache else None
                if cached is not None:
                    uid = cached if cached != "__NONE__" else None
                    _DYNAMIC_UNIPROT_CACHE[g] = uid
                    resolved[g] = uid
                else:
                    to_query.append(g)

        if to_query:
            logger.info(
                f"Resolving {len(to_query)} gene→UniProt IDs via API "
                f"(already cached: {len(resolved)})"
            )
            for i, g in enumerate(to_query):
                uid = resolve_uniprot_id(g, self.cache)
                resolved[g] = uid
                if (i + 1) % 100 == 0:
                    logger.info(
                        f"  UniProt resolution: {i+1}/{len(to_query)} "
                        f"({sum(1 for v in resolved.values() if v is not None)} mapped)"
                    )

        n_mapped = sum(1 for v in resolved.values() if v is not None)
        n_unmapped = sum(1 for v in resolved.values() if v is None)
        logger.info(
            f"UniProt resolution complete: {n_mapped} mapped, "
            f"{n_unmapped} unmappable out of {len(genes)} genes"
        )
        return resolved

    def score_gene(self, gene: str) -> ProteinDruggabilityScore:
        """Compute the full multi-omics druggability for a single gene.

        **v3 scoring improvements (on top of v2):**

        1. *Partial scoring* — when UniProt ID is unavailable, layers
           2 (abundance), 3 (degradability from PROTAC-DB only),
           5 (RNA expression), and 6 (concordance) are still computed
           using gene symbols.  Only layers 1 (structural) and 4 (PPI)
           fall back to defaults.  This rescues the 96.7% of genes that
           were previously getting flat 0.3 scores.

        2. *Lab-grade concordance* — when Gygi Table S4 pre-computed
           Spearman correlations are available (12,000+ genes, 375 cell
           lines), they replace the per-cancer computed values.

        3. *Replicate confidence* — Gygi Table S3 biological replicate
           CV modulates abundance confidence weight.

        4. *Mutation-aware degradability* — Table S7 mutation→protein
           associations boost degradability for genes with significant
           mutation effects (FDR < 0.1).
        """
        # Try static dict first, then dynamic UniProt resolution
        uniprot_id = resolve_uniprot_id(gene, self.cache, self.proteomics_dir)
        has_uniprot = bool(uniprot_id)

        # ---- Layer 1: Structural druggability (needs UniProt) ----
        if has_uniprot:
            structural = compute_structural_score(gene, uniprot_id, self.cache)
            af_data = fetch_alphafold_plddt(uniprot_id, self.cache)
        else:
            structural = StructuralDruggability(
                gene=gene, uniprot_id="",
                mean_plddt=0.0, domain_plddt={},
                n_pdb_structures=0, n_ligand_bound=0,
                has_pocket=False, structural_score=0.3,
            )
            af_data = None

        # ---- Layer 3: Degradability ----
        self._ensure_degrader_targets()
        self._ensure_gygi_mutations()
        if has_uniprot:
            degradability = compute_degradability_score(
                gene, uniprot_id, af_data, self._degrader_targets,
            )
        else:
            # Gene-symbol-only degradability: PROTAC-DB + mutation evidence
            degradability = self._compute_degradability_gene_only(gene)

        # ---- Layer 4: PPI surface accessibility (needs UniProt) ----
        if has_uniprot:
            ppi = compute_ppi_score(gene, uniprot_id, af_data, self.cache)
        else:
            ppi = PPIAccessibility(
                gene=gene, n_pdb_complexes=0,
                has_interface_data=False, disordered_fraction=0.3,
                ppi_score=0.3,
            )

        # ---- Layer 5: RNA expression (gene symbol only) ----
        self._ensure_expression()
        rna_expr = compute_rna_expression_score(
            gene, self.cancer_type,
            self._expression_df, self._expr_cancer_map,
        )

        # ---- Layer 6: RNA/protein concordance ----
        # Prefer lab-grade Gygi Table S4 correlations when available
        self._ensure_gygi_correlations()
        concordance = self._compute_concordance_with_lab(gene)

        # ---- Layer 2: Protein abundance (gene symbol only) ----
        self._ensure_proteomics()
        self._ensure_gygi_replicate_cv()
        abundance = compute_abundance_score(
            gene, self.cancer_type,
            self._proteomics_df, self._cancer_map,
            rna_expression=rna_expr,
            concordance=concordance,
        )
        # Modulate abundance confidence by replicate CV.
        # Prefer raw (non-normalized) CV when available — it reflects
        # genuine measurement variability (median ≈ 0.44) without
        # normalization artifacts (normalized median ≈ 1.20).
        if abundance is not None:
            cv = None
            if self._gygi_raw_replicate_cv is not None:
                cv = self._gygi_raw_replicate_cv.get(gene)
            if cv is None and self._gygi_replicate_cv is not None:
                cv = self._gygi_replicate_cv.get(gene)
            if cv is not None:
                # High CV → reduce confidence; CV=0 → full confidence
                # w_rep = 1 / (1 + CV), typical range 0.5–1.0
                rep_confidence = 1.0 / (1.0 + cv)
                # Combine sample-size and replicate confidence (geometric mean)
                abundance = ProteinAbundance(
                    gene=abundance.gene,
                    cancer_type=abundance.cancer_type,
                    n_cell_lines=abundance.n_cell_lines,
                    n_detected=abundance.n_detected,
                    mean_abundance=abundance.mean_abundance,
                    detection_fraction=abundance.detection_fraction,
                    abundance_score=abundance.abundance_score,
                    confidence_weight=round(
                        (abundance.confidence_weight * rep_confidence) ** 0.5, 4
                    ),
                    rna_imputed=abundance.rna_imputed,
                )

        # ---- Confidence-weighted composite protein score ----
        # Each layer contributes (weight, score, confidence).
        # Layers with UniProt data get confidence 1.0; layers without
        # UniProt get reduced confidence (0.5) for structural/PPI defaults.
        structural_conf = 1.0 if has_uniprot else 0.5
        ppi_conf = 1.0 if has_uniprot else 0.5

        scores: List[Tuple[float, float, float]] = [
            (STRUCTURAL_WEIGHT, structural.structural_score, structural_conf),
            (DEGRADABILITY_WEIGHT, degradability.degradability_score, 1.0),
            (PPI_WEIGHT, ppi.ppi_score, ppi_conf),
        ]
        if abundance is not None:
            scores.append((ABUNDANCE_WEIGHT, abundance.abundance_score,
                           abundance.confidence_weight))
        if rna_expr is not None:
            scores.append((RNA_EXPRESSION_WEIGHT, rna_expr.expression_score, 1.0))
        if concordance is not None and concordance.concordance_tier != 'insufficient':
            scores.append((RNA_PROTEIN_CONCORDANCE_WEIGHT,
                           concordance.concordance_score, 1.0))

        total_weight = sum(w * c for w, _, c in scores)
        protein_score = sum(w * c * s for w, s, c in scores) / max(total_weight, 1e-9)
        protein_score = max(0.0, min(1.0, protein_score))

        # ---- Adaptive blending with discordance penalty ----
        gene_d = self.gene_druggability_fn(gene)
        blended = _adaptive_blend(gene_d, protein_score, self.alpha)

        return ProteinDruggabilityScore(
            gene=gene,
            structural=structural,
            abundance=abundance,
            degradability=degradability,
            ppi=ppi,
            rna_expression=rna_expr,
            rna_protein_concordance=concordance,
            protein_score=round(protein_score, 4),
            blended_score=round(blended, 4),
            gene_druggability=round(gene_d, 4),
        )

    def _compute_degradability_gene_only(self, gene: str) -> DegradabilityScore:
        """Compute degradability using only gene symbol (no AlphaFold).

        Uses PROTAC-DB known degrader status and Gygi Table S7 mutation
        associations as evidence.  Surface lysine count is unavailable
        without AlphaFold, so we use a moderate default of 3.
        """
        targets = self._degrader_targets if self._degrader_targets else KNOWN_DEGRADER_TARGETS
        known = targets.get(gene)

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
            # Without AlphaFold we can't estimate surface lysines.
            # Use a moderate baseline (0.3) that can be boosted by
            # mutation evidence from Table S7.
            score = 0.3

        # Mutation-aware boost: if Table S7 shows significant mutation→protein
        # effects (FDR < 0.1), the gene is a better PROTAC candidate because
        # the mutant protein is functionally important and worth degrading.
        if self._gygi_mutations and gene in self._gygi_mutations:
            sig_mutations = [
                m for m in self._gygi_mutations[gene]
                if m.get('fdr', 1.0) < 0.1
            ]
            if sig_mutations:
                # Boost by up to 0.15 based on strongest mutation effect
                max_coeff = max(abs(m.get('coefficient', 0)) for m in sig_mutations)
                mutation_boost = min(0.15, max_coeff * 0.1)
                score = min(1.0, score + mutation_boost)
                if not exemplar:
                    exemplar = f"mutation-associated ({len(sig_mutations)} sig.)"

        return DegradabilityScore(
            gene=gene,
            has_known_degrader=(known is not None),
            degrader_status=status,
            degrader_exemplar=exemplar,
            n_surface_lysines=0,  # unknown without AlphaFold
            degradability_score=round(score, 4),
        )

    def _compute_concordance_with_lab(self, gene: str) -> Optional[RNAProteinConcordance]:
        """Compute RNA/protein concordance, preferring lab-grade Gygi values.

        If Table S4 provides pre-computed Spearman ρ for this gene
        (computed across 375 CCLE cell lines), use that instead of
        the per-cancer-type computed value which may only have ~20
        cell lines and correspondingly higher variance.
        """
        # Check Gygi lab pre-computed correlations first
        if self._gygi_correlations and gene in self._gygi_correlations:
            lab = self._gygi_correlations[gene]
            rho = lab.get('spearman', lab.get('pearson', 0.0))

            if rho >= CONCORDANCE_HIGH:
                tier = 'high'
                score = 1.0
            elif rho >= CONCORDANCE_MODERATE:
                tier = 'moderate'
                score = 0.6
            else:
                tier = 'low'
                score = 0.3

            return RNAProteinConcordance(
                gene=gene,
                n_matched_lines=375,  # Gygi dataset covers ~375 lines
                spearman_rho=round(float(rho), 4),
                spearman_pvalue=0.0,  # lab pre-computed, assume significant
                concordance_tier=tier,
                concordance_score=score,
            )

        # Fall back to per-cancer computed concordance
        if self._id_map is None and self._expression_df is not None and self._proteomics_df is not None:
            self._id_map = _build_ach_to_ccle_map(self.proteomics_dir)
        return compute_rna_protein_concordance(
            gene, self._expression_df, self._proteomics_df, self._id_map,
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
        """Fallback when scoring completely fails (exception handler).

        Uses gene-level druggability and whatever gene-symbol-based
        data is available.  This is a last resort — the main
        ``score_gene`` path now handles partial scoring for genes
        without UniProt IDs.
        """
        gene_d = self.gene_druggability_fn(gene)
        structural = StructuralDruggability(
            gene=gene, uniprot_id="",
            mean_plddt=0.0, domain_plddt={},
            n_pdb_structures=0, n_ligand_bound=0,
            has_pocket=False, structural_score=0.3,
        )
        # Use PROTAC-DB even in fallback
        targets = self._degrader_targets if self._degrader_targets else KNOWN_DEGRADER_TARGETS
        known = targets.get(gene)
        if known:
            deg_score = 0.9 if known.get("status") in ("approved", "phase1") else 0.7
            deg_status = known.get("status", "preclinical")
            deg_exemplar = known.get("exemplar", "")
            deg_known = True
        else:
            deg_score = 0.3
            deg_status = "none"
            deg_exemplar = ""
            deg_known = False
        degradability = DegradabilityScore(
            gene=gene, has_known_degrader=deg_known,
            degrader_status=deg_status,
            degrader_exemplar=deg_exemplar,
            n_surface_lysines=0, degradability_score=deg_score,
        )
        ppi = PPIAccessibility(
            gene=gene, n_pdb_complexes=0,
            has_interface_data=False, disordered_fraction=0.3,
            ppi_score=0.3,
        )
        protein_score = 0.3
        blended = _adaptive_blend(gene_d, protein_score, self.alpha)
        return ProteinDruggabilityScore(
            gene=gene, structural=structural, abundance=None,
            degradability=degradability, ppi=ppi,
            rna_expression=None,
            rna_protein_concordance=None,
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
        row = {
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
            # RNA expression layer
            "rna_expression_score": r.rna_expression.expression_score if r.rna_expression else None,
            "rna_expression_fraction": r.rna_expression.expression_fraction if r.rna_expression else None,
            "rna_mean_tpm": r.rna_expression.mean_expression if r.rna_expression else None,
            "rna_n_cell_lines": r.rna_expression.n_cell_lines if r.rna_expression else None,
            # RNA/protein concordance layer
            "concordance_rho": r.rna_protein_concordance.spearman_rho if r.rna_protein_concordance else None,
            "concordance_pvalue": r.rna_protein_concordance.spearman_pvalue if r.rna_protein_concordance else None,
            "concordance_tier": r.rna_protein_concordance.concordance_tier if r.rna_protein_concordance else None,
            "concordance_score": r.rna_protein_concordance.concordance_score if r.rna_protein_concordance else None,
        }
        rows.append(row)

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
    print(f"MULTI-OMICS DRUGGABILITY SCORING SUMMARY")
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
        description="ALIN Multi-Omics Druggability Scoring"
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
