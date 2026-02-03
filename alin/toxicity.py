#!/usr/bin/env python3
"""
Toxicity Enhancement Module for Cost Function
=============================================
Enhances toxicity scoring with external data sources.

1. OpenTargets API - off-target toxicity, safety liabilities
2. Tissue expression (OpenTargets baseline expression) - weight by expression in healthy tissue
3. FDA FAERS (MedWatch) - known adverse drug reactions via OpenFDA API
"""

import json
import logging
from pathlib import Path

import requests
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# OpenTargets GraphQL endpoint
OPENTARGETS_GRAPHQL = "https://api.platform.opentargets.org/api/v4/graphql"

# Gene symbol -> Ensembl ID mapping (common cancer targets; expand as needed)
GENE_TO_ENSEMBL: Dict[str, str] = {
    'EGFR': 'ENSG00000146648', 'KRAS': 'ENSG00000133703', 'BRAF': 'ENSG00000157764',
    'PIK3CA': 'ENSG00000121879', 'TP53': 'ENSG00000141510', 'ERBB2': 'ENSG00000141736',
    'CDK4': 'ENSG00000135446', 'CDK6': 'ENSG00000105810', 'STAT3': 'ENSG00000115415',
    'SRC': 'ENSG00000197122', 'FYN': 'ENSG00000010810', 'MET': 'ENSG00000005968',
    'BCL2': 'ENSG00000171791', 'MCL1': 'ENSG00000143384', 'MTOR': 'ENSG00000198793',
    'JAK1': 'ENSG00000162434', 'JAK2': 'ENSG00000096968', 'ALK': 'ENSG00000171094',
    'MAP2K1': 'ENSG00000169032', 'MAP2K2': 'ENSG00000126934', 'FGFR1': 'ENSG00000077782',
}

# Gene symbol -> drug names (for OpenFDA FAERS queries; representative inhibitors)
GENE_TO_DRUGS: Dict[str, List[str]] = {
    'EGFR': ['erlotinib', 'gefitinib', 'osimertinib', 'cetuximab'],
    'KRAS': ['sotorasib', 'adagrasib'],
    'BRAF': ['vemurafenib', 'dabrafenib', 'encorafenib'],
    'PIK3CA': ['alpelisib', 'copanlisib'],
    'ERBB2': ['trastuzumab', 'lapatinib', 'pertuzumab'],
    'CDK4': ['palbociclib', 'ribociclib', 'abemaciclib'],
    'CDK6': ['palbociclib', 'ribociclib', 'abemaciclib'],
    'STAT3': ['napabucasin', 'stattic'],
    'MET': ['crizotinib', 'capmatinib', 'cabozantinib'],
    'BCL2': ['venetoclax'],
    'MTOR': ['everolimus', 'temsirolimus'],
    'JAK1': ['ruxolitinib'], 'JAK2': ['ruxolitinib', 'fedratinib'],
    'ALK': ['crizotinib', 'alectinib', 'ceritinib'],
    'MAP2K1': ['trametinib', 'cobimetinib'], 'MAP2K2': ['trametinib', 'cobimetinib'],
    'FGFR1': ['erdafitinib', 'pemigatinib'],
}


def _fetch_opentargets_safety(ensembl_id: str) -> Optional[Dict]:
    """Fetch target safety from OpenTargets GraphQL API."""
    try:
        query = """
        query TargetSafety($ensemblId: String!) {
          target(ensemblId: $ensemblId) {
            id
            approvedSymbol
            safetyLiabilities {
              event
              eventId
              sideEffect {
                name
              }
            }
          }
        }
        """
        resp = requests.post(
            OPENTARGETS_GRAPHQL,
            json={"query": query, "variables": {"ensemblId": ensembl_id}},
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            return data.get("data", {}).get("target")
    except Exception as e:
        logger.debug(f"OpenTargets fetch failed for {ensembl_id}: {e}")
    return None


def get_opentargets_toxicity(gene: str, cache_dir: Optional[str] = None) -> Optional[float]:
    """
    Get toxicity score (0-1) from OpenTargets safety data.
    Returns None if unavailable; otherwise 0-1 (higher = more safety liabilities).
    """
    cache_path = None
    if cache_dir:
        Path(cache_dir).mkdir(exist_ok=True)
        cache_path = Path(cache_dir) / f"opentargets_safety_{gene}.json"
        if cache_path.exists():
            try:
                data = json.loads(cache_path.read_text())
                if data and "safetyLiabilities" in data:
                    n = len(data["safetyLiabilities"])
                    return min(1.0, 0.3 + n * 0.1)
                return data.get("toxicity", 0.5)
            except Exception:
                pass
    
    ensembl_id = GENE_TO_ENSEMBL.get(gene)
    if not ensembl_id:
        return None
    
    target_data = _fetch_opentargets_safety(ensembl_id)
    if not target_data:
        return None
    
    liabilities = target_data.get("safetyLiabilities") or []
    n = len(liabilities)
    toxicity = min(1.0, 0.3 + n * 0.1)
    
    if cache_path:
        try:
            cache_path.write_text(json.dumps({
                "gene": gene,
                "safetyLiabilities": liabilities,
                "toxicity": toxicity,
            }, indent=2))
        except Exception:
            pass
    
    return toxicity


def _fetch_opentargets_expression(ensembl_id: str) -> Optional[Dict]:
    """Fetch baseline (tissue) expression from OpenTargets GraphQL."""
    try:
        query = """
        query TargetExpression($ensemblId: String!) {
          target(ensemblId: $ensemblId) {
            id
            approvedSymbol
            expression {
              tissue {
                id
                name
              }
              level
              unit
            }
          }
        }
        """
        resp = requests.post(
            OPENTARGETS_GRAPHQL,
            json={"query": query, "variables": {"ensemblId": ensembl_id}},
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            return data.get("data", {}).get("target")
    except Exception as e:
        logger.debug(f"OpenTargets expression fetch failed for {ensembl_id}: {e}")
    return None


def get_tissue_expression_weight(gene: str, tissue: str = "liver") -> float:
    """
    Weight toxicity by tissue expression (OpenTargets baseline expression).
    Higher expression in the specified healthy tissue = higher weight (more concern).
    Returns a value in [0.5, 1.5]; 1.0 if data unavailable (no adjustment).
    """
    ensembl_id = GENE_TO_ENSEMBL.get(gene)
    if not ensembl_id:
        return 1.0
    target_data = _fetch_opentargets_expression(ensembl_id)
    if not target_data or not target_data.get("expression"):
        return 1.0
    tissue_lower = tissue.lower()
    for item in target_data.get("expression", []):
        t = item.get("tissue") or {}
        name = (t.get("name") or "").lower()
        if tissue_lower in name or name in tissue_lower:
            level = item.get("level")
            if level is not None:
                # Normalize: assume level in TPM-like scale; map to weight 0.5--1.5
                try:
                    x = float(level)
                    weight = 0.5 + min(1.0, x / 100.0)
                    return round(weight, 2)
                except (TypeError, ValueError):
                    pass
            break
    return 1.0


def get_fda_medwatch_adrs(gene: str, limit_per_drug: int = 50) -> List[str]:
    """
    Get known ADRs from FDA FAERS (OpenFDA API) for drugs targeting the gene.
    Returns list of unique MedDRA preferred terms; empty if unavailable.
    """
    drugs = GENE_TO_DRUGS.get(gene, [])
    if not drugs:
        return []
    OPENFDA_EVENTS = "https://api.fda.gov/drug/event.json"
    seen: Set[str] = set()
    for drug in drugs[:5]:  # cap to 5 drugs per gene
        try:
            # OpenFDA: search by medicinal product (drug name)
            params = {
                "search": f'patient.drug.medicinalproduct:"{drug}"',
                "limit": limit_per_drug,
            }
            response = requests.get(OPENFDA_EVENTS, params=params, timeout=10)
            if response.status_code != 200:
                continue
            data = response.json()
            for rec in data.get("results", []):
                for r in rec.get("patient", {}).get("reaction", []):
                    term = r.get("reactionmeddrapt")
                    if term and term.strip():
                        seen.add(term.strip())
        except Exception as e:
            logger.debug(f"OpenFDA query failed for {drug}: {e}")
    return sorted(seen)
