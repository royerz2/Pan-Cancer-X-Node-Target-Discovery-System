#!/usr/bin/env python3
"""
Toxicity Enhancement Module for Cost Function
=============================================
Enhances toxicity scoring with external data sources.

1. OpenTargets API - off-target toxicity, safety liabilities
2. Tissue expression (GCN portal) - weight by expression in healthy tissue (placeholder)
3. FDA MedWatch ADRs - known adverse drug reactions (placeholder)
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
}


def _fetch_opentargets_safety(ensembl_id: str) -> Optional[Dict]:
    """Fetch target safety from OpenTargets GraphQL API."""
    try:
        import requests
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


def get_tissue_expression_weight(gene: str, tissue: str = "liver") -> float:
    """
    Weight toxicity by tissue expression (GCN portal placeholder).
    Higher expression in healthy tissue = higher weight (more concern).
    Returns 1.0 if data unavailable (no adjustment).
    """
    # Placeholder: GCN portal integration would fetch tissue-specific expression
    # and return weight 0.5-1.5 based on expression level
    return 1.0


def get_fda_medwatch_adrs(gene: str) -> List[str]:
    """
    Get known ADRs from FDA MedWatch (placeholder).
    Returns empty list if unavailable; otherwise list of ADR terms.
    """
    # Placeholder: FDA FAERS/MedWatch integration would query by drug/target
    # For now, return empty - DrugTargetDB toxicities serve as built-in ADR source
    return []
