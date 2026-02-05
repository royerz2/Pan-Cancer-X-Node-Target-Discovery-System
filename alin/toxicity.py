#!/usr/bin/env python3
"""
Toxicity Enhancement Module for Cost Function
=============================================
Enhances toxicity scoring with external data sources.

**Per-Target Toxicity:**
1. OpenTargets API - off-target toxicity, safety liabilities
2. Tissue expression (OpenTargets baseline expression) - weight by expression in healthy tissue
3. FDA FAERS (MedWatch) - known adverse drug reactions via OpenFDA API

**Combination-Level Toxicity (Combo-Tox):**
4. FAERS co-adverse event signals - drug pairs administered together with elevated ADR rates
5. Known contraindication graphs - drug-drug interactions from DrugBank/clinical sources
6. Overlapping toxicity profiles - additive risk when drugs share toxicities (e.g., both cause neutropenia)
"""

import json
import logging
from pathlib import Path
from collections import Counter
from itertools import combinations

import requests
from typing import Dict, List, Optional, Set, Tuple, FrozenSet

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


# =============================================================================
# COMBINATION-LEVEL TOXICITY (COMBO-TOX)
# =============================================================================

# Known drug-drug interactions / contraindications (curated from clinical sources)
# Format: frozenset({drug1, drug2}) -> {'severity': 'major'|'moderate'|'minor', 'mechanism': str}
KNOWN_DDI: Dict[FrozenSet[str], Dict] = {
    # QT prolongation combinations (additive cardiac risk)
    frozenset({'trametinib', 'dasatinib'}): {
        'severity': 'major', 'mechanism': 'QT prolongation (additive)', 
        'effect': 'Increased risk of Torsades de Pointes'
    },
    frozenset({'ribociclib', 'dasatinib'}): {
        'severity': 'major', 'mechanism': 'QT prolongation (additive)',
        'effect': 'Both cause QTc prolongation; cardiac monitoring required'
    },
    frozenset({'sotorasib', 'venetoclax'}): {
        'severity': 'moderate', 'mechanism': 'CYP3A4 interaction',
        'effect': 'Sotorasib induces CYP3A4, may reduce venetoclax exposure'
    },
    # Myelosuppression combinations (additive hematologic toxicity)
    frozenset({'palbociclib', 'venetoclax'}): {
        'severity': 'major', 'mechanism': 'Additive myelosuppression',
        'effect': 'Both cause severe neutropenia; high risk of infections'
    },
    frozenset({'palbociclib', 'dasatinib'}): {
        'severity': 'major', 'mechanism': 'Additive myelosuppression',
        'effect': 'Neutropenia + thrombocytopenia; requires dose modification'
    },
    frozenset({'ruxolitinib', 'venetoclax'}): {
        'severity': 'major', 'mechanism': 'Additive cytopenias',
        'effect': 'Both cause cytopenias; severe bone marrow suppression risk'
    },
    # Hepatotoxicity combinations
    frozenset({'sotorasib', 'alpelisib'}): {
        'severity': 'moderate', 'mechanism': 'Additive hepatotoxicity',
        'effect': 'Both cause elevated transaminases; monitor LFTs closely'
    },
    frozenset({'trametinib', 'alpelisib'}): {
        'severity': 'moderate', 'mechanism': 'Additive hepatotoxicity',
        'effect': 'Elevated risk of hepatic injury'
    },
    # GI toxicity combinations
    frozenset({'erlotinib', 'alpelisib'}): {
        'severity': 'moderate', 'mechanism': 'Additive GI toxicity',
        'effect': 'Severe diarrhea risk; prophylactic anti-diarrheals recommended'
    },
    frozenset({'sotorasib', 'napabucasin'}): {
        'severity': 'minor', 'mechanism': 'Additive GI toxicity',
        'effect': 'Both cause GI symptoms; manageable with supportive care'
    },
    # Pneumonitis combinations
    frozenset({'everolimus', 'osimertinib'}): {
        'severity': 'major', 'mechanism': 'Additive pulmonary toxicity',
        'effect': 'Both cause ILD/pneumonitis; avoid combination or close monitoring'
    },
}

# Toxicity classes for overlap detection
TOXICITY_CLASSES: Dict[str, Set[str]] = {
    'myelosuppression': {'neutropenia', 'thrombocytopenia', 'anemia', 'leukopenia', 
                         'pancytopenia', 'myelosuppression', 'bone marrow suppression'},
    'hepatotoxicity': {'hepatotoxicity', 'elevated transaminases', 'ALT increased', 
                       'AST increased', 'liver injury', 'hepatic failure'},
    'cardiotoxicity': {'QT prolongation', 'cardiotoxicity', 'heart failure', 
                       'cardiomyopathy', 'arrhythmia', 'Torsades de Pointes'},
    'pulmonary': {'pneumonitis', 'ILD', 'interstitial lung disease', 'pulmonary toxicity'},
    'gi_toxicity': {'diarrhea', 'nausea', 'vomiting', 'mucositis', 'colitis'},
    'dermatologic': {'rash', 'dermatitis', 'photosensitivity', 'hand-foot syndrome'},
    'nephrotoxicity': {'nephrotoxicity', 'renal failure', 'proteinuria', 'creatinine increased'},
    'thrombosis': {'thrombosis', 'VTE', 'PE', 'DVT', 'embolism'},
    'hemorrhage': {'hemorrhage', 'bleeding', 'GI bleeding'},
    'hyperglycemia': {'hyperglycemia', 'diabetes', 'glucose intolerance'},
}

# Drug-specific toxicity profiles (expanded for combo analysis)
DRUG_TOXICITY_PROFILE: Dict[str, Set[str]] = {
    'palbociclib': {'neutropenia', 'myelosuppression'},
    'ribociclib': {'neutropenia', 'QT prolongation', 'hepatotoxicity'},
    'abemaciclib': {'diarrhea', 'neutropenia'},
    'sotorasib': {'diarrhea', 'hepatotoxicity', 'nausea'},
    'adagrasib': {'diarrhea', 'QT prolongation', 'hepatotoxicity'},
    'vemurafenib': {'rash', 'photosensitivity', 'arthralgia', 'QT prolongation'},
    'dabrafenib': {'pyrexia', 'rash', 'arthralgia'},
    'trametinib': {'rash', 'cardiomyopathy', 'retinopathy', 'hepatotoxicity'},
    'cobimetinib': {'diarrhea', 'rash', 'photosensitivity'},
    'alpelisib': {'hyperglycemia', 'diarrhea', 'rash', 'hepatotoxicity'},
    'everolimus': {'mucositis', 'pneumonitis', 'hyperglycemia', 'myelosuppression'},
    'temsirolimus': {'mucositis', 'pneumonitis', 'hyperglycemia'},
    'erlotinib': {'rash', 'diarrhea', 'pneumonitis'},
    'gefitinib': {'rash', 'diarrhea', 'hepatotoxicity'},
    'osimertinib': {'diarrhea', 'rash', 'pneumonitis', 'QT prolongation'},
    'napabucasin': {'diarrhea', 'nausea', 'fatigue'},
    'dasatinib': {'myelosuppression', 'pleural effusion', 'QT prolongation'},
    'bosutinib': {'diarrhea', 'rash', 'hepatotoxicity', 'myelosuppression'},
    'venetoclax': {'neutropenia', 'tumor lysis syndrome', 'infections'},
    'ruxolitinib': {'myelosuppression', 'thrombocytopenia', 'infections'},
    'capmatinib': {'edema', 'nausea', 'pneumonitis'},
    'erdafitinib': {'hyperphosphatemia', 'stomatitis', 'diarrhea'},
    'trastuzumab': {'cardiotoxicity', 'infusion reactions'},
    'lapatinib': {'diarrhea', 'rash', 'hepatotoxicity'},
}


def get_known_ddi(drug1: str, drug2: str) -> Optional[Dict]:
    """
    Check for known drug-drug interaction between two drugs.
    Returns interaction details if known, None otherwise.
    """
    key = frozenset({drug1.lower(), drug2.lower()})
    return KNOWN_DDI.get(key)


def get_overlapping_toxicities(drugs: List[str]) -> Dict[str, int]:
    """
    Identify overlapping toxicity classes across multiple drugs.
    Returns dict of toxicity_class -> number of drugs sharing that toxicity.
    """
    class_counts: Counter = Counter()
    
    for drug in drugs:
        drug_lower = drug.lower()
        drug_toxicities = DRUG_TOXICITY_PROFILE.get(drug_lower, set())
        
        # Map individual toxicities to classes
        for tox in drug_toxicities:
            tox_lower = tox.lower()
            for tox_class, class_terms in TOXICITY_CLASSES.items():
                if tox_lower in class_terms or any(term in tox_lower for term in class_terms):
                    class_counts[tox_class] += 1
                    break
    
    # Return only classes with ≥2 drugs (overlapping)
    return {k: v for k, v in class_counts.items() if v >= 2}


def query_faers_coadministration(drug1: str, drug2: str, limit: int = 100) -> Optional[Dict]:
    """
    Query FDA FAERS for adverse events when two drugs are co-administered.
    Returns signal metrics if available.
    
    Note: This is a proxy for combination toxicity; elevated ADR rates for
    co-administered drugs suggest potential additive/synergistic toxicity.
    """
    OPENFDA_EVENTS = "https://api.fda.gov/drug/event.json"
    
    try:
        # Query for reports containing BOTH drugs
        search_query = (
            f'patient.drug.medicinalproduct:"{drug1}" AND '
            f'patient.drug.medicinalproduct:"{drug2}"'
        )
        params = {"search": search_query, "limit": limit}
        response = requests.get(OPENFDA_EVENTS, params=params, timeout=15)
        
        if response.status_code != 200:
            return None
            
        data = response.json()
        results = data.get("results", [])
        
        if not results:
            return None
        
        # Aggregate adverse events
        adverse_events: Counter = Counter()
        serious_count = 0
        death_count = 0
        
        for rec in results:
            # Check seriousness
            if rec.get("serious") == "1":
                serious_count += 1
            if rec.get("seriousnessdeath") == "1":
                death_count += 1
            
            # Collect reactions
            for r in rec.get("patient", {}).get("reaction", []):
                term = r.get("reactionmeddrapt")
                if term:
                    adverse_events[term] += 1
        
        return {
            "drug_pair": (drug1, drug2),
            "total_reports": len(results),
            "serious_reports": serious_count,
            "death_reports": death_count,
            "serious_rate": serious_count / len(results) if results else 0,
            "top_adverse_events": adverse_events.most_common(10),
        }
        
    except Exception as e:
        logger.debug(f"FAERS co-administration query failed for {drug1}+{drug2}: {e}")
        return None


def compute_combo_toxicity_score(
    genes: List[str],
    cache_dir: Optional[str] = None,
    use_faers: bool = False,  # FAERS queries can be slow; optional
) -> Dict:
    """
    Compute combination-level toxicity score for a set of target genes.
    
    Components:
    1. Known DDI penalty: major interactions between drug pairs
    2. Overlapping toxicity: additive risk when drugs share toxicity classes
    3. FAERS co-admin signals: elevated serious ADR rates (optional, slow)
    
    Returns:
        {
            'combo_tox_score': float (0-1, higher = more combo toxicity concern),
            'ddi_penalties': list of known DDIs found,
            'overlapping_toxicities': dict of shared toxicity classes,
            'faers_signals': list of FAERS co-admin results (if enabled),
            'component_scores': breakdown of score components,
        }
    """
    # Map genes to representative drugs
    drugs = []
    for gene in genes:
        gene_drugs = GENE_TO_DRUGS.get(gene, [])
        if gene_drugs:
            drugs.append(gene_drugs[0])  # Use first (most common) drug
    
    if len(drugs) < 2:
        return {
            'combo_tox_score': 0.0,
            'ddi_penalties': [],
            'overlapping_toxicities': {},
            'faers_signals': [],
            'component_scores': {'ddi': 0, 'overlap': 0, 'faers': 0},
        }
    
    # 1. Check known DDIs
    ddi_penalties = []
    ddi_score = 0.0
    for d1, d2 in combinations(drugs, 2):
        ddi = get_known_ddi(d1, d2)
        if ddi:
            ddi_penalties.append({
                'drugs': (d1, d2),
                'severity': ddi['severity'],
                'mechanism': ddi['mechanism'],
            })
            # Score: major=0.4, moderate=0.2, minor=0.1
            if ddi['severity'] == 'major':
                ddi_score += 0.4
            elif ddi['severity'] == 'moderate':
                ddi_score += 0.2
            else:
                ddi_score += 0.1
    ddi_score = min(1.0, ddi_score)  # Cap at 1.0
    
    # 2. Check overlapping toxicities
    overlaps = get_overlapping_toxicities(drugs)
    overlap_score = 0.0
    for tox_class, count in overlaps.items():
        # High-risk classes get higher weight
        if tox_class in ('myelosuppression', 'cardiotoxicity', 'hepatotoxicity'):
            overlap_score += 0.15 * (count - 1)  # -1 because overlap starts at 2
        elif tox_class in ('pulmonary', 'nephrotoxicity'):
            overlap_score += 0.12 * (count - 1)
        else:
            overlap_score += 0.08 * (count - 1)
    overlap_score = min(1.0, overlap_score)
    
    # 3. FAERS co-administration signals (optional)
    faers_signals = []
    faers_score = 0.0
    if use_faers:
        for d1, d2 in combinations(drugs, 2):
            signal = query_faers_coadministration(d1, d2)
            if signal and signal['total_reports'] >= 5:
                faers_signals.append(signal)
                # Score based on serious event rate
                if signal['serious_rate'] > 0.5:
                    faers_score += 0.2
                elif signal['serious_rate'] > 0.3:
                    faers_score += 0.1
        faers_score = min(1.0, faers_score)
    
    # Combine scores (weighted)
    # DDI is most reliable, overlap is heuristic, FAERS is noisy
    combo_tox_score = (
        0.5 * ddi_score +
        0.35 * overlap_score +
        0.15 * faers_score
    )
    
    return {
        'combo_tox_score': round(combo_tox_score, 3),
        'ddi_penalties': ddi_penalties,
        'overlapping_toxicities': overlaps,
        'faers_signals': faers_signals,
        'component_scores': {
            'ddi': round(ddi_score, 3),
            'overlap': round(overlap_score, 3),
            'faers': round(faers_score, 3),
        },
    }


def get_combo_tox_for_triple(
    gene1: str, gene2: str, gene3: str,
    cache_dir: Optional[str] = None,
) -> float:
    """
    Convenience function to get combo toxicity score for a triple.
    Returns float 0-1 (higher = more toxicity concern).
    """
    result = compute_combo_toxicity_score([gene1, gene2, gene3], cache_dir)
    return result['combo_tox_score']
