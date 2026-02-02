#!/usr/bin/env python3
"""
Enhanced API-Based Validators for X-Node Target Discovery
==========================================================
Real implementations of PubMed and STRING API validation.

Features:
- PubMed: Full text search, citation counts, recent publications
- STRING: Protein interactions, functional enrichment, network analysis
- BioGRID: Additional PPI validation
- Rate limiting and caching for efficiency
"""

import requests
import time
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
import logging
import xml.etree.ElementTree as ET

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# CACHING UTILITY
# ============================================================================

class APICache:
    """Simple disk-based cache for API responses"""
    
    def __init__(self, cache_dir: str = "./api_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
    def _get_cache_key(self, prefix: str, params: dict) -> str:
        """Generate cache key from parameters"""
        param_str = json.dumps(params, sort_keys=True)
        hash_val = hashlib.md5(param_str.encode()).hexdigest()[:12]
        return f"{prefix}_{hash_val}"
    
    def get(self, prefix: str, params: dict) -> Optional[dict]:
        """Get cached result if exists and not expired"""
        key = self._get_cache_key(prefix, params)
        cache_file = self.cache_dir / f"{key}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    # Check if cache is less than 7 days old
                    if time.time() - data.get('_cached_at', 0) < 7 * 24 * 3600:
                        return data.get('result')
            except:
                pass
        return None
    
    def set(self, prefix: str, params: dict, result: dict):
        """Cache result"""
        key = self._get_cache_key(prefix, params)
        cache_file = self.cache_dir / f"{key}.json"
        
        data = {
            '_cached_at': time.time(),
            'result': result
        }
        
        with open(cache_file, 'w') as f:
            json.dump(data, f)

# ============================================================================
# PUBMED API VALIDATOR
# ============================================================================

@dataclass
class PubMedResult:
    """PubMed search result"""
    total_count: int
    pmids: List[str]
    articles: List[Dict]  # Title, authors, journal, year, abstract
    query: str
    
@dataclass
class PubMedEvidence:
    """Structured evidence from PubMed"""
    total_publications: int
    recent_publications: int  # Last 5 years
    high_impact_count: int    # In top journals
    relevant_reviews: int
    combination_mentions: int  # Papers specifically about combinations
    clinical_mentions: int     # Papers with clinical data
    score: float
    key_papers: List[Dict]


class PubMedValidator:
    """Enhanced PubMed API validator with detailed analysis"""
    
    # High-impact oncology journals
    HIGH_IMPACT_JOURNALS = {
        'nature', 'science', 'cell', 'cancer cell', 'nature medicine',
        'nat med', 'cancer discov', 'cancer discovery', 'j clin oncol',
        'jco', 'lancet oncol', 'lancet oncology', 'nejm', 'n engl j med',
        'nature communications', 'nat commun', 'cancer res', 'clinical cancer res',
        'annals of oncology', 'jama oncol', 'jama oncology'
    }
    
    def __init__(self, email: str = "xnode-discovery@example.com", cache_dir: str = "./api_cache"):
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self.email = email  # Required by NCBI
        self.cache = APICache(cache_dir)
        self._rate_limit_delay = 0.4  # NCBI allows ~3 requests/second
        
    def search(self, query: str, max_results: int = 100) -> PubMedResult:
        """
        Search PubMed with full query support
        
        Returns:
            PubMedResult with counts, PMIDs, and article details
        """
        # Check cache first
        cached = self.cache.get('pubmed_search', {'query': query, 'max': max_results})
        if cached:
            return PubMedResult(**cached)
        
        try:
            # Step 1: Search for PMIDs
            search_url = f"{self.base_url}/esearch.fcgi"
            search_params = {
                'db': 'pubmed',
                'term': query,
                'retmax': max_results,
                'retmode': 'json',
                'email': self.email,
                'sort': 'relevance'
            }
            
            response = requests.get(search_url, params=search_params, timeout=15)
            time.sleep(self._rate_limit_delay)
            
            if response.status_code != 200:
                logger.warning(f"PubMed search failed: {response.status_code}")
                return PubMedResult(0, [], [], query)
            
            data = response.json()
            result = data.get('esearchresult', {})
            total_count = int(result.get('count', 0))
            pmids = result.get('idlist', [])
            
            if not pmids:
                return PubMedResult(total_count, [], [], query)
            
            # Step 2: Fetch article details
            articles = self._fetch_article_details(pmids[:50])  # Limit to 50 for speed
            
            result = PubMedResult(
                total_count=total_count,
                pmids=pmids,
                articles=articles,
                query=query
            )
            
            # Cache result
            self.cache.set('pubmed_search', {'query': query, 'max': max_results}, {
                'total_count': result.total_count,
                'pmids': result.pmids,
                'articles': result.articles,
                'query': result.query
            })
            
            return result
            
        except Exception as e:
            logger.warning(f"PubMed search error: {e}")
            return PubMedResult(0, [], [], query)
    
    def _fetch_article_details(self, pmids: List[str]) -> List[Dict]:
        """Fetch detailed article information"""
        if not pmids:
            return []
        
        try:
            fetch_url = f"{self.base_url}/efetch.fcgi"
            params = {
                'db': 'pubmed',
                'id': ','.join(pmids),
                'retmode': 'xml',
                'email': self.email
            }
            
            response = requests.get(fetch_url, params=params, timeout=30)
            time.sleep(self._rate_limit_delay)
            
            if response.status_code != 200:
                return []
            
            # Parse XML
            articles = []
            root = ET.fromstring(response.content)
            
            for article in root.findall('.//PubmedArticle'):
                try:
                    medline = article.find('.//MedlineCitation')
                    article_elem = medline.find('.//Article')
                    
                    # Get PMID
                    pmid = medline.find('PMID').text if medline.find('PMID') is not None else ''
                    
                    # Get title
                    title_elem = article_elem.find('.//ArticleTitle')
                    title = title_elem.text if title_elem is not None else ''
                    
                    # Get journal
                    journal_elem = article_elem.find('.//Journal/Title')
                    journal = journal_elem.text if journal_elem is not None else ''
                    
                    # Get year
                    year_elem = article_elem.find('.//Journal/JournalIssue/PubDate/Year')
                    year = year_elem.text if year_elem is not None else ''
                    
                    # Get abstract
                    abstract_elem = article_elem.find('.//Abstract/AbstractText')
                    abstract = abstract_elem.text if abstract_elem is not None else ''
                    
                    articles.append({
                        'pmid': pmid,
                        'title': title,
                        'journal': journal,
                        'year': year,
                        'abstract': abstract[:500] if abstract else ''  # Truncate
                    })
                except Exception as e:
                    continue
            
            return articles
            
        except Exception as e:
            logger.warning(f"Article fetch error: {e}")
            return []
    
    def validate_combination(self, targets: List[str], cancer_type: str) -> PubMedEvidence:
        """
        Comprehensive PubMed validation for a target combination
        
        Performs multiple searches:
        1. All targets + cancer type + combination
        2. Each target pair + synergy/combination
        3. Recent publications (last 5 years)
        4. High-impact journals only
        """
        scores = []
        all_articles = []
        
        # Search 1: Main combination search
        target_str = ' AND '.join([f'"{t}"' for t in targets])
        main_query = f'({target_str}) AND ("{cancer_type}" OR cancer) AND (combination OR synergy OR inhibitor)'
        main_result = self.search(main_query, max_results=100)
        
        total_pubs = main_result.total_count
        all_articles.extend(main_result.articles)
        
        # Search 2: Recent publications (last 5 years)
        recent_query = f'{main_query} AND ("2021"[Date - Publication] : "3000"[Date - Publication])'
        recent_result = self.search(recent_query, max_results=50)
        recent_count = recent_result.total_count
        
        # Search 3: Combination-specific
        combo_query = f'({target_str}) AND ("drug combination" OR "combination therapy" OR "dual inhibition" OR "triple combination")'
        combo_result = self.search(combo_query, max_results=30)
        combo_mentions = combo_result.total_count
        
        # Search 4: Clinical evidence
        clinical_query = f'({target_str}) AND (clinical trial OR patient OR phase I OR phase II OR phase III)'
        clinical_result = self.search(clinical_query, max_results=30)
        clinical_mentions = clinical_result.total_count
        
        # Count high-impact publications
        high_impact_count = sum(
            1 for a in all_articles 
            if any(j in a.get('journal', '').lower() for j in self.HIGH_IMPACT_JOURNALS)
        )
        
        # Count reviews
        review_count = sum(
            1 for a in all_articles 
            if 'review' in a.get('title', '').lower() or 'review' in a.get('abstract', '').lower()
        )
        
        # Calculate score (0-1)
        score = min(1.0, (
            0.3 * min(1.0, total_pubs / 50) +  # Total publications
            0.25 * min(1.0, recent_count / 20) +  # Recent publications
            0.2 * min(1.0, combo_mentions / 10) +  # Combination-specific
            0.15 * min(1.0, clinical_mentions / 10) +  # Clinical evidence
            0.1 * min(1.0, high_impact_count / 5)  # High-impact
        ))
        
        # Select key papers
        key_papers = sorted(
            all_articles,
            key=lambda x: (
                any(j in x.get('journal', '').lower() for j in self.HIGH_IMPACT_JOURNALS),
                x.get('year', '0')
            ),
            reverse=True
        )[:5]
        
        return PubMedEvidence(
            total_publications=total_pubs,
            recent_publications=recent_count,
            high_impact_count=high_impact_count,
            relevant_reviews=review_count,
            combination_mentions=combo_mentions,
            clinical_mentions=clinical_mentions,
            score=score,
            key_papers=key_papers
        )


# ============================================================================
# STRING DATABASE VALIDATOR
# ============================================================================

@dataclass
class STRINGInteraction:
    """Single protein-protein interaction from STRING"""
    protein_a: str
    protein_b: str
    combined_score: float
    experimental_score: float
    database_score: float
    textmining_score: float
    
@dataclass
class STRINGEvidence:
    """Structured evidence from STRING"""
    interactions: List[STRINGInteraction]
    average_score: float
    network_density: float  # How connected the proteins are
    enriched_pathways: List[Dict]  # KEGG/Reactome pathways
    enriched_processes: List[Dict]  # GO biological processes
    functional_partners: Dict[str, List[str]]  # Additional interactors for each target
    hub_genes: List[str]  # Which targets are network hubs
    score: float


class STRINGValidator:
    """Enhanced STRING database validator"""
    
    def __init__(self, species: int = 9606, cache_dir: str = "./api_cache"):
        self.base_url = "https://string-db.org/api"
        self.species = species  # 9606 = human
        self.cache = APICache(cache_dir)
        self._rate_limit_delay = 0.5
        
    def get_interactions(self, proteins: List[str], required_score: int = 400) -> List[STRINGInteraction]:
        """
        Get protein-protein interactions from STRING
        
        Args:
            proteins: Gene/protein symbols
            required_score: Minimum combined score (0-1000)
        
        Returns:
            List of interactions
        """
        cached = self.cache.get('string_network', {'proteins': sorted(proteins), 'score': required_score})
        if cached:
            return [STRINGInteraction(**i) for i in cached]
        
        try:
            url = f"{self.base_url}/json/network"
            params = {
                'identifiers': '%0d'.join(proteins),
                'species': self.species,
                'required_score': required_score,
                'caller_identity': 'xnode_discovery'
            }
            
            response = requests.get(url, params=params, timeout=15)
            time.sleep(self._rate_limit_delay)
            
            if response.status_code != 200:
                logger.warning(f"STRING API error: {response.status_code}")
                return []
            
            data = response.json()
            
            interactions = []
            for edge in data:
                interactions.append(STRINGInteraction(
                    protein_a=edge.get('preferredName_A', ''),
                    protein_b=edge.get('preferredName_B', ''),
                    combined_score=edge.get('score', 0) / 1000.0,
                    experimental_score=edge.get('escore', 0) / 1000.0,
                    database_score=edge.get('dscore', 0) / 1000.0,
                    textmining_score=edge.get('tscore', 0) / 1000.0
                ))
            
            # Cache
            self.cache.set('string_network', {'proteins': sorted(proteins), 'score': required_score},
                          [i.__dict__ for i in interactions])
            
            return interactions
            
        except Exception as e:
            logger.warning(f"STRING network error: {e}")
            return []
    
    def get_functional_enrichment(self, proteins: List[str]) -> Tuple[List[Dict], List[Dict]]:
        """
        Get functional enrichment (pathways and GO terms)
        
        Returns:
            (KEGG/Reactome pathways, GO biological processes)
        """
        cached = self.cache.get('string_enrichment', {'proteins': sorted(proteins)})
        if cached:
            return cached.get('pathways', []), cached.get('processes', [])
        
        try:
            url = f"{self.base_url}/json/enrichment"
            params = {
                'identifiers': '%0d'.join(proteins),
                'species': self.species,
                'caller_identity': 'xnode_discovery'
            }
            
            response = requests.get(url, params=params, timeout=15)
            time.sleep(self._rate_limit_delay)
            
            if response.status_code != 200:
                return [], []
            
            data = response.json()
            
            pathways = []
            processes = []
            
            for item in data:
                category = item.get('category', '')
                input_genes = item.get('inputGenes', '')
                if isinstance(input_genes, str):
                    genes_list = input_genes.split(',') if input_genes else []
                else:
                    genes_list = input_genes if input_genes else []
                
                enrichment = {
                    'term': item.get('term', ''),
                    'description': item.get('description', ''),
                    'p_value': item.get('p_value', 1.0),
                    'fdr': item.get('fdr', 1.0),
                    'genes': genes_list
                }
                
                if category in ['KEGG', 'Reactome']:
                    pathways.append(enrichment)
                elif category == 'Process':
                    processes.append(enrichment)
            
            # Sort by FDR
            pathways = sorted(pathways, key=lambda x: x['fdr'])[:10]
            processes = sorted(processes, key=lambda x: x['fdr'])[:10]
            
            # Cache
            self.cache.set('string_enrichment', {'proteins': sorted(proteins)},
                          {'pathways': pathways, 'processes': processes})
            
            return pathways, processes
            
        except Exception as e:
            logger.warning(f"STRING enrichment error: {e}")
            return [], []
    
    def get_interaction_partners(self, protein: str, limit: int = 10) -> List[str]:
        """Get top interaction partners for a protein"""
        try:
            url = f"{self.base_url}/json/interaction_partners"
            params = {
                'identifiers': protein,
                'species': self.species,
                'limit': limit,
                'caller_identity': 'xnode_discovery'
            }
            
            response = requests.get(url, params=params, timeout=10)
            time.sleep(self._rate_limit_delay)
            
            if response.status_code != 200:
                return []
            
            data = response.json()
            partners = [edge.get('preferredName_B', '') for edge in data]
            return partners
            
        except Exception as e:
            return []
    
    def validate_combination(self, targets: List[str]) -> STRINGEvidence:
        """
        Comprehensive STRING validation for target combination
        
        Analyzes:
        - Direct interactions between targets
        - Functional enrichment (shared pathways)
        - Network topology (hub genes, density)
        - Additional interaction partners
        """
        # Get interactions
        interactions = self.get_interactions(targets, required_score=400)
        
        # Calculate network metrics
        if interactions:
            avg_score = sum(i.combined_score for i in interactions) / len(interactions)
            # Network density = actual edges / possible edges
            n = len(targets)
            max_edges = n * (n - 1) / 2
            density = len(interactions) / max_edges if max_edges > 0 else 0
        else:
            avg_score = 0.0
            density = 0.0
        
        # Get enrichment
        pathways, processes = self.get_functional_enrichment(targets)
        
        # Get interaction partners for each target
        functional_partners = {}
        hub_genes = []
        
        for target in targets:
            partners = self.get_interaction_partners(target, limit=10)
            functional_partners[target] = partners
            
            # Hub genes have many partners
            if len(partners) >= 8:
                hub_genes.append(target)
        
        # Calculate overall score
        score = min(1.0, (
            0.4 * avg_score +  # Direct interaction strength
            0.2 * density +    # Network connectivity
            0.2 * min(1.0, len(pathways) / 3) +  # Pathway enrichment
            0.2 * min(1.0, len(processes) / 3)   # Process enrichment
        ))
        
        return STRINGEvidence(
            interactions=interactions,
            average_score=avg_score,
            network_density=density,
            enriched_pathways=pathways,
            enriched_processes=processes,
            functional_partners=functional_partners,
            hub_genes=hub_genes,
            score=score
        )


# ============================================================================
# BIOGRID VALIDATOR (Additional PPI source)
# ============================================================================

class BioGRIDValidator:
    """BioGRID protein interaction validator"""
    
    def __init__(self, access_key: str = None, cache_dir: str = "./api_cache"):
        # BioGRID requires access key for API
        # Get one at: https://wiki.thebiogrid.org/doku.php/biogridrest
        self.access_key = access_key
        self.base_url = "https://webservice.thebiogrid.org"
        self.cache = APICache(cache_dir)
        
    def get_interactions(self, genes: List[str]) -> List[Dict]:
        """Get interactions from BioGRID"""
        if not self.access_key:
            logger.info("BioGRID validation requires access key (thebiogrid.org)")
            return []
        
        cached = self.cache.get('biogrid', {'genes': sorted(genes)})
        if cached:
            return cached
        
        try:
            url = f"{self.base_url}/interactions"
            params = {
                'accessKey': self.access_key,
                'format': 'json',
                'geneList': '|'.join(genes),
                'searchNames': 'true',
                'taxId': 9606,  # Human
                'interSpeciesExcluded': 'true',
                'selfInteractionsExcluded': 'true',
                'includeHeader': 'true'
            }
            
            response = requests.get(url, params=params, timeout=15)
            
            if response.status_code != 200:
                return []
            
            data = response.json()
            
            interactions = []
            for key, interaction in data.items():
                if key == 'header':
                    continue
                interactions.append({
                    'gene_a': interaction.get('OFFICIAL_SYMBOL_A', ''),
                    'gene_b': interaction.get('OFFICIAL_SYMBOL_B', ''),
                    'method': interaction.get('EXPERIMENTAL_SYSTEM', ''),
                    'pmid': interaction.get('PUBMED_ID', ''),
                    'throughput': interaction.get('THROUGHPUT', '')
                })
            
            self.cache.set('biogrid', {'genes': sorted(genes)}, interactions)
            return interactions
            
        except Exception as e:
            logger.warning(f"BioGRID error: {e}")
            return []


# ============================================================================
# COMBINED API VALIDATOR
# ============================================================================

@dataclass
class APIValidationResult:
    """Combined result from all API validators"""
    pubmed_evidence: Optional[PubMedEvidence] = None
    string_evidence: Optional[STRINGEvidence] = None
    biogrid_interactions: int = 0
    overall_score: float = 0.0
    confidence: str = "unknown"
    summary: str = ""


class CombinedAPIValidator:
    """Orchestrates all API-based validators"""
    
    def __init__(self, cache_dir: str = "./api_cache", biogrid_key: str = None):
        self.pubmed = PubMedValidator(cache_dir=cache_dir)
        self.string = STRINGValidator(cache_dir=cache_dir)
        self.biogrid = BioGRIDValidator(access_key=biogrid_key, cache_dir=cache_dir)
        
    def validate(self, targets: List[str], cancer_type: str, 
                 enable_pubmed: bool = True,
                 enable_string: bool = True,
                 enable_biogrid: bool = False) -> APIValidationResult:
        """
        Run all API validations
        
        Args:
            targets: Gene symbols
            cancer_type: Cancer type name
            enable_*: Enable specific validators
        
        Returns:
            Combined validation result
        """
        result = APIValidationResult()
        scores = []
        
        # PubMed validation
        if enable_pubmed:
            logger.info(f"Validating via PubMed: {targets}")
            result.pubmed_evidence = self.pubmed.validate_combination(targets, cancer_type)
            scores.append(result.pubmed_evidence.score)
        
        # STRING validation
        if enable_string:
            logger.info(f"Validating via STRING: {targets}")
            result.string_evidence = self.string.validate_combination(targets)
            scores.append(result.string_evidence.score)
        
        # BioGRID validation
        if enable_biogrid:
            logger.info(f"Validating via BioGRID: {targets}")
            biogrid_data = self.biogrid.get_interactions(targets)
            result.biogrid_interactions = len(biogrid_data)
            if biogrid_data:
                scores.append(min(1.0, len(biogrid_data) / 20))
        
        # Calculate overall score
        if scores:
            result.overall_score = sum(scores) / len(scores)
        
        # Assign confidence
        if result.overall_score >= 0.7:
            result.confidence = "HIGH"
        elif result.overall_score >= 0.4:
            result.confidence = "MEDIUM"
        elif result.overall_score >= 0.2:
            result.confidence = "LOW"
        else:
            result.confidence = "VERY LOW"
        
        # Generate summary
        summaries = []
        if result.pubmed_evidence:
            pe = result.pubmed_evidence
            summaries.append(f"PubMed: {pe.total_publications} papers ({pe.recent_publications} recent)")
        if result.string_evidence:
            se = result.string_evidence
            summaries.append(f"STRING: {len(se.interactions)} interactions, {len(se.enriched_pathways)} pathways")
        
        result.summary = "; ".join(summaries)
        
        return result


# ============================================================================
# CLI FOR TESTING
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test API validators")
    parser.add_argument('--targets', type=str, nargs='+', default=['KRAS', 'CDK6', 'STAT3'])
    parser.add_argument('--cancer', type=str, default='Pancreatic Adenocarcinoma')
    
    args = parser.parse_args()
    
    print(f"\nTesting API validation for: {args.targets}")
    print(f"Cancer type: {args.cancer}\n")
    
    validator = CombinedAPIValidator()
    result = validator.validate(args.targets, args.cancer)
    
    print("="*60)
    print("API VALIDATION RESULTS")
    print("="*60)
    print(f"Overall Score: {result.overall_score:.3f}")
    print(f"Confidence: {result.confidence}")
    print(f"Summary: {result.summary}")
    
    if result.pubmed_evidence:
        pe = result.pubmed_evidence
        print(f"\nPubMed Details:")
        print(f"  Total publications: {pe.total_publications}")
        print(f"  Recent (5 years): {pe.recent_publications}")
        print(f"  High-impact: {pe.high_impact_count}")
        print(f"  Combination papers: {pe.combination_mentions}")
        print(f"  Clinical papers: {pe.clinical_mentions}")
        if pe.key_papers:
            print(f"  Key paper: {pe.key_papers[0].get('title', '')[:60]}...")
    
    if result.string_evidence:
        se = result.string_evidence
        print(f"\nSTRING Details:")
        print(f"  Interactions: {len(se.interactions)}")
        print(f"  Avg score: {se.average_score:.3f}")
        print(f"  Network density: {se.network_density:.3f}")
        print(f"  Hub genes: {se.hub_genes}")
        if se.enriched_pathways:
            print(f"  Top pathway: {se.enriched_pathways[0].get('description', '')[:50]}...")
