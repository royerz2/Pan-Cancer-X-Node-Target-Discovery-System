#!/usr/bin/env python3
"""
Clinical Trial Matcher for ALIN Framework (Adaptive Lethal Intersection Network)
===================================================
Searches ClinicalTrials.gov for trials matching discovered drug combinations.

Features:
- Search by drug names and cancer types
- Match single, double, and triple combinations
- Identify gaps (no existing trials = opportunity!)
- Export matched trials with status and phase
"""

import pandas as pd
import requests
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
import logging
from collections import defaultdict

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs): return x

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# DRUG NAME MAPPINGS
# ============================================================================

# Map gene targets to their drug names (for searching trials)
TARGET_TO_DRUGS = {
    'KRAS': ['sotorasib', 'adagrasib', 'MRTX849', 'AMG510', 'KRAS G12C'],
    'BRAF': ['vemurafenib', 'dabrafenib', 'encorafenib', 'BRAF inhibitor'],
    'MAP2K1': ['trametinib', 'cobimetinib', 'binimetinib', 'MEK inhibitor', 'MEK1'],
    'MEK': ['trametinib', 'cobimetinib', 'binimetinib', 'MEK inhibitor'],
    'EGFR': ['erlotinib', 'gefitinib', 'osimertinib', 'afatinib', 'EGFR inhibitor'],
    'CDK4': ['palbociclib', 'ribociclib', 'abemaciclib', 'CDK4/6 inhibitor'],
    'CDK6': ['palbociclib', 'ribociclib', 'abemaciclib', 'CDK4/6 inhibitor'],
    'CDK2': ['dinaciclib', 'CDK2 inhibitor'],
    'MET': ['capmatinib', 'tepotinib', 'crizotinib', 'MET inhibitor', 'c-MET'],
    'STAT3': ['napabucasin', 'TTI-101', 'STAT3 inhibitor', 'BBI608'],
    'FGFR1': ['erdafitinib', 'pemigatinib', 'FGFR inhibitor'],
    'FGFR2': ['erdafitinib', 'pemigatinib', 'FGFR inhibitor'],
    'BCL2': ['venetoclax', 'navitoclax', 'BCL-2 inhibitor'],
    'MCL1': ['AMG-176', 'S64315', 'MCL-1 inhibitor'],
    'PIK3CA': ['alpelisib', 'idelalisib', 'copanlisib', 'PI3K inhibitor'],
    'AKT1': ['capivasertib', 'ipatasertib', 'AKT inhibitor'],
    'MTOR': ['everolimus', 'temsirolimus', 'mTOR inhibitor'],
    'SRC': ['dasatinib', 'bosutinib', 'saracatinib', 'SRC inhibitor'],
    'FYN': ['dasatinib', 'saracatinib'],
    'JAK1': ['ruxolitinib', 'tofacitinib', 'baricitinib', 'JAK inhibitor'],
    'JAK2': ['ruxolitinib', 'fedratinib', 'JAK2 inhibitor'],
    'ALK': ['crizotinib', 'alectinib', 'brigatinib', 'lorlatinib', 'ALK inhibitor'],
    'RET': ['selpercatinib', 'pralsetinib', 'RET inhibitor'],
    'ERBB2': ['trastuzumab', 'pertuzumab', 'lapatinib', 'HER2', 'ERBB2'],
}

# Cancer type to search terms
CANCER_SEARCH_TERMS = {
    'Pancreatic Adenocarcinoma': ['pancreatic cancer', 'pancreatic adenocarcinoma', 'PDAC'],
    'Non-Small Cell Lung Cancer': ['NSCLC', 'non-small cell lung cancer', 'lung adenocarcinoma'],
    'Melanoma': ['melanoma', 'metastatic melanoma'],
    'Colorectal Adenocarcinoma': ['colorectal cancer', 'colon cancer', 'CRC'],
    'Ovarian Epithelial Tumor': ['ovarian cancer', 'ovarian carcinoma'],
    'Breast Invasive Carcinoma': ['breast cancer', 'invasive breast carcinoma'],
    'Acute Myeloid Leukemia': ['AML', 'acute myeloid leukemia'],
    'Anaplastic Thyroid Cancer': ['anaplastic thyroid', 'thyroid cancer'],
    'Pleural Mesothelioma': ['mesothelioma', 'pleural mesothelioma'],
    'Hepatocellular Carcinoma': ['hepatocellular carcinoma', 'HCC', 'liver cancer'],
}

@dataclass
class ClinicalTrial:
    """Represents a clinical trial"""
    nct_id: str
    title: str
    status: str
    phase: str
    conditions: List[str]
    interventions: List[str]
    start_date: str = ""
    completion_date: str = ""
    enrollment: int = 0
    url: str = ""
    
@dataclass  
class TrialMatch:
    """Match between our combination and a clinical trial"""
    cancer_type: str
    our_targets: Tuple[str, str, str]
    our_drugs: Tuple[str, str, str]
    trial: ClinicalTrial
    match_type: str  # 'exact_triple', 'double', 'single', 'cancer_only'
    matched_drugs: List[str]
    match_score: float  # 0-1, higher = better match


class ClinicalTrialMatcher:
    """Search and match clinical trials to our combinations"""
    
    def __init__(self, cache_dir: str = "./trial_cache"):
        self.api_base = "https://clinicaltrials.gov/api/v2/studies"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self._cache = {}
        
    def search_trials(self, query: str, max_results: int = 50) -> List[ClinicalTrial]:
        """
        Search ClinicalTrials.gov API
        
        Args:
            query: Search query string
            max_results: Maximum number of results
            
        Returns:
            List of ClinicalTrial objects
        """
        # Check cache first - sanitize filename (remove/replace problematic chars)
        cache_key = query.lower().replace(' ', '_').replace('/', '_').replace('\\', '_')
        cache_key = ''.join(c if c.isalnum() or c == '_' else '_' for c in cache_key)[:80]
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                cached = json.load(f)
                return [ClinicalTrial(**t) for t in cached]
        
        trials = []
        
        try:
            params = {
                'query.term': query,
                'filter.overallStatus': 'RECRUITING,ACTIVE_NOT_RECRUITING,ENROLLING_BY_INVITATION,COMPLETED',
                'pageSize': min(max_results, 100),
                'format': 'json'
            }
            
            response = requests.get(self.api_base, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                studies = data.get('studies', [])
                
                for study in studies:
                    protocol = study.get('protocolSection', {})
                    id_module = protocol.get('identificationModule', {})
                    status_module = protocol.get('statusModule', {})
                    design_module = protocol.get('designModule', {})
                    conditions_module = protocol.get('conditionsModule', {})
                    interventions_module = protocol.get('armsInterventionsModule', {})
                    
                    # Extract interventions
                    interventions = []
                    for intervention in interventions_module.get('interventions', []):
                        interventions.append(intervention.get('name', ''))
                    
                    trial = ClinicalTrial(
                        nct_id=id_module.get('nctId', ''),
                        title=id_module.get('briefTitle', ''),
                        status=status_module.get('overallStatus', ''),
                        phase=','.join(design_module.get('phases', ['Unknown'])),
                        conditions=conditions_module.get('conditions', []),
                        interventions=interventions,
                        start_date=status_module.get('startDateStruct', {}).get('date', ''),
                        enrollment=design_module.get('enrollmentInfo', {}).get('count', 0),
                        url=f"https://clinicaltrials.gov/study/{id_module.get('nctId', '')}"
                    )
                    trials.append(trial)
                
                # Cache results
                with open(cache_file, 'w') as f:
                    json.dump([t.__dict__ for t in trials], f)
                    
        except Exception as e:
            logger.warning(f"Trial search failed for '{query}': {e}")
        
        return trials
    
    def find_matching_trials(self, 
                             targets: Tuple[str, str, str],
                             drugs: Tuple[str, str, str],
                             cancer_type: str,
                             delay: float = 0.5) -> List[TrialMatch]:
        """
        Find trials matching a triple combination
        
        Searches for:
        1. All three drugs together (exact match)
        2. Pairs of drugs
        3. Single drugs in the cancer type
        """
        matches = []
        
        # Get search terms for cancer
        cancer_terms = CANCER_SEARCH_TERMS.get(cancer_type, [cancer_type])
        
        # Build drug search terms
        all_drug_terms = []
        for target, drug in zip(targets, drugs):
            terms = TARGET_TO_DRUGS.get(target, [drug])
            all_drug_terms.append(terms)
        
        # Search 1: Triple combination
        for cancer_term in cancer_terms[:1]:  # Use first cancer term
            query = f"{drugs[0]} AND {drugs[1]} AND {drugs[2]} AND {cancer_term}"
            trials = self.search_trials(query, max_results=20)
            
            for trial in trials:
                matches.append(TrialMatch(
                    cancer_type=cancer_type,
                    our_targets=targets,
                    our_drugs=drugs,
                    trial=trial,
                    match_type='exact_triple',
                    matched_drugs=list(drugs),
                    match_score=1.0
                ))
            
            time.sleep(delay)
        
        # Search 2: Double combinations (if no triple found)
        if len(matches) == 0:
            pairs = [
                (drugs[0], drugs[1]),
                (drugs[0], drugs[2]),
                (drugs[1], drugs[2])
            ]
            
            for d1, d2 in pairs:
                for cancer_term in cancer_terms[:1]:
                    query = f"{d1} AND {d2} AND {cancer_term}"
                    trials = self.search_trials(query, max_results=10)
                    
                    for trial in trials:
                        # Check if not already matched
                        existing_ids = [m.trial.nct_id for m in matches]
                        if trial.nct_id not in existing_ids:
                            matches.append(TrialMatch(
                                cancer_type=cancer_type,
                                our_targets=targets,
                                our_drugs=drugs,
                                trial=trial,
                                match_type='double',
                                matched_drugs=[d1, d2],
                                match_score=0.66
                            ))
                    
                    time.sleep(delay)
        
        # Search 3: Single drugs (if still few matches)
        if len(matches) < 3:
            for drug in drugs:
                for cancer_term in cancer_terms[:1]:
                    query = f"{drug} AND {cancer_term}"
                    trials = self.search_trials(query, max_results=5)
                    
                    for trial in trials:
                        existing_ids = [m.trial.nct_id for m in matches]
                        if trial.nct_id not in existing_ids:
                            matches.append(TrialMatch(
                                cancer_type=cancer_type,
                                our_targets=targets,
                                our_drugs=drugs,
                                trial=trial,
                                match_type='single',
                                matched_drugs=[drug],
                                match_score=0.33
                            ))
                    
                    time.sleep(delay)
        
        return matches
    
    def match_all_combinations(self, 
                               combinations_file: str,
                               max_combos: int = None,
                               delay: float = 1.0) -> Dict[str, List[TrialMatch]]:
        """
        Match all combinations from a CSV file to clinical trials
        
        Args:
            combinations_file: Path to triple_combinations.csv
            max_combos: Maximum number of combinations to search
            delay: Delay between API calls (rate limiting)
            
        Returns:
            Dict mapping cancer_type to list of TrialMatches
        """
        df = pd.read_csv(combinations_file)
        
        if max_combos:
            df = df.head(max_combos)
        
        all_matches = {}
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Searching trials"):
            cancer = row['Cancer_Type']
            targets = (row['Target_1'], row['Target_2'], row['Target_3'])
            drugs = (row['Drug_1'], row['Drug_2'], row['Drug_3'])
            
            matches = self.find_matching_trials(targets, drugs, cancer, delay=delay)
            all_matches[cancer] = matches
            
            logger.info(f"{cancer}: Found {len(matches)} matching trials")
        
        return all_matches


def generate_trial_report(all_matches: Dict[str, List[TrialMatch]]) -> str:
    """Generate formatted report of trial matches"""
    
    report = """
================================================================================
CLINICAL TRIAL MATCHING REPORT
================================================================================
ALIN Framework (Adaptive Lethal Intersection Network)
Generated: 2026-02-01

This report matches our discovered triple combinations against existing
clinical trials on ClinicalTrials.gov to identify:
  - Combinations already being tested (validation)
  - Gaps where new trials could be proposed (opportunity)

================================================================================
"""
    
    # Summary stats
    total_combos = len(all_matches)
    combos_with_exact = sum(1 for m in all_matches.values() if any(x.match_type == 'exact_triple' for x in m))
    combos_with_double = sum(1 for m in all_matches.values() if any(x.match_type == 'double' for x in m))
    combos_no_match = sum(1 for m in all_matches.values() if len(m) == 0)
    
    report += f"""
SUMMARY:
--------
Total combinations searched: {total_combos}
Exact triple matches found:  {combos_with_exact}
Double combination matches:  {combos_with_double}
NO matches (OPPORTUNITY!):   {combos_no_match}

"""
    
    # Exact matches (validation)
    report += """
================================================================================
EXACT TRIPLE MATCHES (Our combinations already in trials!)
================================================================================
"""
    
    for cancer, matches in all_matches.items():
        exact = [m for m in matches if m.match_type == 'exact_triple']
        if exact:
            m = exact[0]
            report += f"""
{cancer}
  Our combo: {' + '.join(m.our_drugs)}
  Trial: {m.trial.nct_id} - {m.trial.title[:60]}...
  Phase: {m.trial.phase} | Status: {m.trial.status}
  URL: {m.trial.url}
"""
    
    # No matches (opportunity)
    report += """
================================================================================
NO MATCHES FOUND - NOVEL COMBINATION OPPORTUNITIES!
================================================================================
These combinations have no existing trials = potential for new research!

"""
    
    for cancer, matches in all_matches.items():
        if len(matches) == 0:
            report += f"  ★ {cancer}\n"
    
    # Double matches
    report += """
================================================================================
PARTIAL MATCHES (2 of 3 drugs being tested)
================================================================================
"""
    
    for cancer, matches in all_matches.items():
        doubles = [m for m in matches if m.match_type == 'double']
        if doubles and not any(m.match_type == 'exact_triple' for m in matches):
            m = doubles[0]
            report += f"""
{cancer}
  Our combo: {' + '.join(m.our_drugs)}
  Partial match: {' + '.join(m.matched_drugs)}
  Trial: {m.trial.nct_id} ({m.trial.phase})
  Missing drug: {set(m.our_drugs) - set(m.matched_drugs)}
"""
    
    return report


def export_trial_matches(all_matches: Dict[str, List[TrialMatch]], output_path: Path):
    """Export trial matches to CSV and report"""
    
    output_path.mkdir(exist_ok=True, parents=True)
    
    # CSV export
    rows = []
    for cancer, matches in all_matches.items():
        if matches:
            for m in matches:
                rows.append({
                    'Cancer_Type': cancer,
                    'Our_Targets': ' + '.join(m.our_targets),
                    'Our_Drugs': ' + '.join(m.our_drugs),
                    'Match_Type': m.match_type,
                    'Match_Score': m.match_score,
                    'Matched_Drugs': ' + '.join(m.matched_drugs),
                    'NCT_ID': m.trial.nct_id,
                    'Trial_Title': m.trial.title,
                    'Phase': m.trial.phase,
                    'Status': m.trial.status,
                    'URL': m.trial.url
                })
        else:
            rows.append({
                'Cancer_Type': cancer,
                'Our_Targets': '',
                'Our_Drugs': '',
                'Match_Type': 'NO_MATCH',
                'Match_Score': 0,
                'Matched_Drugs': '',
                'NCT_ID': '',
                'Trial_Title': 'NO EXISTING TRIALS - NOVEL OPPORTUNITY',
                'Phase': '',
                'Status': '',
                'URL': ''
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path / "clinical_trial_matches.csv", index=False)
    
    # Report
    report = generate_trial_report(all_matches)
    with open(output_path / "clinical_trial_report.txt", 'w') as f:
        f.write(report)
    
    logger.info(f"Trial matches exported to {output_path}")


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Match X-Node combinations to clinical trials")
    parser.add_argument('--combinations', type=str, default='results_triples/triple_combinations.csv',
                        help='Path to triple combinations CSV')
    parser.add_argument('--output', type=str, default='clinical_trials',
                        help='Output directory')
    parser.add_argument('--max', type=int, default=20,
                        help='Maximum combinations to search (rate limiting)')
    parser.add_argument('--delay', type=float, default=1.0,
                        help='Delay between API calls in seconds')
    
    args = parser.parse_args()
    
    matcher = ClinicalTrialMatcher()
    
    print(f"\nSearching clinical trials for top {args.max} combinations...")
    print("This may take a few minutes due to API rate limiting.\n")
    
    matches = matcher.match_all_combinations(
        args.combinations,
        max_combos=args.max,
        delay=args.delay
    )
    
    export_trial_matches(matches, Path(args.output))
    
    # Print summary
    print("\n" + "="*60)
    print("CLINICAL TRIAL MATCHING COMPLETE")
    print("="*60)
    
    exact = sum(1 for m in matches.values() if any(x.match_type == 'exact_triple' for x in m))
    none = sum(1 for m in matches.values() if len(m) == 0)
    
    print(f"Combinations with exact trial matches: {exact}")
    print(f"Combinations with NO trials (NOVEL!): {none}")
    print(f"\nResults saved to: {args.output}/")
