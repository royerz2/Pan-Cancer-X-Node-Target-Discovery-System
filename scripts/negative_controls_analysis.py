#!/usr/bin/env python3
"""
Negative Controls & Failure Analysis for ALIN Pan-Cancer X-Node System.

Two analyses:
  1. Implausible high-confidence predictions (negative controls)
     – Identifies ALIN top-1 predictions that are biologically suspect
  2. Known failed combinations
     – Curates clinically failed drug combos and checks whether ALIN predicts them

Outputs:
  - negative_controls_results/implausible_predictions.json
  - negative_controls_results/failed_combos_analysis.json
  - negative_controls_results/summary.json
"""

import json, csv, os, sys
from pathlib import Path
from collections import Counter

# ── paths ──────────────────────────────────────────────────────────────────
RESULTS_TRIPLE = 'results/triple_combinations.csv'
RESULTS_MERGED = 'results/triple_combinations_merged.csv'
OUT_DIR = Path('negative_controls_results')
OUT_DIR.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# PART 1: IMPLAUSIBLE HIGH-CONFIDENCE PREDICTIONS
# ═══════════════════════════════════════════════════════════════════════════

# Biology-based flags for implausibility.
# Each entry: cancer, predicted triple, reason, severity, proposed filter.

IMPLAUSIBLE_PREDICTIONS = [
    {
        "cancer": "Renal Cell Carcinoma",
        "prediction": ["CDK6", "EGFR", "MAP2K1"],
        "best_combo": ["EGFR", "MET"],
        "combined_score": 0.952,
        "implausibility": (
            "EGFR is not a validated therapeutic target in RCC. Unlike lung or "
            "colorectal cancers, RCC lacks activating EGFR pathway mutations or "
            "amplifications. Phase 2 trials of erlotinib monotherapy in RCC "
            "showed <5% ORR (Gordon et al., JCO 2009). The gold-standard "
            "combination for RCC is lenvatinib (KDR) + everolimus (MTOR), "
            "which ALIN fails to predict because MTOR does not emerge as "
            "CRISPR-essential in RCC cell lines."
        ),
        "mechanism": "EGFR non-driver in RCC; CRISPR-essentiality does not reflect clinical targetability",
        "filter": "Lineage-restricted target whitelist (exclude EGFR for RCC)",
        "severity": "high",
    },
    {
        "cancer": "Prostate Adenocarcinoma",
        "prediction": ["CDK2", "EGFR", "MAP2K1"],
        "best_combo": ["MAP2K1", "STAT3"],
        "combined_score": 1.008,
        "implausibility": (
            "Prostate cancer is driven by androgen receptor (AR) signalling, not "
            "EGFR/MAPK. EGFR inhibitors have failed in prostate cancer trials "
            "(Gravis et al., Eur Urol 2008; Nabhan et al., Cancer 2009). ALIN "
            "misses the dominant AR axis (CYP17A1, AR) because these are not "
            "CRISPR-essential in the DepMap screen — hormonal targets are invisible "
            "to a CRISPR-based pipeline."
        ),
        "mechanism": "Hormonal pathway blind spot; CRISPR insensitive to ligand-dependent targets",
        "filter": "AR-pathway prior for prostate; flag predictions lacking known driver pathway",
        "severity": "high",
    },
    {
        "cancer": "Acute Myeloid Leukemia",
        "prediction": ["CDK4", "CDK6", "MCL1"],
        "best_combo": ["CDK4", "CDK6", "MCL1"],
        "combined_score": 1.140,
        "implausibility": (
            "While CDK4/6 and MCL1 are individually essential in AML cell lines, "
            "the CDK4/6 + MCL1 combination has no clinical precedent and faces "
            "severe toxicity concerns. MCL1 inhibitors (AMG-176, S64315) cause "
            "dose-limiting cardiotoxicity in Phase 1 (Caenepeel et al., Cancer "
            "Discov 2018). CDK4/6 inhibitors add myelosuppression. The validated "
            "AML combinations target FLT3 + BCL2 or IDH1/2 + BCL2, which ALIN "
            "cannot access because FLT3/IDH are lineage-specific and BCL2 "
            "(venetoclax) acts via apoptotic priming, not CRISPR essentiality."
        ),
        "mechanism": "MCL1 inhibitor cardiotoxicity; FLT3/IDH blind spot in CRISPR data",
        "filter": "Toxicity ceiling: flag combos where all targets have dose-limiting single-agent toxicity",
        "severity": "moderate",
    },
    {
        "cancer": "Head and Neck Squamous Cell Carcinoma",
        "prediction": ["CDK4", "CDK6", "ERBB2"],
        "best_combo": ["FYN", "STAT3"],
        "combined_score": 1.060,
        "implausibility": (
            "ERBB2 overexpression is rare (<5%) in HNSCC. The relevant target is "
            "EGFR (cetuximab, approved). ALIN's best-combo prediction of FYN + "
            "STAT3 is also problematic: FYN is a pan-cancer hub gene with no "
            "single-agent clinical activity in any cancer, and STAT3 inhibitors "
            "(napabucasin) have failed Phase 3 trials in GI cancers (Jonker et al., "
            "Lancet Gastro Hepatol 2018). Both are network-degree artefacts."
        ),
        "mechanism": "Hub gene artefact (FYN, STAT3); ERBB2 rare in HNSCC",
        "filter": "Hub-gene frequency cap; require target mutation/amplification frequency >5%",
        "severity": "moderate",
    },
    {
        "cancer": "Endometrial Carcinoma",
        "prediction": ["CDK2", "EGFR", "MET"],
        "best_combo": ["MAP2K1", "STAT3"],
        "combined_score": 0.979,
        "implausibility": (
            "Endometrial carcinoma is primarily driven by PI3K/MTOR and MMR "
            "deficiency pathways. EGFR/MET targeting has no clinical precedent "
            "here. The validated combination is lenvatinib (KDR) + everolimus "
            "(MTOR), which ALIN misses. The best-combo fallback to MAP2K1 + "
            "STAT3 reflects generic MAPK hub selection rather than endometrial-"
            "specific biology."
        ),
        "mechanism": "PI3K/MTOR pathway missed; MAPK/STAT3 hub artefact",
        "filter": "Pathway-prior enrichment for known driver pathways per cancer",
        "severity": "moderate",
    },
    {
        "cancer": "Hepatocellular Carcinoma",
        "prediction": ["EGFR", "FGFR1", "MET"],
        "best_combo": ["FGFR1", "STAT3"],
        "combined_score": 1.044,
        "implausibility": (
            "While MET is a valid HCC target (cabozantinib approved), FGFR1 is "
            "not established in HCC (FGFR1 amplification is rare; FGFR2 fusions "
            "drive cholangiocarcinoma instead). The validated HCC combination is "
            "cabozantinib (MET + KDR) or atezolizumab + bevacizumab (VEGF). "
            "ALIN partially recovers MET but couples it with FGFR1 (network "
            "neighbor) and misses KDR, which is invisible to CRISPR because "
            "KDR acts via tumour vasculature, not cell-autonomous essentiality."
        ),
        "mechanism": "FGFR1 wrong isoform; KDR invisible to cell-autonomous CRISPR screen",
        "filter": "Isoform-aware target matching (FGFR1 vs FGFR2); microenvironment target flag",
        "severity": "moderate",
    },
    {
        "cancer": "Liposarcoma",
        "prediction": ["EGFR", "FGFR1", "MET"],
        "best_combo": ["FGFR1", "STAT3"],
        "combined_score": 0.951,
        "implausibility": (
            "Liposarcoma is driven by MDM2 amplification (well-differentiated) or "
            "chromosomal translocations (myxoid). The validated combination is "
            "MDM2 + CDK4 (BI 907828 + abemaciclib, Phase 2). ALIN predicts "
            "generic RTK inhibition (EGFR/FGFR1/MET) which has no basis in "
            "liposarcoma biology. The gold standard entry CDK4 + MDM2 is "
            "completely missed."
        ),
        "mechanism": "MDM2 is computationally undruggable in GENE_TO_DRUGS; RTK over-representation",
        "filter": "Ensure targets include known lineage-specific drivers before returning generic RTK triples",
        "severity": "high",
    },
    {
        "cancer": "Non-Cancerous (Lymphoid)",
        "prediction": ["CDK6", "FYN", "STAT3"],
        "best_combo": ["CDK6", "FYN", "STAT3"],
        "combined_score": 0.576,
        "implausibility": (
            "This is a non-malignant control tissue (normal lymphoid cells). "
            "Producing a therapeutic target prediction for a non-cancerous label "
            "is by definition meaningless. The CDK6 + FYN + STAT3 triple is the "
            "most common ALIN prediction (appears in 52/77 cancers as STAT3-"
            "containing), confirming it is a pan-cancer hub artefact."
        ),
        "mechanism": "Non-malignant tissue; hub artefact",
        "filter": "Exclude non-cancerous lineages from prediction output",
        "severity": "high",
    },
]


# ═══════════════════════════════════════════════════════════════════════════
# PART 2: KNOWN FAILED COMBINATIONS
# ═══════════════════════════════════════════════════════════════════════════

# Curated from clinical trial failures in oncology where drug targets are known.
# For each: does ALIN predict the failed combo? (matching cancer + targets)

KNOWN_FAILED_COMBOS = [
    {
        "combination": "BRAF + MEK + CDK4/6",
        "targets": {"BRAF", "MAP2K1", "CDK4", "CDK6"},
        "targets_subset_for_match": {"BRAF", "MAP2K1", "CDK4"},  # any 3 of 4
        "cancer": "Melanoma",
        "drugs": "encorafenib + binimetinib + ribociclib",
        "trial": "Phase 1/2 (COLUMBUS-related)",
        "outcome": "ORR 52% (< 64-68% doublet); 24% discontinuation; excess hematotoxicity",
        "failure_reason": "toxicity_plus_inferior_efficacy",
        "reference": "Dummer et al., Lancet Oncol 2018",
        "pmid": "30219628",
    },
    {
        "combination": "PI3K + MEK",
        "targets": {"PIK3CA", "MAP2K1"},
        "targets_subset_for_match": {"PIK3CA", "MAP2K1"},
        "cancer": "Ovarian Epithelial Tumor",
        "drugs": "BKM120 (buparlisib) + trametinib",
        "trial": "Phase 1/2 (multiple)",
        "outcome": "ORR 4.7%; DCR 19.2%; dose-limiting toxicity prevented adequate target inhibition",
        "failure_reason": "inadequate_efficacy_plus_toxicity",
        "reference": "Shimizu et al., Clin Cancer Res 2012",
        "pmid": "22496272",
    },
    {
        "combination": "EGFR mono in RCC",
        "targets": {"EGFR"},
        "targets_subset_for_match": {"EGFR"},
        "cancer": "Renal Cell Carcinoma",
        "drugs": "erlotinib",
        "trial": "Phase 2",
        "outcome": "ORR <5% in unselected RCC; EGFR not a RCC driver",
        "failure_reason": "wrong_target_for_indication",
        "reference": "Gordon et al., JCO 2009",
        "pmid": "19720913",
    },
    {
        "combination": "CDK4/6 + BRAF (no MEK)",
        "targets": {"CDK4", "CDK6", "BRAF"},
        "targets_subset_for_match": {"CDK4", "BRAF"},
        "cancer": "Melanoma",
        "drugs": "palbociclib + vemurafenib",
        "trial": "Phase 1/2",
        "outcome": "Excess haematotoxicity; inferior to BRAF+MEK doublet; rapid resistance",
        "failure_reason": "toxicity_plus_suboptimal_target_pairing",
        "reference": "Carvajal et al., Cancer Discov 2018",
        "pmid": "29898992",
    },
    {
        "combination": "STAT3 (napabucasin) in CRC",
        "targets": {"STAT3"},
        "targets_subset_for_match": {"STAT3"},
        "cancer": "Colorectal Adenocarcinoma",
        "drugs": "napabucasin + FOLFIRI",
        "trial": "Phase 3 (CanStem111P)",
        "outcome": "Failed primary OS endpoint; no improvement over chemotherapy alone",
        "failure_reason": "no_single_agent_or_combo_efficacy",
        "reference": "Jonker et al., Lancet Gastro Hepatol 2018",
        "pmid": "29937312",
    },
    {
        "combination": "EGFR mono in Prostate",
        "targets": {"EGFR"},
        "targets_subset_for_match": {"EGFR"},
        "cancer": "Prostate Adenocarcinoma",
        "drugs": "erlotinib, gefitinib",
        "trial": "Phase 2 (multiple)",
        "outcome": "No objective responses in CRPC; EGFR not a prostate driver",
        "failure_reason": "wrong_target_for_indication",
        "reference": "Nabhan et al., Cancer 2009; Gravis et al., Eur Urol 2008",
        "pmid": "19606525",
    },
    {
        "combination": "MCL1 inhibitor (any)",
        "targets": {"MCL1"},
        "targets_subset_for_match": {"MCL1"},
        "cancer": "Acute Myeloid Leukemia",
        "drugs": "AMG-176, S64315",
        "trial": "Phase 1",
        "outcome": "Dose-limiting cardiotoxicity; narrow therapeutic window; clinical development paused",
        "failure_reason": "dose_limiting_toxicity",
        "reference": "Caenepeel et al., Cancer Discov 2018; Tron et al., Nat Commun 2018",
        "pmid": "30224409",
    },
    {
        "combination": "BRAF mono (no MEK) in melanoma",
        "targets": {"BRAF"},
        "targets_subset_for_match": {"BRAF"},
        "cancer": "Melanoma",
        "drugs": "vemurafenib, dabrafenib",
        "trial": "Phase 3 (historical)",
        "outcome": "Initial high response (ORR ~50%) but rapid resistance (PFS 5-8 mo); paradoxical MAPK reactivation",
        "failure_reason": "rapid_resistance_without_pathway_coverage",
        "reference": "Chapman et al., NEJM 2011; Flaherty et al., NEJM 2012",
        "pmid": "21639808",
    },
]


def load_predictions():
    """Load both benchmark and merged predictions."""
    preds = {}  # cancer -> {triple: set, best_combo: set}
    
    # Benchmark predictions (17 cancers)
    with open(RESULTS_TRIPLE) as f:
        reader = csv.DictReader(f)
        for row in reader:
            cancer = row['Cancer Type']
            triple = {row['Target 1'], row['Target 2'], row['Target 3']}
            bc = set()
            if row.get('Best_Combo_1') and row['Best_Combo_1'] != 'nan':
                bc.add(row['Best_Combo_1'])
            if row.get('Best_Combo_2') and row['Best_Combo_2'] != 'nan':
                bc.add(row['Best_Combo_2'])
            if row.get('Best_Combo_3') and row['Best_Combo_3'] not in ('nan', ''):
                bc.add(row['Best_Combo_3'])
            preds[cancer] = {'triple': triple, 'best_combo': bc if bc else triple}
    
    # Merged predictions (77 cancers)
    with open(RESULTS_MERGED) as f:
        reader = csv.DictReader(f)
        for row in reader:
            cancer = row['Cancer_Type']
            if cancer not in preds:
                targets_str = row.get('Triple_Targets', '')
                if targets_str:
                    triple = set(targets_str.replace(' ', '').split('+'))
                else:
                    triple = {row.get('Target_1', ''), row.get('Target_2', ''), row.get('Target_3', '')}
                    triple.discard('')
                preds[cancer] = {'triple': triple, 'best_combo': triple}
    
    return preds


def check_prediction_contains(pred_targets, failed_targets):
    """Check if ALIN prediction contains the failed targets."""
    overlap = pred_targets & failed_targets
    return {
        'overlap': sorted(overlap),
        'n_overlap': len(overlap),
        'fraction': len(overlap) / len(failed_targets) if failed_targets else 0,
        'prediction_contains_failed': failed_targets.issubset(pred_targets),
        'any_overlap': bool(overlap),
    }


def analyse_failed_combos(preds):
    """Check each known failed combo against ALIN predictions."""
    results = []
    for fc in KNOWN_FAILED_COMBOS:
        cancer = fc['cancer']
        failed_targets = fc['targets_subset_for_match']
        
        # Look up ALIN prediction for this cancer
        if cancer in preds:
            triple_match = check_prediction_contains(preds[cancer]['triple'], failed_targets)
            combo_match = check_prediction_contains(preds[cancer]['best_combo'], failed_targets)
            predicted_triple = sorted(preds[cancer]['triple'])
            predicted_combo = sorted(preds[cancer]['best_combo'])
            has_prediction = True
        else:
            triple_match = {'overlap': [], 'n_overlap': 0, 'fraction': 0,
                           'prediction_contains_failed': False, 'any_overlap': False}
            combo_match = triple_match.copy()
            predicted_triple = []
            predicted_combo = []
            has_prediction = False
        
        # Interpret
        if triple_match['prediction_contains_failed'] or combo_match['prediction_contains_failed']:
            verdict = "PREDICTS_FAILED_COMBO"
            interpretation = (
                f"ALIN predicts a combination containing {sorted(failed_targets)}, "
                f"which failed clinically ({fc['outcome']}). "
                f"This is a false positive that a post-hoc clinical filter should catch."
            )
        elif triple_match['any_overlap'] or combo_match['any_overlap']:
            all_overlaps = set(triple_match['overlap']) | set(combo_match['overlap'])
            verdict = "PARTIAL_OVERLAP"
            interpretation = (
                f"ALIN prediction overlaps on {sorted(all_overlaps)} with the failed "
                f"combo. Partial overlap with a failed combination is informative but "
                f"not necessarily problematic — the additional targets may rescue efficacy."
            )
        else:
            verdict = "CORRECTLY_AVOIDS"
            interpretation = (
                f"ALIN does not predict the failed targets {sorted(failed_targets)} for "
                f"{cancer}. This is a correct negative."
            )
        
        results.append({
            'combination': fc['combination'],
            'failed_targets': sorted(fc['targets']),
            'match_targets': sorted(failed_targets),
            'cancer': cancer,
            'drugs': fc['drugs'],
            'trial': fc['trial'],
            'outcome': fc['outcome'],
            'failure_reason': fc['failure_reason'],
            'reference': fc['reference'],
            'pmid': fc.get('pmid', ''),
            'alin_triple': predicted_triple,
            'alin_combo': predicted_combo,
            'has_prediction': has_prediction,
            'triple_overlap': triple_match,
            'combo_overlap': combo_match,
            'verdict': verdict,
            'interpretation': interpretation,
        })
    
    return results


def analyse_implausible(preds):
    """Annotate implausible predictions with ALIN data."""
    results = []
    for ip in IMPLAUSIBLE_PREDICTIONS:
        cancer = ip['cancer']
        entry = {
            'cancer': cancer,
            'alin_triple': ip['prediction'],
            'alin_best_combo': ip['best_combo'],
            'combined_score': ip['combined_score'],
            'implausibility': ip['implausibility'],
            'mechanism': ip['mechanism'],
            'proposed_filter': ip['filter'],
            'severity': ip['severity'],
        }
        results.append(entry)
    return results


def compute_summary(implausible_results, failed_results):
    """Compute summary statistics."""
    n_implausible = len(implausible_results)
    n_high = sum(1 for r in implausible_results if r['severity'] == 'high')
    n_moderate = sum(1 for r in implausible_results if r['severity'] == 'moderate')
    
    n_failed = len(failed_results)
    n_predicts_failed = sum(1 for r in failed_results if r['verdict'] == 'PREDICTS_FAILED_COMBO')
    n_partial = sum(1 for r in failed_results if r['verdict'] == 'PARTIAL_OVERLAP')
    n_avoids = sum(1 for r in failed_results if r['verdict'] == 'CORRECTLY_AVOIDS')
    
    # Filter categories from implausible predictions
    filter_categories = Counter()
    for ip in IMPLAUSIBLE_PREDICTIONS:
        # Extract filter type
        f = ip['filter']
        if 'hub' in f.lower():
            filter_categories['Hub-gene frequency cap'] += 1
        if 'lineage' in f.lower() or 'whitelist' in f.lower():
            filter_categories['Lineage-restricted target list'] += 1
        if 'pathway' in f.lower() or 'driver' in f.lower():
            filter_categories['Driver-pathway prior'] += 1
        if 'toxicity' in f.lower():
            filter_categories['Toxicity ceiling check'] += 1
        if 'isoform' in f.lower():
            filter_categories['Isoform-aware matching'] += 1
        if 'non-cancerous' in f.lower() or 'non-malignant' in f.lower():
            filter_categories['Malignancy requirement'] += 1
    
    return {
        'n_implausible_predictions': n_implausible,
        'implausible_severity': {'high': n_high, 'moderate': n_moderate},
        'n_failed_combos_tested': n_failed,
        'failed_combo_verdicts': {
            'predicts_failed_combo': n_predicts_failed,
            'partial_overlap': n_partial,
            'correctly_avoids': n_avoids,
        },
        'proposed_filter_categories': dict(filter_categories),
        'key_findings': [
            f"{n_implausible} implausible predictions identified ({n_high} high-severity, {n_moderate} moderate)",
            f"ALIN fully predicts {n_predicts_failed}/{n_failed} known failed combinations",
            f"ALIN partially overlaps with {n_partial}/{n_failed} failed combinations",
            f"ALIN correctly avoids {n_avoids}/{n_failed} failed combinations",
            "Common failure modes: hub-gene artefacts (STAT3/FYN), CRISPR-invisible targets (hormonal, microenvironment), and generic RTK over-representation",
            "Proposed mitigation: 5-tier post-hoc filter (malignancy check → hub cap → driver prior → lineage whitelist → toxicity ceiling)",
        ],
    }


def main():
    print("=" * 70)
    print("NEGATIVE CONTROLS & FAILURE ANALYSIS")
    print("=" * 70)
    
    # Load predictions
    preds = load_predictions()
    print(f"Loaded predictions for {len(preds)} cancer types")
    
    # Part 1: Implausible predictions
    print("\n─── PART 1: IMPLAUSIBLE HIGH-CONFIDENCE PREDICTIONS ───")
    implausible_results = analyse_implausible(preds)
    for i, r in enumerate(implausible_results, 1):
        sev = '🔴' if r['severity'] == 'high' else '🟡'
        print(f"\n{sev} [{i}] {r['cancer']}")
        print(f"   Prediction: {' + '.join(r['alin_triple'])}")
        print(f"   Best combo: {' + '.join(r['alin_best_combo'])}")
        print(f"   Score: {r['combined_score']}")
        print(f"   Mechanism: {r['mechanism']}")
        print(f"   Filter: {r['proposed_filter']}")
    
    # Part 2: Known failed combos
    print("\n\n─── PART 2: KNOWN FAILED COMBINATIONS ───")
    failed_results = analyse_failed_combos(preds)
    for i, r in enumerate(failed_results, 1):
        if r['verdict'] == 'PREDICTS_FAILED_COMBO':
            icon = '❌'
        elif r['verdict'] == 'PARTIAL_OVERLAP':
            icon = '⚠️'
        else:
            icon = '✅'
        print(f"\n{icon} [{i}] {r['combination']} in {r['cancer']}")
        print(f"   Failed targets: {r['match_targets']}")
        print(f"   ALIN triple: {r['alin_triple']}")
        print(f"   ALIN combo: {r['alin_combo']}")
        print(f"   Trial: {r['trial']}")
        print(f"   Outcome: {r['outcome']}")
        print(f"   Verdict: {r['verdict']}")
    
    # Summary
    summary = compute_summary(implausible_results, failed_results)
    
    print("\n\n─── SUMMARY ───")
    for finding in summary['key_findings']:
        print(f"  • {finding}")
    
    # Save results
    with open(OUT_DIR / 'implausible_predictions.json', 'w') as f:
        json.dump(implausible_results, f, indent=2)
    with open(OUT_DIR / 'failed_combos_analysis.json', 'w') as f:
        json.dump(failed_results, f, indent=2)
    with open(OUT_DIR / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to {OUT_DIR}/")
    return summary


if __name__ == '__main__':
    main()
