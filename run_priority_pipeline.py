#!/usr/bin/env python3
"""
Run full pipeline on priority combinations from lab_protocols/priority_combinations.csv
=====================================================================================
1. API validation (PubMed, STRING)
2. Clinical trial matching
3. Patient stratification
4. Drug sensitivity (with DepMap)
5. Combined report
"""

import pandas as pd
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

from utils import sanitize_cancer_name, load_depmap_crispr_subset


def run_drug_sensitivity_and_report(df: pd.DataFrame, base: Path, output_dir: Path) -> None:
    """Run drug sensitivity validation and write combined report. Reusable by finish_pipeline."""
    genes_needed = {row['Target_1'] for _, row in df.iterrows()} | {row['Target_2'] for _, row in df.iterrows()} | {row['Target_3'] for _, row in df.iterrows()}
    depmap_subset = load_depmap_crispr_subset(base, genes_needed, fallback_genes={'KRAS', 'CDK6', 'STAT3', 'BRAF', 'EGFR', 'MET', 'CDK2'})
    
    from drug_sensitivity_module import DrugSensitivityValidator, generate_sensitivity_report
    validator = DrugSensitivityValidator(data_dir=str(base / "drug_sensitivity_data"), depmap_data=depmap_subset)
    (output_dir / "drug_sensitivity").mkdir(exist_ok=True)
    
    sens_results = []
    for i, row in df.iterrows():
        targets = [row['Target_1'], row['Target_2'], row['Target_3']]
        drugs = [row['Drug_1'], row['Drug_2'], row['Drug_3']]
        cancer = row['Cancer_Type']
        r = validator.validate_combination(targets, drugs, cancer, skip_biomarkers=True)
        sens_results.append(r)
        report = generate_sensitivity_report(r)
        safe_name = sanitize_cancer_name(cancer, max_len=50)
        (output_dir / "drug_sensitivity" / f"{safe_name}_sensitivity.txt").write_text(report)
        print(f"  {cancer[:45]:45} sens={r.validation_score:.2f}")
    
    sens_summary = [{'Cancer_Type': r.cancer_type, 'Targets': ' + '.join(r.targets),
                    'Sens_Score': r.validation_score, 'Sens_Confidence': r.confidence,
                    'N_Profiles': len(r.drug_profiles), 'N_Correlations': len(r.correlations)}
                   for r in sens_results]
    pd.DataFrame(sens_summary).to_csv(output_dir / "drug_sensitivity_summary.csv", index=False)
    
    combined = df.copy()
    if (output_dir / "api_validation.csv").exists():
        api_df = pd.read_csv(output_dir / "api_validation.csv")
        combined = combined.merge(api_df[['Cancer_Type', 'API_Score', 'API_Confidence']], on='Cancer_Type', how='left')
    combined = combined.merge(pd.DataFrame(sens_summary)[['Cancer_Type', 'Sens_Score', 'Sens_Confidence']], on='Cancer_Type', how='left')
    combined.to_csv(output_dir / "priority_combined_summary.csv", index=False)


def main():
    base = Path(__file__).parent
    priority_file = base / "lab_protocols" / "priority_combinations.csv"
    output_dir = base / "priority_pipeline_results"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    df = pd.read_csv(priority_file)
    print(f"\n{'='*60}")
    print("PRIORITY PIPELINE - Running on", len(df), "combinations")
    print("="*60)
    
    # 1. API Validation (PubMed, STRING)
    print("\n[1/4] API Validation (PubMed + STRING)...")
    try:
        from api_validators import CombinedAPIValidator
        api_val = CombinedAPIValidator(cache_dir=str(base / "api_cache"))
        api_results = []
        for i, row in df.iterrows():
            targets = [row['Target_1'], row['Target_2'], row['Target_3']]
            cancer = row['Cancer_Type']
            r = api_val.validate(targets, cancer, enable_pubmed=True, enable_string=True)
            api_results.append({
                'Cancer_Type': cancer,
                'Targets': ' + '.join(targets),
                'API_Score': r.overall_score,
                'API_Confidence': r.confidence,
                'API_Summary': r.summary
            })
            print(f"  {cancer[:40]:40} API score: {r.overall_score:.3f}")
        pd.DataFrame(api_results).to_csv(output_dir / "api_validation.csv", index=False)
        print("  -> Saved api_validation.csv")
    except Exception as e:
        print(f"  API validation failed: {e}")
    
    # 2. Clinical Trial Matching
    print("\n[2/4] Clinical Trial Matching...")
    try:
        from clinical_trial_matcher import ClinicalTrialMatcher, export_trial_matches
        matcher = ClinicalTrialMatcher(cache_dir=str(base / "trial_cache"))
        all_matches = {}
        for i, row in df.iterrows():
            targets = (row['Target_1'], row['Target_2'], row['Target_3'])
            drugs = (row['Drug_1'], row['Drug_2'], row['Drug_3'])
            cancer = row['Cancer_Type']
            matches = matcher.find_matching_trials(targets, drugs, cancer, delay=0.5)
            all_matches[cancer] = matches
        export_trial_matches(all_matches, output_dir / "clinical_trials")
        print("  -> Saved clinical_trials/")
    except Exception as e:
        print(f"  Clinical trial matching failed: {e}")
    
    # 3. Patient Stratification
    print("\n[3/4] Patient Stratification...")
    try:
        from patient_stratification_module import PatientStratifier, StratificationResult, export_stratification_results
        stratifier = PatientStratifier()
        strat_results = []
        for i, row in df.iterrows():
            targets = [row['Target_1'], row['Target_2'], row['Target_3']]
            drugs = [row['Drug_1'], row['Drug_2'], row['Drug_3']]
            cancer = row['Cancer_Type']
            r = stratifier.stratify(targets, drugs, cancer)
            strat_results.append(r)
        export_stratification_results(strat_results, output_dir / "stratification")
        print("  -> Saved stratification/")
    except Exception as e:
        print(f"  Patient stratification failed: {e}")
    
    # 4. Drug Sensitivity (with DepMap)
    print("\n[4/4] Drug Sensitivity (PRISM + DepMap)...")
    try:
        run_drug_sensitivity_and_report(df, base, output_dir)
        print("  -> Saved drug_sensitivity/")
    except Exception as e:
        print(f"  Drug sensitivity failed: {e}")
        import traceback
        traceback.print_exc()
    
    # 5. Combined summary
    print("\n[5/5] Generating combined report...")
    try:
        combined = df.copy()
        if (output_dir / "api_validation.csv").exists():
            api_df = pd.read_csv(output_dir / "api_validation.csv")
            combined = combined.merge(api_df[['Cancer_Type', 'API_Score', 'API_Confidence']], on='Cancer_Type', how='left')
        if (output_dir / "drug_sensitivity_summary.csv").exists():
            sens_df = pd.read_csv(output_dir / "drug_sensitivity_summary.csv")
            combined = combined.merge(sens_df[['Cancer_Type', 'Sens_Score', 'Sens_Confidence']], on='Cancer_Type', how='left')
        
        combined.to_csv(output_dir / "priority_combined_summary.csv", index=False)
        
        report = f"""
{'='*80}
PRIORITY PIPELINE - COMPLETE
{'='*80}
Combinations processed: {len(df)}
Output directory: {output_dir}

Generated files:
  - api_validation.csv
  - clinical_trials/ (clinical_trial_matches.csv, clinical_trial_report.txt)
  - stratification/ (stratification_summary.csv, individual reports)
  - drug_sensitivity/ (individual reports)
  - drug_sensitivity_summary.csv
  - priority_combined_summary.csv

{'='*80}
"""
        (output_dir / "PIPELINE_COMPLETE.txt").write_text(report)
        print(report)
    except Exception as e:
        print(f"  Combined report failed: {e}")
    
    print("Done.\n")

if __name__ == "__main__":
    main()
