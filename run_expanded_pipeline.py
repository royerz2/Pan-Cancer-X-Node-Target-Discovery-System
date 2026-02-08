#!/usr/bin/env python3
"""
Re-run ALIN pipeline for gold standard cancer types only.
Uses the expanded candidate pool (druggable viability-path genes).

Saves predictions to results_triples_expanded/triple_combinations.csv

Flags:
  --no-known-synergies  Disable KNOWN_SYNERGIES lookup in synergy scoring.
                        Saves to results_triples_expanded/triple_combinations_no_ks.csv
"""
import sys
import time
import logging
import argparse
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.WARNING)

# Target cancer types with gold standard entries that ALSO exist in pipeline
GOLD_STANDARD_CANCERS = [
    'Melanoma',
    'Non-Small Cell Lung Cancer',
    'Anaplastic Thyroid Cancer',
    'Colorectal Adenocarcinoma',
    'Invasive Breast Carcinoma',
    'Renal Cell Carcinoma',
    'Acute Myeloid Leukemia',
    'Head and Neck Squamous Cell Carcinoma',
    'Ovarian Epithelial Tumor',
    'Prostate Adenocarcinoma',
    'Pancreatic Adenocarcinoma',
    'Bladder Urothelial Carcinoma',
    'Endometrial Carcinoma',
    'Hepatocellular Carcinoma',
    'Esophagogastric Adenocarcinoma',  # maps to Stomach Adenocarcinoma
    'Liposarcoma',
    'Diffuse Glioma',  # maps to Low-Grade/High-Grade Glioma
]

def main():
    parser = argparse.ArgumentParser(description='Re-run ALIN pipeline for gold standard cancers')
    parser.add_argument('--no-known-synergies', action='store_true',
                        help='Disable KNOWN_SYNERGIES in synergy scoring')
    args = parser.parse_args()

    from pan_cancer_xnode import PanCancerXNodeAnalyzer, generate_triple_summary_table

    use_ks = not args.no_known_synergies
    mode_label = 'WITHOUT KNOWN_SYNERGIES' if not use_ks else 'with KNOWN_SYNERGIES'
    print(f"Initializing ALIN pipeline ({mode_label})...", flush=True)
    t0 = time.time()
    analyzer = PanCancerXNodeAnalyzer(use_known_synergies=use_ks)
    t_init = time.time() - t0
    print(f"Initialization: {t_init:.1f}s", flush=True)

    output_dir = Path('results_triples_expanded')
    output_dir.mkdir(exist_ok=True)

    results = {}
    for i, ct in enumerate(GOLD_STANDARD_CANCERS, 1):
        print(f"\n[{i}/{len(GOLD_STANDARD_CANCERS)}] {ct}...", flush=True)
        t1 = time.time()
        try:
            analysis = analyzer.analyze_cancer_type(ct)
            results[ct] = analysis
            if analysis.best_triple:
                targets = sorted(analysis.best_triple.targets)
                print(f"  -> {' + '.join(targets)} (score={analysis.best_triple.combined_score:.3f})",
                      flush=True)
            else:
                print("  -> No triple found", flush=True)
        except Exception as e:
            print(f"  -> ERROR: {e}", flush=True)
        t2 = time.time()
        print(f"  [{t2-t1:.1f}s]", flush=True)

    # Generate output CSV
    if results:
        df = generate_triple_summary_table(results)
        csv_name = 'triple_combinations_no_ks.csv' if not use_ks else 'triple_combinations.csv'
        csv_path = output_dir / csv_name
        df.to_csv(csv_path, index=False)
        print(f"\n{'='*70}")
        print(f"Saved {len(df)} predictions to {csv_path}")
        print(f"{'='*70}")

        # Show comparison of old vs new predictions
        try:
            old_csv = 'results_triples/triple_combinations.csv'
            old_df = pd.read_csv(old_csv)
            # Normalize column names to underscored form
            old_df.columns = [c.replace(' ', '_') for c in old_df.columns]
            new_df = df.copy()
            new_df.columns = [c.replace(' ', '_') for c in new_df.columns]
            print(f"\nOLD ({old_csv}) vs NEW predictions for gold standard cancers:")
            print(f"{'Cancer Type':<45} {'OLD Triple':<30} {'NEW Triple':<30}")
            print("-" * 105)
            for ct in GOLD_STANDARD_CANCERS:
                old_row = old_df[old_df['Cancer_Type'] == ct]
                new_row = new_df[new_df['Cancer_Type'] == ct]
                old_t = f"{old_row.iloc[0]['Target_1']}+{old_row.iloc[0]['Target_2']}+{old_row.iloc[0]['Target_3']}" if len(old_row) > 0 else "N/A"
                new_t = f"{new_row.iloc[0]['Target_1']}+{new_row.iloc[0]['Target_2']}+{new_row.iloc[0]['Target_3']}" if len(new_row) > 0 else "N/A"
                changed = " ***" if old_t != new_t else ""
                print(f"  {ct:<43} {old_t:<30} {new_t:<30}{changed}")
        except Exception as e:
            print(f"(Could not load old predictions for comparison: {e})")

    print(f"\nTotal time: {time.time()-t0:.1f}s")

if __name__ == '__main__':
    main()
