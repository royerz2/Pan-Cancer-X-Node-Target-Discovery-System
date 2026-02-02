#!/usr/bin/env python3
"""Finish drug sensitivity and combined report for priority pipeline (resume after timeout)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from run_priority_pipeline import run_drug_sensitivity_and_report

base = Path(__file__).parent
output_dir = base / "priority_pipeline_results"
priority_file = base / "lab_protocols" / "priority_combinations.csv"

if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv(priority_file)
    print(f"\nFinishing pipeline for {len(df)} combinations...")
    run_drug_sensitivity_and_report(df, base, output_dir)
    (output_dir / "PIPELINE_COMPLETE.txt").write_text(f"PRIORITY PIPELINE - COMPLETE\nCombinations: {len(df)}\nOutput: {output_dir}\n")
    print("\nDone. Combined report saved.")
