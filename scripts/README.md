# Scripts

The `scripts/` directory contains both the public reproducibility runner and a larger set of analysis helpers used to generate manuscript figures and secondary evaluation artifacts.

## Main Public Script

- `run_pipeline.py` — run the ALIN pipeline for the gold-standard cancer set used in benchmark evaluation

## Supporting Analysis Scripts

Most other scripts in this directory fall into one of these groups:

- benchmark and ablation studies
- calibration and null-model analyses
- lineage and hub-strategy comparisons
- manuscript figure and table generation
- exploratory inspection or one-off interpretation utilities

These scripts are intentionally preserved for transparency, but they are supporting research utilities rather than the primary public entry surface.

## Recommended Public Workflow

1. Run `pan_cancer_xnode.py` for discovery outputs.
2. Run `benchmarking_module.py` or `scripts/run_pipeline.py` for benchmark-oriented evaluation.
3. Run `pharmacological_validation.py` when you need the validation JSON export under `outputs/reports/validation_reports/`.