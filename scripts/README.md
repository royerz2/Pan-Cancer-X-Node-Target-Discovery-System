# Scripts Guide

This directory contains the supported public workflow surface for the pipeline-only repository.

Supported entry points:

- `run_pipeline.py` — focused benchmark-cancer runner
- `compare_modes.py` — actionable vs exploratory comparison for two result directories
- `sensitivity_exploratory.py` — exploratory-mode biology-weight sweep
- `post_pipeline_validation.py` — post-run validation and report generation
- `pipelines/run_strategy_arm_comparison.py` — latest fresh strategy-arm comparison workflow
- `pipelines/run_benchmark_viability_audit.py` — focused second-layer benchmark audit for one prediction set

Notes:

- `run_full_pipeline.sh` and `run_full_pipeline.ps1` call `pipelines/run_strategy_arm_comparison.py` with `--skip-historical` by default because the dev-only historical comparison directories are not bundled in this public repo.
