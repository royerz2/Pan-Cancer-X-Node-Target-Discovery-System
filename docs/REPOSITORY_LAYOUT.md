# Repository Layout

This public repository is scoped to the runnable ALIN pipeline. It keeps the source tree, setup helpers, tests, and a small supported workflow surface. Manuscript assets, release-era analysis bundles, and precomputed result directories are intentionally excluded.

## Primary Entry Points

- `pan_cancer_xnode.py` — main discovery CLI for single-cancer and pan-cancer runs
- `scripts/run_pipeline.py` — focused benchmark-cancer runner
- `scripts/pipelines/run_strategy_arm_comparison.py` — public comparison workflow for actionable and exploratory runs
- `scripts/pipelines/run_benchmark_viability_audit.py` — focused benchmark viability audit for one prediction set
- `run_full_pipeline.sh` / `run_full_pipeline.ps1` — cross-platform wrappers for the public comparison workflow
- `pharmacological_validation.py` — pharmacological validation export to `outputs/reports/validation_reports/`

## Retained Top-Level Areas

- `alin/` — core library package
- `core/` — shared data structures and statistics utilities
- `docs/` — pipeline documentation and data availability notes
- `outputs/` — canonical home for generated comparisons, benchmark audits, and validation reports
- `scripts/` — supported workflow entry points documented in [scripts/README.md](../scripts/README.md)
- `tests/` — unit and integration tests for the supported public surface

## Output Conventions

- Quick-start discovery runs still default to `results/`
- Benchmark CLI examples still default to `benchmark_results/`
- Auxiliary comparisons, audits, and validation exports should go under `outputs/`
- Avoid writing new logs, JSON captures, or ad hoc reports into the repository root