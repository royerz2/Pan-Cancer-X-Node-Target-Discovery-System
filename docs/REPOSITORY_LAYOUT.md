# Repository Layout

This public repository preserves both the runnable ALIN framework and the main result bundles that support the manuscript, benchmark analyses, and follow-on validation work.

## Primary Entry Points

- `pan_cancer_xnode.py` — main discovery CLI for single-cancer and pan-cancer runs
- `scripts/run_pipeline.py` — reproducibility runner for the gold-standard benchmark cancers
- `benchmarking_module.py` — benchmark and baseline evaluation over generated triples
- `pharmacological_validation.py` — pharmacological validation export, now writing JSON reports under `outputs/reports/validation_reports/`
- `run_full_pipeline.sh` — compact shell entrypoint for the public reproducibility path

## Stable Top-Level Areas

- `alin/` — core library package
- `core/` — lower-level data structures and statistics utilities
- `docs/` — release-facing documentation
- `manuscript/` — paper and supplementary source files
- `figures/` — generated figure assets used in the manuscript
- `tests/` — unit and integration tests
- `results/` — default quick-start discovery output location
- `benchmark_results/` — default benchmark output location when the benchmark CLI is run locally
- `outputs/` — canonical home for newly generated auxiliary reports and non-root runtime artifacts

## Preserved Published Artifact Bundles

These top-level directories are intentionally kept in the public repository because they back manuscript figures, ablation studies, calibration experiments, lineage analyses, and other release-era evaluations:

- `ablation_results/`
- `benchmark_hardening_results/`
- `calibration_results/`
- `feasibility_results/`
- `hub_strategy_results/`
- `lineage_control_results/`
- `lineage_evaluation_results/`
- `mhs_nonuniqueness_results/`
- `mhs_triple_results/`
- `negative_controls_results/`
- `pearson_comparison_results/`
- `results_discovery/`
- `validation_results/`

Treat these as preserved release artifacts, not as the preferred destination for new ad hoc runtime outputs.

## Script Surface

Most files under `scripts/` are analysis, figure-generation, or study-specific helper scripts. They remain public for transparency and reproducibility, but they are not all equal entry points.

Use [scripts/README.md](../scripts/README.md) to identify the small subset that matters for routine public use.

## Output Hygiene

- Keep new auxiliary validation exports under `outputs/`
- Avoid writing one-off logs and JSON captures into the repository root
- Leave preserved published artifact directories in place unless there is a deliberate release curation pass