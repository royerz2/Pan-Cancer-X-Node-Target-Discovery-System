# Outputs

`outputs/` is reserved for newly generated auxiliary artifacts that should not be written into the repository root.

Current canonical subpaths:

- `comparisons/` — strategy-arm comparison runs produced by `scripts/pipelines/run_strategy_arm_comparison.py`
- `benchmark_audits/` — focused viability audits produced by `scripts/pipelines/run_benchmark_viability_audit.py`
- `reports/validation_reports/` — JSON reports produced by `pharmacological_validation.py`
- `reports/validation_results/` — standalone exports from `alin/validation.py`
- `outcome_benchmark/` — outcome benchmark figures and summary artifacts when `outcome_benchmark.py` is run directly

Notes:

- Primary quick-start discovery outputs still default to `results/` and benchmark exports to `benchmark_results/`.
- New one-off logs, captures, and auxiliary reports should be kept under `outputs/` rather than added at the root.
