# Outputs

`outputs/` is reserved for newly generated auxiliary artifacts that should not be written into the repository root.

Current canonical subpaths:

- `reports/validation_reports/` — JSON reports produced by `pharmacological_validation.py`

Notes:

- Primary quick-start discovery outputs still default to `results/` and benchmark exports to `benchmark_results/`.
- The many committed top-level `*_results/` directories in this public repository are preserved release artifacts used for manuscript figures, benchmarking, and calibration analyses.
- New one-off logs, captures, and auxiliary reports should be kept under `outputs/` rather than added at the root.
