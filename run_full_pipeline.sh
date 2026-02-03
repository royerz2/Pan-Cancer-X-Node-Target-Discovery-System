#!/bin/bash
# Full reproducibility pipeline for ALIN Framework (Adaptive Lethal Intersection Network)
# Run from project root: bash run_full_pipeline.sh

set -e
echo "=== ALIN Framework Full Pipeline ==="
echo ""

# 1. Discovery (requires DepMap data in depmap_data/)
echo "[1/4] Pan-cancer discovery (DepMap + OmniPath -> triples)..."
echo "      Progress and per-cancer steps will print below."
echo ""
python pan_cancer_xnode.py --all-cancers --triples --output results_triples/ --no-api
echo ""
echo "[1/4] Discovery done."
echo ""

# 2. Benchmark
echo "[2/4] Benchmarking vs gold standard + baselines..."
python benchmarking_module.py --triples results_triples/triple_combinations.csv --baselines --n-trials 30 --output benchmark_results/
echo "[2/4] Benchmark done."
echo ""

# 3. Priority pipeline (validation, trials, stratification, drug sensitivity)
echo "[3/4] Priority pipeline (validation, trials, stratification, PRISM)..."
python run_priority_pipeline.py || python finish_pipeline.py
echo "[3/4] Priority pipeline done."
echo ""

# 4. Summary
echo "[4/4] Done."
echo ""
echo "Outputs:"
echo "  - results_triples/     (discovery)"
echo "  - benchmark_results/  (benchmark)"
echo "  - priority_pipeline_results/ (validation)"
