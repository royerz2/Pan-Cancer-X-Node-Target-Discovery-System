#!/bin/bash
# Full reproducibility pipeline for ALIN Framework (Adaptive Lethal Intersection Network)
# Run from project root: bash run_full_pipeline.sh

set -e
echo "=== ALIN Framework Full Pipeline ==="

# 1. Discovery (requires DepMap data in depmap_data/)
echo "[1/5] Running pan-cancer discovery..."
python pan_cancer_xnode.py --all-cancers --triples --output results_triples/ --no-api 2>/dev/null || true

# 2. Benchmark
echo "[2/5] Running benchmark with baselines..."
python benchmarking_module.py --triples results_triples/triple_combinations.csv --baselines --n-trials 30 --output benchmark_results/

# 3. Priority pipeline (validation, trials, stratification, drug sensitivity)
echo "[3/5] Running priority pipeline..."
python run_priority_pipeline.py 2>/dev/null || python finish_pipeline.py

# 4. Figures
echo "[4/5] Generating figures..."
python generate_figures.py

# 5. Summary
echo "[5/5] Done."
echo ""
echo "Outputs:"
echo "  - results_triples/     (discovery)"
echo "  - benchmark_results/  (benchmark)"
echo "  - priority_pipeline_results/ (validation)"
echo "  - figures/            (publication figures)"
