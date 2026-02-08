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
python pan_cancer_xnode.py --all-cancers --triples --output results/ --no-api
echo ""
echo "[1/4] Discovery done."
echo ""

# 2. Benchmark
echo "[2/4] Benchmarking vs gold standard + baselines..."
python benchmarking_module.py --triples results/triple_combinations.csv --baselines --n-trials 30 --output benchmark_results/
echo "[2/4] Benchmark done."
echo ""

# 3. Pathway shifting simulation (X-node vs three-axis comparison)
echo "[3/4] Pathway shifting simulation (ODE-based X-node vs three-axis triple)..."
python pathway_shifting_simulation.py
echo "[3/4] Simulation done."
echo ""

# 4. Summary
echo "[4/4] Done."
echo ""
echo "Outputs:"
echo "  - results/           (discovery)"
echo "  - benchmark_results/ (benchmark)"
echo "  - simulation_results/ (ODE simulation figures + metrics)"
