#!/usr/bin/env bash
# Public wrapper for the latest strategy-arm comparison workflow.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON_BIN="${ALIN_PYTHON:-python}"

echo "=== ALIN Public Strategy-Arm Workflow ==="
echo "Running fresh actionable/exploratory arm comparisons without dev-only historical baselines."
echo

"$PYTHON_BIN" scripts/pipelines/run_strategy_arm_comparison.py \
	--skip-historical \
	--no-api \
	--stream-subprocess-output \
	"$@"
