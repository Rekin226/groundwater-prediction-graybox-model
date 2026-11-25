#!/usr/bin/env bash
# Structural sensitivity runs with full variant re-selection (figures skipped):
#   sens_ups_offset  upstream head shifted to the target's calibration mean
#   sens_implicit    implicit Euler step
#   sens_dry_b1/b2   rainfall scaled by trailing 365-day dryness, beta = 1, 2
# Usage: bash scripts/analysis/run_structure_batch.sh [INPUT_CSV]
set -euo pipefail
cd "$(dirname "$0")/../.."
INPUT=${1:-data/gray_box_input_cal_optimized.csv}
export PYTHONPATH="$(cd .. && pwd)${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg GW_SKIP_PLOTS=1

run() {  # run <run_id> [VAR=value ...]
  local rid=$1; shift
  echo "=== $(date '+%F %T') start ${rid}"
  env "$@" poetry run python srcs/03_run_model.py --input "$INPUT" --run-id "$rid" --force \
    > "workspace/results/${rid}.log" 2>&1
  echo "=== $(date '+%F %T') done  ${rid}"
}

run sens_ups_offset GW_UPS_OFFSET=cal
run sens_implicit GW_INTEGRATOR=implicit
run sens_dry_b1 GW_RAIN_DRYNESS_BETA=1
run sens_dry_b2 GW_RAIN_DRYNESS_BETA=2
echo "=== $(date '+%F %T') batch complete"
