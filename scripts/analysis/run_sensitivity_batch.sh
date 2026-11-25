#!/usr/bin/env bash
# Sequential batch of sensitivity / benchmark runs (figures skipped).
# Usage: bash scripts/analysis/run_sensitivity_batch.sh [REFERENCE_INPUT] [SENS_DIR]
set -euo pipefail
cd "$(dirname "$0")/../.."
REF_INPUT=${1:-data/gray_box_input_cal_optimized.csv}
SENS=${2:-data/sens}
export PYTHONPATH="$(cd .. && pwd)${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg GW_SKIP_PLOTS=1

run() {  # run <input_csv> <run_id> [extra args...]
  local input=$1 rid=$2; shift 2
  echo "=== $(date '+%F %T') start $rid"
  poetry run python srcs/03_run_model.py --input "$input" --run-id "$rid" --force "$@" > "workspace/results/${rid}.log" 2>&1
  echo "=== $(date '+%F %T') done  $rid"
}

for f in "$SENS"/seed_*.csv; do
  s=$(basename "$f" .csv)
  run "$f" "sens_${s}"
done
run "$SENS/reduced.csv" bench_reduced
for v in ups_nearest ups_bestcorr ups_none; do
  run "$SENS/${v}.csv" "sens_${v}"
done
GW_STFT_ALIGN=trailing run "$REF_INPUT" sens_stft_trailing
echo "=== $(date '+%F %T') batch complete"
