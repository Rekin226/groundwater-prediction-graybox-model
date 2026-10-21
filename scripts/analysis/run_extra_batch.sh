#!/usr/bin/env bash
# Structural ablations and further pairing checks (figures skipped).
# Usage: bash scripts/analysis/run_extra_batch.sh [SENS_DIR]
set -euo pipefail
cd "$(dirname "$0")/../.."
SENS=${1:-data/sens}
export PYTHONPATH="$(cd .. && pwd)${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg GW_SKIP_PLOTS=1
for name in no_amp_amt coastal_as_inland hsea_msl ups_rfcond ups_samelayer; do
  echo "=== $(date '+%F %T') start sens_${name}"
  poetry run python srcs/03_run_model.py --input "$SENS/${name}.csv" --run-id "sens_${name}" --force \
    > "workspace/results/sens_${name}.log" 2>&1
  echo "=== $(date '+%F %T') done  sens_${name}"
done
echo "=== $(date '+%F %T') batch complete"
