#!/usr/bin/env bash
# Refit selected stations inside existing sensitivity runs after an input change.
# Usage: bash scripts/analysis/refit_stations.sh "st2 st60" [SENS_DIR]
# Each data/sens/<name>.csv is paired with workspace/results/sens_<name> (or
# bench_reduced for reduced.csv, sens_stft_trailing for the reference input).
set -euo pipefail
cd "$(dirname "$0")/../.."
STATIONS=$1
SENS=${2:-data/sens}
export PYTHONPATH="$(cd .. && pwd)${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg GW_SKIP_PLOTS=1

refit() {  # refit <input_csv> <run_id> [env...]
  local input=$1 rid=$2
  [ -d "workspace/results/$rid" ] || { echo "skip $rid (no run)"; return; }
  for st in $STATIONS; do
    if ! awk -F, -v s="$st" 'NR==1 {for (i=1;i<=NF;i++) if ($i=="st_id") c=i; next} $c==s {found=1} END {exit !found}' "$input"; then continue; fi
    echo "=== $(date '+%T') $rid $st"
    poetry run python srcs/03_run_model.py --input "$input" --run-id "$rid" --station "$st" --force \
      >> "workspace/results/${rid}_refit.log" 2>&1
  done
}

for f in "$SENS"/*.csv; do
  name=$(basename "$f" .csv)
  case $name in
    reduced) refit "$f" bench_reduced ;;
    *) refit "$f" "sens_${name}" ;;
  esac
done
GW_STFT_ALIGN=trailing refit data/gray_box_input_cal_optimized.csv sens_stft_trailing
echo "=== $(date '+%T') refit complete"
