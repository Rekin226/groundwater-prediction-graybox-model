"""Orchestrator — fit subsidence ODE for active stations in parallel.

Per-station logic lives in `subsidence/sub_runner.py` (testable module).
This script handles CLI parsing, station list filtering, and the
multiprocessing pool.

Reads:
    subsidence/data/sub_station_master.csv (active=1 rows)
    subsidence/data/h_drivers/<sub_id>.parquet
    data/ls_cache/clean/<dataset>__<url_encoded_sub_id>.parquet
Writes:
    workspace/results_sub/<run_id>/per_station/<sub_id>.csv
    workspace/results_sub/<run_id>/sub_fit_results.csv      (best variant per station)
    workspace/results_sub/<run_id>/all_variants_results.csv (all variants per station)

Run:
    poetry run python subsidence/05_run_subsidence.py --run-id initial
    poetry run python subsidence/05_run_subsidence.py --run-id initial --station TKJS
"""
from __future__ import annotations
import argparse
import multiprocessing as mp
import sys
from pathlib import Path
import pandas as pd

# Ensure project root on sys.path so `from subsidence...` works under multiprocessing
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from subsidence.sub_runner import (
    _process,
    EXCLUDED_DATASETS,
    EXCLUDED_STATIONS,
)


def _merge(run_id: str):
    """Concat per-station CSVs into run-level summary tables.

    Skips *_mlcw_layer.csv files (different schema — per-layer compaction,
    datetime-indexed).  Writes:
        workspace/results_sub/{run_id}/sub_fit_results.csv      — best variant per station
        workspace/results_sub/{run_id}/all_variants_results.csv — all variants per station
    """
    base = Path(f"workspace/results_sub/{run_id}/per_station")
    rows = []
    for f in sorted(base.glob("*.csv")):
        if f.stem.endswith("_mlcw_layer"):
            continue
        if f.stem.endswith("_gpr"):
            continue  # Time-series-keyed GPR CSV (different schema)
        rows.append(pd.read_csv(f))
    if not rows:
        return
    all_df = pd.concat(rows, ignore_index=True)
    # Ensure is_best is boolean (pandas reads "True"/"False" strings as object)
    all_df["is_best"] = all_df["is_best"].astype(bool)
    all_df.to_csv(f"workspace/results_sub/{run_id}/all_variants_results.csv", index=False)
    best = all_df[all_df["is_best"]].copy()
    best.to_csv(f"workspace/results_sub/{run_id}/sub_fit_results.csv", index=False)
    print(f"\nMerged: {len(best)} stations × {best['variant'].nunique()} variants (best)")
    print(f"  → workspace/results_sub/{run_id}/sub_fit_results.csv")
    print(f"  → workspace/results_sub/{run_id}/all_variants_results.csv")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", default="initial")
    p.add_argument("--station", default=None)
    p.add_argument("--workers", type=int, default=mp.cpu_count() - 1)
    args = p.parse_args(argv)

    master = pd.read_csv("subsidence/data/sub_station_master.csv")
    master = master[master["active"] == 1]
    master = master[~master["sub_dataset"].isin(EXCLUDED_DATASETS)]
    master = master[~master["sub_id"].isin(EXCLUDED_STATIONS)]
    if args.station:
        master = master[master["sub_id"] == args.station]
    if master.empty:
        print("No matching active stations."); return

    tasks = [(r["sub_id"], r["sub_dataset"], args.run_id) for _, r in master.iterrows()]
    if args.workers <= 1 or args.station:
        for t in tasks:
            print(_process(*t))
    else:
        with mp.Pool(args.workers) as pool:
            for msg in pool.starmap(_process, tasks):
                print(msg)

    _merge(args.run_id)


if __name__ == "__main__":
    sys.exit(main())
