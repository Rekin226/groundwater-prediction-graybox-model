"""Subsidence diagnostic report — flags weak-skill stations.

Reads:
    workspace/results_sub/<run_id>/sub_fit_results.csv
    subsidence/data/sub_station_master.csv  (for n_cal/n_val/span_days)
Writes:
    workspace/diagnostics_sub/<run_id>/diag_report.csv

Run:
    poetry run python subsidence/06_diag_report.py --run-id initial
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import pandas as pd

WEAK_THRESHOLD = 0.5
GOOD_THRESHOLD = 0.7


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Generate diagnostics report for weak-skill subsidence stations."
    )
    p.add_argument("--run-id", required=True)
    p.add_argument(
        "--threshold",
        type=float,
        default=WEAK_THRESHOLD,
        help="kge_val below this counts as weak (default 0.5)",
    )
    args = p.parse_args(argv)

    fit_csv = Path(f"workspace/results_sub/{args.run_id}/sub_fit_results.csv")
    if not fit_csv.exists():
        print(f"ERR: {fit_csv} not found. Run 05_run_subsidence.py first.")
        return 1

    df = pd.read_csv(fit_csv)

    # Keep only best-variant rows
    if "is_best" in df.columns:
        df = df[df["is_best"] == True].copy()

    master_csv = Path("subsidence/data/sub_station_master.csv")
    master = pd.read_csv(master_csv) if master_csv.exists() else pd.DataFrame()

    # Tier counts (NaN kge_val reported separately, not counted as weak)
    n_total = len(df)
    n_nan = int(df["kge_val"].isna().sum())
    n_good = int((df["kge_val"] >= GOOD_THRESHOLD).sum())
    n_med = int(
        ((df["kge_val"] >= args.threshold) & (df["kge_val"] < GOOD_THRESHOLD)).sum()
    )
    n_weak = int((df["kge_val"] < args.threshold).sum())

    print(f"Run {args.run_id}: {n_total} stations")
    print(f"  good  (kge_val >= {GOOD_THRESHOLD}):                      {n_good}")
    print(
        f"  medium ({args.threshold} <= kge_val < {GOOD_THRESHOLD}): {n_med}"
    )
    print(f"  weak  (kge_val < {args.threshold}):                       {n_weak}")
    print(f"  nan   kge_val:                                            {n_nan}")

    # Weak-station detail (exclude NaN kge_val)
    weak = df[df["kge_val"].notna() & (df["kge_val"] < args.threshold)].copy()
    if weak.empty:
        print("\nNo weak stations.")
        return 0

    # Merge in n_cal / n_val / span_days from master
    if not master.empty and "sub_id" in master.columns:
        merge_cols = [c for c in ["sub_id", "n_cal", "n_val", "span_days"] if c in master.columns]
        weak = weak.merge(master[merge_cols], on="sub_id", how="left")

    # Build ordered column list
    diag_columns = [
        "sub_id",
        "sub_dataset",
        "variant",
        "tau_underidentified",
        "kge_cal",
        "kge_val",
        "kge_rate_val",
        "rmse_val",
        "r2_val",
        "bias_val",
        "n_cal",
        "n_val",
        "span_days",
        "Sk_e",
        "Sk_v",
        "h_ref",
    ]
    # Optional variant-dependent cols
    for c in ("v_tect", "v0", "v1", "tau"):
        if c in weak.columns:
            diag_columns.append(c)

    diag = weak[[c for c in diag_columns if c in weak.columns]].copy()
    diag = diag.sort_values("kge_val", ascending=True)

    out = Path(f"workspace/diagnostics_sub/{args.run_id}/diag_report.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    diag.to_csv(out, index=False)
    print(f"\nWeak-station diagnostic report -> {out}")
    show_cols = [
        c for c in ["sub_id", "variant", "kge_val", "kge_rate_val", "rmse_val"] if c in diag.columns
    ]
    print(diag[show_cols].to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
