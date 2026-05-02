"""Non-stationarity diagnostic — does the val period sample new historical h minima?

For each station with an h_driver, compute:
- h_min_cal       : min(h) over cal period
- h_min_full_pre_val : min(h) up to val_start (cal + buffer)
- h_min_val       : min(h) over val period
- new_min_in_val  : True if h_min_val < h_min_full_pre_val

Inelastic compaction in Riley/IBS is path-dependent on h_min_hist.  When the
val period crosses a new historical minimum, the inelastic term fires for the
first time outside the cal window — Sk_v calibrated on cal cannot have been
informed by that response, and any cal/val drop is structural (not noise).
Stations flagged here are candidates for a longer cal period or a different
calibration framing.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

CAL_START = pd.Timestamp("2020-01-01")
CAL_END = pd.Timestamp("2022-12-31")
VAL_START = pd.Timestamp("2024-01-01")
VAL_END = pd.Timestamp("2025-03-31")


def _diag_one(h_path: Path) -> dict | None:
    sub_id = h_path.stem
    df = pd.read_parquet(h_path)
    if "h_driver" not in df.columns or df.empty:
        return None
    s = df["h_driver"].dropna()
    if s.empty:
        return None
    s_cal = s.loc[(s.index >= CAL_START) & (s.index <= CAL_END)]
    s_pre_val = s.loc[s.index < VAL_START]
    s_val = s.loc[(s.index >= VAL_START) & (s.index <= VAL_END)]
    if s_cal.empty or s_val.empty:
        return None
    h_min_cal = float(s_cal.min())
    h_min_pre_val = float(s_pre_val.min())
    h_min_val = float(s_val.min())
    new_min = h_min_val < h_min_pre_val
    delta = h_min_val - h_min_pre_val  # negative when new min in val
    return {
        "sub_id": sub_id,
        "h_min_cal_m": h_min_cal,
        "h_min_pre_val_m": h_min_pre_val,
        "h_min_val_m": h_min_val,
        "delta_min_m": delta,
        "new_min_in_val": new_min,
        "n_cal_days": int(s_cal.size),
        "n_val_days": int(s_val.size),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", default="initial",
                   help="Output is written to workspace/results_sub/<run_id>/")
    p.add_argument("--h-driver-dir", default="subsidence/data/h_drivers")
    args = p.parse_args(argv)

    h_dir = Path(args.h_driver_dir)
    rows = []
    for f in sorted(h_dir.glob("*.parquet")):
        d = _diag_one(f)
        if d is not None:
            rows.append(d)
    if not rows:
        print("No h_drivers found.")
        return 1
    df = pd.DataFrame(rows).sort_values("delta_min_m")
    out_dir = Path(f"workspace/results_sub/{args.run_id}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "h_min_hist_diag.csv"
    df.to_csv(out_path, index=False)
    n_total = len(df)
    n_new = int(df["new_min_in_val"].sum())
    print(f"Wrote {out_path}  ({n_total} stations)")
    print(f"  Stations with NEW h_min in val period: {n_new}/{n_total}  "
          f"({100*n_new/n_total:.0f}%)")
    if n_new:
        print("\nTop 10 by largest new-minimum drop (most non-stationary):")
        top = df[df["new_min_in_val"]].head(10)
        cols = ["sub_id", "h_min_pre_val_m", "h_min_val_m", "delta_min_m"]
        print(top[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
