#!/usr/bin/env python3
"""Two-station smoke test for the subsidence pipeline.

Mirrors scripts/verify_cal_val_plots.py — runs a representative GNSS station and
a representative MLCW station end-to-end, verifies CSV outputs and figure files.

Run:
    LS_USER=... LS_PASS=... poetry run python subsidence/scripts/verify_subsidence.py
"""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
RUN_ID = "verify"


def run_station(sub_id: str) -> bool:
    cmd = [sys.executable, "subsidence/05_run_subsidence.py",
           "--run-id", RUN_ID, "--station", sub_id]
    print(f"\n>>> Fitting {sub_id} ...")
    res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"  FAIL: nonzero exit\n{res.stderr[:500]}")
        return False
    print(res.stdout.strip())
    return True


def verify_station(sub_id: str, expect_n_variants: int) -> bool:
    csv = ROOT / f"workspace/results_sub/{RUN_ID}/per_station/{sub_id}.csv"
    if not csv.exists():
        print(f"  FAIL: CSV not written: {csv}"); return False
    df = pd.read_csv(csv)
    if len(df) != expect_n_variants:
        print(f"  FAIL: expected {expect_n_variants} rows, got {len(df)}"); return False
    n_best = df["is_best"].sum() if df["is_best"].dtype != object else (df["is_best"] == True).sum()
    if n_best != 1:
        print(f"  FAIL: expected exactly 1 is_best row, got {n_best}"); return False
    best_row = df[df["is_best"] == True].iloc[0] if df["is_best"].dtype == object else df[df["is_best"]].iloc[0]
    if pd.isna(best_row["kge_val"]):
        print(f"  FAIL: best-row kge_val is NaN"); return False
    fig = ROOT / f"workspace/results_sub/{RUN_ID}/figures/M1/sub_fit_{sub_id}.tiff"
    if not fig.exists():
        print(f"  FAIL: figure missing: {fig}"); return False
    print(f"  OK: {sub_id} | best={best_row['variant']} kge_val={best_row['kge_val']:.3f}")
    return True


def main():
    ok = True
    for sub_id, n_variants in [("TKJS", 4), ("客厝國小", 2)]:
        if not run_station(sub_id):
            ok = False; continue
        if not verify_station(sub_id, n_variants):
            ok = False
    print("\n" + ("VERIFY OK" if ok else "VERIFY FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
