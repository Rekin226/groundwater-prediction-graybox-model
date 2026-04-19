"""
Stage 1 verification: run modified pipeline on st3 (worst cal/val disconnect,
low KGE) and st14 (good-fit baseline) so the plot overhaul can be reviewed
before the full 61-station rerun.

Usage:
    PYTHONPATH=../:srcs python scripts/verify_cal_val_plots.py

Outputs (per station, 6 plots each; rklib_savefig normalizes the extension to .tiff):
    workspace/results/final/figures/{base,filtered,base_tz,filtered_tz}/gw_fit_{st}.tiff
    workspace/results/final/figures/comparison/gw_compare_{st}.tiff
    workspace/results/final/figures/full_subplots/gw_fit_{st}.tiff
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "srcs"))
sys.path.insert(0, str(ROOT.parent))  # so `import rklib` works

import pandas as pd
import gw_shell

INPUT_CSV = ROOT / "data" / "gray_box_input_optimized.csv"
OUTPUT_ROOT = ROOT / "workspace" / "results" / "final"
STATIONS = ["st3", "st14"]
MODELS = "base,filtered,base_tz,filtered_tz"


def build_args_for(row: pd.Series) -> dict:
    """Mirror srcs/03_run_model.py::build_station_arguments, bypass active filter."""
    args = {
        "gw_st": str(row["gw_st"]),
        "st_id": str(row["st_id"]),
        "gw_x": str(row["gw_TM_X97"]),
        "gw_y": str(row["gw_TM_Y97"]),
        "rf_id": str(row["rf_id"]),
        "lag_days": str(row["lag_days"]),
        "group_name": str(row["group"]),
        "output_root": str(OUTPUT_ROOT),
        "models": MODELS,
    }
    ups_val = row.get("ups_id")
    args["ups_id"] = "none" if pd.isna(ups_val) else str(ups_val)
    return args


def main() -> None:
    if not INPUT_CSV.exists():
        raise SystemExit(f"Input CSV not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    for st_id in STATIONS:
        match = df[df["st_id"] == st_id]
        if match.empty:
            print(f"[{st_id}] SKIP — not in {INPUT_CSV.name}")
            continue
        args = build_args_for(match.iloc[0])
        print(f"\n[{st_id}] running (group={args['group_name']}, ups={args['ups_id']})")
        gw_shell.run_station(args)

    print("\n=== Verification outputs ===")
    for st_id in STATIONS:
        print(f"\nStation {st_id}:")
        checks = [
            ("base",        OUTPUT_ROOT / "figures" / "base"          / f"gw_fit_{st_id}.tiff"),
            ("filtered",    OUTPUT_ROOT / "figures" / "filtered"      / f"gw_fit_{st_id}.tiff"),
            ("base_tz",     OUTPUT_ROOT / "figures" / "base_tz"       / f"gw_fit_{st_id}.tiff"),
            ("filtered_tz", OUTPUT_ROOT / "figures" / "filtered_tz"   / f"gw_fit_{st_id}.tiff"),
            ("comparison",  OUTPUT_ROOT / "figures" / "comparison"    / f"gw_compare_{st_id}.tiff"),
            ("full",        OUTPUT_ROOT / "figures" / "full_subplots" / f"gw_fit_{st_id}.tiff"),
        ]
        for name, path in checks:
            mark = "✓" if path.exists() else "✗"
            print(f"  {mark} {name:12s} {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
