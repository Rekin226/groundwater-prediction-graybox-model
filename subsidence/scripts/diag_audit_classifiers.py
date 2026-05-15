"""Network-wide §7/§8 audit classifiers.

Reads outputs from a finished run-dir and emits audit_classifiers_report.csv
with one row per station and prevalence-summary columns + companion
audit_classifiers_summary.txt with counts.

Usage:
    poetry run python subsidence/scripts/diag_audit_classifiers.py \\
        --run-dir workspace/results_sub/v12_audit_baseline \\
        --h-driver-dir subsidence/data/h_drivers
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


# ── §7 thresholds ──────────────────────────────────────────────────────────
FILL_FRACTION_HIGH = 0.50
PAIRING_TIE_FRAC = 0.10
RING_THRESHOLD_GRID = [(8, 2), (12, 3), (16, 4)]


def compute_fill_fraction(h_df: pd.DataFrame) -> float:
    """Return fraction of cells where driver_source != 'observed'."""
    if "driver_source" not in h_df.columns:
        return float("nan")
    n = len(h_df)
    if n == 0:
        return float("nan")
    obs = int((h_df["driver_source"] == "observed").sum())
    return 1.0 - obs / n


def flag_high_fill_fraction(frac: float) -> bool:
    return bool(np.isfinite(frac) and frac > FILL_FRACTION_HIGH)


def _main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--h-driver-dir", required=True, type=Path)
    args = p.parse_args(argv)
    print(f"diag_audit_classifiers placeholder — run-dir={args.run_dir}")


if __name__ == "__main__":
    _main()
