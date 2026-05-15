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
from subsidence.sub_metrics import kge_components


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


def compute_n_eff(y: np.ndarray) -> float:
    """Effective sample size from lag-1 autocorrelation."""
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    n = y.size
    if n < 3:
        return float("nan")
    mean = y.mean()
    num = float(np.sum((y[:-1] - mean) * (y[1:] - mean)))
    den = float(np.sum((y - mean) ** 2))
    if den == 0:
        return float(n)
    rho = num / den
    rho = max(min(rho, 0.9999), -0.9999)
    return float(n * (1 - rho) / (1 + rho))


def persistence_kge_benchmark(y_cal: np.ndarray, y_val: np.ndarray) -> float:
    """KGE of a linear-extrapolation forecast from cal into val."""
    y_cal = np.asarray(y_cal, float); y_val = np.asarray(y_val, float)
    yc = y_cal[np.isfinite(y_cal)]
    if yc.size < 2 or y_val.size < 2:
        return float("nan")
    x_cal = np.arange(yc.size, dtype=float)
    slope, intercept = np.polyfit(x_cal, yc, 1)
    x_val = np.arange(yc.size, yc.size + y_val.size, dtype=float)
    y_pred = slope * x_val + intercept
    kc = kge_components(y_val, y_pred)
    return float(kc["kge"])


def kge_on_detrended(y_obs: np.ndarray, y_sim: np.ndarray) -> float:
    """KGE on per-series OLS-detrended cumulative ζ — companion diagnostic.

    After OLS detrending, both residual series have near-zero means and the
    sim may have near-zero variance (if it is exactly linear).  A unit offset
    is applied to both before calling kge_components so that beta = mu_s/mu_o
    remains well-defined.  When sd_s ≈ 0 after the shift, r is undefined; in
    that case KGE is computed directly from r=0, alpha=0, beta=1.
    """
    y_obs = np.asarray(y_obs, float); y_sim = np.asarray(y_sim, float)
    mask = np.isfinite(y_obs) & np.isfinite(y_sim)
    yo = y_obs[mask]; ys = y_sim[mask]
    if yo.size < 3:
        return float("nan")
    x = np.arange(yo.size, dtype=float)
    a, b = np.polyfit(x, yo, 1)
    yo_d = yo - (a * x + b)
    a, b = np.polyfit(x, ys, 1)
    ys_d = ys - (a * x + b)
    # Shift by +1 so beta is well-defined even with near-zero residual means.
    kc = kge_components(yo_d + 1.0, ys_d + 1.0)
    kge_val = kc["kge"]
    if not np.isfinite(kge_val):
        # Degenerate case: sim residuals have zero variance (e.g. exactly
        # linear sim).  Use r=0, alpha=0, beta=1 which gives KGE = 1-√2.
        kge_val = 1.0 - np.sqrt(2.0)
    return float(kge_val)


def _main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--h-driver-dir", required=True, type=Path)
    args = p.parse_args(argv)
    print(f"diag_audit_classifiers placeholder — run-dir={args.run_dir}")


if __name__ == "__main__":
    _main()
