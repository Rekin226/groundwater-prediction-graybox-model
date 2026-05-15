"""Network-wide GPR pathology classifier.

CLI to scan a run directory and emit gpr_pathology_report.csv with five
deterministic flags per station (over_smoothing, outlier_spike,
extrapolation_drift, render_dominance, narrow_band) plus
gpr_pathology_summary.txt with prevalence counts.

Usage:
    poetry run python subsidence/scripts/diag_gpr_pathology.py \\
        --run-dir workspace/results_sub/v11_cleaned
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


# ── Classifier thresholds (population-derived; see spec §1.5) ─────────────
OVER_SMOOTHING_RATIO_THRESHOLD = 0.30
OUTLIER_RANGE_MULTIPLIER = 2.0
EXTRAPOLATION_NEAREST_OBS_FRAC = 0.25
EXTRAPOLATION_SIGMA_MULTIPLIER = 5.0
RENDER_DOMINANCE_FRAC = 0.30
NARROW_BAND_FRAC_OF_SIGMA = 0.05


def flag_over_smoothing(obs: np.ndarray,
                         gpr_mean: np.ndarray,
                         imputed_mask: np.ndarray) -> tuple[bool, float]:
    """Flag True if std(obs − GPR_mean) / std(obs) < threshold,
    evaluated on observed cells only.
    Returns (flag, ratio).
    """
    obs = np.asarray(obs, float)
    gpr_mean = np.asarray(gpr_mean, float)
    obs_cells = ~np.asarray(imputed_mask, dtype=bool)
    yo = obs[obs_cells]
    ym = gpr_mean[obs_cells]
    finite = np.isfinite(yo) & np.isfinite(ym)
    yo = yo[finite]; ym = ym[finite]
    if yo.size < 5 or float(np.std(yo)) == 0:
        return (False, float("nan"))
    ratio = float(np.std(yo - ym) / np.std(yo))
    return (ratio < OVER_SMOOTHING_RATIO_THRESHOLD, ratio)


def _main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--station", default=None,
                   help="Single-station spot-check mode")
    args = p.parse_args(argv)
    print(f"diag_gpr_pathology placeholder — run-dir={args.run_dir}")


if __name__ == "__main__":
    _main()
