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


def flag_outlier_spike(obs: np.ndarray, gpr_mean: np.ndarray,
                        imputed_mask: np.ndarray) -> tuple[bool, float]:
    obs = np.asarray(obs, float)
    gpr_mean = np.asarray(gpr_mean, float)
    obs_cells = ~np.asarray(imputed_mask, dtype=bool)
    yo = obs[obs_cells]
    yo = yo[np.isfinite(yo)]
    if yo.size < 2:
        return (False, float("nan"))
    obs_min, obs_max = float(yo.min()), float(yo.max())
    rng = obs_max - obs_min
    mean_finite = gpr_mean[np.isfinite(gpr_mean)]
    if mean_finite.size == 0:
        return (False, 0.0)
    deviation = max(mean_finite.max() - obs_max,
                    obs_min - mean_finite.min(), 0.0)
    fired = bool(deviation > OUTLIER_RANGE_MULTIPLIER * rng)
    return (fired, float(deviation / rng) if rng > 0 else float("nan"))


def flag_extrapolation_drift(obs: np.ndarray, gpr_mean: np.ndarray,
                              imputed_mask: np.ndarray,
                              l_train_days: float,
                              sigma_obs: float) -> tuple[bool, float]:
    obs = np.asarray(obs, float)
    gpr_mean = np.asarray(gpr_mean, float)
    imputed_mask = np.asarray(imputed_mask, dtype=bool)
    n = obs.shape[0]
    obs_idx = np.where(np.isfinite(obs))[0]
    if obs_idx.size == 0:
        return (False, float("nan"))
    last_obs = float(obs[obs_idx[-1]])
    max_gap = EXTRAPOLATION_NEAREST_OBS_FRAC * l_train_days
    cells = np.arange(n)
    nearest_dist = np.full(n, np.inf)
    for i in cells:
        nearest_dist[i] = float(np.min(np.abs(obs_idx - i)))
    far_cells = nearest_dist > max_gap
    deviation_sigma = (
        np.abs(gpr_mean - last_obs) / max(sigma_obs, 1e-12)
    )
    fired_mask = imputed_mask & far_cells & (
        deviation_sigma > EXTRAPOLATION_SIGMA_MULTIPLIER
    )
    return (bool(fired_mask.any()), float(deviation_sigma.max()))


def flag_render_dominance(imputed_mask: np.ndarray,
                           render_mask: np.ndarray) -> tuple[bool, float]:
    imputed_mask = np.asarray(imputed_mask, dtype=bool)
    render_mask = np.asarray(render_mask, dtype=bool)
    n = max(imputed_mask.size, 1)
    rendered_imp = int((imputed_mask & render_mask).sum())
    frac = rendered_imp / n
    return (frac > RENDER_DOMINANCE_FRAC, float(frac))


def flag_narrow_band(obs: np.ndarray, sigma: np.ndarray,
                      imputed_mask: np.ndarray) -> tuple[bool, float]:
    obs = np.asarray(obs, float)
    sigma = np.asarray(sigma, float)
    imputed_mask = np.asarray(imputed_mask, dtype=bool)
    yo = obs[~imputed_mask]
    yo = yo[np.isfinite(yo)]
    if yo.size < 2:
        return (False, float("nan"))
    sigma_obs = float(np.std(yo))
    if sigma_obs == 0:
        return (False, float("nan"))
    sigma_in_gap = sigma[imputed_mask]
    sigma_in_gap = sigma_in_gap[np.isfinite(sigma_in_gap)]
    if sigma_in_gap.size == 0:
        return (False, float("nan"))
    median_sigma = float(np.median(sigma_in_gap))
    ratio = median_sigma / sigma_obs
    return (ratio < NARROW_BAND_FRAC_OF_SIGMA, ratio)


def _main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--station", default=None,
                   help="Single-station spot-check mode")
    args = p.parse_args(argv)
    print(f"diag_gpr_pathology placeholder — run-dir={args.run_dir}")


if __name__ == "__main__":
    _main()
