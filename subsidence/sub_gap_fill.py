"""Gaussian-Process gap-filling for cumulative ζ observation series.

Visualization only — never feed imputed values into the optimizer.
The composite kernel (RBF + ExpSineSquared(annual) + WhiteKernel)
captures multi-month trend, the annual hydrologic cycle, and daily
measurement noise so imputed segments inherit realistic fluctuation
amplitude.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, ExpSineSquared, WhiteKernel, ConstantKernel,
)

MIN_OBS_FOR_GPR = 30
SIGMA_FLOOR_M = 1e-4
RANDOM_STATE = 42
N_RESTARTS = 8

# ────────────────────────────────────────────────────────────────────────
# Derive-don't-tune constants (see spec §1, Threshold derivations).
# Adjust here; never inline these values at call sites.
# ────────────────────────────────────────────────────────────────────────
MAX_GAP_FRAC_OF_TRAIN = 0.25      # gap longer than 25% of training span → masked
ALPHA_NOISE_MULT = 1.0            # Tikhonov α = (k · σ_obs_short)², k = 1 σ
MAX_PLAUSIBLE_RATE_M_PER_YR = 0.10  # conservative Zhuoshui Fan compaction-rate cap
SIGMA_RENDER_FLOOR_M = 0.005      # 5 mm absolute floor on sigma band
SIGMA_RENDER_FLOOR_FRAC = 0.10    # OR 10% of per-station σ_obs — whichever larger
RBF_LENGTH_SCALE_MAX_DAYS = 120.0   # sub-seasonal RBF; annual cycle in ExpSineSquared


def _estimate_obs_noise(y: np.ndarray, window: int = 30) -> float:
    """MAD-based short-window noise σ estimate, outlier-robust.

    Computes per-window median absolute deviation on first differences,
    then converts to σ via MAD × 1.4826.  Returns the network-pooled
    median across windows.  Fallback to global std on too-few windows.
    """
    if window < 1:
        raise ValueError(f"window must be ≥ 1, got {window}")
    y = np.asarray(y, float)
    finite = np.isfinite(y)
    yf = y[finite]
    if yf.size < 4:
        return float("nan")
    dy = np.diff(yf)  # first differences live on noise scale; trend cancels
    if dy.size < window:
        # Too short for windowed; use global MAD
        mad = float(np.median(np.abs(dy - np.median(dy))))
        return 1.4826 * mad / np.sqrt(2.0)
    mads = []
    for i in range(0, dy.size - window + 1, window):
        seg = dy[i:i + window]
        mads.append(float(np.median(np.abs(seg - np.median(seg)))))
    mad = float(np.median(mads))
    # Divide by sqrt(2) since dy = N(0, 2σ²) for white noise.
    return 1.4826 * mad / np.sqrt(2.0)


def _build_kernel(length_scale_max: float = RBF_LENGTH_SCALE_MAX_DAYS,
                  noise_lower: float = 1e-5):
    """Composite kernel: trend × (RBF + annual periodic) + white noise.

    length_scale_max — upper bound on RBF correlation length in days.
        Caller passes a population-derived value (default sub-seasonal).
        Must be ≥ 7.0 (RBF lower bound).
    noise_lower — lower bound on WhiteKernel noise_level; caller passes
        the network-median MAD-derived noise floor.
    """
    if length_scale_max < 7.0:
        raise ValueError(
            f"length_scale_max must be ≥ 7.0 (RBF lower bound), got {length_scale_max}"
        )
    return (
        ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3))
        * (
            RBF(length_scale=max(7.0, min(60.0, length_scale_max / 2)),
                length_scale_bounds=(7.0, length_scale_max))
            + ExpSineSquared(
                length_scale=1.0,
                periodicity=365.25,
                length_scale_bounds=(0.5, 5.0),
                periodicity_bounds="fixed",
            )
        )
        + WhiteKernel(noise_level=max(1e-4, noise_lower),
                      noise_level_bounds=(noise_lower, 1e-1))
    )


def gpr_fill(
    t: pd.DatetimeIndex,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Gap-fill ζ_obs with a Gaussian Process for visualization only.

    Returns
    -------
    y_filled : ndarray, (N,) — observed cells unchanged; NaN cells replaced
        by GP posterior mean. If MIN_OBS or sanity gate rejects, NaN cells
        stay NaN.
    sigma : ndarray, (N,) — posterior std (floored), NaN where fit skipped.
    imputed_mask : ndarray bool, (N,) — True where original y was NaN.
    render_mask : ndarray bool, (N,) — True where the imputation line should
        be drawn.  False where gap is too long, too far from any observation,
        or the physical-sanity gate has globally rejected the fill.  In the
        rejected case, render_mask is all-False so plotting code suppresses
        the legend entry.
    """
    y = np.asarray(y, dtype=float)
    imputed_mask = np.isnan(y)
    n_finite = int((~imputed_mask).sum())
    if n_finite < MIN_OBS_FOR_GPR:
        sigma_nan = np.full_like(y, np.nan)
        render_mask = np.zeros_like(y, dtype=bool)
        return y.copy(), sigma_nan, imputed_mask, render_mask

    t_days = (t - t[0]).days.values.astype(float).reshape(-1, 1)
    obs = ~imputed_mask
    X_obs = t_days[obs]
    y_obs = y[obs]

    # Per-station noise estimate (drives α and noise_lower)
    sigma_obs_short = _estimate_obs_noise(y_obs)
    if not np.isfinite(sigma_obs_short) or sigma_obs_short <= 0:
        sigma_obs_short = max(1e-4, 0.01 * float(np.std(y_obs)))
    alpha_tikhonov = (ALPHA_NOISE_MULT * sigma_obs_short) ** 2

    # Detrend (cumulative is non-stationary)
    slope, intercept = np.polyfit(X_obs.ravel(), y_obs, 1)
    trend_obs = slope * X_obs.ravel() + intercept
    y_obs_detr = y_obs - trend_obs

    gpr = GaussianProcessRegressor(
        kernel=_build_kernel(
            length_scale_max=RBF_LENGTH_SCALE_MAX_DAYS,
            noise_lower=max(1e-7, sigma_obs_short ** 2),
        ),
        n_restarts_optimizer=N_RESTARTS,
        random_state=RANDOM_STATE,
        normalize_y=False,
        alpha=alpha_tikhonov,
    )
    gpr.fit(X_obs, y_obs_detr)
    mean_detr, sigma = gpr.predict(t_days, return_std=True)
    trend_full = slope * t_days.ravel() + intercept
    mean = mean_detr + trend_full

    # Sigma render floor (visible band)
    sigma_floor = max(SIGMA_RENDER_FLOOR_M,
                      SIGMA_RENDER_FLOOR_FRAC * float(np.std(y_obs)))
    sigma = np.maximum(sigma, sigma_floor)

    # Compose y_filled (observed bit-identical; only NaN cells filled)
    y_filled = y.copy()
    y_filled[imputed_mask] = mean[imputed_mask]

    # ── Physical-sanity GLOBAL reject ──────────────────────────────────────
    # Δ_phys = ċ_max × (gap duration in years).  Use the full record span
    # as a conservative ceiling on the maximum gap.
    record_span_days = float(X_obs.max() - X_obs.min())
    delta_phys = (
        MAX_PLAUSIBLE_RATE_M_PER_YR * (record_span_days / 365.25)
    )
    obs_min, obs_max = float(y_obs.min()), float(y_obs.max())
    sanity_low = obs_min - delta_phys
    sanity_high = obs_max + delta_phys
    if (mean.min() < sanity_low) or (mean.max() > sanity_high):
        render_mask = np.zeros_like(y, dtype=bool)
        return y_filled, sigma, imputed_mask, render_mask

    # ── Per-cell render mask (gap-too-long / too-far-from-obs) ────────────
    L_train = record_span_days
    max_gap_days = MAX_GAP_FRAC_OF_TRAIN * L_train
    # Build "distance to nearest observation" array.
    obs_positions = X_obs.ravel()
    cell_positions = t_days.ravel()
    # vectorised nearest-obs distance via searchsorted
    idx = np.searchsorted(obs_positions, cell_positions)
    left = np.clip(idx - 1, 0, len(obs_positions) - 1)
    right = np.clip(idx, 0, len(obs_positions) - 1)
    dist_left = np.abs(cell_positions - obs_positions[left])
    dist_right = np.abs(cell_positions - obs_positions[right])
    nearest_dist = np.minimum(dist_left, dist_right)

    render_mask = np.ones_like(y, dtype=bool)
    render_mask &= (nearest_dist <= max_gap_days)
    # Observed cells are always "renderable" in mask sense (plotter checks
    # imputed_mask separately), but we keep them True for clarity.
    render_mask[obs] = True

    return y_filled, sigma, imputed_mask, render_mask
