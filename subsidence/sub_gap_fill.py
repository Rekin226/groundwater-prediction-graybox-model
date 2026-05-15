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


def _build_kernel():
    """Composite kernel: trend × (RBF + annual periodic) + white noise."""
    return (
        ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3))
        * (
            RBF(length_scale=180.0, length_scale_bounds=(30.0, 720.0))
            + ExpSineSquared(
                length_scale=1.0,
                periodicity=365.25,
                length_scale_bounds=(0.5, 5.0),
                periodicity_bounds="fixed",
            )
        )
        + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-6, 1e-1))
    )


def gpr_fill(
    t: pd.DatetimeIndex,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gap-fill ζ_obs with a Gaussian Process for visualization only.

    Parameters
    ----------
    t : pd.DatetimeIndex
        Full time axis (length N) — typically daily.
    y : np.ndarray
        Cumulative ζ values aligned to t (length N), NaN where missing.

    Returns
    -------
    y_filled : ndarray, shape (N,)
        y with NaN cells replaced by GP posterior mean. Real
        observations are returned bit-identically — only NaN cells
        are altered. If the fit is skipped (n_finite < MIN_OBS_FOR_GPR),
        y_filled is a copy of y with its NaN cells unchanged.
    sigma : ndarray, shape (N,)
        Posterior standard deviation at every t. NaN if fit was
        skipped (too few observations). Otherwise floored at
        ``max(SIGMA_FLOOR_M, 0.01 * std(y_obs))`` so fill_between
        envelopes stay visible at observation points.
    imputed_mask : ndarray of bool, shape (N,)
        True where the original y was NaN (i.e. GP-filled).
    """
    y = np.asarray(y, dtype=float)
    imputed_mask = np.isnan(y)
    n_finite = int((~imputed_mask).sum())
    if n_finite < MIN_OBS_FOR_GPR:
        return y.copy(), np.full_like(y, np.nan), imputed_mask

    # Days-since-first as the GP's x-axis (float, kernel-friendly units).
    t_days = (t - t[0]).days.values.astype(float).reshape(-1, 1)
    obs = ~imputed_mask
    X_obs = t_days[obs]
    y_obs = y[obs]

    # Detrend before fitting: cumulative ζ is non-stationary; GPR with RBF
    # works best on stationary residuals. Subtract the OLS line on observed
    # points and re-add it after prediction.
    slope, intercept = np.polyfit(X_obs.ravel(), y_obs, 1)
    trend_obs = slope * X_obs.ravel() + intercept
    y_obs_detr = y_obs - trend_obs

    gpr = GaussianProcessRegressor(
        kernel=_build_kernel(),
        n_restarts_optimizer=N_RESTARTS,
        random_state=RANDOM_STATE,
        normalize_y=False,
        alpha=0.0,  # WhiteKernel handles noise explicitly
    )
    gpr.fit(X_obs, y_obs_detr)

    mean_detr, sigma = gpr.predict(t_days, return_std=True)
    trend_full = slope * t_days.ravel() + intercept
    mean = mean_detr + trend_full

    # Apply sigma floor so fill_between bands stay visible.
    sigma_floor = max(SIGMA_FLOOR_M, 0.01 * float(np.nanstd(y_obs)))
    sigma = np.maximum(sigma, sigma_floor)

    # Preserve observed cells bit-identically; only fill NaN cells.
    y_filled = y.copy()
    y_filled[imputed_mask] = mean[imputed_mask]

    return y_filled, sigma, imputed_mask
