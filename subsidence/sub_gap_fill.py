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

MIN_OBS_FOR_GPR = 30


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
        are altered.
    sigma : ndarray, shape (N,)
        Posterior standard deviation at every t. NaN if fit was
        skipped (too few observations).
    imputed_mask : ndarray of bool, shape (N,)
        True where the original y was NaN (i.e. GP-filled).
    """
    y = np.asarray(y, dtype=float)
    imputed_mask = np.isnan(y)
    n_finite = int((~imputed_mask).sum())
    if n_finite < MIN_OBS_FOR_GPR:
        sigma = np.full_like(y, np.nan)
        return y.copy(), sigma, imputed_mask

    raise NotImplementedError("GPR fit not yet implemented")  # filled in Task 2
