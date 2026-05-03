"""Pure-function detection, classification, and cleaning for LS observations.

All public functions are stateless; I/O is handled by 03b_clean_ls.py.

Public API:
    compute_robust_sigma     — MAD-based σ of day-to-day diffs
    detect_jumps             — flag points exceeding n_sigma threshold
    classify_jump            — 5-branch decision tree per spec §3.3
    clean_station            — iterative orchestrator
"""
from __future__ import annotations
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
import pandas as pd

MAD_TO_SIGMA = 1.4826


def compute_robust_sigma(z: pd.Series) -> float:
    """Median-absolute-deviation σ of z.diff(), robust to outliers in z.

    Returns σ in the same units as z.
    """
    dz = z.diff().dropna()
    if dz.empty:
        return 0.0
    median_dz = dz.median()
    mad = (dz - median_dz).abs().median()
    return float(MAD_TO_SIGMA * mad)


def detect_jumps(z: pd.Series,
                 n_sigma: float = 6.0,
                 sigma_floor_cm: float = 1.0) -> pd.DataFrame:
    """Detect jump candidates: |Δζ| ≥ max(n_sigma·σ, sigma_floor_cm/100).

    Returns DataFrame with columns: date, magnitude_m (signed), sigma_m, n_sigma.
    """
    sigma = compute_robust_sigma(z)
    threshold_m = max(n_sigma * sigma, sigma_floor_cm / 100.0)
    dz = z.diff()
    flags = dz.abs() >= threshold_m
    flagged = dz[flags]
    if flagged.empty:
        return pd.DataFrame(columns=["date", "magnitude_m", "sigma_m", "n_sigma"])
    return pd.DataFrame({
        "date": flagged.index,
        "magnitude_m": flagged.values,
        "sigma_m": sigma,
        "n_sigma": flagged.abs().values / max(sigma, 1e-12),
    }).reset_index(drop=True)


def count_agreeing_neighbors(jump_date: pd.Timestamp,
                              neighbor_series: List[pd.Series],
                              magnitude_sign: float,
                              n_sigma: float = 4.0,
                              time_window_days: int = 1) -> int:
    """Count how many neighbor series show a same-sign jump > n_sigma·σ
    within ±time_window_days of jump_date.
    """
    n_agree = 0
    for n in neighbor_series:
        sigma = compute_robust_sigma(n)
        threshold = n_sigma * sigma
        if threshold <= 0:
            continue
        dz = n.diff()
        window_lo = jump_date - pd.Timedelta(days=time_window_days)
        window_hi = jump_date + pd.Timedelta(days=time_window_days)
        in_window = dz.loc[(dz.index >= window_lo) & (dz.index <= window_hi)]
        same_sign = in_window[np.sign(in_window.values) == np.sign(magnitude_sign)]
        if (same_sign.abs() >= threshold).any():
            n_agree += 1
    return n_agree
