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


from subsidence.eq_catalog import match_jump_to_event


def _fit_slope_cmyr(z: pd.Series) -> Optional[float]:
    """OLS slope of z (m) over its index (days), expressed in cm/year."""
    if z.dropna().shape[0] < 5:
        return None
    t_days = (z.index - z.index[0]).total_seconds().values / 86400.0
    y = z.values
    mask = np.isfinite(y)
    if mask.sum() < 5:
        return None
    coef = np.polyfit(t_days[mask], y[mask], 1)[0]   # m/day
    return float(coef * 100.0 * 365.25)              # cm/yr


def classify_jump(z: pd.Series, *,
                   jump_date: pd.Timestamp,
                   magnitude_m: float,
                   sigma: float,
                   station_lat_lon: Tuple[float, float],
                   eq_catalog: pd.DataFrame,
                   neighbor_series: List[pd.Series],
                   eq_distance_km: float = 50.0,
                   eq_time_window_days: int = 2,
                   eq_min_magnitude: float = 5.0,
                   neighbor_min_agree: int = 2,
                   neighbor_n_sigma: float = 4.0,
                   slope_window_days: int = 30,
                   parallel_slope_tol_cmyr: float = 0.5,
                   ) -> Dict[str, Any]:
    """Classify a detected jump into one of six categories per spec §3.3.

    Returns dict: classification, action, eq_id, eq_distance_km, eq_magnitude,
    eq_depth_km, n_neighbors_agree, slope_pre_cmyr, slope_post_cmyr.
    """
    out: Dict[str, Any] = {
        "classification": None, "action": None,
        "eq_id": None, "eq_distance_km": None, "eq_magnitude": None, "eq_depth_km": None,
        "n_neighbors_agree": 0, "slope_pre_cmyr": None, "slope_post_cmyr": None,
    }

    # Step 1: Earthquake match
    eq = match_jump_to_event(station_lat_lon, jump_date, eq_catalog,
                              distance_km=eq_distance_km,
                              time_window_days=eq_time_window_days,
                              min_magnitude=eq_min_magnitude)
    if eq is not None:
        out.update(classification="co_seismic", action="nan_event_day",
                   eq_id=eq["id"], eq_distance_km=float(eq["_dist_km"]),
                   eq_magnitude=float(eq["mag"]), eq_depth_km=float(eq["depth"]))
        return out

    # Step 2: Neighbor coherence
    n_agree = count_agreeing_neighbors(jump_date, neighbor_series,
                                        magnitude_sign=magnitude_m,
                                        n_sigma=neighbor_n_sigma)
    out["n_neighbors_agree"] = n_agree
    if n_agree >= neighbor_min_agree:
        out.update(classification="regional_event", action="nan_event_day")
        return out

    # Step 3: Snap-back glitch test
    try:
        i = z.index.get_loc(jump_date)
    except KeyError:
        out.update(classification="boundary_uncertain", action="flag_only")
        return out
    if 0 < i < len(z) - 1:
        dz_next = z.iloc[i + 1] - z.iloc[i]
        if (np.isfinite(dz_next) and abs(dz_next) >= 4.0 * sigma
                and np.sign(dz_next) != np.sign(magnitude_m)
                and abs(z.iloc[i + 1] - z.iloc[i - 1]) <= 2.0 * sigma):
            out.update(classification="glitch", action="nan_spike_day")
            return out

    # Steps 4 + 5: Slope comparison or boundary
    pre_lo = jump_date - pd.Timedelta(days=slope_window_days)
    pre_hi = jump_date - pd.Timedelta(days=1)
    post_lo = jump_date + pd.Timedelta(days=1)
    post_hi = jump_date + pd.Timedelta(days=slope_window_days)
    z_pre = z.loc[(z.index >= pre_lo) & (z.index <= pre_hi)].dropna()
    z_post = z.loc[(z.index >= post_lo) & (z.index <= post_hi)].dropna()
    # Spec §3.3 step 5: "fewer than 30 days either side" → boundary_uncertain.
    # Enforce slope_window_days minimum on each side (not just 5).
    if len(z_pre) < slope_window_days or len(z_post) < slope_window_days:
        out.update(classification="boundary_uncertain", action="flag_only")
        return out
    slope_pre = _fit_slope_cmyr(z_pre)
    slope_post = _fit_slope_cmyr(z_post)
    out["slope_pre_cmyr"] = slope_pre
    out["slope_post_cmyr"] = slope_post
    if slope_pre is None or slope_post is None:
        out.update(classification="boundary_uncertain", action="flag_only")
        return out
    if abs(slope_post - slope_pre) <= parallel_slope_tol_cmyr:
        out.update(classification="datum_reset", action="rebaseline")
    else:
        out.update(classification="drift_onset", action="flag_only")
    return out


VALID_ACTIONS = {"nan_event_day", "nan_spike_day", "rebaseline", "flag_only"}


def apply_action(z: pd.Series, *, jump_date: pd.Timestamp,
                  magnitude_m: float, action: str) -> pd.Series:
    """Return modified series per the given action.

    Pure function — does not mutate input.
    """
    if action not in VALID_ACTIONS:
        raise ValueError(f"unknown action: {action!r}; expected one of {VALID_ACTIONS}")
    out = z.copy()
    if action in ("nan_event_day", "nan_spike_day"):
        out.loc[jump_date] = np.nan
    elif action == "rebaseline":
        out.loc[out.index >= jump_date] -= magnitude_m
    # flag_only: no modification
    return out
