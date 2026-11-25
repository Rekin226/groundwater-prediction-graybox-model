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

PHYS_PLAUSIBLE_INSTANTANEOUS_M = 0.30
BOXCAR_MAX_GAP_DAYS = 400
BOXCAR_MAG_RATIO_TOL = 0.5
BOXCAR_LONG_GAP_MIN_MAG_M = 1.0
BOXCAR_SHORT_GAP_DAYS = 30
BOXCAR_CLUSTER_MAX_GAP_DAYS = 3


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


def detect_boxcar_anomalies(z: pd.Series, *,
                             n_sigma: float = 6.0,
                             sigma_floor_cm: float = 1.0,
                             min_magnitude_m: float = PHYS_PLAUSIBLE_INSTANTANEOUS_M,
                             max_gap_days: int = BOXCAR_MAX_GAP_DAYS,
                             mag_ratio_tol: float = BOXCAR_MAG_RATIO_TOL,
                             ) -> List[Tuple[pd.Timestamp, pd.Timestamp, float]]:
    """Find opposite-sign jump pairs bracketing a hardware-glitch segment.

    A boxcar anomaly is one jump up by ~X m and one jump down by ~X m within
    max_gap_days. The interval between them is the bogus segment (antenna
    swap, firmware reset). Magnitudes below min_magnitude_m are physically
    plausible ground motion and excluded.

    Returns list of (start_date, end_date, averaged_magnitude_m) tuples.
    """
    jumps = detect_jumps(z, n_sigma=n_sigma, sigma_floor_cm=sigma_floor_cm)
    if jumps.empty:
        return []
    jumps = jumps[jumps["magnitude_m"].abs() >= min_magnitude_m].sort_values("date")
    if len(jumps) < 2:
        return []
    # Consolidate consecutive same-sign jumps within BOXCAR_CLUSTER_MAX_GAP_DAYS
    # so e.g. [-0.89, -1.70, +2.61] across 3 days becomes [-2.59, +2.61] (a
    # well-matched pair). Without this, a multi-day glitch with split sub-jumps
    # leaves the down side fragmented and unpairable.
    consolidated: List[Dict[str, Any]] = []
    cur_total = float(jumps.iloc[0]["magnitude_m"])
    cur_start = jumps.iloc[0]["date"]
    for k in range(1, len(jumps)):
        nxt = jumps.iloc[k]
        same_sign = np.sign(nxt["magnitude_m"]) == np.sign(cur_total)
        within_cluster = (nxt["date"] - cur_start).days <= BOXCAR_CLUSTER_MAX_GAP_DAYS
        if same_sign and within_cluster:
            cur_total += float(nxt["magnitude_m"])
        else:
            consolidated.append({"date": cur_start, "magnitude_m": cur_total})
            cur_total = float(nxt["magnitude_m"])
            cur_start = nxt["date"]
    consolidated.append({"date": cur_start, "magnitude_m": cur_total})
    rows = pd.DataFrame(consolidated)
    if len(rows) < 2:
        return []
    # Score-based pairing: enumerate all candidate pairs, score each by gap
    # (smaller = better) and magnitude ratio (closer to 1 = better), then
    # greedily accept best-scoring pairs. Prevents a lone jump from latching
    # onto a distant, weakly-matched partner.
    candidates: List[Tuple[float, int, int]] = []
    for i in range(len(rows) - 1):
        for j in range(i + 1, len(rows)):
            a, b = rows.iloc[i], rows.iloc[j]
            gap_days = (b["date"] - a["date"]).days
            if gap_days > max_gap_days:
                break
            if np.sign(a["magnitude_m"]) == np.sign(b["magnitude_m"]):
                continue
            mag_a, mag_b = abs(a["magnitude_m"]), abs(b["magnitude_m"])
            ratio = min(mag_a, mag_b) / max(mag_a, mag_b)
            if ratio < (1.0 - mag_ratio_tol):
                continue
            # Magnitude-dependent gap: small (< BOXCAR_LONG_GAP_MIN_MAG_M)
            # boxcars must be tight in time; larger magnitudes can span longer
            # (e.g., a 6-month antenna swap leaves a >1 m offset).
            if max(mag_a, mag_b) < BOXCAR_LONG_GAP_MIN_MAG_M and gap_days > BOXCAR_SHORT_GAP_DAYS:
                continue
            score = (1.0 - gap_days / max_gap_days) + ratio
            candidates.append((score, i, j))
    candidates.sort(reverse=True)
    pairs: List[Tuple[pd.Timestamp, pd.Timestamp, float]] = []
    used: set[int] = set()
    for _, i, j in candidates:
        if i in used or j in used:
            continue
        a, b = rows.iloc[i], rows.iloc[j]
        mag = (abs(a["magnitude_m"]) + abs(b["magnitude_m"])) / 2.0
        pairs.append((a["date"], b["date"], mag))
        used.add(i); used.add(j)
    pairs.sort(key=lambda t: t[0])
    return pairs


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


def clean_station(*, z: pd.Series,
                   station_lat_lon: Tuple[float, float],
                   eq_catalog: pd.DataFrame,
                   neighbor_series: List[pd.Series],
                   n_sigma: float = 6.0,
                   sigma_floor_cm: float = 1.0,
                   max_iterations: int = 20,
                   **classify_kwargs,
                   ) -> Tuple[pd.Series, pd.DataFrame]:
    """Iteratively detect → classify → apply until no jumps remain.

    Returns: (cleaned_series, qc_dataframe).
    QC dataframe columns: jump_date, magnitude_m, sigma_m, n_sigma,
    classification, action, eq_id, eq_distance_km, eq_magnitude, eq_depth_km,
    n_neighbors_agree, slope_pre_cmyr, slope_post_cmyr.
    """
    z_curr = z.copy()
    qc_rows: List[Dict[str, Any]] = []

    # Boxcar pre-pass: NaN out hardware-glitch segments before the iterative
    # jump-by-jump loop runs. The iterative path can't NaN spans, only single
    # days — so a +1 m / −1 m pair months apart (antenna swap) would leave a
    # bogus elevated segment in the middle.
    for start_date, end_date, mag in detect_boxcar_anomalies(
            z_curr, n_sigma=n_sigma, sigma_floor_cm=sigma_floor_cm):
        seg_mask = (z_curr.index >= start_date) & (z_curr.index <= end_date)
        qc_rows.append({
            "jump_date": start_date, "magnitude_m": float(mag),
            "sigma_m": float("nan"), "n_sigma": float("nan"),
            "classification": "boxcar_anomaly", "action": "nan_segment",
            "eq_id": None, "eq_distance_km": None, "eq_magnitude": None,
            "eq_depth_km": None, "n_neighbors_agree": 0,
            "slope_pre_cmyr": None, "slope_post_cmyr": None,
        })
        z_curr = z_curr.copy()
        z_curr.loc[z_curr.index[seg_mask]] = np.nan

    for iteration in range(max_iterations):
        jumps = detect_jumps(z_curr, n_sigma=n_sigma, sigma_floor_cm=sigma_floor_cm)
        if jumps.empty:
            break
        # Process the EARLIEST jump first (so subsequent σ recomputation is meaningful)
        first = jumps.sort_values("date").iloc[0]
        result = classify_jump(
            z=z_curr,
            jump_date=first["date"],
            magnitude_m=float(first["magnitude_m"]),
            sigma=float(first["sigma_m"]),
            station_lat_lon=station_lat_lon,
            eq_catalog=eq_catalog,
            neighbor_series=neighbor_series,
            **classify_kwargs,
        )
        qc_rows.append({
            "jump_date": first["date"],
            "magnitude_m": float(first["magnitude_m"]),
            "sigma_m": float(first["sigma_m"]),
            "n_sigma": float(first["n_sigma"]),
            **result,
        })
        if result["action"] == "flag_only":
            # Don't iterate on flag_only — it'd loop forever. Mark and skip.
            # Strategy: temporarily NaN the day so we don't re-detect this jump,
            # then restore at the end via the flag_only_dates set.
            qc_rows[-1]["_skip"] = True
            z_curr = z_curr.copy()
            z_curr.loc[first["date"]] = np.nan
        else:
            z_curr = apply_action(z_curr,
                                  jump_date=first["date"],
                                  magnitude_m=float(first["magnitude_m"]),
                                  action=result["action"])

    # Restore flag_only days that were temporarily NaN'd for iteration
    flag_only_dates = [r["jump_date"] for r in qc_rows if r.get("_skip")]
    for d in flag_only_dates:
        z_curr.loc[d] = z.loc[d]
    for r in qc_rows:
        r.pop("_skip", None)

    qc = pd.DataFrame(qc_rows)
    return z_curr, qc
