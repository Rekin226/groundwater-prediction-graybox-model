"""Tests for subsidence.clean_ls."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def clean_series():
    """Stationary series with σ ≈ 1 cm = 0.01 m of day-to-day noise."""
    rng = np.random.default_rng(42)
    idx = pd.date_range("2020-01-01", periods=400, freq="1D")
    z = np.cumsum(rng.normal(0.0, 0.01, size=len(idx)))
    return pd.Series(z, index=idx, name="z_m")


def test_compute_robust_sigma_clean_series(clean_series):
    from subsidence.clean_ls import compute_robust_sigma
    sigma = compute_robust_sigma(clean_series)
    # Expect ≈ 0.01 m (1 cm) within ±50% (small sample tolerance)
    assert 0.005 <= sigma <= 0.015


def test_compute_robust_sigma_resists_outliers(clean_series):
    """Inject a single 50 cm spike — MAD should barely move."""
    from subsidence.clean_ls import compute_robust_sigma
    sigma_clean = compute_robust_sigma(clean_series)
    spiked = clean_series.copy()
    spiked.iloc[200] += 0.50
    sigma_spiked = compute_robust_sigma(spiked)
    # MAD-based estimator should change by < 20%
    assert abs(sigma_spiked - sigma_clean) / sigma_clean < 0.20


def test_compute_robust_sigma_handles_nans(clean_series):
    from subsidence.clean_ls import compute_robust_sigma
    z = clean_series.copy()
    z.iloc[50:60] = np.nan
    sigma = compute_robust_sigma(z)
    assert sigma > 0  # should not crash, returns finite estimate


def test_detect_jumps_clean_series(clean_series):
    from subsidence.clean_ls import detect_jumps
    out = detect_jumps(clean_series, n_sigma=6.0, sigma_floor_cm=1.0)
    assert len(out) == 0


def test_detect_jumps_finds_injected_spike(clean_series):
    from subsidence.clean_ls import detect_jumps
    spiked = clean_series.copy()
    spike_date = spiked.index[200]
    spiked.iloc[200] += 0.20  # 20 cm spike
    out = detect_jumps(spiked, n_sigma=6.0, sigma_floor_cm=1.0)
    assert len(out) >= 1
    assert spike_date in set(out["date"])


def test_detect_jumps_respects_sigma_floor():
    """A near-zero-σ series shouldn't flag every point — floor protects it."""
    from subsidence.clean_ls import detect_jumps
    idx = pd.date_range("2020-01-01", periods=100, freq="1D")
    z = pd.Series(np.linspace(0, 1, 100) * 0.001, index=idx)  # σ ≈ 0
    z.iloc[50] += 0.005  # 0.5 cm — would exceed 6σ but below 1cm floor
    out = detect_jumps(z, n_sigma=6.0, sigma_floor_cm=1.0)
    assert len(out) == 0


def test_detect_jumps_returns_expected_columns(clean_series):
    from subsidence.clean_ls import detect_jumps
    spiked = clean_series.copy()
    spiked.iloc[200] += 0.20
    out = detect_jumps(spiked, n_sigma=6.0, sigma_floor_cm=1.0)
    assert {"date", "magnitude_m", "sigma_m", "n_sigma"}.issubset(out.columns)
    # magnitude is signed
    assert (out["magnitude_m"].abs() > 0.01).all()


def test_count_agreeing_neighbors_one_agrees():
    """One neighbor shows same-sign jump >4σ on the same day."""
    from subsidence.clean_ls import count_agreeing_neighbors
    idx = pd.date_range("2020-01-01", periods=100, freq="1D")
    rng = np.random.default_rng(0)
    n1 = pd.Series(np.cumsum(rng.normal(0, 0.01, len(idx))), index=idx)
    n2 = pd.Series(np.cumsum(rng.normal(0, 0.01, len(idx))), index=idx)
    n1.iloc[50] += 0.10  # 10 cm jump same direction as candidate
    jump_date = idx[50]
    n_agree = count_agreeing_neighbors(jump_date, [n1, n2], +1.0,
                                       n_sigma=4.0, time_window_days=1)
    assert n_agree == 1


def test_count_agreeing_neighbors_opposite_sign_doesnt_count():
    from subsidence.clean_ls import count_agreeing_neighbors
    idx = pd.date_range("2020-01-01", periods=100, freq="1D")
    rng = np.random.default_rng(0)
    n1 = pd.Series(np.cumsum(rng.normal(0, 0.01, len(idx))), index=idx)
    n1.iloc[50:] -= 0.10  # opposite sign (persistent level shift avoids rebound diff)
    n_agree = count_agreeing_neighbors(idx[50], [n1], +1.0,
                                       n_sigma=4.0, time_window_days=1)
    assert n_agree == 0


def test_count_agreeing_neighbors_within_time_window():
    """Jump on D+1 (within ±1 day window) still counts."""
    from subsidence.clean_ls import count_agreeing_neighbors
    idx = pd.date_range("2020-01-01", periods=100, freq="1D")
    rng = np.random.default_rng(0)
    n1 = pd.Series(np.cumsum(rng.normal(0, 0.01, len(idx))), index=idx)
    n1.iloc[51] += 0.10
    n_agree = count_agreeing_neighbors(idx[50], [n1], +1.0,
                                       n_sigma=4.0, time_window_days=1)
    assert n_agree == 1


# ---------------------------------------------------------------------------
# classify_jump tests (Task 7)
# ---------------------------------------------------------------------------

def _build_z_with_step(step_at_idx, step_m, n=200, sigma=0.0001, seed=0):
    """Helper: clean series + sustained step starting at step_at_idx."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="1D")
    z = np.cumsum(rng.normal(0, sigma, len(idx)))
    z[step_at_idx:] += step_m
    return pd.Series(z, index=idx)


def test_classify_jump_co_seismic_when_eq_match():
    from subsidence.clean_ls import classify_jump, compute_robust_sigma
    z = _build_z_with_step(100, 0.10)
    cat = pd.DataFrame({
        "time": pd.to_datetime(["2020-04-10T00:00:00"]),
        "latitude": [23.78], "longitude": [120.39],
        "depth": [10.0], "mag": [6.0], "id": ["E1"],
    })
    result = classify_jump(
        z, jump_date=z.index[100], magnitude_m=0.10,
        sigma=compute_robust_sigma(z),
        station_lat_lon=(23.78, 120.39),
        eq_catalog=cat, neighbor_series=[],
    )
    assert result["classification"] == "co_seismic"
    assert result["action"] == "nan_event_day"
    assert result["eq_id"] == "E1"


def test_classify_jump_regional_when_neighbors_agree():
    from subsidence.clean_ls import classify_jump, compute_robust_sigma
    z = _build_z_with_step(100, 0.10)
    n1 = _build_z_with_step(100, 0.08)
    n2 = _build_z_with_step(100, 0.06)
    result = classify_jump(
        z, jump_date=z.index[100], magnitude_m=0.10,
        sigma=compute_robust_sigma(z),
        station_lat_lon=(23.78, 120.39),
        eq_catalog=pd.DataFrame(columns=["time","latitude","longitude","depth","mag","id"]),
        neighbor_series=[n1, n2],
    )
    assert result["classification"] == "regional_event"
    assert result["action"] == "nan_event_day"


def test_classify_jump_glitch_snapback():
    """Single-day spike that returns to baseline."""
    from subsidence.clean_ls import classify_jump, compute_robust_sigma
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-01", periods=200, freq="1D")
    z = pd.Series(np.cumsum(rng.normal(0, 0.01, len(idx))), index=idx)
    z.iloc[100] += 0.10  # spike on day 100 only — snaps back day 101+
    result = classify_jump(
        z, jump_date=z.index[100], magnitude_m=0.10,
        sigma=compute_robust_sigma(z),
        station_lat_lon=(23.78, 120.39),
        eq_catalog=pd.DataFrame(columns=["time","latitude","longitude","depth","mag","id"]),
        neighbor_series=[],
    )
    assert result["classification"] == "glitch"
    assert result["action"] == "nan_spike_day"


def test_classify_jump_datum_reset_parallel_slopes():
    """Sustained step, post-jump slope ≈ pre-jump slope."""
    from subsidence.clean_ls import classify_jump, compute_robust_sigma
    z = _build_z_with_step(100, 0.10)  # sustained step, no slope change
    result = classify_jump(
        z, jump_date=z.index[100], magnitude_m=0.10,
        sigma=compute_robust_sigma(z),
        station_lat_lon=(23.78, 120.39),
        eq_catalog=pd.DataFrame(columns=["time","latitude","longitude","depth","mag","id"]),
        neighbor_series=[],
    )
    assert result["classification"] == "datum_reset"
    assert result["action"] == "rebaseline"


def test_classify_jump_drift_onset_diverging_slopes():
    """Sustained step + clear slope change post-jump."""
    from subsidence.clean_ls import classify_jump, compute_robust_sigma
    rng = np.random.default_rng(0)
    n = 200
    idx = pd.date_range("2020-01-01", periods=n, freq="1D")
    z = np.cumsum(rng.normal(0, 0.01, n))
    z[100:] += 0.10
    z[100:] += np.linspace(0, 0.05, n - 100)  # 5cm/100d post-jump drift
    result = classify_jump(
        pd.Series(z, index=idx), jump_date=idx[100], magnitude_m=0.10,
        sigma=compute_robust_sigma(pd.Series(z, index=idx)),
        station_lat_lon=(23.78, 120.39),
        eq_catalog=pd.DataFrame(columns=["time","latitude","longitude","depth","mag","id"]),
        neighbor_series=[],
    )
    assert result["classification"] == "drift_onset"
    assert result["action"] == "flag_only"


def test_classify_jump_boundary_uncertain_near_start():
    from subsidence.clean_ls import classify_jump, compute_robust_sigma
    # Jump at idx 5 in a 50-day series → only 5 days of pre-window
    # (slope_window_days=30 default) → boundary_uncertain.
    z = _build_z_with_step(5, 0.10, n=50)
    result = classify_jump(
        z, jump_date=z.index[5], magnitude_m=0.10,
        sigma=compute_robust_sigma(z),
        station_lat_lon=(23.78, 120.39),
        eq_catalog=pd.DataFrame(columns=["time","latitude","longitude","depth","mag","id"]),
        neighbor_series=[],
    )
    assert result["classification"] == "boundary_uncertain"
    assert result["action"] == "flag_only"


# ---------------------------------------------------------------------------
# apply_action tests (Task 8)
# ---------------------------------------------------------------------------

def test_apply_action_nan_event_day():
    from subsidence.clean_ls import apply_action
    idx = pd.date_range("2020-01-01", periods=10, freq="1D")
    z = pd.Series(np.arange(10, dtype=float), index=idx)
    out = apply_action(z, jump_date=idx[5], magnitude_m=0.10,
                       action="nan_event_day")
    assert np.isnan(out.iloc[5])
    assert out.iloc[4] == 4.0  # unchanged
    assert out.iloc[6] == 6.0  # unchanged


def test_apply_action_rebaseline_subtracts_jump():
    from subsidence.clean_ls import apply_action
    idx = pd.date_range("2020-01-01", periods=10, freq="1D")
    z = pd.Series(np.arange(10, dtype=float), index=idx)
    z.iloc[5:] += 100.0  # 100m datum reset at index 5
    out = apply_action(z, jump_date=idx[5], magnitude_m=100.0,
                       action="rebaseline")
    # Post-jump days should be subtracted by magnitude — back to original
    np.testing.assert_array_equal(out.values, np.arange(10, dtype=float))


def test_apply_action_flag_only_unchanged():
    from subsidence.clean_ls import apply_action
    idx = pd.date_range("2020-01-01", periods=10, freq="1D")
    z = pd.Series(np.arange(10, dtype=float), index=idx)
    out = apply_action(z, jump_date=idx[5], magnitude_m=0.10,
                       action="flag_only")
    np.testing.assert_array_equal(out.values, z.values)


def test_apply_action_unknown_raises():
    from subsidence.clean_ls import apply_action
    idx = pd.date_range("2020-01-01", periods=10, freq="1D")
    z = pd.Series(np.arange(10, dtype=float), index=idx)
    with pytest.raises(ValueError):
        apply_action(z, jump_date=idx[5], magnitude_m=0.10, action="bogus")


# ---------------------------------------------------------------------------
# clean_station tests (Task 9)
# ---------------------------------------------------------------------------

def test_clean_station_clean_input_no_changes(clean_series):
    from subsidence.clean_ls import clean_station
    out_z, qc = clean_station(
        z=clean_series, station_lat_lon=(23.78, 120.39),
        eq_catalog=pd.DataFrame(columns=["time","latitude","longitude","depth","mag","id"]),
        neighbor_series=[],
    )
    np.testing.assert_array_equal(out_z.values, clean_series.values)
    assert len(qc) == 0


def test_clean_station_handles_single_datum_reset():
    from subsidence.clean_ls import clean_station
    z = _build_z_with_step(100, 0.50)  # 50cm sustained step
    out_z, qc = clean_station(
        z=z, station_lat_lon=(23.78, 120.39),
        eq_catalog=pd.DataFrame(columns=["time","latitude","longitude","depth","mag","id"]),
        neighbor_series=[],
    )
    assert len(qc) == 1
    assert qc.iloc[0]["classification"] == "datum_reset"
    assert qc.iloc[0]["action"] == "rebaseline"
    # After rebaseline, post-jump days should align with pre-jump trajectory
    assert abs(out_z.iloc[101] - out_z.iloc[99]) < 0.05  # within ~σ


def test_clean_station_iterates_multiple_jumps():
    """Two sustained steps should both be detected (iteration after rebaseline)."""
    from subsidence.clean_ls import clean_station
    rng = np.random.default_rng(0)
    n = 400
    idx = pd.date_range("2020-01-01", periods=n, freq="1D")
    z = np.cumsum(rng.normal(0, 0.0001, n))  # use the same low-noise scale as Task 7
    z[100:] += 0.20    # first step
    z[300:] += -0.30   # second step
    out_z, qc = clean_station(
        z=pd.Series(z, index=idx), station_lat_lon=(23.78, 120.39),
        eq_catalog=pd.DataFrame(columns=["time","latitude","longitude","depth","mag","id"]),
        neighbor_series=[],
    )
    assert len(qc) >= 2


def test_detect_boxcar_anomalies_finds_antenna_swap():
    """+1m / -1m pair ~6 months apart = hardware glitch, not real motion."""
    from subsidence.clean_ls import detect_boxcar_anomalies
    rng = np.random.default_rng(0)
    idx = pd.date_range("2022-01-01", periods=800, freq="1D")
    z = pd.Series(np.cumsum(rng.normal(0, 0.001, 800)), index=idx)
    z.iloc[200:380] += 1.05   # 6-month elevated segment
    pairs = detect_boxcar_anomalies(z)
    assert len(pairs) == 1
    start, end, mag = pairs[0]
    assert start == idx[200]
    assert end == idx[380]
    assert 1.0 <= mag <= 1.10


def test_clean_station_nans_boxcar_segment():
    """clean_station should NaN the entire bogus segment, not just endpoints."""
    from subsidence.clean_ls import clean_station
    rng = np.random.default_rng(0)
    idx = pd.date_range("2022-01-01", periods=800, freq="1D")
    z = pd.Series(np.cumsum(rng.normal(0, 0.001, 800)), index=idx)
    z.iloc[200:380] += 1.05
    out_z, qc = clean_station(
        z=z, station_lat_lon=(23.78, 120.39),
        eq_catalog=pd.DataFrame(columns=["time","latitude","longitude","depth","mag","id"]),
        neighbor_series=[],
    )
    # Entire bogus segment should be NaN
    assert out_z.iloc[200:381].isna().all()
    # Surrounding clean data should be preserved
    assert out_z.iloc[:200].notna().all()
    assert out_z.iloc[381:].notna().all()
    # QC should record the boxcar
    assert (qc["classification"] == "boxcar_anomaly").any()


def test_detect_boxcar_anomalies_ignores_real_step():
    """A single sustained step (datum reset) is NOT a boxcar — leave it alone."""
    from subsidence.clean_ls import detect_boxcar_anomalies
    rng = np.random.default_rng(0)
    idx = pd.date_range("2022-01-01", periods=400, freq="1D")
    z = pd.Series(np.cumsum(rng.normal(0, 0.001, 400)), index=idx)
    z.iloc[200:] += 0.50  # sustained, no return
    assert detect_boxcar_anomalies(z) == []


def test_detect_boxcar_prefers_nearest_partner_over_distant():
    """Lone jump near other unrelated jumps shouldn't pair across hundreds of days.

    Scenario: a tight -7m/+7m pair on day 200/202 (obvious 2-day glitch),
    plus a lone -0.5m on day 100 and a lone +0.5m on day 400. The lone
    pair is 300 days apart with matching magnitude — old greedy first-match
    would pair them and NaN a huge swath. Score-based pairing should NaN
    the tight pair but leave the lone jumps alone (they're 300 days apart,
    a much worse score than the perfect day-200/202 pair).
    """
    from subsidence.clean_ls import detect_boxcar_anomalies
    rng = np.random.default_rng(0)
    n = 500
    idx = pd.date_range("2022-01-01", periods=n, freq="1D")
    z = pd.Series(np.cumsum(rng.normal(0, 0.001, n)), index=idx)
    # Tight glitch (the real boxcar)
    z.iloc[200:203] += 7.0
    # Lone sustained steps far apart that shouldn't pair (no immediate revert)
    z.iloc[100:200] -= 0.5
    z.iloc[400:] += 0.5
    pairs = detect_boxcar_anomalies(z)
    # Should find exactly the tight pair (days 200-202)
    assert len(pairs) == 1
    start, end, _ = pairs[0]
    assert start == idx[200]
    assert end == idx[203]


def test_detect_boxcar_consolidates_split_subjumps():
    """3-day cluster like [-0.89, -1.70, +2.61] should pair as boxcar.

    Without consolidation, the lone -0.89 vs +2.61 fails ratio test (0.34 < 0.5),
    and the cluster's down side is split across 2 sub-jumps. The fix sums
    consecutive same-sign jumps within 3 days into a single effective jump.
    """
    from subsidence.clean_ls import detect_boxcar_anomalies
    rng = np.random.default_rng(0)
    n = 500
    idx = pd.date_range("2022-01-01", periods=n, freq="1D")
    z = pd.Series(np.cumsum(rng.normal(0, 0.001, n)), index=idx)
    # Simulate split-subjump glitch: day 200 -0.89, day 201 -1.70, day 202 +2.61
    z.iloc[200:] -= 0.89
    z.iloc[201:] -= 1.70
    z.iloc[202:] += 2.61
    pairs = detect_boxcar_anomalies(z)
    assert len(pairs) == 1
    start, end, _ = pairs[0]
    assert start == idx[200]
    assert end == idx[202]
