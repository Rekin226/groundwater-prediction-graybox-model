"""Unit tests for the poroelastic storage core.

Strategy: build synthetic daily surface + head with KNOWN trend, seasonal
amplitude, phase, and storativity, then check the core recovers them.
"""
import numpy as np
import pandas as pd
import pytest

from subsidence.poroelastic import (
    decompose, couple, phase_lag_days, storativity, specific_storage,
    analyze_station, is_elastic_valid, common_origin, bootstrap_ci,
)

W = 2 * np.pi  # 1 cycle / year (t in years)


def _daily(n_days=5 * 365, start="2020-01-01"):
    idx = pd.date_range(start, periods=n_days, freq="1D")
    t = (idx - idx[0]).total_seconds().to_numpy() / (365.25 * 86400.0)
    return idx, t


def _series(values, idx):
    return pd.Series(values, index=idx)


# ---------- decompose ----------

def test_decompose_recovers_amplitude_and_phase():
    idx, t = _daily()
    amp, phase = 0.012, 0.7
    y = 30.0 - 0.05 * t + amp * np.sin(W * t + phase)  # offset + linear trend + harmonic
    d = decompose(_series(y, idx))
    assert d.amplitude == pytest.approx(amp, abs=2e-4)
    # phase as atan2(B, A) of [sin, cos] basis: recovered phase == input phase
    assert d.phase == pytest.approx(phase, abs=0.02)
    assert d.trend_rate == pytest.approx(-0.05, abs=2e-3)


def test_decompose_seasonal_r2_high_for_harmonic_low_for_trend():
    idx, t = _daily()
    harmonic = _series(0.01 * np.sin(W * t) + 1e-5 * np.random.default_rng(0).standard_normal(len(t)), idx)
    pure_trend = _series(0.1 * t + 1e-4 * np.random.default_rng(1).standard_normal(len(t)), idx)
    assert decompose(harmonic).seasonal_r2 > 0.9
    assert decompose(pure_trend).seasonal_r2 < 0.1


def test_decompose_quadratic_trend_does_not_leak_into_seasonal():
    # a pure quadratic ramp (no harmonic) must yield ~zero seasonal amplitude
    idx, t = _daily()
    y = _series(0.2 * t + 0.3 * t**2, idx)
    assert decompose(y).amplitude < 1e-3


def test_decompose_preserves_nan_positions():
    idx, t = _daily(n_days=800)
    y = 0.01 * np.sin(W * t)
    y[100:120] = np.nan
    d = decompose(_series(y, idx))
    assert d.residual.iloc[100:120].isna().all()
    assert d.n == len(t) - 20


def test_decompose_raises_on_too_few_points():
    idx = pd.date_range("2020-01-01", periods=4, freq="1D")
    with pytest.raises(ValueError):
        decompose(pd.Series([1.0, 2.0, 3.0, 4.0], index=idx))


# ---------- couple ----------

def test_couple_in_phase_is_positive():
    idx, t = _daily()
    a = _series(0.01 * np.sin(W * t), idx)
    b = _series(2.0 * np.sin(W * t), idx)
    r, n = couple(a, b)
    assert r == pytest.approx(1.0, abs=1e-6)
    assert n > 1000


def test_couple_antiphase_is_negative():
    idx, t = _daily()
    a = _series(0.01 * np.sin(W * t), idx)
    b = _series(2.0 * np.sin(W * t + np.pi), idx)
    r, _ = couple(a, b)
    assert r == pytest.approx(-1.0, abs=1e-6)


def test_couple_handles_no_overlap():
    idx_a = pd.date_range("2020-01-01", periods=100, freq="1D")
    idx_b = pd.date_range("2030-01-01", periods=100, freq="1D")
    r, n = couple(pd.Series(np.arange(100.0), idx_a),
                  pd.Series(np.arange(100.0), idx_b))
    assert np.isnan(r) and n == 0


# ---------- storativity / specific storage ----------

def test_storativity_is_amplitude_ratio():
    assert storativity(0.01, 2.0) == pytest.approx(0.005)


def test_storativity_guards_zero_head():
    assert np.isnan(storativity(0.01, 0.0))


def test_specific_storage_and_compressibility():
    ss = specific_storage(0.005, thickness_m=50.0)
    assert ss == pytest.approx(1e-4)
    from subsidence.poroelastic import skeletal_compressibility
    assert skeletal_compressibility(ss) == pytest.approx(1e-4 / (1000.0 * 9.80665))


# ---------- analyze_station (end-to-end on synthetic) ----------

def test_analyze_station_recovers_known_s_ke():
    idx, t = _daily()
    s_ke_true, head_amp = 0.006, 3.0
    head = _series(20.0 - 0.4 * t + head_amp * np.sin(W * t), idx)
    disp = _series(30.0 - 0.05 * t + s_ke_true * head_amp * np.sin(W * t), idx)  # in-phase
    out = analyze_station(disp, head)
    assert out["s_ke"] == pytest.approx(s_ke_true, abs=3e-4)
    assert out["coupling_r"] == pytest.approx(1.0, abs=1e-3)
    assert abs(out["phase_lag_days"]) < 3.0  # in-phase => ~0 lag


def test_analyze_station_phase_lag_sign():
    idx, t = _daily()
    # surface peaks ~30 days BEFORE head -> negative lag
    shift = W * (30.0 / 365.25)
    head = _series(2.0 * np.sin(W * t), idx)
    disp = _series(0.01 * np.sin(W * t + shift), idx)
    lag = phase_lag_days(decompose(disp), decompose(head))
    assert lag == pytest.approx(-30.0, abs=2.0)


# ---------- phase epoch ----------

def _offset_pair(disp_start, head_start="2010-01-01", lag_days=20.0):
    """Head from `head_start`, displacement from `disp_start`, sharing one absolute
    annual cycle in which the surface lags head by `lag_days`."""
    h_idx = pd.date_range(head_start, "2025-03-31", freq="1D")
    d_idx = pd.date_range(disp_start, "2025-03-31", freq="1D")
    epoch = pd.Timestamp(head_start)
    yrs = lambda ix: (ix - epoch).total_seconds().to_numpy() / (365.25 * 86400.0)
    lag = W * (lag_days / 365.25)
    head = _series(2.0 * np.sin(W * yrs(h_idx)), h_idx)
    disp = _series(0.01 * np.sin(W * yrs(d_idx) - lag), d_idx)
    return disp, head


def test_phase_lag_is_invariant_to_record_start_offsets():
    """Regression: series starting on different dates must not shift the lag. A
    fractional-year offset between record starts previously leaked straight in."""
    for disp_start in ("2010-01-01", "2014-06-10", "2017-12-21", "2020-01-01"):
        disp, head = _offset_pair(disp_start, lag_days=20.0)
        t0 = common_origin(disp, head)
        lag = phase_lag_days(decompose(disp, t0=t0), decompose(head, t0=t0))
        assert lag == pytest.approx(20.0, abs=1.0), f"disp_start={disp_start}"


def test_phase_lag_rejects_mismatched_epochs():
    disp, head = _offset_pair("2014-06-10")
    with pytest.raises(ValueError, match="different epochs"):
        phase_lag_days(decompose(disp), decompose(head))


def test_amplitude_and_residual_are_invariant_to_t0():
    idx, t = _daily()
    y = _series(30.0 - 0.05 * t + 0.012 * np.sin(W * t + 0.7), idx)
    a = decompose(y)
    b = decompose(y, t0=pd.Timestamp("2010-01-01"))
    assert b.amplitude == pytest.approx(a.amplitude, rel=1e-9)
    assert b.residual.equals(a.residual) or np.allclose(b.residual, a.residual)
    assert b.phase != pytest.approx(a.phase, abs=1e-6)  # phase alone moves


# ---------- bootstrap block ----------

def test_bootstrap_block_is_per_series():
    """A longer block must widen the interval, so the head block has to be settable
    independently of the displacement block."""
    idx, t = _daily()
    rng = np.random.default_rng(3)
    head = _series(2.0 * np.sin(W * t) + 0.4 * np.cumsum(rng.standard_normal(len(t))) / 30, idx)
    disp = _series(0.01 * np.sin(W * t) + 1e-3 * rng.standard_normal(len(t)), idx)
    narrow = bootstrap_ci(disp, head, block_disp=30, block_head=3, n_boot=300, seed=0)
    wide = bootstrap_ci(disp, head, block_disp=30, block_head=30, n_boot=300, seed=0)
    span = lambda ci: ci[1] - ci[0]
    assert span(wide["head_amp_ci"]) > span(narrow["head_amp_ci"])
    assert wide["disp_amp"] == pytest.approx(narrow["disp_amp"])  # point est. unchanged


# ---------- QC gate ----------

def test_is_elastic_valid_gate():
    assert is_elastic_valid(0.5, 0.2) is True
    assert is_elastic_valid(0.3, 0.2) is False     # r below threshold
    assert is_elastic_valid(0.5, 0.05) is False    # seasonal r2 too low
    assert is_elastic_valid(-0.6, 0.5) is False    # antiphase rejected
    assert is_elastic_valid(np.nan, 0.5) is False
