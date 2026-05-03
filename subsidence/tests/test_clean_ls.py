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
