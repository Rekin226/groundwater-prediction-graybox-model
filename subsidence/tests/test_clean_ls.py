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
