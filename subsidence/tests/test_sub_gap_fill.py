"""Tests for subsidence/sub_gap_fill.py — GPR-based ζ_obs gap-filling.

These tests verify visualization-only gap-fill behavior. The optimizer
in sub_shell.py is NOT exercised here; pipeline non-leakage is tested
separately in test_no_pipeline_leakage / test_pipeline_byte_identity.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest

from subsidence.sub_gap_fill import gpr_fill


def _make_t(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=n, freq="D")


def test_gap_fill_handles_too_few_observations():
    """< 30 finite samples → return (y, nan, isnan(y)); no GPR fit, no exception."""
    t = _make_t(100)
    y = np.full(100, np.nan)
    y[:25] = np.linspace(0.0, 0.05, 25)  # only 25 finite samples

    y_filled, sigma, mask = gpr_fill(t, y)

    assert y_filled.shape == (100,)
    assert sigma.shape == (100,)
    assert mask.shape == (100,)
    assert mask.dtype == bool
    np.testing.assert_array_equal(np.isnan(y_filled), np.isnan(y))
    assert np.all(np.isnan(sigma))
    np.testing.assert_array_equal(mask, np.isnan(y))


def test_gap_fill_handles_no_gaps():
    """Input with no NaN → imputed_mask all False, y_filled == y exactly."""
    t = _make_t(120)
    rng = np.random.default_rng(0)
    y = np.cumsum(0.001 + rng.normal(0, 0.0005, 120))

    y_filled, sigma, mask = gpr_fill(t, y)

    assert mask.sum() == 0
    np.testing.assert_array_equal(y_filled, y)
    assert sigma.shape == (120,)
    # sigma may be small but should be finite at observation points.
    assert np.all(np.isfinite(sigma))


def test_gap_fill_preserves_observed():
    """Real observations are returned bit-identically; only NaN cells altered."""
    t = _make_t(365)
    rng = np.random.default_rng(1)
    y_truth = np.cumsum(0.0008 + rng.normal(0, 0.0006, 365))
    y = y_truth.copy()
    # Knock out two windows
    y[60:90]   = np.nan   # 30-day gap
    y[200:240] = np.nan   # 40-day gap

    y_filled, sigma, mask = gpr_fill(t, y)

    # The observed cells are byte-identical
    obs_idx = ~mask
    np.testing.assert_array_equal(y_filled[obs_idx], y[obs_idx])
    # The imputed cells are no longer NaN
    assert np.all(np.isfinite(y_filled[mask]))
    # Mask matches original NaN pattern
    np.testing.assert_array_equal(mask, np.isnan(y))


def test_gap_fill_recovers_known_signal():
    """Synthetic linear-trend + annual-cycle signal: GPR posterior mean
    in a 60-day knockout window matches truth within 2σ everywhere.
    """
    n = 1500  # ~4 years daily
    t = _make_t(n)
    t_days = np.arange(n, dtype=float)
    truth = (
        0.001 * t_days
        + 0.005 * np.sin(2 * np.pi * t_days / 365.25)
    )
    rng = np.random.default_rng(2)
    y = truth + rng.normal(0, 0.0005, n)

    # Knock out a 60-day window in the middle (well-supported on both sides).
    gap_start, gap_end = 700, 760
    y[gap_start:gap_end] = np.nan

    y_filled, sigma, mask = gpr_fill(t, y)

    err = np.abs(y_filled[gap_start:gap_end] - truth[gap_start:gap_end])
    band = 2.0 * sigma[gap_start:gap_end]
    n_outside = int((err > band).sum())
    # Allow ≤5% of in-gap points to fall outside ±2σ (statistical headroom).
    assert n_outside <= 3, (
        f"{n_outside}/60 in-gap points outside ±2σ; max err = {err.max():.4e}"
    )


def test_estimate_obs_noise_mad_based():
    """MAD-based short-window σ — robust to outliers, matches σ on clean data."""
    from subsidence.sub_gap_fill import _estimate_obs_noise
    rng = np.random.default_rng(7)
    n = 600
    y_clean = np.cumsum(0.001 + rng.normal(0, 0.0008, n))

    sigma_clean = _estimate_obs_noise(y_clean, window=30)
    # MAD-derived σ should be within 50% of the true normal-noise σ (0.0008)
    assert 0.0004 < sigma_clean < 0.0016, f"clean σ = {sigma_clean}"

    # Inject an outlier — MAD must not blow up
    y_outlier = y_clean.copy()
    y_outlier[300] += 0.05  # huge outlier
    sigma_outlier = _estimate_obs_noise(y_outlier, window=30)
    assert abs(sigma_outlier - sigma_clean) / sigma_clean < 0.20, (
        f"MAD shifted >20% by single outlier: clean={sigma_clean}, "
        f"with outlier={sigma_outlier}"
    )


def test_module_constants_exposed():
    """The five derive-don't-tune constants must live at module scope."""
    import subsidence.sub_gap_fill as g
    for name in ("MAX_GAP_FRAC_OF_TRAIN", "ALPHA_NOISE_MULT",
                 "MAX_PLAUSIBLE_RATE_M_PER_YR",
                 "SIGMA_RENDER_FLOOR_M", "SIGMA_RENDER_FLOOR_FRAC"):
        assert hasattr(g, name), f"missing constant: {name}"
