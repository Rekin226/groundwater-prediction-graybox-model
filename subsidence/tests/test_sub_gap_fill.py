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
