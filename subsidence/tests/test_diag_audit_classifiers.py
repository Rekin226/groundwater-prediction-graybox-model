"""Unit tests for §7/§8 audit classifiers."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
import importlib.util, sys, pathlib


def _mod():
    p = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "diag_audit_classifiers.py"
    spec = importlib.util.spec_from_file_location("diag_audit_classifiers", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules["diag_audit_classifiers"] = m
    spec.loader.exec_module(m)
    return m


def test_fill_fraction_counts_synthetic_columns_correctly():
    m = _mod()
    df = pd.DataFrame({
        "driver_source": ["observed"] * 60 + ["model_fill"] * 30 + ["taper"] * 10
    })
    frac = m.compute_fill_fraction(df)
    # 40/100 = 0.40 non-observed
    assert abs(frac - 0.40) < 1e-9
    assert m.flag_high_fill_fraction(frac) is False


def test_high_fill_fraction_fires_when_above_50pct():
    m = _mod()
    df = pd.DataFrame({"driver_source": ["model_fill"] * 60 + ["observed"] * 40})
    frac = m.compute_fill_fraction(df)
    assert frac == 0.60
    assert m.flag_high_fill_fraction(frac) is True


def test_n_eff_from_lag1_autocorrelation():
    m = _mod()
    rng = np.random.default_rng(0)
    y = np.cumsum(0.001 + rng.normal(0, 0.0005, 1000))
    n_eff = m.compute_n_eff(y)
    # Lag-1 ρ for a cumulative random walk approaches 1 → n_eff is much
    # smaller than n
    assert n_eff < 50, f"n_eff = {n_eff} should be ≪ 1000 for cumulative"


def test_persistence_kge_outperforms_zero_on_drifting_series():
    m = _mod()
    n = 500
    y_train = np.linspace(0.0, 0.05, 300)
    y_val = np.linspace(0.05, 0.10, 200)
    kge = m.persistence_kge_benchmark(y_train, y_val)
    assert kge > 0.5


def test_kge_detrended_strips_total_trend():
    m = _mod()
    n = 300
    y_obs = np.linspace(0, 0.10, n) + 0.01 * np.sin(np.arange(n) / 30)
    y_sim = np.linspace(0, 0.10, n)
    kge_raw = m.kge_on_detrended(y_obs, y_sim)
    assert kge_raw < 0.9
