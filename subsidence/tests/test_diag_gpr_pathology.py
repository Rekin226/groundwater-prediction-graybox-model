"""Unit tests for subsidence/scripts/diag_gpr_pathology.py — five GPR
pathology classifier flags. Each test feeds synthetic obs + GPR_mean
arrays and asserts the flag fires (or stays silent) per the spec
signature definitions.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest


def _import_classifier():
    import importlib.util, sys, pathlib
    p = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "diag_gpr_pathology.py"
    spec = importlib.util.spec_from_file_location("diag_gpr_pathology", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["diag_gpr_pathology"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_over_smoothing_fires_when_residual_too_small():
    """std(obs − GPR_mean) / std(obs) < 0.30 → flag True."""
    mod = _import_classifier()
    rng = np.random.default_rng(0)
    n = 500
    truth = np.cumsum(0.001 + rng.normal(0, 0.001, n))
    obs = truth + rng.normal(0, 0.001, n)
    gpr_mean = truth
    imputed_mask = np.zeros(n, dtype=bool)
    flag, ratio = mod.flag_over_smoothing(obs, gpr_mean, imputed_mask)
    assert flag is True
    assert 0.0 < ratio < 0.30


def test_over_smoothing_stays_silent_when_gpr_honors_noise():
    """A noisy GPR (residual std close to obs std) → flag False."""
    mod = _import_classifier()
    rng = np.random.default_rng(1)
    n = 500
    obs = rng.normal(0, 0.01, n)
    gpr_mean = obs + rng.normal(0, 0.01, n)
    imputed_mask = np.zeros(n, dtype=bool)
    flag, ratio = mod.flag_over_smoothing(obs, gpr_mean, imputed_mask)
    assert flag is False
    assert ratio >= 0.30


def test_outlier_spike_fires_when_gpr_exits_obs_range():
    mod = _import_classifier()
    obs = np.linspace(0.0, 0.05, 100)
    gpr = obs.copy()
    gpr[50] = obs.max() + 3 * (obs.max() - obs.min())
    mask = np.zeros(100, dtype=bool); mask[50] = True
    flag, val = mod.flag_outlier_spike(obs, gpr, mask)
    assert flag is True


def test_outlier_spike_silent_within_range():
    mod = _import_classifier()
    obs = np.linspace(0.0, 0.05, 100)
    gpr = obs.copy() + 0.005
    mask = np.zeros(100, dtype=bool)
    flag, _ = mod.flag_outlier_spike(obs, gpr, mask)
    assert flag is False


def test_extrapolation_drift_fires_far_from_obs_with_big_offset():
    mod = _import_classifier()
    n = 1000
    obs = np.full(n, np.nan)
    obs[:200] = 0.0
    gpr_mean = np.zeros(n)
    gpr_mean[800] = 0.5
    imputed_mask = np.isnan(obs)
    flag, _ = mod.flag_extrapolation_drift(
        obs, gpr_mean, imputed_mask,
        l_train_days=200.0, sigma_obs=0.001,
    )
    assert flag is True


def test_extrapolation_drift_silent_when_close_or_small_offset():
    mod = _import_classifier()
    n = 500
    obs = np.full(n, np.nan)
    obs[200:300] = 0.0
    gpr_mean = np.zeros(n)
    gpr_mean[210] = 0.001
    imputed_mask = np.isnan(obs)
    flag, _ = mod.flag_extrapolation_drift(
        obs, gpr_mean, imputed_mask,
        l_train_days=100.0, sigma_obs=0.001,
    )
    assert flag is False


def test_render_dominance_fires_when_imputed_rendered_exceeds_30pct():
    mod = _import_classifier()
    n = 1000
    imputed_mask = np.zeros(n, dtype=bool)
    imputed_mask[:400] = True
    render_mask = np.ones(n, dtype=bool)
    flag, frac = mod.flag_render_dominance(imputed_mask, render_mask)
    assert flag is True
    assert frac > 0.30


def test_render_dominance_silent_when_most_masked_out():
    mod = _import_classifier()
    n = 1000
    imputed_mask = np.zeros(n, dtype=bool)
    imputed_mask[:400] = True
    render_mask = np.ones(n, dtype=bool)
    render_mask[:350] = False
    flag, frac = mod.flag_render_dominance(imputed_mask, render_mask)
    assert flag is False
    assert frac < 0.30


def test_narrow_band_fires_when_sigma_too_small():
    mod = _import_classifier()
    n = 500
    obs = np.random.default_rng(0).normal(0, 0.01, n)
    sigma = np.full(n, 0.0001)
    imputed_mask = np.zeros(n, dtype=bool)
    imputed_mask[100:300] = True
    flag, ratio = mod.flag_narrow_band(obs, sigma, imputed_mask)
    assert flag is True


def test_narrow_band_silent_when_sigma_reasonable():
    mod = _import_classifier()
    n = 500
    obs = np.random.default_rng(0).normal(0, 0.01, n)
    sigma = np.full(n, 0.005)
    imputed_mask = np.zeros(n, dtype=bool)
    imputed_mask[100:300] = True
    flag, _ = mod.flag_narrow_band(obs, sigma, imputed_mask)
    assert flag is False
