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
