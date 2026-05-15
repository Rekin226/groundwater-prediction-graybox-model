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
