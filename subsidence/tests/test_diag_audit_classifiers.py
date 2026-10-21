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


def test_audit_cli_end_to_end_on_tmpdir(tmp_path):
    """Wire-up: minimal run-dir + h-driver-dir → CSV + summary written."""
    m = _mod()
    run_dir = tmp_path / "run"
    h_dir = tmp_path / "h_drivers"
    (run_dir / "per_station").mkdir(parents=True)
    h_dir.mkdir()

    # Minimal sub_fit_results.csv with one best-row
    pd.DataFrame([{
        "sub_id": "TEST", "sub_dataset": "ls-wra-gnss-obs", "variant": "M1",
        "is_best": True, "kge_val": 0.3, "rate_loss_active": True,
    }]).to_csv(run_dir / "sub_fit_results.csv", index=False)

    # h-driver with mixed source labels
    pd.DataFrame({
        "driver_source": ["observed"] * 60 + ["model_fill"] * 40,
    }).to_csv(h_dir / "TEST.csv", index=False)

    # _gpr.csv (need to span the cal/val windows)
    n = 1200
    dates = pd.date_range("2019-12-01", periods=n, freq="D")
    pd.DataFrame({
        "date": dates,
        "zeta_obs": np.cumsum(np.random.default_rng(0).normal(0, 0.0005, n)),
        "zeta_gpr_mean": 0.0,
        "zeta_gpr_sigma": 0.001,
        "imputed_mask": False,
        "render_mask": True,
        "sim_best": np.cumsum(np.random.default_rng(1).normal(0, 0.0005, n)),
    }).to_csv(run_dir / "per_station" / "TEST_gpr.csv", index=False)

    m._main(["--run-dir", str(run_dir), "--h-driver-dir", str(h_dir)])

    report = run_dir / "audit_classifiers_report.csv"
    summary = run_dir / "audit_classifiers_summary.txt"
    assert report.exists()
    assert summary.exists()
    out = pd.read_csv(report)
    assert out.shape[0] == 1
    expected_cols = {"sub_id", "fill_fraction", "fill_fraction_high",
                     "n_eff", "persistence_kge_val", "model_beats_persistence",
                     "kge_detrended_val", "rate_loss_active",
                     "val_obs_fraction", "driver_synthetic_val",
                     "nonstationary_coupling"}
    assert expected_cols.issubset(set(out.columns))


def test_driver_synthetic_val_fires_below_threshold():
    m = _mod()
    idx = pd.date_range("2024-01-01", "2025-03-31", freq="D")
    # Fully model_fill in the val window → 0% observed → flagged.
    synth = pd.DataFrame({"driver_source": ["model_fill"] * len(idx)}, index=idx)
    frac = m.compute_val_obs_fraction(synth, "2024-01-01", "2025-03-31")
    assert frac == 0.0
    assert m.flag_driver_synthetic_val(frac) is True
    # Mostly observed → not flagged.
    obs = pd.DataFrame({"driver_source": ["obs"] * len(idx)}, index=idx)
    frac2 = m.compute_val_obs_fraction(obs, "2024-01-01", "2025-03-31")
    assert frac2 == 1.0
    assert m.flag_driver_synthetic_val(frac2) is False


def test_driver_synthetic_val_nan_without_datetime_index():
    m = _mod()
    df = pd.DataFrame({"driver_source": ["model_fill"] * 10})  # no datetime index
    frac = m.compute_val_obs_fraction(df, "2024-01-01", "2025-03-31")
    assert np.isnan(frac)
    assert m.flag_driver_synthetic_val(frac) is False


def test_nonstationary_coupling_separates_breakdown_from_healthy():
    m = _mod()
    # High cal r, collapsed val r → flagged (the r-failure signature).
    assert m.flag_nonstationary_coupling(0.90, -0.31) is True
    assert m.flag_nonstationary_coupling(0.84, 0.15) is True
    # Healthy station keeps r_val ≥ 0.2 → never flagged.
    assert m.flag_nonstationary_coupling(0.85, 0.40) is False
    # Low cal r (model never matched) → not this family.
    assert m.flag_nonstationary_coupling(0.30, -0.10) is False
    # NaN (e.g. deep-retired, no val) → not flagged.
    assert m.flag_nonstationary_coupling(0.9, float("nan")) is False
