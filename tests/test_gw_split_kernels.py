"""Tests for memory kernels, datum-invariant metrics and the cal/sel/test split.

The leakage tests perturb every record after SEL_END and check that nothing
decided from earlier periods changes: lags, fitted parameters, calibration and
selection metrics, and the selected variant.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "srcs"))
sys.path.insert(0, str(REPO.parent))  # rklib lives next to the repo

import gw_shell  # noqa: E402
import gw_subroutine as sub  # noqa: E402


# --------------------------------------------------------------------------
# Kernels
# --------------------------------------------------------------------------

@pytest.mark.parametrize("tau", [0.1, 1.0, 5.0, 30.0])
def test_kernel_has_unit_sum(tau):
    w = sub.exp_kernel(tau, 90)
    assert w.sum() == pytest.approx(1.0)
    assert np.all(np.diff(w) <= 0)


@pytest.mark.parametrize("tau", [0.1, 5.0, 30.0])
def test_constant_head_passes_through_unchanged(tau):
    h = np.full(400, 12.3)
    out = sub.memory_convolve(h, tau, 90, pad_value=float(h[0]))
    np.testing.assert_allclose(out, 12.3)


def test_rainfall_pulse_response_conserves_mass():
    rain = np.zeros(400)
    rain[10] = 50.0
    out = sub.memory_convolve(rain, 5.0, 90, pad_value=0.0)
    assert out[:10].sum() == 0.0  # causal
    assert out.sum() == pytest.approx(50.0)


def test_memory_convolve_matches_direct_sum():
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    tau, L = 4.0, 30
    w = sub.exp_kernel(tau, L)
    out = sub.memory_convolve(x, tau, L, pad_value=x[0])
    i = 120
    direct = sum(w[k] * x[i - k] for k in range(L + 1))
    assert out[i] == pytest.approx(direct)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def _obs_pred():
    rng = np.random.default_rng(1)
    obs = np.sin(np.linspace(0, 20, 500)) + rng.normal(scale=0.1, size=500)
    pred = 0.9 * obs + 0.05 + rng.normal(scale=0.1, size=500)
    return obs, pred


def test_kge_ab_is_datum_invariant_and_classic_kge_is_not():
    obs, pred = _obs_pred()
    m0 = gw_shell.compute_metrics(obs, pred)
    m1 = gw_shell.compute_metrics(obs + 100.0, pred + 100.0)
    assert m1["kge_ab"] == pytest.approx(m0["kge_ab"])
    assert m1["rmse"] == pytest.approx(m0["rmse"])
    assert m1["nse"] == pytest.approx(m0["nse"])
    assert abs(m1["kge"] - m0["kge"]) > 1e-3


def test_kge_ab_perfect_fit_and_bias_term():
    obs, _ = _obs_pred()
    assert gw_shell.compute_metrics(obs, obs)["kge_ab"] == pytest.approx(1.0)
    shifted = gw_shell.compute_metrics(obs, obs + np.std(obs))
    assert shifted["kge_ab_beta"] == pytest.approx(1.0)
    assert shifted["kge_ab"] == pytest.approx(0.0)


def test_short_period_metrics_are_nan():
    m = gw_shell.period_metrics(np.arange(5.0), np.arange(5.0))
    assert all(np.isnan(v) for v in m.values())


def test_selection_score_ignores_test_period():
    def res(cal, sel, test):
        return {"metrics": {"cal": {"kge_ab": cal}, "sel": {"kge_ab": sel}, "test": {"kge_ab": test}}}
    assert gw_shell.selection_score(res(0.9, 0.5, 0.1)) == 0.5
    assert gw_shell.selection_score(res(0.9, 0.5, 0.99)) == 0.5
    assert gw_shell.selection_score(res(0.9, np.nan, 0.99)) == 0.9


def test_period_bounds():
    idx = pd.date_range("2018-12-30", "2021-01-02", freq="D")
    n_cal, n_sel_end = gw_shell.period_bounds(idx)
    assert idx[n_cal - 1] == pd.Timestamp(gw_shell.CAL_END)
    assert idx[n_cal] == pd.Timestamp("2019-01-01")
    assert idx[n_sel_end - 1] == pd.Timestamp(gw_shell.SEL_END)
    assert idx[n_sel_end] == pd.Timestamp("2021-01-01")


# --------------------------------------------------------------------------
# STFT alignment
# --------------------------------------------------------------------------

def _hourly_series(seed=2):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-01", "2019-06-30 23:00", freq="h")
    t = np.arange(len(idx)) / 24.0
    y = (0.05 * np.sin(2 * np.pi * 1.0 * t) + 0.03 * np.sin(2 * np.pi * 1.9323 * t)
         + rng.normal(scale=0.01, size=len(idx)))
    return pd.Series(y, index=idx)


@pytest.mark.parametrize("align, causal", [("trailing", True), ("centre", False)])
def test_stft_alignment_causality(monkeypatch, align, causal):
    monkeypatch.setattr(gw_shell, "STFT_ALIGN", align)
    s = _hourly_series()
    cut = pd.Timestamp("2018-12-31")
    s2 = s.copy()
    s2[s2.index > cut + pd.Timedelta(hours=23)] *= 5.0
    amp1, _ = gw_shell._compute_amp_amt(s)
    amp2, _ = gw_shell._compute_amp_amt(s2)
    a1 = amp1.loc[:cut, "amp"].dropna()
    a2 = amp2.loc[:cut, "amp"].dropna()
    unchanged = np.allclose(a1.values, a2.values)
    assert unchanged == causal


# --------------------------------------------------------------------------
# End-to-end leakage: perturbing the test period changes nothing upstream of it
# --------------------------------------------------------------------------

def _synthetic_merge(seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2016-07-01", "2022-12-31", freq="D")
    n = len(idx)
    rf = rng.gamma(0.3, 20.0, size=n) * (rng.random(n) < 0.3)
    amp = 0.02 + 0.005 * rng.random(n)
    amt = 0.01 + 0.005 * rng.random(n)
    ups = 20.0 + np.cumsum(rng.normal(scale=0.05, size=n)) + 0.5 * np.sin(2 * np.pi * idx.dayofyear / 365.25)
    params = (0.05, 10.0, 0.02, 0.5, 0.05, 5.0, 1.0, 0.0, 0.0)
    gwl = sub.simulate_inland(params, np.arange(n), rf, amp, ups, h0=10.0,
                              doy=idx.dayofyear.values.astype(float))
    gwl = gwl + rng.normal(scale=0.02, size=n)
    return pd.DataFrame({"gwl": gwl, "ups_gwl": ups, "rf": rf, "amp": amp, "amt": amt}, index=idx)


def _perturb_test(df):
    out = df.copy()
    test = out.index > pd.Timestamp(gw_shell.SEL_END)
    out.loc[test, "gwl"] += 5.0
    out.loc[test, "ups_gwl"] -= 3.0
    out.loc[test, "rf"] *= 4.0
    return out


@pytest.mark.slow
def test_test_period_does_not_leak_into_fit_or_selection():
    df = _synthetic_merge()
    df_p = _perturb_test(df)

    lags = gw_shell.estimate_lags_calibration(df, no_upstream=False, max_lag=45)
    lags_p = gw_shell.estimate_lags_calibration(df_p, no_upstream=False, max_lag=45)
    assert lags == lags_p

    fits, fits_p = [], []
    for model in ("base", "base_tz"):
        kw = dict(group_name="inland", rain_lag_days=lags[0], up_lag_days=lags[1],
                  no_upstream=False, model_name=model, n_starts=3)
        fits.append(gw_shell._fit_model(df, **kw))
        fits_p.append(gw_shell._fit_model(df_p, **kw))

    for r, rp in zip(fits, fits_p):
        np.testing.assert_array_equal(r["params"], rp["params"])
        n = r["n_sel_end"]
        np.testing.assert_array_equal(r["y_full"][:n], rp["y_full"][:n])
        for p in ("cal", "sel"):
            assert r["metrics"][p] == rp["metrics"][p]
        assert r["metrics"]["test"] != rp["metrics"]["test"]

    best = max(fits, key=gw_shell.selection_score)["model"]
    best_p = max(fits_p, key=gw_shell.selection_score)["model"]
    assert best == best_p


@pytest.mark.slow
def test_fixed_parameters_are_held_and_parse():
    assert gw_shell.parse_fixed("c=0; k_link=0.0") == {"c": 0.0, "k_link": 0.0}
    assert gw_shell.parse_fixed(float("nan")) == {}
    df = _synthetic_merge()
    r = gw_shell._fit_model(df, group_name="inland", rain_lag_days=0, up_lag_days=0,
                            no_upstream=False, model_name="base", n_starts=3,
                            fixed={"c": 0.0, "tau_rain": 12.0})
    p = dict(zip(r["param_names"], r["params"]))
    assert p["c"] == pytest.approx(0.0, abs=1e-8)
    assert p["tau_rain"] == pytest.approx(12.0, rel=1e-8)
    with pytest.raises(KeyError):
        gw_shell._fit_model(df, group_name="inland", rain_lag_days=0, up_lag_days=0,
                            no_upstream=False, model_name="base", n_starts=3,
                            fixed={"k_sgd": 0.1})
