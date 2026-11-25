"""Test-period comparison of the selected grey-box model against simple baselines.

Baselines (all built from records before the test period only, except
seasonal-naive which by definition uses the observation one year earlier):

  persistence     last observed head before the test period, held constant
  climatology     pre-test mean plus a smoothed day-of-year anomaly cycle
  seasonal_naive  observed head 365 days earlier (one-year-ahead forecast)
  ar1             AR(1) on daily anomalies from climatology, fitted before
                  the test period and iterated forward from its last value
  reduced         rainfall-only single tank (separate run; optional)
  regression      information-matched linear regression: the same observed
                  inputs as the grey-box model (upstream head, rainfall through
                  exponential kernels of 3/10/30/90 d, AMP, AMT at coastal
                  stations, annual harmonics), fitted by least squares on the
                  calibration period and run over the test period without
                  using observed target heads

Note: the grey-box model is driven by observed rainfall, upstream heads and
AMP/AMT during the test period (reconstruction mode); the statistical
baselines see no test-period forcing. The reduced model is the like-for-like
comparison of structure.

Usage:
    python scripts/analysis/benchmarks.py [--run ...] [--reduced workspace/results/bench_reduced]
"""
import argparse

import numpy as np
import pandas as pd

from common import REPO, analysis_dir, gw_shell, load_results, load_series, run_dir, station_sort_key


def climatology(dates: pd.Series, obs: np.ndarray, train: np.ndarray) -> np.ndarray:
    """Mean + 31-day circularly smoothed DOY anomaly cycle from the training rows."""
    doy = dates.dt.dayofyear.clip(upper=365).to_numpy()
    mu = obs[train].mean()
    anom = pd.Series(obs[train] - mu).groupby(doy[train]).mean().reindex(range(1, 366)).interpolate(limit_direction="both")
    a = anom.to_numpy()
    k = 15
    ext = np.concatenate([a[-k:], a, a[:k]])
    smooth = np.convolve(ext, np.ones(2 * k + 1) / (2 * k + 1), mode="valid")
    return mu + smooth[doy - 1]


def seasonal_naive(dates: pd.Series, obs: np.ndarray) -> np.ndarray:
    s = pd.Series(obs, index=pd.DatetimeIndex(dates))
    lagged = s.reindex(s.index - pd.Timedelta(days=365))
    return lagged.to_numpy()


def ar1_forecast(clim: np.ndarray, obs: np.ndarray, train: np.ndarray, test: np.ndarray) -> np.ndarray:
    anom = obs - clim
    x = anom[train]
    phi = float(np.corrcoef(x[:-1], x[1:])[0, 1])
    last = x[-1]
    steps = np.arange(1, test.sum() + 1)
    out = np.full(len(obs), np.nan)
    out[test] = clim[test] + last * phi ** steps
    return out


RAIN_TAUS = (3.0, 10.0, 30.0, 90.0)


def regression_benchmark(res_row: pd.Series, st: str) -> pd.Series:
    """Calibration-fitted linear regression on the grey-box model's own inputs."""
    ups = res_row.get("ups_id", "none")
    df_merge, no_up = gw_shell.prepare_data({"st_id": st, "ups_id": ups, "rf_id": res_row["rf_id"]})
    t, rain, amp, amt, h_up, h_obs, idx, doy = gw_shell.fit_preprocess(
        df_merge, int(res_row["rain_lag_days"]), int(res_row["up_lag_days"]))
    n_cal, _ = gw_shell.period_bounds(idx)
    cols = [np.ones(len(t))]
    cols += [gw_shell.sub.memory_convolve(rain, tau, 180, 0.0) for tau in RAIN_TAUS]
    if not no_up:
        cols.append(h_up)
    cols.append(amp)
    if str(res_row["group_name"]).lower() == "coastal":
        cols.append(amt)
    ang = 2 * np.pi * doy / 365.25
    cols += [np.sin(ang), np.cos(ang)]
    X = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(X[:n_cal], h_obs[:n_cal], rcond=None)
    return pd.Series(X @ beta, index=idx)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    ap.add_argument("--reduced", default="workspace/results/bench_reduced")
    args = ap.parse_args()
    run = run_dir(args.run)
    res = load_results(run)
    red_dir = REPO / args.reduced
    red = load_results(red_dir) if (red_dir / "gw_fit_results.csv").exists() else None

    rows = []
    for st in sorted(res.index, key=station_sort_key):
        s = load_series(run, st)
        obs = s["obs"].to_numpy()
        test = (s["period"] == "test").to_numpy()
        train = ~test
        preds = {
            "greybox": s[f"sim_{res.loc[st, 'model']}"].to_numpy(),
            "persistence": np.where(test, obs[train][-1], np.nan),
        }
        clim = climatology(s["date"], obs, train)
        preds["climatology"] = clim
        preds["seasonal_naive"] = seasonal_naive(s["date"], obs)
        preds["ar1"] = ar1_forecast(clim, obs, train, test)
        if red is not None and st in red.index:
            sr = load_series(red_dir, st).set_index("date")["sim_base"]
            # Different lags drop different leading rows, so align on date.
            preds["reduced"] = sr.reindex(s["date"]).to_numpy()
        try:
            preds["regression"] = regression_benchmark(res.loc[st], st).reindex(s["date"]).to_numpy()
        except Exception as exc:  # keep the other baselines if inputs fail
            print(f"{st}: regression benchmark failed ({exc})")
        for name, pred in preds.items():
            m_ok = test & np.isfinite(pred)
            m = gw_shell.period_metrics(obs[m_ok], pred[m_ok])
            rows.append({"st_id": st, "group": res.loc[st, "group_name"], "model": name,
                         "n": int(m_ok.sum()), **{k: m[k] for k in ("kge_ab", "nse", "rmse", "mae")}})

    df = pd.DataFrame(rows)
    out = analysis_dir(run)
    df.to_csv(out / "benchmarks_per_station.csv", index=False)

    summ = df.groupby("model")[["kge_ab", "nse", "rmse", "mae"]].median()
    wide = df.pivot(index="st_id", columns="model", values="rmse")
    for b in wide.columns:
        if b != "greybox":
            summ.loc[b, "greybox_better_rmse"] = int((wide["greybox"] < wide[b]).sum())
    summ.loc["greybox", "greybox_better_rmse"] = np.nan
    summ.to_csv(out / "benchmarks_summary.csv")
    print("Test-period medians across stations (greybox_better_rmse = stations where grey-box RMSE is lower):")
    print(summ.round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
