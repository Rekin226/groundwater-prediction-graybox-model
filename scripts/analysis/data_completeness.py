"""Per-station data completeness of the daily groundwater and rainfall inputs.

For each station (and its assigned rainfall gauge and upstream station):
percentage of days with data per period, number of gaps, longest gap, the
number of days that drop out of the model record, and hourly runs of exactly
constant head lasting >= 48 h. The input series carry no explicit gaps, so such
held-constant runs are the visible trace of gap filling (or a stuck sensor)
upstream of this pipeline. Daily rows missing any input are removed rather than
imputed; hourly heads are interpolated only inside the STFT used for AMP/AMT.

Usage:
    python scripts/analysis/data_completeness.py [--run ...]
"""
import argparse

import numpy as np
import pandas as pd

from common import analysis_dir, gw_shell, load_results, run_dir, station_sort_key

START, END = "2012-01-01", "2022-12-31"


def gaps(valid: pd.Series) -> tuple[int, int]:
    """Number of runs of missing days and the longest run length."""
    miss = (~valid).astype(int).to_numpy()
    if miss.sum() == 0:
        return 0, 0
    edges = np.diff(np.concatenate([[0], miss, [0]]))
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
    lengths = ends - starts
    return int(len(lengths)), int(lengths.max())


def held_constant(h: np.ndarray, min_hours: int = 48) -> tuple[int, int]:
    """Number and total length (hours) of runs where the hourly head does not change."""
    flat = np.concatenate([[0], (np.abs(np.diff(h)) < 1e-9).astype(int), [0]])
    edges = np.diff(flat)
    lengths = np.where(edges == -1)[0] - np.where(edges == 1)[0]
    long = lengths[lengths >= min_hours]
    return int(len(long)), int(long.sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    args = ap.parse_args()
    run = run_dir(args.run)
    res = load_results(run)

    gw = pd.read_csv(gw_shell.GW_DATA_PATH, parse_dates=["date time"]).set_index("date time")
    hourly_n = gw.loc[START:END].notna().resample("D").sum()
    gw_daily = gw.resample("D").mean().loc[START:END]
    rf = pd.read_csv(gw_shell.RF_DATA_PATH, parse_dates=["date time"]).set_index("date time").loc[START:END]
    days = pd.date_range(START, END, freq="D")
    periods = {"cal": (START, gw_shell.CAL_END), "sel": ("2019-01-01", gw_shell.SEL_END), "test": ("2021-01-01", END)}

    rows = []
    for st in sorted(res.index, key=station_sort_key):
        g = gw_daily[st].reindex(days).notna()
        r = rf[res.loc[st, "rf_id"]].reindex(days).notna()
        ups = str(res.loc[st, "ups_id"])
        u = gw_daily[ups].reindex(days).notna() if ups in gw_daily.columns else pd.Series(True, index=days)
        model_ok = g & r & u
        n_gaps, longest = gaps(g)
        n_flat, flat_h = held_constant(gw.loc[START:END, st].to_numpy())
        row = {"st_id": st, "group": res.loc[st, "group_name"], "rf_id": res.loc[st, "rf_id"], "ups_id": ups,
               "gw_days_pct": 100 * g.mean(), "rf_days_pct": 100 * r.mean(), "ups_days_pct": 100 * u.mean(),
               "gw_gaps": n_gaps, "gw_longest_gap_d": longest,
               "held_constant_runs_48h": n_flat, "held_constant_pct": 100 * flat_h / len(gw.loc[START:END]),
               "median_hourly_obs_per_day": float(hourly_n[st].loc[g[g].index].median()) if g.any() else np.nan,
               "model_days": int(model_ok.sum()), "model_days_dropped": int((~model_ok).sum())}
        for p, (a, b) in periods.items():
            row[f"model_days_pct_{p}"] = 100 * model_ok.loc[a:b].mean()
        rows.append(row)

    df = pd.DataFrame(rows)
    path = analysis_dir(run) / "data_completeness.csv"
    df.to_csv(path, index=False)
    num = df.select_dtypes("number")
    print(num.describe(percentiles=[0.1, 0.5, 0.9]).T[["min", "10%", "50%", "90%", "max"]].round(1).to_string())
    print(f"\nStations with <90% model days in any period: "
          f"{df.loc[df[[c for c in df if c.startswith('model_days_pct_')]].min(axis=1) < 90, 'st_id'].tolist()}")
    print(f"Stations with >1% held-constant hours: "
          f"{df.loc[df.held_constant_pct > 1, ['st_id', 'held_constant_pct']].round(1).values.tolist()}")
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
