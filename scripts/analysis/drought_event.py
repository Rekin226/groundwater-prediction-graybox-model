"""Event-based look at the 2020-2021 drought.

Windows (spanning the selection/test boundary):
  pre      2019-01-01 .. 2020-06-30
  drought  2020-07-01 .. 2021-06-30
  post     2021-07-01 .. 2022-12-31

Per station: rainfall at the assigned gauge as a percentage of the
calibration-period (2012-2018) mean for the same calendar days, and the
selected model's mean bias (sim - obs) and RMSE in each window. Network-level:
how many stations are biased high (model above observations) in each window,
and the across-station correlation between rainfall deficit and bias.

Usage:
    python scripts/analysis/drought_event.py [--run ...]
"""
import argparse

import numpy as np
import pandas as pd
from scipy import stats

from common import analysis_dir, load_results, load_series, run_dir, station_sort_key

WINDOWS = {
    "pre": ("2019-01-01", "2020-06-30"),
    "drought": ("2020-07-01", "2021-06-30"),
    "post": ("2021-07-01", "2022-12-31"),
}


def rainfall_pct_of_normal(s: pd.DataFrame, a: str, b: str) -> float:
    """Window rainfall total as % of the mean 2012-2018 total over the same calendar days."""
    rf = s.set_index("date")["rf"]
    win = rf.loc[a:b]
    doy = np.minimum(win.index.dayofyear.to_numpy(), 365)
    cal = rf.loc[:"2018-12-31"]
    clim = cal.groupby(np.minimum(cal.index.dayofyear.to_numpy(), 365)).mean()
    expected = clim.reindex(doy).to_numpy().sum()
    return 100.0 * win.sum() / expected if expected > 0 else np.nan


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    args = ap.parse_args()
    run = run_dir(args.run)
    res = load_results(run)

    rows = []
    for st in sorted(res.index, key=station_sort_key):
        s = load_series(run, st)
        sim = s[f"sim_{res.loc[st, 'model']}"]
        row = {"st_id": st, "group": res.loc[st, "group_name"], "rf_id": res.loc[st, "rf_id"]}
        for w, (a, b) in WINDOWS.items():
            m = (s["date"] >= a) & (s["date"] <= b)
            err = (sim - s["obs"])[m]
            row[f"rain_pct_{w}"] = rainfall_pct_of_normal(s, a, b)
            row[f"bias_{w}"] = float(err.mean())
            row[f"rmse_{w}"] = float(np.sqrt((err ** 2).mean()))
        rows.append(row)
    df = pd.DataFrame(rows)
    out = analysis_dir(run) / "drought_event.csv"
    df.to_csv(out, index=False)

    print("Window medians across stations:")
    for w in WINDOWS:
        b = df[f"bias_{w}"]
        print(f"  {w:>7}: rainfall {df[f'rain_pct_{w}'].median():5.0f}% of normal "
              f"(range {df[f'rain_pct_{w}'].min():.0f}-{df[f'rain_pct_{w}'].max():.0f}%), "
              f"bias median {b.median():+.2f} m, biased high at {(b > 0).sum()}/{len(b)}, "
              f"RMSE median {df[f'rmse_{w}'].median():.2f} m")
    d_bias = df["bias_drought"] - df["bias_pre"]
    rho, p = stats.spearmanr(df["rain_pct_drought"], d_bias)
    print(f"Change in bias pre->drought: median {d_bias.median():+.2f} m; increased at {(d_bias > 0).sum()}/{len(df)}")
    print(f"Spearman(rainfall % of normal in drought, bias change) = {rho:.2f} (p = {p:.3f})")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
