"""Test-period skill with the model state restarted from the observed head.

The production run carries the simulated state from 2012 through the
selection period into the test period. Here each selected variant is rerun
over the test period only, starting from the observed head on the first test
day, with the rainfall and upstream memory convolutions (and the filtered
upstream state) computed on the full record so that only the head state is
reset. The difference from the production test scores shows how much of the
test error is carried in from earlier periods.

Usage:
    python scripts/analysis/reinit_test.py [--run ...]
"""
import argparse

import numpy as np
import pandas as pd

from common import analysis_dir, gw_shell, load_results, run_dir, station_sort_key

sub = gw_shell.sub
_orig_conv = sub.memory_convolve


def param_value(row: pd.Series, name: str) -> float:
    if name == "z" and pd.isna(row.get("z")):
        return float(row["z0"])
    return float(row[name])


def restart_sim(res_row: pd.Series, st: str) -> tuple[np.ndarray, np.ndarray, int]:
    ups = res_row["ups_id"] if pd.notna(res_row["ups_id"]) else "none"
    df, no_up = gw_shell.prepare_data({"st_id": st, "ups_id": ups, "rf_id": res_row["rf_id"]})
    setup = gw_shell.setup_fit(df, res_row["group_name"], int(res_row["rain_lag_days"]),
                               int(res_row["up_lag_days"]), no_up, res_row["model"])
    p = [param_value(res_row, n) for n in setup["param_names"]]
    k = setup["n_sel_end"]
    full = {"rainfall": setup["rainfall"], "h_up": setup["h_up"]}
    sliced = {name: arr[k:] for name, arr in full.items()}

    def conv(x, tau, max_lag, pad_value):
        # convolve the full input, return the part that matches the test slice
        for name, arr in sliced.items():
            if x is arr:
                return _orig_conv(full[name], tau, max(max_lag, 90), float(full[name][0]) if name == "h_up" else 0.0)[k:]
        return _orig_conv(x, tau, max_lag, pad_value)

    kw = dict(rainfall=sliced["rainfall"], amp=setup["amp"][k:],
              amt=setup["amt"][k:] if setup["is_coastal"] else None,
              h_up=sliced["h_up"], h0=float(setup["h_obs"][k]), is_coastal=setup["is_coastal"],
              doy=setup["doy"][k:])
    sub.memory_convolve = conv
    try:
        # keep absolute time: z(t) = z0 + z1 t depends on it
        sim = setup["model_func"](setup["t"][k:], *p, **kw)
    finally:
        sub.memory_convolve = _orig_conv
    return setup["h_obs"][k:], sim, k


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    args = ap.parse_args()
    run = run_dir(args.run)
    res = load_results(run)
    rows = []
    for st in sorted(res.index, key=station_sort_key):
        obs, sim, _ = restart_sim(res.loc[st], st)
        m = gw_shell.period_metrics(obs, sim)
        rows.append({"st_id": st, "group": res.loc[st, "group_name"],
                     "kge_ab_restart": m["kge_ab"], "nse_restart": m["nse"], "rmse_restart": m["rmse"],
                     "bias_restart": m["bias"], "kge_ab_carried": res.loc[st, "kge_ab_test"],
                     "nse_carried": res.loc[st, "nse_test"], "rmse_carried": res.loc[st, "rmse_test"],
                     "bias_carried": res.loc[st, "bias_test"]})
    df = pd.DataFrame(rows)
    df.to_csv(analysis_dir(run) / "reinit_test.csv", index=False)
    for m in ("kge_ab", "nse", "rmse", "bias"):
        print(f"{m:>6}: carried median {df[f'{m}_carried'].median():+.3f} | restart median {df[f'{m}_restart'].median():+.3f}")
    d = df["kge_ab_restart"] - df["kge_ab_carried"]
    print(f"KGE_ab change with restart: median {d.median():+.3f}; better by >0.05 at {(d > 0.05).sum()}, "
          f"worse by >0.05 at {(d < -0.05).sum()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
