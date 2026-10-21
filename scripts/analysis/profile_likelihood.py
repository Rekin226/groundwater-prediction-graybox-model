"""Profile likelihoods for selected parameters of the selected variant.

At each grid value the profiled parameter is held fixed and all others are
re-optimised on the calibration period (same optimiser as production). The
profile is reported as the calibration RMSE and as
dNLL = n/2 * ln(SSE / SSE_min) (Gaussian, independent errors), with 1.92 the
approximate 95% threshold for one parameter. Daily residuals are strongly
autocorrelated, so this threshold is optimistic; flat profiles are the robust
signal of non-identifiability.

Stations / parameters:
  coastal stations            k_sgd, h_sea
  five inland stations at the 10/30/50/70/90th percentiles of test KGE_ab
                              z (z0 for z(t) variants), tau_rain, tau_up

Usage:
    python scripts/analysis/profile_likelihood.py [--run ...] [--input ...] [--workers 14]
"""
import argparse
import contextlib
import io
from multiprocessing import Pool

import numpy as np
import pandas as pd

from common import REPO, analysis_dir, gw_shell, load_results, run_dir, station_sort_key

KSGD_GRID = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
HSEA_OFFSETS = [-50, -40, -30, -20, -10, -5, 0, 5, 10]          # m relative to calibration mean head
Z_RANGE_FACTORS = [0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0]      # z = h_min - factor x range
TAU_RAIN_GRID = [0.5, 1, 2, 5, 10, 20, 30, 45, 60]
TAU_UP_GRID = [0.1, 0.5, 1, 2, 5, 10, 30]


def station_inputs(st: str, inp: pd.DataFrame, res: pd.DataFrame):
    row = inp.loc[st]
    args = {"st_id": st, "ups_id": row["ups_id"] if pd.notna(row["ups_id"]) else "none", "rf_id": row["rf_id"]}
    with contextlib.redirect_stdout(io.StringIO()):
        df, no_up = gw_shell.prepare_data(args)
    return df, no_up


def fit_point(task: dict) -> dict:
    with contextlib.redirect_stdout(io.StringIO()):
        df, no_up = gw_shell.prepare_data(task["data_args"])
        r = gw_shell._fit_model(df, group_name=task["group"], rain_lag_days=task["rain_lag"],
                                up_lag_days=task["up_lag"], no_upstream=no_up,
                                model_name=task["model"], fixed={task["param"]: task["value"]})
    out = {k: task[k] for k in ("st_id", "group", "model", "param", "value")}
    if r is None:
        return out
    m = r["metrics"]
    out.update({"rmse_cal": m["cal"]["rmse"], "n_cal": r["n_cal"],
                "kge_ab_sel": m["sel"]["kge_ab"], "kge_ab_test": m["test"]["kge_ab"]})
    out.update({f"p_{n}": float(v) for n, v in zip(r["param_names"], r["params"])})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    ap.add_argument("--input", default="data/gray_box_input_cal_optimized.csv")
    ap.add_argument("--workers", type=int, default=14)
    args = ap.parse_args()
    run = run_dir(args.run)
    res = load_results(run)
    inp = pd.read_csv(REPO / args.input).astype({"st_id": str}).set_index("st_id")

    coastal = sorted(res.index[res.group_name == "coastal"], key=station_sort_key)
    inland = res[res.group_name == "inland"].sort_values("kge_ab_test")
    picks = [inland.index[int(round(q * (len(inland) - 1)))] for q in (0.1, 0.3, 0.5, 0.7, 0.9)]

    tasks = []
    for st in coastal + picks:
        df, _ = station_inputs(st, inp, res)
        cal = gw_shell.calibration_slice(df)["gwl"]
        model = res.loc[st, "model"]
        base = {"st_id": st, "group": res.loc[st, "group_name"], "model": model,
                "rain_lag": int(res.loc[st, "rain_lag_days"]), "up_lag": int(res.loc[st, "up_lag_days"]),
                "data_args": {"st_id": st, "rf_id": inp.loc[st, "rf_id"],
                              "ups_id": inp.loc[st, "ups_id"] if pd.notna(inp.loc[st, "ups_id"]) else "none"}}
        grids = {}
        if st in coastal:
            grids["k_sgd"] = KSGD_GRID
            grids["h_sea"] = [float(cal.mean() + o) for o in HSEA_OFFSETS]
        else:
            zname = "z0" if model.endswith("_tz") else "z"
            rng = float(cal.max() - cal.min())
            grids[zname] = [float(cal.min() - f * rng) for f in Z_RANGE_FACTORS]
            grids["tau_rain"] = TAU_RAIN_GRID
            grids["tau_up"] = TAU_UP_GRID
        for param, values in grids.items():
            for v in values:
                tasks.append({**base, "param": param, "value": float(v)})

    print(f"{len(tasks)} profile fits on {len(coastal)} coastal + {len(picks)} inland stations ({picks})")
    with Pool(args.workers) as pool:
        rows = pool.map(fit_point, tasks, chunksize=1)

    df = pd.DataFrame(rows)
    sse = df["rmse_cal"] ** 2 * df["n_cal"]
    opt_sse = (res.loc[df["st_id"], "rmse_cal"].to_numpy() ** 2) * df["n_cal"]
    sse_min = pd.concat([sse.groupby([df.st_id, df.param]).transform("min"),
                         pd.Series(opt_sse, index=df.index)], axis=1).min(axis=1)
    df["dnll"] = 0.5 * df["n_cal"] * np.log(sse / sse_min)
    df["optimum_value"] = [res.loc[s, p] if p in res.columns else np.nan for s, p in zip(df.st_id, df.param)]
    path = analysis_dir(run) / "profile_likelihood.csv"
    df.to_csv(path, index=False)

    summ = (df.assign(within=df.dnll <= 1.92)
              .groupby(["st_id", "param"])
              .agg(n_points=("value", "size"), n_within_95=("within", "sum"),
                   rmse_min=("rmse_cal", "min"), rmse_max=("rmse_cal", "max")))
    summ["rmse_rel_range_pct"] = 100 * (summ.rmse_max - summ.rmse_min) / summ.rmse_min
    summ.to_csv(analysis_dir(run) / "profile_summary.csv")
    print(summ.round(3).to_string())
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
