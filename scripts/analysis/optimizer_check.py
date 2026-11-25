"""Check whether the production calibration reached the best attainable fit.

Refits each station's selected variant with a stronger search: several DE runs
(different seeds, no regression-based seed in the initial population, larger
population and iteration budget) and keeps the best calibration RMSE. Reports
the relative improvement over the production fit, the parameters of the best
run, and its selection/test skill.

Usage:
    python scripts/analysis/optimizer_check.py [--run ...] [--input ...] [--seeds 3] [--popsize 30] [--maxiter 2000]
"""
import argparse
import contextlib
import io
from multiprocessing import Pool

import pandas as pd

from common import REPO, analysis_dir, gw_shell, load_results, run_dir, station_sort_key


def fit_one(task: dict) -> dict:
    with contextlib.redirect_stdout(io.StringIO()):
        df, no_up = gw_shell.prepare_data(task["data_args"])
        r = gw_shell._fit_model(df, group_name=task["group"], rain_lag_days=task["rain_lag"],
                                up_lag_days=task["up_lag"], no_upstream=no_up, model_name=task["model"],
                                n_starts=task["popsize"], seed=task["seed"], maxiter=task["maxiter"],
                                seed_p0=False)
    out = {"st_id": task["st_id"], "seed": task["seed"]}
    if r is not None:
        m = r["metrics"]
        out.update({"rmse_cal": m["cal"]["rmse"], "kge_ab_sel": m["sel"]["kge_ab"],
                    "kge_ab_test": m["test"]["kge_ab"], "nse_test": m["test"]["nse"]})
        out.update({f"p_{n}": float(v) for n, v in zip(r["param_names"], r["params"])})
        out.update({f"atb_{n}": gw_shell._at_bound(float(v), float(lo), float(hi))
                    for n, v, lo, hi in zip(r["param_names"], r["params"], r["lower"], r["upper"])})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    ap.add_argument("--input", default="data/gray_box_input_cal_optimized.csv")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--popsize", type=int, default=30)
    ap.add_argument("--maxiter", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=14)
    args = ap.parse_args()
    run = run_dir(args.run)
    res = load_results(run)
    inp = pd.read_csv(REPO / args.input).astype({"st_id": str}).set_index("st_id")

    tasks = []
    for st in sorted(res.index, key=station_sort_key):
        r = res.loc[st]
        for s in range(1, args.seeds + 1):
            tasks.append({"st_id": st, "group": r["group_name"], "model": r["model"],
                          "rain_lag": int(r["rain_lag_days"]), "up_lag": int(r["up_lag_days"]),
                          "seed": 100 + s, "popsize": args.popsize, "maxiter": args.maxiter,
                          "data_args": {"st_id": st, "rf_id": inp.loc[st, "rf_id"],
                                        "ups_id": inp.loc[st, "ups_id"] if pd.notna(inp.loc[st, "ups_id"]) else "none"}})
    print(f"{len(tasks)} fits ({args.seeds} seeds x {len(res)} stations), popsize {args.popsize}, maxiter {args.maxiter}")
    with Pool(args.workers) as pool:
        rows = pool.map(fit_one, tasks, chunksize=1)
    allfits = pd.DataFrame(rows)
    out = analysis_dir(run)
    allfits.to_csv(out / "optimizer_check_all.csv", index=False)

    best = allfits.loc[allfits.groupby("st_id").rmse_cal.idxmin()].set_index("st_id")
    cmp = pd.DataFrame({
        "group": res.group_name, "model": res.model,
        "rmse_cal_prod": res.rmse_cal, "rmse_cal_best": best.rmse_cal,
        "kge_ab_sel_prod": res.kge_ab_sel, "kge_ab_sel_best": best.kge_ab_sel,
        "kge_ab_test_prod": res.kge_ab_test, "kge_ab_test_best": best.kge_ab_test,
    })
    cmp["improve_pct"] = 100 * (cmp.rmse_cal_prod - cmp.rmse_cal_best) / cmp.rmse_cal_prod
    cmp.to_csv(out / "optimizer_check.csv")
    imp = cmp.improve_pct
    print(f"Calibration RMSE improvement with the stronger search: median {imp.median():.2f}%, "
          f">1% at {(imp > 1).sum()}, >5% at {(imp > 5).sum()} of {len(imp)} stations; worse at {(imp < -0.1).sum()}")
    print(f"Median test KGE_ab: production {cmp.kge_ab_test_prod.median():.3f}, stronger search {cmp.kge_ab_test_best.median():.3f}")
    print(f"Median selection KGE_ab: production {cmp.kge_ab_sel_prod.median():.3f}, stronger search {cmp.kge_ab_sel_best.median():.3f}")
    atb_cols = [c for c in best.columns if c.startswith("atb_")]
    hits = {c[4:]: f"{int((best[c] == True).sum())}/{int(best[c].notna().sum())}" for c in atb_cols}  # noqa: E712
    print(f"Parameters at a bound after the stronger search: {hits}")
    print(cmp.sort_values("improve_pct", ascending=False).head(12).round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
