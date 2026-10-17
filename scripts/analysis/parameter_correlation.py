"""Local parameter correlations of the selected fits from the calibration Jacobian.

For each station the residual Jacobian J (calibration period) is evaluated at
the fitted parameters by central differences, and the approximate parameter
correlation matrix is taken from pinv(J^T J). Parameters held at a bound are
kept (their correlations are still informative about trade-offs). Reported:
the condition number of J^T J (scaled to unit parameter ranges), every pair
with |corr| > 0.9, and network counts per pair.

Usage:
    python scripts/analysis/parameter_correlation.py [--run ...] [--input ...]
"""
import argparse
import contextlib
import io
from collections import Counter

import numpy as np
import pandas as pd

from common import REPO, analysis_dir, gw_shell, load_results, run_dir, station_sort_key

THRESH = 0.9


def jacobian(func, t, p, kwargs, steps):
    base = func(t, *p, **kwargs)
    J = np.empty((len(base), len(p)))
    for j, h in enumerate(steps):
        up, dn = p.copy(), p.copy()
        up[j] += h
        dn[j] -= h
        J[:, j] = (func(t, *up, **kwargs) - func(t, *dn, **kwargs)) / (2 * h)
    return J


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    ap.add_argument("--input", default="data/gray_box_input_cal_optimized.csv")
    args = ap.parse_args()
    run = run_dir(args.run)
    res = load_results(run)
    inp = pd.read_csv(REPO / args.input).astype({"st_id": str}).set_index("st_id")

    pair_rows, st_rows = [], []
    counts, totals = Counter(), Counter()
    for st in sorted(res.index, key=station_sort_key):
        r = res.loc[st]
        data_args = {"st_id": st, "rf_id": inp.loc[st, "rf_id"],
                     "ups_id": inp.loc[st, "ups_id"] if pd.notna(inp.loc[st, "ups_id"]) else "none"}
        with contextlib.redirect_stdout(io.StringIO()):
            df, no_up = gw_shell.prepare_data(data_args)
            s = gw_shell.setup_fit(df, r["group_name"], int(r["rain_lag_days"]), int(r["up_lag_days"]),
                                   no_up, r["model"])
        names = s["param_names"]
        p = np.array([float(r[n]) for n in names])
        rngs = np.maximum(np.array(s["upper"], float) - np.array(s["lower"], float), 1e-9)
        steps = 1e-4 * rngs
        J = jacobian(s["model_func"], s["t_cal"], p, s["model_kwargs"], steps)
        Js = J * rngs                      # sensitivities per unit parameter range
        A = Js.T @ Js
        cond = float(np.linalg.cond(A))
        cov = np.linalg.pinv(A)
        sd = np.sqrt(np.clip(np.diag(cov), 1e-300, None))
        corr = cov / np.outer(sd, sd)
        n_high = 0
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                key = tuple(sorted((names[i], names[j])))
                totals[key] += 1
                c = float(corr[i, j])
                if abs(c) > THRESH:
                    counts[key] += 1
                    n_high += 1
                    pair_rows.append({"st_id": st, "model": r["model"], "p1": key[0], "p2": key[1], "corr": c})
        st_rows.append({"st_id": st, "group": r["group_name"], "model": r["model"],
                        "log10_cond": np.log10(cond), "n_pairs_gt_0.9": n_high})

    out = analysis_dir(run)
    pd.DataFrame(pair_rows).to_csv(out / "parameter_corr_pairs.csv", index=False)
    st_df = pd.DataFrame(st_rows)
    st_df.to_csv(out / "parameter_corr_stations.csv", index=False)
    net = pd.DataFrame([{"pair": f"{a}-{b}", "n_high": counts[(a, b)], "n_fits": totals[(a, b)],
                         "frac_high": counts[(a, b)] / totals[(a, b)]} for (a, b) in totals])
    net = net.sort_values("frac_high", ascending=False)
    net.to_csv(out / "parameter_corr_network.csv", index=False)
    print(f"|corr| > {THRESH}: stations with >= 1 such pair: {(st_df['n_pairs_gt_0.9'] > 0).sum()}/{len(st_df)}; "
          f"median log10 condition number {st_df.log10_cond.median():.1f}")
    print(net.head(15).round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
