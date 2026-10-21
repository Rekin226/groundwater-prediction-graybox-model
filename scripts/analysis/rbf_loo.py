"""Leave-one-station-out check of the parameter-map interpolation.

For each mapped parameter (inland stations): every station is removed in turn,
the interpolant is built from the rest and evaluated at the removed station.
Reported per interpolator: LOO RMSE and the skill score
1 - SSE / SSE_mean, where SSE_mean comes from predicting the mean of the
remaining stations (skill <= 0 means no spatial predictability beyond a
constant). Interpolators: scipy Rbf with several kernels (default shape
parameter, exact interpolation, lon/lat coordinates as in the map figure),
inverse-distance weighting (power 2) and nearest neighbour. Spearman
correlation of the values with distance to the coast and to the fan apex is
added as a simple check of the gradients the maps suggest.

Usage:
    python scripts/analysis/rbf_loo.py [--run ...] [--input data/gray_box_input_cal_optimized.csv]
"""
import argparse

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy import stats
from scipy.interpolate import Rbf

from common import REPO, analysis_dir, load_results, run_dir

PARAMS = ("a", "b", "tau_rain", "k_link")
RBF_KERNELS = ("inverse", "multiquadric", "gaussian", "linear", "thin_plate")


def predict_loo(x, y, v, method):
    pred = np.empty(len(v))
    for i in range(len(v)):
        m = np.arange(len(v)) != i
        if method == "mean":
            pred[i] = v[m].mean()
        elif method == "nearest":
            d = np.hypot(x[m] - x[i], y[m] - y[i])
            pred[i] = v[m][np.argmin(d)]
        elif method == "idw2":
            d = np.hypot(x[m] - x[i], y[m] - y[i])
            w = 1.0 / d ** 2
            pred[i] = (w * v[m]).sum() / w.sum()
        else:
            pred[i] = Rbf(x[m], y[m], v[m], function=method)(x[i], y[i])
    return pred


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    ap.add_argument("--input", default="data/gray_box_input_cal_optimized.csv")
    args = ap.parse_args()
    run = run_dir(args.run)
    res = load_results(run)
    meta = pd.read_csv(REPO / args.input).astype({"st_id": str}).set_index("st_id")
    df = res[res.group_name == "inland"].join(meta[["gw_TM_X97", "gw_TM_Y97"]])
    tr = Transformer.from_crs("EPSG:3828", "EPSG:4326", always_xy=True)
    lon, lat = tr.transform(df["gw_TM_X97"].values, df["gw_TM_Y97"].values)

    # Distance proxies: to the westmost (coastal) and eastmost (apex) station envelope
    xk = df["gw_TM_X97"].to_numpy() / 1000.0
    coast_proxy = xk - xk.min()

    rows = []
    for p in PARAMS:
        v = df[p].to_numpy(float)
        sse_mean = ((predict_loo(lon, lat, v, "mean") - v) ** 2).sum()
        for method in ("mean", "nearest", "idw2") + RBF_KERNELS:
            pred = predict_loo(lon, lat, v, method)
            sse = ((pred - v) ** 2).sum()
            rows.append({"param": p, "method": method, "n": len(v),
                         "loo_rmse": float(np.sqrt(sse / len(v))),
                         "skill_vs_mean": float(1.0 - sse / sse_mean)})
        rho, pv = stats.spearmanr(coast_proxy, v)
        rows.append({"param": p, "method": "spearman_vs_eastward_distance", "n": len(v),
                     "loo_rmse": np.nan, "skill_vs_mean": np.nan, "rho": rho, "p": pv})
    out = pd.DataFrame(rows)
    path = analysis_dir(run) / "rbf_loo.csv"
    out.to_csv(path, index=False)
    print(out.pivot(index="method", columns="param", values="skill_vs_mean").round(3).to_string())
    print(out[out.method.str.startswith("spearman")][["param", "rho", "p"]].round(3).to_string(index=False))
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
