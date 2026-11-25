"""Spatial dependence checks for network summaries and the k_link gradient.

1. Spatial-block bootstrap: stations are grouped into square blocks
   (BLOCK_KM on the TWD97 grid) and whole blocks are resampled, so CIs allow
   for correlation between neighbouring wells. Compared with the station
   bootstrap for the test-period network medians and the coastal - inland
   difference.
2. Moran's I (inverse-distance weights, permutation p) for test KGE_ab,
   test bias and log k_link.
3. Robustness of the k_link - distance-to-coast gradient: all stations;
   stations whose a - k_link Jacobian correlation is below 0.9; partial
   Spearman controlling for a; the coupling share k_link / (a + k_link);
   one value per rain gauge (median); and a spatial-block bootstrap CI of rho.

Usage:
    python scripts/analysis/spatial_dependence.py [--run ...] [--block-km 10]
"""
import argparse

import numpy as np
import pandas as pd
from scipy import stats

from common import REPO, analysis_dir, load_results, run_dir

N_BOOT = 10000
N_PERM = 9999


def load_frame(run) -> pd.DataFrame:
    res = load_results(run)
    meta = pd.read_csv(REPO / "data" / "gray_box_input_cal_optimized.csv").astype({"st_id": str}).set_index("st_id")
    cls = pd.read_csv(REPO / "data" / "intermediate" / "gw_coastal_inland_class.csv").astype({"st_id": str}).set_index("st_id")
    df = res.join(meta[["gw_TM_X97", "gw_TM_Y97"]]).join(cls[["dist_to_coast_m"]])
    pairs = pd.read_csv(run / "analysis" / "parameter_corr_pairs.csv").astype({"st_id": str})
    ak = pairs[((pairs.p1 == "a") & (pairs.p2 == "k_link")) | ((pairs.p1 == "k_link") & (pairs.p2 == "a"))]
    # the pairs file lists only |r| > 0.9, so an absent pair means |r| <= 0.9
    df["corr_a_klink"] = ak.set_index("st_id")["corr"].reindex(df.index).fillna(0.0)
    return df


def blocks(df: pd.DataFrame, km: float) -> np.ndarray:
    gx = np.floor(df["gw_TM_X97"].to_numpy() / (km * 1000)).astype(int)
    gy = np.floor(df["gw_TM_Y97"].to_numpy() / (km * 1000)).astype(int)
    return pd.factorize(pd.Series([f"{x}_{y}" for x, y in zip(gx, gy)]))[0]


def block_resample(block_ids: np.ndarray, rng) -> np.ndarray:
    ub = np.unique(block_ids)
    pick = rng.choice(ub, size=len(ub), replace=True)
    return np.concatenate([np.flatnonzero(block_ids == b) for b in pick])


def ci(vals) -> tuple[float, float]:
    v = np.asarray(vals)
    v = v[np.isfinite(v)]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def morans_i(x: np.ndarray, xy: np.ndarray, rng) -> tuple[float, float]:
    d = np.hypot(*(xy[:, None, :] - xy[None, :, :]).transpose(2, 0, 1))
    w = np.where(d > 0, 1.0 / np.maximum(d, 1.0), 0.0)
    w /= w.sum()

    def stat(v):
        z = v - v.mean()
        return len(v) * (z @ w @ z) / (z @ z)

    obs = stat(x)
    perm = np.array([stat(rng.permutation(x)) for _ in range(N_PERM)])
    p = (1 + np.sum(np.abs(perm - perm.mean()) >= abs(obs - perm.mean()))) / (N_PERM + 1)
    return float(obs), float(p)


def partial_spearman(x, y, z) -> tuple[float, float]:
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))
    res_x = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
    res_y = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    r, p = stats.pearsonr(res_x, res_y)
    return float(r), float(p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    ap.add_argument("--block-km", type=float, default=10.0)
    args = ap.parse_args()
    run = run_dir(args.run)
    df = load_frame(run)
    rng = np.random.default_rng(42)
    bid = blocks(df, args.block_km)
    coastal = (df["group_name"] == "coastal").to_numpy()
    rows = []

    # 1. block vs station bootstrap
    for metric in ("kge_ab_test", "nse_test", "rmse_test"):
        x = df[metric].to_numpy()
        st_meds = [np.median(x[rng.integers(0, len(x), len(x))]) for _ in range(N_BOOT)]
        bl_meds = [np.median(x[block_resample(bid, rng)]) for _ in range(N_BOOT)]
        diffs = []
        for _ in range(N_BOOT):
            idx = block_resample(bid, rng)
            c, i = x[idx][coastal[idx]], x[idx][~coastal[idx]]
            diffs.append(np.median(c) - np.median(i) if len(c) and len(i) else np.nan)
        rows.append({"check": "median", "metric": metric, "value": float(np.median(x)),
                     "station_ci": ci(st_meds), "block_ci": ci(bl_meds)})
        rows.append({"check": "coastal_minus_inland", "metric": metric,
                     "value": float(np.median(x[coastal]) - np.median(x[~coastal])), "block_ci": ci(diffs)})

    # 2. Moran's I
    xy = df[["gw_TM_X97", "gw_TM_Y97"]].to_numpy(dtype=float)
    for name, v in (("kge_ab_test", df["kge_ab_test"]), ("bias_test", df["bias_test"]),
                    ("log_k_link", np.log10(df["k_link"].clip(lower=1e-3)))):
        i_obs, p = morans_i(np.asarray(v, dtype=float), xy, rng)
        rows.append({"check": "morans_i", "metric": name, "value": i_obs, "p": p})

    # 3. k_link gradient robustness
    dist = df["dist_to_coast_m"].to_numpy() / 1000.0
    kl = df["k_link"].to_numpy()
    a = df["a"].to_numpy()
    r, p = stats.spearmanr(dist, kl)
    boot = []
    for _ in range(N_BOOT):
        idx = block_resample(bid, rng)
        boot.append(stats.spearmanr(dist[idx], kl[idx])[0])
    rows.append({"check": "klink_gradient", "metric": "all", "value": r, "p": p, "n": len(df), "block_ci": ci(boot)})
    low = (df["corr_a_klink"].abs() <= 0.9).to_numpy()
    r, p = stats.spearmanr(dist[low], kl[low])
    rows.append({"check": "klink_gradient", "metric": "corr_a_klink<0.9", "value": r, "p": p, "n": int(low.sum())})
    r, p = partial_spearman(dist, kl, a)
    rows.append({"check": "klink_gradient", "metric": "partial_given_a", "value": r, "p": p, "n": len(df)})
    share = kl / (a + kl)
    r, p = stats.spearmanr(dist, share)
    rows.append({"check": "klink_gradient", "metric": "share_klink/(a+klink)", "value": r, "p": p, "n": len(df)})
    r, p = stats.spearmanr(dist, a)
    rows.append({"check": "klink_gradient", "metric": "a_vs_distance", "value": r, "p": p, "n": len(df)})
    g = pd.DataFrame({"rf": df["rf_id"], "d": dist, "k": kl}).groupby("rf").median()
    r, p = stats.spearmanr(g["d"], g["k"])
    rows.append({"check": "klink_gradient", "metric": "median_per_rain_gauge", "value": r, "p": p, "n": len(g)})

    out = pd.DataFrame(rows)
    out.to_csv(analysis_dir(run) / "spatial_dependence.csv", index=False)
    print(f"{len(np.unique(bid))} spatial blocks of {args.block_km:g} km")
    print(out.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
