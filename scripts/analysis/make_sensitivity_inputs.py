"""Build station input tables for sensitivity and benchmark runs.

Starting from the pairing table used for a reference run and that run's
selected variants, writes (to --outdir):

  seed_XX.csv          selected variant per station, optimiser seed XX
  reduced.csv          rainfall-only single tank: base variant, no upstream,
                       inland form (no marine terms), c = 0, k_link = 0
  ups_nearest.csv      upstream = nearest up-gradient station (no Delaunay filter)
  ups_bestcorr.csv     upstream = best lagged correlation among up-gradient
                       stations within --radius-km
  ups_none.csv         no upstream station
  (upstream tables fit each station's selected variant only)

All station-selection statistics (mean heads, gradient, correlations) use the
calibration period only.

Usage:
    python scripts/analysis/make_sensitivity_inputs.py \
        --input data/gray_box_input_cal_optimized.csv \
        --run workspace/results/final_rev1 --outdir data/sens
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "srcs"))
sys.path.insert(0, str(REPO.parent))

import gw_shell  # noqa: E402

MIN_HEAD_DIFF_M = 0.5
MAX_LAG_DAYS = 90


def updip_direction(coords: pd.DataFrame, mean_heads: pd.Series) -> np.ndarray:
    """Unit vector of increasing head from a planar trend surface h = c0 + c1 x + c2 y."""
    X = np.column_stack([np.ones(len(coords)), coords["x"], coords["y"]])
    c, *_ = np.linalg.lstsq(X, mean_heads.loc[coords.index].values, rcond=None)
    g = np.array([c[1], c[2]])
    return g / np.linalg.norm(g)


def upgradient_candidates(st: str, coords: pd.DataFrame, heads: pd.Series, updip: np.ndarray) -> list[str]:
    """Stations up-gradient of st (positive projection on updip) with head higher by MIN_HEAD_DIFF_M."""
    d = coords[["x", "y"]].values - coords.loc[st, ["x", "y"]].values
    proj = d @ updip
    ok = (proj > 0) & (heads.loc[coords.index].values >= heads[st] + MIN_HEAD_DIFF_M)
    ok &= coords.index != st
    return list(coords.index[ok])


def lagged_abs_corr(dh_target: np.ndarray, dh_up: np.ndarray) -> float:
    best = -np.inf
    for lag in range(0, MAX_LAG_DAYS + 1):
        x = dh_up[:len(dh_up) - lag] if lag else dh_up
        y = dh_target[lag:]
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 60 or np.std(x[m]) == 0 or np.std(y[m]) == 0:
            continue
        best = max(best, abs(np.corrcoef(x[m], y[m])[0, 1]))
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/gray_box_input_cal_optimized.csv")
    ap.add_argument("--run", default="workspace/results/final_rev1")
    ap.add_argument("--outdir", default="data/sens")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--radius-km", type=float, default=10.0)
    args = ap.parse_args()

    base = pd.read_csv(REPO / args.input)
    base["st_id"] = base["st_id"].astype(str)
    selected = pd.read_csv(REPO / args.run / "gw_fit_results.csv").set_index("st_id")["model"]
    out = REPO / args.outdir
    out.mkdir(parents=True, exist_ok=True)
    active = base[pd.to_numeric(base["active"], errors="coerce") == 1].copy()

    # Multi-start: selected variant, seeds 1..N
    for s in range(1, args.seeds + 1):
        t = active.copy()
        t["models"] = t["st_id"].map(selected)
        t["seed"] = s
        t.to_csv(out / f"seed_{s:02d}.csv", index=False)

    # Reduced rainfall-only model
    t = active.copy()
    t["ups_id"] = "none"
    t["group"] = "inland"
    t["models"] = "base"
    t["fixed"] = "c=0;k_link=0"
    t.to_csv(out / "reduced.csv", index=False)

    # Upstream alternatives (calibration-period statistics only)
    gw_daily = pd.read_csv(gw_shell.GW_DATA_PATH, parse_dates=["date time"]).set_index("date time")
    gw_daily = gw_shell.calibration_slice(gw_daily.resample("D").mean())
    coords = active.set_index("st_id")[["gw_TM_X97", "gw_TM_Y97"]].rename(columns={"gw_TM_X97": "x", "gw_TM_Y97": "y"})
    heads = gw_daily[coords.index].mean()
    updip = updip_direction(coords, heads)
    dh = gw_daily[coords.index].diff()

    nearest, bestcorr = {}, {}
    for st in coords.index:
        cands = upgradient_candidates(st, coords, heads, updip)
        if not cands:
            nearest[st] = bestcorr[st] = "none"
            continue
        dist = np.hypot(*(coords.loc[cands, ["x", "y"]].values - coords.loc[st, ["x", "y"]].values).T)
        nearest[st] = cands[int(np.argmin(dist))]
        within = [c for c, d in zip(cands, dist) if d <= args.radius_km * 1000.0]
        scores = {c: lagged_abs_corr(dh[st].values, dh[c].values) for c in within}
        scores = {c: v for c, v in scores.items() if np.isfinite(v)}
        bestcorr[st] = max(scores, key=scores.get) if scores else "none"

    for name, mapping in (("ups_nearest", nearest), ("ups_bestcorr", bestcorr),
                          ("ups_none", {st: "none" for st in coords.index})):
        t = active.copy()
        t["ups_id"] = t["st_id"].map(mapping)
        t["models"] = t["st_id"].map(selected)
        t.to_csv(out / f"{name}.csv", index=False)
        same = (t.set_index("st_id")["ups_id"].astype(str) == active.set_index("st_id")["ups_id"].astype(str)).sum()
        print(f"{name}: same upstream as reference at {same}/{len(t)} stations")

    print(f"Updip direction (unit vector, TWD97 x/y): {updip.round(3)}")
    print(f"Wrote tables to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
