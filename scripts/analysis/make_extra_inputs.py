"""Input tables for structural ablations and further pairing checks.

Written to --outdir (default data/sens):
  no_amp_amt.csv       selected variant with c = 0 (and gamma = 0 at coastal
                       stations): no covariates derived from the target record
  coastal_as_inland.csv  coastal stations refitted with the inland structure
                       (all four variants, selection repeated); others inactive
  hsea_msl.csv         coastal stations, selected variant, h_sea fixed at 0 m
                       (mean sea level in the head datum); others inactive
  ups_rfcond.csv       upstream = up-gradient station within --radius-km with
                       the highest lagged correlation after removing each
                       station's rainfall response (calibration period)
  ups_samelayer.csv    adopted pairing, but a cross-layer upstream station is
                       replaced by the best-correlated same-layer up-gradient
                       station within --radius-km (or none)

Usage:
    python scripts/analysis/make_extra_inputs.py [--input ...] [--run ...] [--wells .../well_metadata.csv]
"""
import argparse

import numpy as np
import pandas as pd

from common import REPO, gw_shell, load_results, run_dir
from make_sensitivity_inputs import lagged_abs_corr, updip_direction, upgradient_candidates

RAIN_LAGS = 30


def rain_residual(dh: pd.Series, rain: pd.Series) -> np.ndarray:
    """Residual of daily head change after least squares on rainfall lags 0..RAIN_LAGS."""
    X = np.column_stack([rain.shift(k).to_numpy() for k in range(RAIN_LAGS + 1)] + [np.ones(len(rain))])
    y = dh.to_numpy()
    ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
    coef, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
    out = np.full(len(y), np.nan)
    out[ok] = y[ok] - X[ok] @ coef
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/gray_box_input_cal_optimized.csv")
    ap.add_argument("--run", default=None)
    ap.add_argument("--wells", default=None, help="well_metadata.csv with a 'layer' column")
    ap.add_argument("--outdir", default="data/sens")
    ap.add_argument("--radius-km", type=float, default=10.0)
    args = ap.parse_args()
    run = run_dir(args.run)
    res = load_results(run)
    base = pd.read_csv(REPO / args.input).astype({"st_id": str})
    active = base[pd.to_numeric(base["active"], errors="coerce") == 1].copy()
    out = REPO / args.outdir
    out.mkdir(parents=True, exist_ok=True)
    selected = res["model"]
    coastal = set(res.index[res.group_name == "coastal"])

    t = active.copy()
    t["models"] = t.st_id.map(selected)
    t["fixed"] = np.where(t.st_id.isin(coastal), "c=0;gamma=0", "c=0")
    t.to_csv(out / "no_amp_amt.csv", index=False)

    t = active.copy()
    t["active"] = t.st_id.isin(coastal).astype(int)
    t.loc[t.st_id.isin(coastal), "group"] = "inland"
    t.to_csv(out / "coastal_as_inland.csv", index=False)

    t = active.copy()
    t["active"] = t.st_id.isin(coastal).astype(int)
    t["models"] = t.st_id.map(selected)
    t["fixed"] = "h_sea=0"
    t.to_csv(out / "hsea_msl.csv", index=False)

    # Pairing variants (calibration-period statistics only)
    gw = pd.read_csv(gw_shell.GW_DATA_PATH, parse_dates=["date time"]).set_index("date time")
    gw = gw_shell.calibration_slice(gw.resample("D").mean())
    rf = gw_shell.calibration_slice(pd.read_csv(gw_shell.RF_DATA_PATH, parse_dates=["date time"]).set_index("date time"))
    rf = rf.reindex(gw.index)
    coords = active.set_index("st_id")[["gw_TM_X97", "gw_TM_Y97"]].rename(columns={"gw_TM_X97": "x", "gw_TM_Y97": "y"})
    heads = gw[coords.index].mean()
    updip = updip_direction(coords, heads)
    rf_of = active.set_index("st_id")["rf_id"]
    resid = {st: rain_residual(gw[st].diff(), rf[rf_of[st]]) for st in coords.index}
    wells = pd.read_csv(args.wells or (run / "analysis" / "well_metadata.csv")).astype({"st_id": str}).set_index("st_id")
    layer = wells["layer"]

    rfcond, samelayer = {}, {}
    adopted = active.set_index("st_id")["ups_id"].astype(str)
    for st in coords.index:
        cands = upgradient_candidates(st, coords, heads, updip)
        dist = {c: float(np.hypot(*(coords.loc[c, ["x", "y"]] - coords.loc[st, ["x", "y"]]))) for c in cands}
        within = [c for c in cands if dist[c] <= args.radius_km * 1000.0]
        sc = {c: lagged_abs_corr(resid[st], resid[c]) for c in within}
        sc = {c: v for c, v in sc.items() if np.isfinite(v)}
        rfcond[st] = max(sc, key=sc.get) if sc else "none"
        a = adopted[st]
        if a == "none" or a == "nan" or layer.get(a) == layer.get(st):
            samelayer[st] = a
        else:
            same = [c for c in within if layer.get(c) == layer.get(st)]
            sc2 = {c: lagged_abs_corr(gw[st].diff().to_numpy(), gw[c].diff().to_numpy()) for c in same}
            sc2 = {c: v for c, v in sc2.items() if np.isfinite(v)}
            samelayer[st] = max(sc2, key=sc2.get) if sc2 else "none"

    for name, mapping in (("ups_rfcond", rfcond), ("ups_samelayer", samelayer)):
        t = active.copy()
        t["ups_id"] = t.st_id.map(mapping)
        t["models"] = t.st_id.map(selected)
        t.to_csv(out / f"{name}.csv", index=False)
        same = int((t.set_index("st_id")["ups_id"].astype(str) == adopted).sum())
        print(f"{name}: same upstream as reference at {same}/{len(t)} stations")
    print(f"Wrote tables to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
