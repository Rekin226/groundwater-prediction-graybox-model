"""Synthetic-jump sensitivity diagnostic.

Picks a clean baseline station, refits with and without an injected step in
ζ_obs, and reports parameter / metric drift. Answers: "do undetected jumps
silently bias the current pipeline?"

Run:
    poetry run python subsidence/scripts/diag_jump_sensitivity.py \
        --station YWJS --jump-m 0.03 --jump-date 2021-06-15
"""
from __future__ import annotations
import argparse
import sys
import urllib.parse
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from subsidence.sub_shell import fit_station

CAL_START = pd.Timestamp("2020-01-01")
CAL_END   = pd.Timestamp("2022-12-31")
VAL_START = pd.Timestamp("2024-01-01")
VAL_END   = pd.Timestamp("2025-03-31")
DEFAULT_BOUNDS = {
    "Sk_e": (1e-6, 1e-2), "Sk_v": (1e-6, 1e-1),
    "h_ref": (-50.0, 50.0),
    "v_tect": (-0.015, 0.015),
    "v0": (-0.005, 0.005), "v1": (-0.002, 0.002),
    "tau": (7.0, 1500.0),
}
V_TECT_BOUNDS_BY_DATASET = {
    "ls-wra-gnss-obs": (-0.015, 0.015),
    "ls-wra-mlcw-obs": (-0.005, 0.005),
}
MIN_FORM3_OBS = 36


def _per_station_bounds(h: np.ndarray, sub_dataset: str = "") -> dict:
    out = dict(DEFAULT_BOUNDS)
    h_finite = h[np.isfinite(h)]
    if h_finite.size > 0:
        out["h_ref"] = (float(np.min(h_finite)) - 5.0, float(np.max(h_finite)) + 5.0)
    if sub_dataset in V_TECT_BOUNDS_BY_DATASET:
        out["v_tect"] = V_TECT_BOUNDS_BY_DATASET[sub_dataset]
    return out


def _load(sub_id: str, sub_dataset: str):
    h_df = pd.read_parquet(f"subsidence/data/h_drivers/{sub_id}.parquet")
    sid_encoded = urllib.parse.quote(str(sub_id), safe="")
    raw = pd.read_parquet(f"data/ls_cache/{sub_dataset}__{sid_encoded}.parquet")
    if isinstance(raw, pd.DataFrame) and "value" in raw.columns:
        s = raw["value"].dropna()
    else:
        s = raw.dropna()
    zeta_full = s.iloc[0] - s
    return h_df, zeta_full


def _fit(h_df, zeta_full, sub_dataset, *, jump_m=0.0, jump_date=None):
    idx = h_df.index
    zeta = zeta_full.reindex(idx)
    if jump_m != 0.0 and jump_date is not None:
        jd = pd.Timestamp(jump_date)
        mask = zeta.index >= jd
        zeta.loc[mask] = zeta.loc[mask] + jump_m

    cal_obs = zeta.loc[(zeta.index >= CAL_START) & (zeta.index <= CAL_END)].dropna()
    if not cal_obs.empty:
        zeta = zeta - float(cal_obs.iloc[0])

    h = h_df["h_driver"].values
    t_years = (idx - idx[0]).days.values / 365.25
    cal_idx = np.where((idx >= CAL_START) & (idx <= CAL_END))[0]
    val_idx = np.where((idx >= VAL_START) & (idx <= VAL_END))[0]
    n_obs_cal = int((~np.isnan(zeta.values[cal_idx])).sum())
    form3_ok = sub_dataset != "ls-wra-mlcw-obs" or n_obs_cal >= MIN_FORM3_OBS
    bounds = _per_station_bounds(h, sub_dataset)

    fit = fit_station(h=h, zeta_obs=zeta.values, t_years=t_years,
                      cal_idx=cal_idx, val_idx=val_idx,
                      bounds=bounds, form3_eligible=form3_ok)
    best = fit["best_variant"]
    out = fit["all_variants"][best]
    return {"variant": best, **out["params"],
            "kge_cal": out["kge_cal"], "kge_val": out["kge_val"],
            "rmse_cal": out["rmse_cal"], "rmse_val": out["rmse_val"],
            "kge_rate_val": out["kge_rate_val"]}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--station", default="YWJS")
    p.add_argument("--dataset", default="ls-wra-gnss-obs")
    p.add_argument("--jump-m", type=float, default=0.03)
    p.add_argument("--jump-date", default="2021-06-15")
    args = p.parse_args(argv)

    h_df, zeta_full = _load(args.station, args.dataset)

    print(f"Station: {args.station}  ({args.dataset})")
    print(f"Jump:    +{args.jump_m*100:.1f} cm at {args.jump_date}\n")

    base = _fit(h_df, zeta_full, args.dataset)
    pert = _fit(h_df, zeta_full, args.dataset,
                jump_m=args.jump_m, jump_date=args.jump_date)

    rows = []
    keys = sorted(set(base.keys()) | set(pert.keys()))
    for k in keys:
        b = base.get(k, "—")
        p_ = pert.get(k, "—")
        if isinstance(b, float) and isinstance(p_, float):
            d = p_ - b
            pct = (d / b * 100) if b != 0 else float("nan")
            rows.append((k, f"{b:>10.5g}", f"{p_:>10.5g}",
                         f"{d:>+10.4g}", f"{pct:>+7.1f}%"))
        else:
            rows.append((k, str(b), str(p_), "—", "—"))

    print(f"{'param/metric':<14} {'baseline':>10} {'with_jump':>10} {'Δ':>10} {'%Δ':>8}")
    print("-" * 60)
    for r in rows:
        print(f"{r[0]:<14} {r[1]} {r[2]} {r[3]} {r[4]}")


if __name__ == "__main__":
    sys.exit(main())
