"""Diagnostics raised by the 2026-08-18 review. Produces the numbers quoted in the
revised manuscript and writes ESM 2 (the quality-control sensitivity sweep).

  M4  QC-gate sensitivity: do the reported medians and the scaling exponent depend
      on the r and seasonal-R2 thresholds?
  M2  Can the manuscript's own data separate the two explanations for beta < 1?
      (a) divide the thickness term out of the MLCW estimates
      (b) test the spatial prediction the lithological explanation makes
      (c) negative control on horizontal pairing distance
  M5  The QC statistic r is computed on the full trend-removed residual, the
      storativity on the annual harmonic alone. Recompute r in the seasonal band.
  M7  Five annual cycles of GNSS with a drought in 2023: leave-one-year-out spread
      of the recovered storativity.

Run:
    poetry run python subsidence/scripts/poro_review_diagnostics.py
"""
from __future__ import annotations
import sys
import urllib.parse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats, signal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from subsidence.ls_client import to_api_id
from subsidence.poroelastic import decompose, couple, storativity, common_origin

SUB = Path("workspace/manuscripts/poroelastic_gnss-mlcw/submission")
ESM1 = SUB / "ESM1_poroelastic_per_station_results.csv"
ESM2 = SUB / "ESM2_qc_gate_sensitivity.csv"
CACHE = Path("data/ls_cache")
CLEAN = CACHE / "clean"
GN, ML = "ls-wra-gnss-obs", "ls-wra-mlcw-obs"
DSET = {"GNSS": GN, "MLCW": ML}


def _detz(s: pd.Series) -> pd.Series:
    if getattr(s.index, "tz", None) is not None:
        s.index = s.index.tz_localize(None)
    return s


def load_pair(station: str, network: str):
    """Displacement (m, up positive) and paired head (m) for one station."""
    ds = DSET[network]
    f = CLEAN / f"{ds}__{urllib.parse.quote(station, safe='')}.parquet"
    d = pd.read_parquet(f)
    if ds == GN:
        disp = _detz(d["value"].dropna())
    else:
        rings = sorted((c for c in d.columns if c.startswith("NO")), key=lambda c: int(c[2:]))
        d = _detz(d)
        disp = None
        for c in reversed(rings):
            s = d[c].dropna()
            if len(s) >= 24:
                disp = (d[c] - float(s.iloc[0])).dropna()
                break
    pair = pd.read_csv("subsidence/data/sub_pairing.csv")
    pr = pair[(pair.sub_id == station) & (pair.sub_dataset == ds)]
    if pr.empty:
        pr = pair[pair.sub_id == station]
    head = _detz(pd.read_parquet(CACHE / f"gw__{to_api_id(int(pr.iloc[0].gw_st))}.parquet")
                 ["h_obs_m"].dropna())
    return disp, head


def beta_of(sub: pd.DataFrame):
    """Power-law exponent of displacement amplitude on head amplitude."""
    lr = stats.linregress(np.log(sub.head_seasonal_amp_m), np.log(sub.disp_seasonal_amp_m))
    return lr.slope, lr.stderr


# --------------------------------------------------------------------------- M4
def m4_gate_sensitivity(e: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rmin in (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60):
        for r2min in (0.05, 0.10, 0.15, 0.20):
            s = e[(e.coupling_r > rmin) & (e.disp_seasonal_R2 > r2min)]
            if len(s) < 4:
                continue
            b, se = beta_of(s)
            g = s[s.network == "GNSS"]
            m = s[s.network == "MLCW"]
            rows.append(dict(
                r_min=rmin, R2_min=r2min, n_total=len(s), n_gnss=len(g), n_mlcw=len(m),
                S_ke_median=round(s.S_ke.median(), 5),
                S_ke_median_gnss=round(g.S_ke.median(), 5) if len(g) else np.nan,
                S_ke_median_mlcw=round(m.S_ke.median(), 5) if len(m) else np.nan,
                beta=round(b, 3), beta_se=round(se, 3),
                beta_below_1=bool(b + 1.96 * se < 1.0),
                rho_Ske_headamp=round(stats.spearmanr(s.head_seasonal_amp_m, s.S_ke)[0], 3),
                amp_corr_r=round(stats.pearsonr(s.head_seasonal_amp_m, s.disp_seasonal_amp_m)[0], 3),
            ))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- M2
def m2_separation_tests(e: pd.DataFrame) -> None:
    v = e[e.elastic_valid]
    m = v[v.network == "MLCW"]
    print("(a) thickness divided out (MLCW only, n = %d)" % len(m))
    for lab, col in (("S_ke", m.S_ke), ("Ss = S_ke/b", m.Ss_per_m), ("column depth b", m.deepest_ring_depth_m)):
        r, p = stats.spearmanr(m.head_seasonal_amp_m, col)
        print(f"      rho(head amplitude, {lab:<14}) = {r:+.3f}   p = {p:.3f}")

    master = pd.read_csv("subsidence/data/sub_station_master.csv")
    xy = master.set_index(["sub_id", "sub_dataset"])[["X_3826", "Y_3826"]]
    co = [xy.loc[(r.station, DSET[r.network])] for r in v.itertuples()]
    v = v.assign(X=[c.X_3826 for c in co], Y=[c.Y_3826 for c in co])
    # Fan apex is the north-easternmost extent of the boundary; distance from it is
    # the natural proximal-to-distal coordinate.
    apex = (v.X.max(), v.Y.max())
    v = v.assign(dist_apex=np.hypot(v.X - apex[0], v.Y - apex[1]) / 1000.0)
    print(f"\n(b) spatial prediction of the lithological explanation (n = {len(v)})")
    for lab, col in (("easting", v.X), ("northing", v.Y), ("distance from fan apex", v.dist_apex)):
        r, p = stats.spearmanr(col, v.S_ke)
        print(f"      rho(S_ke, {lab:<22}) = {r:+.3f}   p = {p:.3f}")
    r, p = stats.spearmanr(v.head_seasonal_amp_m, v.S_ke)
    print(f"      rho(S_ke, {'seasonal head amplitude':<22}) = {r:+.3f}   p = {p:.5f}   <- for comparison")

    print(f"\n(c) negative control on horizontal pairing distance (n = {len(v)})")
    r, p = stats.spearmanr(v.paired_well_distance_km, v.S_ke)
    print(f"      rho(S_ke, pairing distance) = {r:+.3f}   p = {p:.3f}")


# --------------------------------------------------------------------------- M5
def _bandpass(s: pd.Series, lo_yr: float, hi_yr: float) -> pd.Series:
    """Retain periods between lo_yr and hi_yr. Gaps are filled by interpolation for
    the filter only; the result is re-masked to the originally observed dates."""
    daily = s.resample("D").mean()
    obs = daily.notna()
    y = daily.interpolate("time").bfill().ffill().to_numpy()
    fs = 365.25  # samples per year
    b, a = signal.butter(2, [1.0 / hi_yr / (fs / 2), 1.0 / lo_yr / (fs / 2)], btype="band")
    out = pd.Series(signal.filtfilt(b, a, y), index=daily.index)
    return out.where(obs)


def m5_seasonal_band_coupling(e: pd.DataFrame) -> pd.DataFrame:
    """Recompute the coupling on the seasonal band (0.5-2 yr) instead of the full
    trend-removed residual, which also carries interannual variance."""
    rows = []
    for r in e[e.network == "GNSS"].itertuples():
        if not np.isfinite(r.coupling_r):
            continue
        disp, head = load_pair(r.station, r.network)
        t0 = common_origin(disp, head)
        sd, hd = decompose(disp, t0=t0), decompose(head, t0=t0)
        rb, _ = couple(_bandpass(sd.residual, 0.5, 2.0), _bandpass(hd.residual, 0.5, 2.0))
        rows.append(dict(station=r.station, r_broadband=round(r.coupling_r, 3),
                         r_seasonal_band=round(rb, 3), seasonal_R2=round(r.disp_seasonal_R2, 3),
                         elastic_valid=bool(r.elastic_valid)))
    d = pd.DataFrame(rows)
    print(d.to_string(index=False))
    both = d.dropna()
    print(f"\n      median r broadband = {both.r_broadband.median():.3f}, "
          f"seasonal band = {both.r_seasonal_band.median():.3f}")
    print(f"      would the gate change? stations passing r > 0.4 on the broadband: "
          f"{(both.r_broadband > 0.4).sum()}; on the seasonal band: {(both.r_seasonal_band > 0.4).sum()}")
    flips = both[(both.r_broadband > 0.4) != (both.r_seasonal_band > 0.4)]
    print(f"      stations that change side of the threshold: "
          f"{flips.station.tolist() if len(flips) else 'none'}")
    return d


# --------------------------------------------------------------------------- M7
def m7_leave_one_year_out(e: pd.DataFrame) -> pd.DataFrame:
    """Refit each elastic-valid GNSS station's harmonic with one calendar year of
    displacement withheld, to test sensitivity to the 2023 drought."""
    rows = []
    for r in e[(e.network == "GNSS") & (e.elastic_valid)].itertuples():
        disp, head = load_pair(r.station, r.network)
        t0 = common_origin(disp, head)
        h_amp = decompose(head, t0=t0).amplitude
        rec = {"station": r.station, "S_ke_all": round(r.S_ke, 5)}
        vals = []
        for yr in sorted({d.year for d in disp.index}):
            sub = disp[disp.index.year != yr]
            if len(sub) < 365:
                continue
            s = storativity(decompose(sub, t0=t0).amplitude, h_amp)
            rec[f"drop{yr}"] = round(s, 5)
            vals.append(s)
        rec["spread_pct"] = round((max(vals) - min(vals)) / r.S_ke * 100, 1)
        rec["drop2023_change_pct"] = round((rec.get("drop2023", np.nan) / r.S_ke - 1) * 100, 1)
        rows.append(rec)
    d = pd.DataFrame(rows)
    print(d.to_string(index=False))
    print(f"\n      median across stations of the leave-one-year-out spread: "
          f"{d.spread_pct.median():.1f}% of S_ke")
    print(f"      dropping 2023 alone moves the network median S_ke by "
          f"{(d.drop2023.median() / d.S_ke_all.median() - 1) * 100:+.1f}%")
    return d


def main() -> None:
    e = pd.read_csv(ESM1)
    line = "=" * 74

    print(line); print("M4 — QC-GATE SENSITIVITY"); print(line)
    sweep = m4_gate_sensitivity(e)
    print(sweep.to_string(index=False))
    sweep.to_csv(ESM2, index=False)
    adopted = sweep[(sweep.r_min == 0.40) & (sweep.R2_min == 0.10)].iloc[0]
    print(f"\n      beta < 1 at every gate setting: {sweep.beta_below_1.all()}"
          f"  (range {sweep.beta.min():.2f} to {sweep.beta.max():.2f})")
    print(f"      adopted gate (r > 0.4, R2 > 0.10) gives beta = {adopted.beta:.2f}, "
          f"rank {int((sweep.beta > adopted.beta).sum()) + 1} of {len(sweep)} "
          f"(1 = closest to unity, i.e. most conservative)")
    print(f"      S_ke median across all gate settings: {sweep.S_ke_median.min():.4f}"
          f" to {sweep.S_ke_median.max():.4f}")
    print(f"      -> {ESM2}")

    print("\n" + line); print("M2 — SEPARATING THE TWO EXPLANATIONS FOR beta < 1"); print(line)
    m2_separation_tests(e)

    print("\n" + line); print("M5 — COUPLING ON THE SEASONAL BAND VS THE FULL RESIDUAL"); print(line)
    m5_seasonal_band_coupling(e)

    print("\n" + line); print("M7 — LEAVE-ONE-YEAR-OUT (2023 DROUGHT SENSITIVITY)"); print(line)
    m7_leave_one_year_out(e)


if __name__ == "__main__":
    main()
