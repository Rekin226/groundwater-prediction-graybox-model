"""Runner — elastic skeletal storage from the seasonal head <-> surface coupling.

Loops the subsidence station set (GNSS + MLCW), calls the I/O-free core in
`subsidence/poroelastic.py`, and writes a results table + per-station trend-removed
residuals for plotting.

Both networks are reduced to the SAME convention: vertical displacement, UP positive.
  GNSS  : `value` is surface elevation (already up-positive).
  MLCW  : ring position; displacement = ring(t) - ring(t0) = -(cumulative compaction),
          so head-up -> displacement-up gives a positive elastic coupling for both.
MLCW additionally yields specific storage Ss = S_ke / depth, because the deepest ring
depth is known; GNSS stays at the dimensionless S_ke (no column-thickness assumption).

This is the poroelastic-storage track (explanatory). It does NOT import or run the
retired groundwater-driven forecasting ODE.

Reads:
    workspace/results_sub/<fit-set>/sub_fit_results.csv   (canonical station set)
    subsidence/data/sub_pairing.csv                       (sub_id -> gw_st)
    subsidence/data/sub_station_master.csv                (coordinates, names)
    data/ls_cache/clean/<dataset>__<url_encoded_sub_id>.parquet
    data/ls_cache/gw__<api_id>.parquet                   (observed head h_obs_m, m)
Writes:
    workspace/results_sub/<run-id>/poroelastic_results.csv
    workspace/results_sub/<run-id>/per_station/<sub_id>_decomp.csv

Run:
    poetry run python subsidence/08_poroelastic_storage.py
    poetry run python subsidence/08_poroelastic_storage.py --run-id poroelastic_v1 --station YWJS
"""
from __future__ import annotations
import argparse
import sys
import urllib.parse
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from subsidence.ls_client import to_api_id
from subsidence.poroelastic import (
    decompose, couple, phase_lag_days, storativity, specific_storage, is_elastic_valid,
    bootstrap_ci, common_origin,
)

GNSS_DATASET = "ls-wra-gnss-obs"
MLCW_DATASET = "ls-wra-mlcw-obs"
NET_LABEL = {GNSS_DATASET: "GNSS", MLCW_DATASET: "MLCW"}
NET_RESAMPLE = {GNSS_DATASET: "D", MLCW_DATASET: "MS"}  # MLCW is ~monthly
# Bootstrap block length, in SAMPLES of the series it applies to, chosen so every
# series carries the same ~90-day block. Ninety days is where the width of the
# bootstrapped amplitude interval stops growing with block length: the residual
# decorrelation scale is roughly seasonal, so a 30-day block resamples within it and
# understates the amplitude uncertainty by about a factor of 1.4. Longer blocks than
# this leave too few blocks in a five-year record for a stable resample.
NET_BLOCK = {GNSS_DATASET: 90, MLCW_DATASET: 3}  # daily / approximately monthly
HEAD_BLOCK = 90  # head is daily in both networks
MIN_OBS = {GNSS_DATASET: 365 * 2, MLCW_DATASET: 24}
MLCW_MIN_RING_OBS = 24
CACHE = Path("data/ls_cache")
CLEAN = CACHE / "clean"
# Study area. Stations outside the fan are dropped: the storage parameters are
# reported for the CRAF aquifer system, and a station beyond the fan margin is
# neither over that system nor near enough to an in-zone well to pair reliably.
BOUNDARY_SHP = Path("data/Zhuoshui Alluvial Fan/Zhuoshui Alluvial Fan.shp")
STATION_EPSG = 3826  # X_3826 / Y_3826 in the station master (TWD97 TM2)
# MLCW stores latitude in lowercase ``latitude_4326``; GNSS in ``Latitude_4326``.
COORD_COLS = ["Longitude_4326", "Latitude_4326", "latitude_4326", "X_3826", "Y_3826",
              "ObservatoryName", "GroundwaterZoneIdentifier"]


def _clean_path(dataset: str, sub_id: str) -> Path:
    return CLEAN / f"{dataset}__{urllib.parse.quote(sub_id, safe='')}.parquet"


def _detz(s: pd.Series) -> pd.Series:
    if getattr(s.index, "tz", None) is not None:
        s.index = s.index.tz_localize(None)
    return s


def load_displacement(dataset: str, sub_id: str) -> tuple[pd.Series | None, float]:
    """Vertical displacement (m, UP positive) and column thickness (m, NaN for GNSS)."""
    f = _clean_path(dataset, sub_id)
    if not f.exists():
        return None, np.nan
    d = pd.read_parquet(f)
    if dataset == GNSS_DATASET:
        return _detz(d["value"].dropna()), np.nan
    # MLCW: deepest ring with enough obs = whole-column compaction observable.
    rings = sorted((c for c in d.columns if c.startswith("NO")), key=lambda c: int(c[2:]))
    d = _detz(d)
    for c in reversed(rings):
        s = d[c].dropna()
        if len(s) >= MLCW_MIN_RING_OBS:
            depth0 = float(s.iloc[0])               # ring install depth (m)
            disp = (d[c] - depth0).dropna()         # = -(cumulative compaction), up positive
            return disp, depth0
    return None, np.nan


def load_head(gw_st) -> pd.Series | None:
    if pd.isna(gw_st):
        return None
    try:
        api = to_api_id(int(gw_st))
    except Exception:
        return None
    f = CACHE / f"gw__{api}.parquet"
    if not f.exists():
        return None
    return _detz(pd.read_parquet(f)["h_obs_m"].dropna())


def inside_study_area(stations: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    """Keep only stations falling inside the alluvial-fan boundary polygon.

    Uses the projected X_3826/Y_3826 coordinates from the station master, not the
    two-decimal lat/lon, which rounds to roughly a kilometre and cannot resolve a
    station near the fan margin.
    """
    import geopandas as gpd
    from shapely.geometry import Point

    poly = gpd.read_file(BOUNDARY_SHP).to_crs(epsg=STATION_EPSG).union_all()
    xy = master.set_index(["sub_id", "sub_dataset"])[["X_3826", "Y_3826"]]
    keep, dropped = [], []
    for sid, dataset in stations.itertuples(index=False):
        try:
            x, y = xy.loc[(sid, dataset)]
        except KeyError:
            keep.append(True)          # no coordinates: cannot test, do not exclude
            continue
        p = Point(float(x), float(y))
        ok = poly.contains(p)
        keep.append(ok)
        if not ok:
            dropped.append((sid, NET_LABEL[dataset], poly.distance(p) / 1000.0))
    for sid, net, km in dropped:
        print(f"  skip {sid} ({net}): {km:.1f} km outside the study-area boundary")
    return stations[pd.Series(keep, index=stations.index)]


def run(run_id: str, fit_set: str, only_station: str | None) -> None:
    out_dir = Path("workspace/results_sub") / run_id
    (out_dir / "per_station").mkdir(parents=True, exist_ok=True)

    best = pd.read_csv(Path("workspace/results_sub") / fit_set / "sub_fit_results.csv")
    pair = pd.read_csv("subsidence/data/sub_pairing.csv")
    master = pd.read_csv("subsidence/data/sub_station_master.csv")

    stations = best[best["sub_dataset"].isin(NET_LABEL)][["sub_id", "sub_dataset"]]
    n_all = len(stations)
    stations = inside_study_area(stations, master)
    print(f"  study-area filter: {len(stations)}/{n_all} stations retained")
    if only_station:
        stations = stations[stations["sub_id"] == only_station]

    rows = []
    for _i, (sid, dataset) in enumerate(stations.itertuples(index=False)):
        disp, thickness = load_displacement(dataset, sid)
        if disp is None or len(disp) < MIN_OBS[dataset]:
            print(f"  skip {sid} ({NET_LABEL[dataset]}): insufficient data")
            continue
        pr = pair[(pair["sub_id"] == sid) & (pair["sub_dataset"] == dataset)]
        if pr.empty:
            pr = pair[pair["sub_id"] == sid]
        gw_st = pr.iloc[0]["gw_st"] if not pr.empty else np.nan
        dist_km = (pr.iloc[0]["distance_m"] / 1000.0) if not pr.empty else np.nan
        head = load_head(gw_st)
        resample = NET_RESAMPLE[dataset]

        # Phase is only meaningful against a shared epoch, so displacement and head
        # are decomposed on a common origin (their records start years apart).
        t0 = common_origin(disp, head) if head is not None else None
        sd = decompose(disp, t0=t0)
        row = {"sub_id": sid, "sub_dataset": dataset, "network": NET_LABEL[dataset],
               "dist_km": dist_km, "thickness_m": thickness,
               "disp_seas_amp_m": sd.amplitude, "disp_seas_r2": sd.seasonal_r2,
               "disp_trend_m_yr": sd.trend_rate, "n_obs": sd.n}

        if head is not None and len(head) >= MIN_OBS[MLCW_DATASET]:
            hd = decompose(head, t0=t0)
            r, n_common = couple(sd.residual, hd.residual, resample=resample)
            s_ke = storativity(sd.amplitude, hd.amplitude)
            ss = specific_storage(s_ke, thickness) if np.isfinite(thickness) else np.nan
            ci = bootstrap_ci(disp, head, block_disp=NET_BLOCK[dataset],
                              block_head=HEAD_BLOCK, seed=_i)
            tk = thickness if np.isfinite(thickness) else np.nan
            row.update({"head_seas_amp_m": hd.amplitude, "head_seas_r2": hd.seasonal_r2,
                        "coupling_r": r, "coupling_n": n_common,
                        "phase_lag_days": phase_lag_days(sd, hd),
                        # phases + their shared epoch, so the fitted harmonics can be
                        # reconstructed downstream (plotting) without a re-fit
                        "disp_phase_rad": sd.phase, "head_phase_rad": hd.phase,
                        "phase_t0": sd.t0.isoformat(),
                        "s_ke": s_ke, "ss_per_m": ss,
                        "disp_amp_lo": ci["disp_amp_ci"][0], "disp_amp_hi": ci["disp_amp_ci"][1],
                        "head_amp_lo": ci["head_amp_ci"][0], "head_amp_hi": ci["head_amp_ci"][1],
                        "s_ke_lo": ci["s_ke_ci"][0], "s_ke_hi": ci["s_ke_ci"][1],
                        "ss_lo": ci["s_ke_ci"][0] / tk, "ss_hi": ci["s_ke_ci"][1] / tk,
                        "elastic_valid": is_elastic_valid(r, sd.seasonal_r2)})
            dd = sd.residual.resample(resample).mean().rename("disp_resid_m")
            hh = hd.residual.resample(resample).mean().rename("head_resid_m")
            pd.concat([dd, hh], axis=1, sort=True).to_csv(out_dir / "per_station" / f"{sid}_decomp.csv")
        else:
            row.update({"head_seas_amp_m": np.nan, "head_seas_r2": np.nan,
                        "coupling_r": np.nan, "coupling_n": 0, "phase_lag_days": np.nan,
                        "s_ke": np.nan, "ss_per_m": np.nan, "elastic_valid": False})
        rows.append(row)

    res = pd.DataFrame(rows)
    keep = ["sub_id", "sub_dataset"] + [c for c in COORD_COLS if c in master.columns]
    res = res.merge(master[keep], on=["sub_id", "sub_dataset"], how="left")
    # unify latitude (GNSS Latitude_4326 / MLCW latitude_4326) into one column
    if "latitude_4326" in res.columns:
        res["Latitude_4326"] = res["Latitude_4326"].fillna(res["latitude_4326"])
        res = res.drop(columns=["latitude_4326"])
    res.to_csv(out_dir / "poroelastic_results.csv", index=False)

    print("=" * 64)
    print(f"POROELASTIC STORAGE — run '{run_id}'  ({len(res)} stations)")
    print("=" * 64)
    for net in ("GNSS", "MLCW"):
        g = res[res["network"] == net]
        v = g[g["elastic_valid"]]
        if g.empty:
            continue
        print(f"  {net}: elastic-valid {len(v)}/{len(g)}", end="")
        if len(v):
            print(f"  | S_ke median={v['s_ke'].median():.4f}"
                  f"  | r median={v['coupling_r'].median():.2f}"
                  f"  | lag median={v['phase_lag_days'].median():+.0f} d", end="")
            if v["ss_per_m"].notna().any():
                print(f"  | Ss median={v['ss_per_m'].median():.2e} /m", end="")
        print()
    print(f"\n  -> {out_dir / 'poroelastic_results.csv'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="poroelastic_v1")
    ap.add_argument("--fit-set", default="v18_ysll_excluded")
    ap.add_argument("--station", default=None, help="run a single station")
    args = ap.parse_args()
    run(args.run_id, args.fit_set, args.station)


if __name__ == "__main__":
    main()
