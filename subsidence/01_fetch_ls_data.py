"""Fetch subsidence observations + station metadata for Zhuoshui zone 50.

Outputs:
    data/ls_cache/<dataset>__meta.parquet         — station metadata table
    data/ls_cache/<dataset>__<sid>.parquet        — per-station time series
    subsidence/data/sub_station_master.csv        — master active-station table

Run:
    LS_USER=... LS_PASS=... poetry run python subsidence/01_fetch_ls_data.py \
        --start 2019-10-01 --end 2025-03-31
"""
from __future__ import annotations
import argparse
import sys
import urllib.parse
from pathlib import Path

import pandas as pd

from subsidence.api_constants import (
    GNSS_WRA, GNSS_NCKU_1D, GNSS_NCKU_7D, MLCW, DBM, ZHUOSHUI_ZONE_ID,
)
from subsidence.ls_client import LSClient, split_to_df

SUBSIDENCE_DATASETS = [GNSS_WRA, GNSS_NCKU_1D, GNSS_NCKU_7D, MLCW, DBM]
CACHE_DIR = Path("data/ls_cache")
MASTER_CSV = Path("subsidence/data/sub_station_master.csv")
ACTIVE_GATES = {
    "daily":   {"min_cal_obs": 365, "min_val_obs": 60, "min_span_days": 730},
    "monthly": {"min_cal_obs":  24, "min_val_obs":  6, "min_span_days": 730},
}


def _zhuoshui_subset(meta_df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    if "GroundwaterZoneIdentifier" in meta_df.columns:
        return meta_df[meta_df["GroundwaterZoneIdentifier"] == ZHUOSHUI_ZONE_ID].copy()
    # NCKU + DBM lack zone column — keep all (operator filters later by coordinates if needed)
    return meta_df.copy()


def _ensure_twd97_columns(meta_df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee X_3826/Y_3826 by reprojecting from Longitude_4326/Latitude_4326
    when needed. EPSG:3826 = TWD97 / TM2 zone 121, matching the existing GW
    model's gw_TM_X97/gw_TM_Y97 convention.

    Also handles NCKU datasets that expose easting/northing as e_twd97/n_twd97.
    """
    df = meta_df.copy()
    has_xy = ("X_3826" in df.columns and "Y_3826" in df.columns
              and df[["X_3826", "Y_3826"]].notna().all(axis=None))
    if has_xy:
        return df
    # NCKU datasets provide e_twd97 (easting) / n_twd97 (northing) — same CRS
    if "e_twd97" in df.columns and "n_twd97" in df.columns:
        df["X_3826"] = df["e_twd97"].astype(float)
        df["Y_3826"] = df["n_twd97"].astype(float)
        return df
    # DBM dataset provides TWD97坐標_X / TWD97坐標_Y (easting/northing in EPSG:3826)
    if "TWD97坐標_X" in df.columns and "TWD97坐標_Y" in df.columns:
        df["X_3826"] = pd.to_numeric(df["TWD97坐標_X"], errors="coerce")
        df["Y_3826"] = pd.to_numeric(df["TWD97坐標_Y"], errors="coerce")
        return df
    lon_col = next((c for c in ("Longitude_4326", "longitude_4326") if c in df.columns), None)
    lat_col = next((c for c in ("Latitude_4326", "latitude_4326") if c in df.columns), None)
    if lon_col is None or lat_col is None:
        raise RuntimeError(
            f"{meta_df.attrs.get('dataset', '?')}: metadata lacks both TWD97 and WGS84 coords"
        )
    from pyproj import Transformer
    tr = Transformer.from_crs(4326, 3826, always_xy=True)
    x, y = tr.transform(df[lon_col].astype(float).values, df[lat_col].astype(float).values)
    df["X_3826"] = x; df["Y_3826"] = y
    return df


def _activeness(ts: pd.DataFrame, cadence: str,
                cal_start, cal_end, val_start, val_end) -> tuple[bool, dict]:
    g = ACTIVE_GATES[cadence]
    if ts.empty:
        return False, {"reason": "empty"}
    cal = ts.loc[cal_start:cal_end]
    val = ts.loc[val_start:val_end]
    span = (ts.index.max() - ts.index.min()).days
    return (
        len(cal) >= g["min_cal_obs"] and len(val) >= g["min_val_obs"] and span >= g["min_span_days"],
        {"n_cal": len(cal), "n_val": len(val), "span_days": span},
    )


def fetch_subsidence(client: LSClient, start: str, end: str,
                     cal_start, cal_end, val_start, val_end) -> pd.DataFrame:
    rows = []
    for ds in SUBSIDENCE_DATASETS:
        try:
            meta_payload = client.get_json(f"/dataset/{ds}/station/", params={"orient": "split"})
            # Preserve the raw station IDs from the split index (they are alphanumeric, not datetimes)
            raw_station_ids = meta_payload.get("index", [])
            meta_df = split_to_df(meta_payload)
            # Attach the real station IDs as a column (the index was coerced to NaT by split_to_df)
            meta_df["_station_id"] = raw_station_ids
            meta_df = meta_df.reset_index(drop=True)
        except Exception as e:
            print(f"  {ds}: meta endpoint failed: {str(e)[:80]}")
            continue
        meta_df.to_parquet(CACHE_DIR / f"{ds}__meta.parquet")
        zhu = _zhuoshui_subset(meta_df, ds)
        # Ensure pairing-ready coordinates (X_3826/Y_3826 reprojected from WGS84 if needed)
        zhu = _ensure_twd97_columns(zhu)
        cadence = "monthly" if ds == MLCW else "daily"
        print(f"\n[{ds}]  candidates in zone 50: {len(zhu)}")
        for _, sid_meta in zhu.iterrows():
            sid = sid_meta["_station_id"]
            # URL-encode station IDs that may contain non-ASCII characters (e.g. MLCW Chinese names)
            sid_encoded = urllib.parse.quote(str(sid), safe="")
            try:
                ts = client.cached_get_dataframe(
                    f"/dataset/{ds}/station/{sid_encoded}/data",
                    params={"orient": "split", "start_datetime": start, "end_datetime": end},
                    cache_dir=CACHE_DIR, cache_key=f"{ds}__{sid_encoded}",
                )
            except Exception as e:
                print(f"  {str(sid)}: ERR {str(e)[:80]}"); continue
            ok, info = _activeness(ts, cadence, cal_start, cal_end, val_start, val_end)
            row = {"sub_id": sid, "sub_dataset": ds, "cadence": cadence,
                   "active": int(ok), **info}
            for k in ("Longitude_4326", "Latitude_4326", "longitude_4326", "latitude_4326",
                     "X_3826", "Y_3826", "GroundwaterZoneIdentifier", "ObservatoryName"):
                if k in sid_meta:
                    row[k] = sid_meta[k]
            rows.append(row)
            print(f"  {str(sid):<25s}  active={int(ok)}  {info}")
    return pd.DataFrame(rows)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2019-10-01")
    p.add_argument("--end", default="2025-03-31")
    p.add_argument("--cal-start", default="2020-01-01")
    p.add_argument("--cal-end", default="2022-12-31")
    p.add_argument("--val-start", default="2024-01-01")
    p.add_argument("--val-end", default="2025-03-31")
    args = p.parse_args(argv)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    MASTER_CSV.parent.mkdir(parents=True, exist_ok=True)

    client = LSClient()
    cal_start, cal_end = pd.Timestamp(args.cal_start), pd.Timestamp(args.cal_end)
    val_start, val_end = pd.Timestamp(args.val_start), pd.Timestamp(args.val_end)
    df = fetch_subsidence(client, args.start, args.end, cal_start, cal_end, val_start, val_end)
    df.to_csv(MASTER_CSV, index=False)
    n_active = int(df["active"].sum()) if "active" in df.columns else 0
    print(f"\nWrote {MASTER_CSV} — {n_active}/{len(df)} active stations.")


if __name__ == "__main__":
    sys.exit(main())
