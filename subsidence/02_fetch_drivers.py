"""Fetch GW + rainfall observations for paired stations.

Reads:
    data/gray_box_input.csv (or _optimized.csv)   — GW station list with rf_id
Writes:
    data/ls_cache/gw__<api_id>.parquet            — daily-mean GW level per station
    data/ls_cache/rain__<rf_id>.parquet           — daily-sum rainfall per station

Run:
    LS_USER=... LS_PASS=... poetry run python subsidence/02_fetch_drivers.py \
        --start 2019-10-01 --end 2025-03-31
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import pandas as pd

from subsidence.api_constants import GW_10MIN, RAIN_10MIN
from subsidence.ls_client import LSClient
from subsidence.ls_resample import clip_negative_to_zero, daily_mean, daily_sum, mask_sentinels

CACHE_DIR = Path("data/ls_cache")
GW_INPUT = Path("data/gray_box_input.csv")
RF_STATIONS = Path("data/rf_stations.csv")


def fetch_gw(client: LSClient, gw_st: int, start: str, end: str) -> pd.DataFrame:
    from subsidence.ls_client import to_api_id
    api_id = to_api_id(gw_st)
    df = client.cached_get_dataframe(
        f"/dataset/{GW_10MIN}/station/{api_id}/data",
        params={"orient": "split", "start_datetime": start, "end_datetime": end},
        cache_dir=CACHE_DIR, cache_key=f"gw__{api_id}__raw",
    )
    if df.empty:
        return df
    clean = mask_sentinels(df["value"])     # drop sentinel values before averaging
    daily = daily_mean(clean).to_frame("h_obs_m")
    daily.to_parquet(CACHE_DIR / f"gw__{api_id}.parquet")
    return daily


def fetch_rain(client: LSClient, rf_id: str, start: str, end: str,
               label: str | None = None) -> pd.DataFrame:
    """Fetch rainfall for API station ``rf_id`` (CWA code).

    ``label`` is the internal project key (e.g. ``rf8``); if given, the
    aggregated parquet is written as ``rain__<label>.parquet`` so it matches
    the join key in gray_box_input.csv.
    """
    out_key = label if label else rf_id
    df = client.cached_get_dataframe(
        f"/dataset/{RAIN_10MIN}/station/{rf_id}/data",
        params={"orient": "split", "start_datetime": start, "end_datetime": end},
        cache_dir=CACHE_DIR, cache_key=f"rain__{rf_id}__raw",
    )
    if df.empty:
        return df
    # 'Now' column is the 10-min reading; clip negatives, then daily-sum
    s = clip_negative_to_zero(df["Now"].astype(float))
    daily = daily_sum(s).to_frame("rainfall_mm")
    daily.to_parquet(CACHE_DIR / f"rain__{out_key}.parquet")
    return daily


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2019-10-01")
    p.add_argument("--end", default="2025-03-31")
    p.add_argument("--input", default=str(GW_INPUT))
    args = p.parse_args(argv)

    df = pd.read_csv(args.input)
    df = df[df.get("active", 1).astype(int) == 1] if "active" in df.columns else df
    client = LSClient()

    print(f"[GW]  fetching {len(df)} stations from {GW_10MIN}")
    for _, r in df.iterrows():
        try:
            out = fetch_gw(client, int(r["gw_st"]), args.start, args.end)
            print(f"  st_id={r['st_id']:6s}  gw_st={int(r['gw_st']):08d}  rows={len(out)}")
        except Exception as e:
            print(f"  st_id={r['st_id']}: ERR {str(e)[:120]}")

    # Build rf_id -> CWA station code mapping from rf_stations.csv
    rf_map: dict[str, str] = {}
    if RF_STATIONS.exists():
        rf_meta = pd.read_csv(RF_STATIONS)
        rf_map = dict(zip(rf_meta["rf_id"].astype(str), rf_meta["rf_num"].astype(str)))

    rf_ids = sorted({str(x) for x in df["rf_id"].dropna()})
    print(f"\n[RAIN]  fetching {len(rf_ids)} rainfall stations")
    for rf in rf_ids:
        api_rf = rf_map.get(rf, rf)  # resolve internal label → CWA code; fallback to rf
        try:
            out = fetch_rain(client, api_rf, args.start, args.end, label=rf)
            print(f"  rf_id={rf:8s}  api_id={api_rf:8s}  rows={len(out)}")
        except Exception as e:
            print(f"  rf_id={rf} (api_id={api_rf}): ERR {str(e)[:120]}")


if __name__ == "__main__":
    sys.exit(main())
