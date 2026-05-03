"""Orchestrator — clean LS observations and write cleaned cache + QC report.

Reads:
    subsidence/data/sub_station_master.csv (active=1 rows)
    data/ls_cache/<dataset>__<sid>.parquet (raw observations)
    data/eq_catalog.csv (auto-fetched on first run)
    data/ls_cache/<dataset>__meta.parquet (station coordinates)
Writes:
    data/ls_cache/clean/<dataset>__<sid>.parquet (cleaned drop-in)
    subsidence/data/qc_report.csv (per-jump audit)
    subsidence/data/qc_summary.csv (per-station summary)

Run:
    poetry run python subsidence/03b_clean_ls.py
    poetry run python subsidence/03b_clean_ls.py --station YWJS
"""
from __future__ import annotations
import argparse
import shutil
import sys
import urllib.parse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from subsidence.clean_ls import clean_station, compute_robust_sigma
from subsidence.eq_catalog import load_catalog

MASTER = Path("subsidence/data/sub_station_master.csv")
CACHE_RAW = Path("data/ls_cache")
OUT_DIR = Path("data/ls_cache/clean")
QC_REPORT = Path("subsidence/data/qc_report.csv")
QC_SUMMARY = Path("subsidence/data/qc_summary.csv")
EQ_CACHE = Path("data/eq_catalog.csv")

EXCLUDED_DATASETS = ("ls-wra-dbm-obs",)
GNSS_DS = "ls-wra-gnss-obs"
MLCW_DS = "ls-wra-mlcw-obs"

NEIGHBOR_RADIUS_KM = 30.0
NEIGHBOR_MAX = 5


def _gnss_meta() -> pd.DataFrame:
    """Return GNSS metadata DataFrame indexed by station id."""
    df = pd.read_parquet(CACHE_RAW / f"{GNSS_DS}__meta.parquet")
    df = df.set_index("_station_id")
    return df


def _load_zeta_m(sub_id: str, ds: str) -> pd.Series:
    """Load ζ_obs in metres. ζ = first_value − raw, anchored at first obs."""
    sid_enc = urllib.parse.quote(sub_id, safe="")
    raw = pd.read_parquet(CACHE_RAW / f"{ds}__{sid_enc}.parquet")
    if "value" not in raw.columns:
        raise RuntimeError(f"{sub_id}: 'value' column missing")
    s = raw["value"].dropna()
    return (s.iloc[0] - s).rename("zeta_m")


def _select_neighbors(target_sid: str, gnss_meta: pd.DataFrame) -> List[pd.Series]:
    """Return ζ series for up to NEIGHBOR_MAX nearest GNSS stations within
    NEIGHBOR_RADIUS_KM of target_sid. Empty list if metadata absent."""
    if target_sid not in gnss_meta.index:
        return []
    from subsidence.eq_catalog import haversine_km
    src = (float(gnss_meta.loc[target_sid, "Latitude_4326"]),
           float(gnss_meta.loc[target_sid, "Longitude_4326"]))
    out: List[tuple[float, str]] = []
    for sid, row in gnss_meta.iterrows():
        if sid == target_sid:
            continue
        try:
            d = haversine_km(src, (float(row["Latitude_4326"]),
                                    float(row["Longitude_4326"])))
        except Exception:
            continue
        if d <= NEIGHBOR_RADIUS_KM:
            out.append((d, sid))
    out.sort()
    out = out[:NEIGHBOR_MAX]
    series = []
    for _, sid in out:
        try:
            series.append(_load_zeta_m(sid, GNSS_DS))
        except Exception:
            continue
    return series


def _process_one(sub_id: str, sub_dataset: str,
                  gnss_meta: pd.DataFrame, eq_catalog: pd.DataFrame) -> dict:
    """Process a single station; write cleaned parquet; return summary row."""
    sid_enc = urllib.parse.quote(sub_id, safe="")
    src_path = CACHE_RAW / f"{sub_dataset}__{sid_enc}.parquet"
    dst_path = OUT_DIR / f"{sub_dataset}__{sid_enc}.parquet"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # MLCW: verbatim copy (cleaning deferred per spec §8)
    if sub_dataset == MLCW_DS:
        shutil.copy2(src_path, dst_path)
        return {"sub_id": sub_id, "sub_dataset": sub_dataset,
                "n_jumps_detected": 0, "deferred": True}

    if sub_dataset != GNSS_DS:
        return {"sub_id": sub_id, "sub_dataset": sub_dataset, "skipped": True}

    raw_full = pd.read_parquet(src_path)
    z = _load_zeta_m(sub_id, sub_dataset)
    sigma_before = compute_robust_sigma(z)

    if sub_id not in gnss_meta.index:
        # No metadata — write verbatim and skip
        shutil.copy2(src_path, dst_path)
        return {"sub_id": sub_id, "sub_dataset": sub_dataset,
                "n_jumps_detected": 0, "no_metadata": True}

    station_ll = (float(gnss_meta.loc[sub_id, "Latitude_4326"]),
                  float(gnss_meta.loc[sub_id, "Longitude_4326"]))
    neighbors = _select_neighbors(sub_id, gnss_meta)

    z_clean, qc = clean_station(
        z=z, station_lat_lon=station_ll,
        eq_catalog=eq_catalog, neighbor_series=neighbors,
        max_iterations=100,
    )
    sigma_after = compute_robust_sigma(z_clean)

    # Map cleaned ζ back to raw value column: value_clean = first_raw_obs - zeta_clean
    first_obs = raw_full["value"].dropna().iloc[0]
    raw_full = raw_full.copy()
    new_values = pd.Series(np.nan, index=raw_full.index)
    new_values.loc[z_clean.index] = first_obs - z_clean.values
    raw_full["value"] = new_values
    raw_full.to_parquet(dst_path)

    qc.insert(0, "sub_id", sub_id)
    qc.insert(1, "sub_dataset", sub_dataset)

    return {"sub_id": sub_id, "sub_dataset": sub_dataset,
            "n_jumps_detected": int(len(qc)),
            "n_co_seismic": int((qc["classification"] == "co_seismic").sum()) if not qc.empty else 0,
            "n_regional": int((qc["classification"] == "regional_event").sum()) if not qc.empty else 0,
            "n_glitch": int((qc["classification"] == "glitch").sum()) if not qc.empty else 0,
            "n_datum_reset": int((qc["classification"] == "datum_reset").sum()) if not qc.empty else 0,
            "n_drift_onset": int((qc["classification"] == "drift_onset").sum()) if not qc.empty else 0,
            "n_boundary": int((qc["classification"] == "boundary_uncertain").sum()) if not qc.empty else 0,
            "n_obs_before": int(z.notna().sum()),
            "n_obs_after": int(z_clean.notna().sum()),
            "sigma_before_cm": float(sigma_before * 100.0),
            "sigma_after_cm": float(sigma_after * 100.0),
            "_qc": qc,
            }


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--station", default=None)
    args = p.parse_args(argv)

    eq_catalog = load_catalog(cache_path=EQ_CACHE)
    gnss_meta = _gnss_meta()

    master = pd.read_csv(MASTER)
    master = master[master["active"] == 1]
    master = master[~master["sub_dataset"].isin(EXCLUDED_DATASETS)]
    if args.station:
        master = master[master["sub_id"] == args.station]
    if master.empty:
        print("No matching stations.")
        return 0

    summaries: List[dict] = []
    qc_frames: List[pd.DataFrame] = []
    for _, r in master.iterrows():
        try:
            summ = _process_one(r["sub_id"], r["sub_dataset"], gnss_meta, eq_catalog)
            qc_df = summ.pop("_qc", None)
            summaries.append(summ)
            if isinstance(qc_df, pd.DataFrame) and not qc_df.empty:
                qc_frames.append(qc_df)
            print(f"  {r['sub_id']:25s}  jumps={summ.get('n_jumps_detected', 0)}")
        except Exception as e:
            print(f"  {r['sub_id']:25s}  ERR {str(e)[:120]}")

    QC_REPORT.parent.mkdir(parents=True, exist_ok=True)
    if qc_frames:
        pd.concat(qc_frames, ignore_index=True).to_csv(QC_REPORT, index=False)
    else:
        # Always write at least an empty file so downstream readers don't crash
        pd.DataFrame(columns=["sub_id", "sub_dataset", "jump_date",
                              "magnitude_m", "classification", "action"]).to_csv(
            QC_REPORT, index=False)
    pd.DataFrame(summaries).to_csv(QC_SUMMARY, index=False)
    print(f"\nWrote {QC_REPORT} and {QC_SUMMARY}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
