"""Data-quality audit for LS observation caches + GW driver caches.

Three scans that close the gaps from the original brainstorm concerns:
    1. Sentinel scan — flag any |value| > 100 m in LS obs (subsidence is cm-scale)
       and any value < -50 m in GW caches (existing convention).
    2. MLCW per-month rate scan — for each MLCW station, max compaction
       between consecutive monthly observations.  Threshold > 50 cm/month
       flags an artefact-class jump.
    3. Per-station coverage — n_obs, n_expected, % missing, longest gap (days),
       observation span.

Outputs:
    workspace/results_sub/data_audit/sentinel_scan.csv
    workspace/results_sub/data_audit/mlcw_monthly_scan.csv
    workspace/results_sub/data_audit/station_coverage.csv
    workspace/results_sub/data_audit/audit_summary.txt
"""
from __future__ import annotations
import sys
import urllib.parse
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

CACHE = Path("data/ls_cache")
MASTER = Path("subsidence/data/sub_station_master.csv")
GW_INPUT = Path("data/gray_box_input.csv")
OUT = Path("workspace/results_sub/data_audit")
OUT.mkdir(parents=True, exist_ok=True)

EXCLUDED_DATASETS = ("ls-wra-dbm-obs",)
LS_SENTINEL_ABS_THRESHOLD_M = 100.0   # subsidence ζ should be < 1 m total
GW_SENTINEL_THRESHOLD_M = -50.0        # below this is API sentinel
MLCW_MONTHLY_RATE_THRESHOLD_CM = 50.0  # > this in 30 days is artefact


def _scan_ls_sentinels():
    master = pd.read_csv(MASTER)
    master = master[master["active"] == 1]
    master = master[~master["sub_dataset"].isin(EXCLUDED_DATASETS)]
    rows = []
    for _, r in master.iterrows():
        sid_enc = urllib.parse.quote(str(r["sub_id"]), safe="")
        path = CACHE / f"{r['sub_dataset']}__{sid_enc}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        # GNSS / DBM single-value column
        if "value" in df.columns:
            v = df["value"].dropna()
            n_extreme = int((v.abs() > LS_SENTINEL_ABS_THRESHOLD_M).sum())
            n_neg_huge = int((v < -1000).sum())
            max_abs = float(v.abs().max()) if len(v) else 0.0
            rows.append({
                "sub_id": r["sub_id"], "ds": r["sub_dataset"],
                "n_obs": len(v),
                "n_extreme_gt_100m": n_extreme,
                "n_lt_minus_1000m": n_neg_huge,
                "max_abs_m": max_abs,
            })
        else:
            # MLCW: per-ring scan (NO* columns)
            no_cols = [c for c in df.columns if c.startswith("NO")]
            if not no_cols:
                continue
            sub = df[no_cols]
            v = sub.values.flatten()
            v = v[np.isfinite(v)]
            n_extreme = int((np.abs(v) > LS_SENTINEL_ABS_THRESHOLD_M).sum())
            n_neg_huge = int((v < -1000).sum())
            max_abs = float(np.abs(v).max()) if len(v) else 0.0
            rows.append({
                "sub_id": r["sub_id"], "ds": r["sub_dataset"],
                "n_obs": int(np.isfinite(sub.values).sum()),
                "n_extreme_gt_100m": n_extreme,
                "n_lt_minus_1000m": n_neg_huge,
                "max_abs_m": max_abs,
            })
    df = pd.DataFrame(rows).sort_values("max_abs_m", ascending=False)
    df.to_csv(OUT / "sentinel_scan.csv", index=False)
    return df


def _scan_mlcw_monthly():
    master = pd.read_csv(MASTER)
    master = master[(master["active"] == 1) & (master["sub_dataset"] == "ls-wra-mlcw-obs")]
    rows = []
    for _, r in master.iterrows():
        sid_enc = urllib.parse.quote(str(r["sub_id"]), safe="")
        path = CACHE / f"ls-wra-mlcw-obs__{sid_enc}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        no_cols = sorted([c for c in df.columns if c.startswith("NO")],
                         key=lambda c: int(c[2:]))
        # Use deepest viable ring (≥12 cal obs) — same as 05_run_subsidence
        chosen = None
        for c in reversed(no_cols):
            s = df[c].dropna()
            if len(s) >= 12:
                chosen = c
                break
        if chosen is None:
            continue
        s = df[chosen].dropna()
        if len(s) < 2:
            continue
        # ζ in metres (first observation as reference)
        zeta = (s.iloc[0] - s) * 100  # cm
        # Month-to-month rate (cm per actual gap days, scaled to 30 days)
        dt_days = pd.Series(s.index, index=s.index).diff().dt.days.dropna()
        dz_cm = zeta.diff().dropna()
        rate_cm_per_30d = (dz_cm.abs() / dt_days * 30.0).fillna(0)
        n_extreme = int((rate_cm_per_30d > MLCW_MONTHLY_RATE_THRESHOLD_CM).sum())
        rows.append({
            "sub_id": r["sub_id"],
            "ring_used": chosen,
            "n_observations": len(s),
            "max_30d_rate_cm": float(rate_cm_per_30d.max()),
            "median_30d_rate_cm": float(rate_cm_per_30d.median()),
            "n_extreme_gt_50cm_per_30d": n_extreme,
        })
    df = pd.DataFrame(rows).sort_values("max_30d_rate_cm", ascending=False)
    df.to_csv(OUT / "mlcw_monthly_scan.csv", index=False)
    return df


def _scan_coverage():
    master = pd.read_csv(MASTER)
    master = master[master["active"] == 1]
    master = master[~master["sub_dataset"].isin(EXCLUDED_DATASETS)]
    cal_start = pd.Timestamp("2020-01-01")
    val_end = pd.Timestamp("2025-03-31")
    rows = []
    for _, r in master.iterrows():
        sid_enc = urllib.parse.quote(str(r["sub_id"]), safe="")
        path = CACHE / f"{r['sub_dataset']}__{sid_enc}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if "value" in df.columns:
            s = df["value"]
        else:
            no_cols = sorted([c for c in df.columns if c.startswith("NO")],
                             key=lambda c: int(c[2:]))
            chosen = None
            for c in reversed(no_cols):
                if df[c].dropna().shape[0] >= 12:
                    chosen = c
                    break
            if chosen is None:
                continue
            s = df[chosen]
        s = s[(s.index >= cal_start) & (s.index <= val_end)]
        finite = s.dropna()
        if len(finite) == 0:
            continue
        # Build expected daily index (for daily series) or monthly (for MLCW)
        if r["sub_dataset"] == "ls-wra-mlcw-obs":
            # MLCW is monthly cadence
            expected = pd.date_range(finite.index.min(), finite.index.max(), freq="MS")
            n_expected = len(expected)
        else:
            expected = pd.date_range(finite.index.min(), finite.index.max(), freq="1D")
            n_expected = len(expected)
        n_obs = len(finite)
        # Largest gap between consecutive observations (in days)
        gaps_days = pd.Series(finite.index).diff().dt.days.dropna()
        max_gap = int(gaps_days.max()) if len(gaps_days) else 0
        rows.append({
            "sub_id": r["sub_id"],
            "ds": r["sub_dataset"],
            "n_obs": n_obs,
            "n_expected": n_expected,
            "pct_missing": round(100.0 * (1 - n_obs / max(n_expected, 1)), 1),
            "max_gap_days": max_gap,
            "first_obs": finite.index.min().date().isoformat(),
            "last_obs":  finite.index.max().date().isoformat(),
            "span_days": (finite.index.max() - finite.index.min()).days,
        })
    df = pd.DataFrame(rows).sort_values(["ds", "pct_missing"], ascending=[True, False])
    df.to_csv(OUT / "station_coverage.csv", index=False)
    return df


def _scan_gw_sentinels():
    """Audit existing GW caches for actual sentinel values found post-mask."""
    rows = []
    paths = sorted(CACHE.glob("gw__*.parquet"))
    for path in paths:
        if path.name.endswith("__raw.parquet"):
            continue
        df = pd.read_parquet(path)
        if "h_obs_m" not in df.columns:
            continue
        v = df["h_obs_m"]
        finite = v.dropna()
        if len(finite) == 0:
            continue
        # Already-masked: should never see < -50 in cleaned cache
        n_below_threshold = int((finite < GW_SENTINEL_THRESHOLD_M).sum())
        n_at_minus_999998 = int(np.isclose(finite.values, -999998.0).sum())
        rows.append({
            "file": path.name,
            "n_obs": len(finite),
            "min_value_m": float(finite.min()),
            "max_value_m": float(finite.max()),
            "n_below_minus_50m_after_mask": n_below_threshold,
            "n_exact_minus_999998": n_at_minus_999998,
        })
    # Also check the *__raw.parquet caches if present (pre-mask data)
    raw_rows = []
    raw_paths = sorted(CACHE.glob("gw__*__raw.parquet"))
    for path in raw_paths:
        df = pd.read_parquet(path)
        if "value" not in df.columns:
            continue
        v = df["value"]
        finite = v.dropna()
        if len(finite) == 0:
            continue
        n_below_threshold = int((finite < GW_SENTINEL_THRESHOLD_M).sum())
        n_at_minus_999998 = int(np.isclose(finite.values, -999998.0).sum())
        raw_rows.append({
            "file": path.name,
            "n_obs": len(finite),
            "min_value_m": float(finite.min()),
            "max_value_m": float(finite.max()),
            "n_below_minus_50m_in_raw": n_below_threshold,
            "n_exact_minus_999998_raw": n_at_minus_999998,
        })
    cleaned_df = pd.DataFrame(rows)
    raw_df = pd.DataFrame(raw_rows)
    cleaned_df.to_csv(OUT / "gw_sentinel_scan_postmask.csv", index=False)
    if not raw_df.empty:
        raw_df.to_csv(OUT / "gw_sentinel_scan_raw.csv", index=False)
    return cleaned_df, raw_df


def main():
    print("=" * 70)
    print("LS sentinel scan")
    print("=" * 70)
    s = _scan_ls_sentinels()
    print(s.head(10).to_string(index=False))
    flagged = s[s["n_extreme_gt_100m"] > 0]
    print(f"\n{len(flagged)} stations flagged with |value|>100m: "
          f"{list(flagged['sub_id']) if len(flagged) else 'NONE'}")

    print("\n" + "=" * 70)
    print("MLCW per-month rate scan")
    print("=" * 70)
    m = _scan_mlcw_monthly()
    print(m.to_string(index=False))
    flagged = m[m["n_extreme_gt_50cm_per_30d"] > 0]
    print(f"\n{len(flagged)} MLCW stations flagged with monthly rate > 50 cm/30d: "
          f"{list(flagged['sub_id']) if len(flagged) else 'NONE'}")

    print("\n" + "=" * 70)
    print("Station coverage")
    print("=" * 70)
    c = _scan_coverage()
    # Group totals
    print(c.groupby("ds")[["n_obs","pct_missing","max_gap_days"]].agg(
        n_stations=("n_obs", "count"),
        median_n_obs=("n_obs", "median"),
        median_pct_missing=("pct_missing", "median"),
        max_gap_days_max=("max_gap_days", "max"),
    ).to_string())
    print("\nWorst-coverage stations (highest %missing):")
    print(c.sort_values("pct_missing", ascending=False).head(8).to_string(index=False))

    print("\n" + "=" * 70)
    print("GW sentinel audit (existing caches)")
    print("=" * 70)
    gc, gr = _scan_gw_sentinels()
    print(f"\nGW post-mask caches ({len(gc)} files):")
    if not gc.empty:
        print(f"  Any values still ≤ -50 m (post-mask): "
              f"{int(gc['n_below_minus_50m_after_mask'].sum())}")
        print(f"  Any -999998 sentinels in cleaned cache: "
              f"{int(gc['n_exact_minus_999998'].sum())}")
        print(f"  Min observed value across all stations: "
              f"{gc['min_value_m'].min():.2f} m")
        print(f"  Max observed value across all stations: "
              f"{gc['max_value_m'].max():.2f} m")
    if not gr.empty:
        n_raw_sentinels = int(gr["n_exact_minus_999998_raw"].sum())
        print(f"\nGW raw caches (pre-mask, {len(gr)} files):")
        print(f"  Total -999998 sentinels in raw GW data: {n_raw_sentinels}")
        per_station_raw = gr[gr["n_exact_minus_999998_raw"] > 0]
        if not per_station_raw.empty:
            print(f"  Stations with sentinels: {len(per_station_raw)} / {len(gr)}")
            print(per_station_raw[["file","n_obs","n_exact_minus_999998_raw"]].head(10).to_string(index=False))
    else:
        print("  (no __raw.parquet caches found — re-fetch with cache_key='_raw' to inspect)")

    # Also write a summary text file
    summary = OUT / "audit_summary.txt"
    with open(summary, "w") as f:
        f.write("Data-quality audit (2026-05-03)\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"LS sentinel scan: {len(s)} stations checked, "
                f"{(s['n_extreme_gt_100m'] > 0).sum()} flagged\n")
        f.write(f"MLCW monthly scan: {len(m)} stations, "
                f"{(m['n_extreme_gt_50cm_per_30d'] > 0).sum()} flagged\n")
        gnss_med_miss = c[c['ds']=='ls-wra-gnss-obs']['pct_missing'].median() if (c['ds']=='ls-wra-gnss-obs').any() else 0
        mlcw_med_miss = c[c['ds']=='ls-wra-mlcw-obs']['pct_missing'].median() if (c['ds']=='ls-wra-mlcw-obs').any() else 0
        f.write(f"Coverage: GNSS median %missing={gnss_med_miss:.1f}, "
                f"MLCW median %missing={mlcw_med_miss:.1f}\n")
        f.write(f"GW sentinel audit: ")
        if not gr.empty:
            f.write(f"{int(gr['n_exact_minus_999998_raw'].sum())} -999998 sentinels in raw caches\n")
        else:
            f.write("no raw caches (cleanup already complete)\n")
    print(f"\nWrote summary to {summary}")
    print(f"Wrote CSVs to {OUT}/")


if __name__ == "__main__":
    main()
