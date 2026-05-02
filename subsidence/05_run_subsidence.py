"""Orchestrator — fit subsidence ODE for active stations in parallel.

Reads:
    subsidence/data/sub_pairing.csv
    subsidence/data/sub_station_master.csv (active=1 rows)
    subsidence/data/h_drivers/<sub_id>.parquet
    data/ls_cache/<dataset>__<url_encoded_sub_id>.parquet  (observation series)
Writes:
    workspace/results_sub/<run_id>/per_station/<sub_id>.csv
    workspace/results_sub/<run_id>/per_station/<sub_id>_mlcw_layer.csv  (MLCW only)
    workspace/results_sub/<run_id>/sub_fit_results.csv      (best variant per station)
    workspace/results_sub/<run_id>/all_variants_results.csv (all variants per station)

Run:
    poetry run python subsidence/05_run_subsidence.py --run-id initial
    poetry run python subsidence/05_run_subsidence.py --run-id initial --station TKJS
"""
from __future__ import annotations
import argparse
import multiprocessing as mp
import sys
import urllib.parse
from pathlib import Path
import numpy as np
import pandas as pd

# Ensure project root on sys.path so `from subsidence...` works under multiprocessing
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from subsidence.sub_shell import fit_station

CAL_START = pd.Timestamp("2020-01-01")
CAL_END   = pd.Timestamp("2022-12-31")
VAL_START = pd.Timestamp("2024-01-01")
VAL_END   = pd.Timestamp("2025-03-31")
DEFAULT_BOUNDS = {
    "Sk_e": (1e-6, 1e-2), "Sk_v": (1e-6, 1e-1),
    "h_ref": (-50.0, 50.0),    # widened; per-station tightened from h range
    "v_tect": (-0.005, 0.005),
    "v0": (-0.005, 0.005), "v1": (-0.002, 0.002),
    "tau": (7.0, 1500.0),
}
MIN_FORM3_OBS = 36
# DBM (Deep Borehole Marker) measures bedrock motion, not aquifer compaction.
# Riley/IBS poroelastic physics is a model misspecification for these stations,
# evidenced by val α ≈ 0.001, β ≈ 0.001 (model collapses to flat-zero) for all
# 6 DBM stations in the initial run.  Excluded from calibration.
EXCLUDED_DATASETS = ("ls-wra-dbm-obs",)


def _build_zeta(raw, sub_dataset: str, sub_id: str, run_id: str):
    """Build the cumulative ζ observable from a raw observation series.
    For MLCW also writes per-layer compaction CSV (spec §9 requirement)."""
    if sub_dataset == "ls-wra-mlcw-obs":
        df = raw if isinstance(raw, pd.DataFrame) else raw.to_frame()
        cols = [c for c in df.columns if c.startswith("NO")]
        cols.sort(key=lambda c: int(c[2:]))
        if not cols:
            return pd.Series(dtype=float)
        # Per-layer cumulative compaction relative to t_0 (each ring's value at first valid date)
        per_layer = pd.DataFrame(index=df.index)
        for c in cols:
            s = df[c].dropna()
            if not s.empty:
                per_layer[c] = s.iloc[0] - df[c]
        per_layer_path = Path(f"workspace/results_sub/{run_id}/per_station/{sub_id}_mlcw_layer.csv")
        per_layer_path.parent.mkdir(parents=True, exist_ok=True)
        per_layer.to_csv(per_layer_path)
        deepest_col = cols[-1]
        s = df[deepest_col].dropna()
        if s.empty:
            return pd.Series(dtype=float)
        return s.iloc[0] - df[deepest_col]
    # GNSS / DBM single-value series
    s = raw["value"] if (isinstance(raw, pd.DataFrame) and "value" in raw.columns) else raw
    s = s.dropna()
    if s.empty:
        return pd.Series(dtype=float)
    return s.iloc[0] - s


def _per_station_bounds(h: np.ndarray) -> dict:
    out = dict(DEFAULT_BOUNDS)
    h_finite = h[np.isfinite(h)]
    if h_finite.size > 0:
        out["h_ref"] = (float(np.min(h_finite)) - 5.0, float(np.max(h_finite)) + 5.0)
    return out


def _process(sub_id: str, sub_dataset: str, run_id: str) -> str:
    out_dir = Path(f"workspace/results_sub/{run_id}/per_station")
    out_dir.mkdir(parents=True, exist_ok=True)
    h_path = Path(f"subsidence/data/h_drivers/{sub_id}.parquet")
    if not h_path.exists():
        return f"  {sub_id}: no h_driver; skip"
    h_df = pd.read_parquet(h_path)
    # URL-encode sub_id for cache file lookup (MLCW station names are Chinese)
    sid_encoded = urllib.parse.quote(str(sub_id), safe="")
    obs_path = Path(f"data/ls_cache/{sub_dataset}__{sid_encoded}.parquet")
    if not obs_path.exists():
        return f"  {sub_id}: no observations; skip"
    raw = pd.read_parquet(obs_path)
    # Daily resample for DBM (hourly raw)
    if sub_dataset == "ls-wra-dbm-obs":
        raw = raw.resample("1D").mean()
    zeta_full = _build_zeta(raw, sub_dataset, sub_id, run_id)
    if zeta_full.empty:
        return f"  {sub_id}: empty zeta; skip"
    idx = h_df.index
    zeta = zeta_full.reindex(idx)
    h = h_df["h_driver"].values

    t_years = (idx - idx[0]).days.values / 365.25
    cal_mask = (idx >= CAL_START) & (idx <= CAL_END)
    val_mask = (idx >= VAL_START) & (idx <= VAL_END)
    cal_idx = np.where(cal_mask)[0]
    val_idx = np.where(val_mask)[0]
    if cal_idx.size == 0 or val_idx.size == 0:
        return f"  {sub_id}: cal/val period has no data; skip"
    n_obs_cal = int((~np.isnan(zeta.values[cal_idx])).sum())
    form3_ok = sub_dataset != "ls-wra-mlcw-obs" or n_obs_cal >= MIN_FORM3_OBS

    bounds = _per_station_bounds(h)
    try:
        fit = fit_station(h=h, zeta_obs=zeta.values, t_years=t_years,
                          cal_idx=cal_idx, val_idx=val_idx,
                          bounds=bounds, form3_eligible=form3_ok)
    except Exception as e:
        return f"  {sub_id}: fit failed — {str(e)[:120]}"

    # Write per-station CSV
    rows = []
    for v, f in fit["all_variants"].items():
        row = {"sub_id": sub_id, "sub_dataset": sub_dataset, "variant": v,
               "is_best": v == fit["best_variant"],
               "tau_underidentified": (v.endswith("_tau") and not form3_ok),
               **f["params"],
               **{k: f[k] for k in f if k.startswith(("kge_", "rmse_", "r2_", "bias_"))}}
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / f"{sub_id}.csv", index=False)

    # ------------------------------------------------------------------
    # Plots (rklib-style, TIFF 300 DPI)
    # ------------------------------------------------------------------
    try:
        from subsidence.sub_plotting import (
            plot_per_variant,
            plot_comparison,
            plot_full_subplots,
            plot_mlcw_layer_profile,
        )
        fig_dir = Path(f"workspace/results_sub/{run_id}/figures")

        # Per-variant figures
        for v, f in fit["all_variants"].items():
            plot_per_variant(
                sub_id=sub_id,
                variant=v,
                t=idx,
                zeta_obs=zeta.values,
                sim=f["sim_full"],
                cal_idx=cal_idx,
                val_idx=val_idx,
                metrics={k: f[k] for k in f
                         if k.startswith(("kge_", "rmse_", "r2_", "bias_"))},
                out_path=fig_dir / v / f"sub_fit_{sub_id}.tiff",
            )

        # Comparison overlay + metrics table
        plot_comparison(
            sub_id=sub_id,
            t=idx,
            zeta_obs=zeta.values,
            fits=fit["all_variants"],
            cal_idx=cal_idx,
            val_idx=val_idx,
            out_path=fig_dir / "comparison" / f"sub_compare_{sub_id}.tiff",
        )

        # Best-variant overview (ζ + h_driver + rainfall)
        best = fit["best_variant"]
        sim_best = fit["all_variants"][best]["sim_full"]
        plot_full_subplots(
            sub_id=sub_id,
            t=idx,
            zeta_obs=zeta.values,
            sim_best=sim_best,
            h_driver=h,
            driver_source=h_df["driver_source"].values
            if "driver_source" in h_df.columns else None,
            rainfall=None,  # rainfall integration deferred to Phase 9
            out_path=fig_dir / "full_subplots" / f"sub_fit_{sub_id}.tiff",
        )

        # MLCW per-layer compaction profile (only for MLCW dataset)
        if sub_dataset == "ls-wra-mlcw-obs":
            layer_csv = out_dir / f"{sub_id}_mlcw_layer.csv"
            if layer_csv.exists():
                plot_mlcw_layer_profile(
                    sub_id=sub_id,
                    layer_csv_path=layer_csv,
                    out_path=fig_dir / "mlcw_layer_profiles" / f"{sub_id}.tiff",
                )
    except Exception as _plot_err:
        import traceback
        print(f"  [plot warning] {sub_id}: {_plot_err}")
        traceback.print_exc()

    best_kge = fit["all_variants"][fit["best_variant"]]["kge_val"]
    return f"  {sub_id}: best={fit['best_variant']}  kge_val={best_kge:.3f}"


def _merge(run_id: str):
    """Concat per-station CSVs into run-level summary tables.

    Skips *_mlcw_layer.csv files (different schema — per-layer compaction,
    datetime-indexed).  Writes:
        workspace/results_sub/{run_id}/sub_fit_results.csv      — best variant per station
        workspace/results_sub/{run_id}/all_variants_results.csv — all variants per station
    """
    base = Path(f"workspace/results_sub/{run_id}/per_station")
    rows = []
    for f in sorted(base.glob("*.csv")):
        if f.stem.endswith("_mlcw_layer"):
            continue
        rows.append(pd.read_csv(f))
    if not rows:
        return
    all_df = pd.concat(rows, ignore_index=True)
    # Ensure is_best is boolean (pandas reads "True"/"False" strings as object)
    all_df["is_best"] = all_df["is_best"].astype(bool)
    all_df.to_csv(f"workspace/results_sub/{run_id}/all_variants_results.csv", index=False)
    best = all_df[all_df["is_best"]].copy()
    best.to_csv(f"workspace/results_sub/{run_id}/sub_fit_results.csv", index=False)
    print(f"\nMerged: {len(best)} stations × {best['variant'].nunique()} variants (best)")
    print(f"  → workspace/results_sub/{run_id}/sub_fit_results.csv")
    print(f"  → workspace/results_sub/{run_id}/all_variants_results.csv")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", default="initial")
    p.add_argument("--station", default=None)
    p.add_argument("--workers", type=int, default=mp.cpu_count() - 1)
    args = p.parse_args(argv)

    master = pd.read_csv("subsidence/data/sub_station_master.csv")
    master = master[master["active"] == 1]
    master = master[~master["sub_dataset"].isin(EXCLUDED_DATASETS)]
    if args.station:
        master = master[master["sub_id"] == args.station]
    if master.empty:
        print("No matching active stations."); return

    tasks = [(r["sub_id"], r["sub_dataset"], args.run_id) for _, r in master.iterrows()]
    if args.workers <= 1 or args.station:
        for t in tasks:
            print(_process(*t))
    else:
        with mp.Pool(args.workers) as pool:
            for msg in pool.starmap(_process, tasks):
                print(msg)

    _merge(args.run_id)


if __name__ == "__main__":
    sys.exit(main())
