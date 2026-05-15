"""Per-station fit + plotting logic, factored out of 05_run_subsidence.py.

Why this exists:
- Python module names cannot begin with a digit, so the orchestrator script
  05_run_subsidence.py is unimportable for testing. This sibling module holds
  the testable per-station logic.
- fit_station is referenced via `sub_shell.fit_station(...)` (module attribute,
  late-bound) so `unittest.mock.patch.object(sub_shell, 'fit_station', ...)`
  intercepts the call cleanly.
"""
from __future__ import annotations
import sys
import urllib.parse
from pathlib import Path
import numpy as np
import pandas as pd

# Module-level imports of subsidence sub-modules (NOT `from … import name`),
# so tests can patch attributes on the module objects.
from subsidence import sub_shell
from subsidence.sub_gap_fill import gpr_fill

CAL_START = pd.Timestamp("2020-01-01")
CAL_END   = pd.Timestamp("2022-12-31")
VAL_START = pd.Timestamp("2024-01-01")
VAL_END   = pd.Timestamp("2025-03-31")
DEFAULT_BOUNDS = {
    "Sk_e": (1e-6, 1e-2), "Sk_v": (1e-6, 1e-1),
    "h_ref": (-50.0, 50.0),    # widened; per-station tightened from h range
    # v_tect bound is dataset-specific (see V_TECT_BOUNDS_BY_DATASET).
    # Default is the wider GNSS bound; MLCW overrides to a tighter bound.
    "v_tect": (-0.015, 0.015),
    "v0": (-0.005, 0.005), "v1": (-0.002, 0.002),
    "tau": (7.0, 1500.0),
}
# v_tect bound varies by dataset.  GNSS (daily, dense): ±0.015 m/yr — a lumped
# regional-drift parameter that captures non-poroelastic long-term sinking
# (mantle/isostatic creep, sub-grid storage, frame drift); cumulative-loss
# under-prediction in v4 forced this widening.  MLCW (monthly, sparse): ±0.005
# m/yr — tight — because the looser GNSS bound lets v_tect absorb the cal
# trend on sparse data, leaving Sk_v unconstrained and producing val blowouts
# (僑義國小 KGE -1.4 → -22 in v8).  Reported transparently as dataset-specific
# regularisation; the choice is empirical, not arbitrary.
V_TECT_BOUNDS_BY_DATASET = {
    "ls-wra-gnss-obs": (-0.015, 0.015),
    "ls-wra-mlcw-obs": (-0.005, 0.005),
}
MIN_FORM3_OBS = 36
# DBM (Deep Borehole Marker) measures bedrock motion, not aquifer compaction.
# Riley/IBS poroelastic physics is a model misspecification for these stations,
# evidenced by val α ≈ 0.001, β ≈ 0.001 (model collapses to flat-zero) for all
# 6 DBM stations in the initial run.  Excluded from calibration.
EXCLUDED_DATASETS = ("ls-wra-dbm-obs",)
# Station-level data-quality exclusions.
# LNJS: station reset / antenna swap event 2024-08-31 → 2024-11-04 introduces
# discrete metres-scale jumps (top day-on-day Δ = 46.8 m) and a permanent
# +5.6 m datum shift within the val window.  Cal period (2020-2022) is clean
# but val is unrecoverable; rate = -438 cm/yr is a metadata artefact, not
# subsidence.
EXCLUDED_STATIONS = ("LNJS",)


MLCW_MIN_CAL_OBS = 12   # ~1 year of monthly samples
MLCW_MIN_VAL_OBS = 3


def _build_zeta(raw, sub_dataset: str, sub_id: str, run_id: str):
    """Build the cumulative ζ observable from a raw observation series.
    For MLCW also writes per-layer compaction CSV (spec §9 requirement).

    MLCW deepest-ring selection: prefer the deepest ring that has adequate
    cal+val coverage.  Some stations have one or two deepest rings retired
    mid-record (e.g. 僑義國小: NO30/31 stop 2021-12 while NO1-29 continue
    through 2025-03).  Falling through to the next-deepest viable ring
    preserves a near-total-compaction signal at the cost of missing the
    truly deepest interval — the alternative (deepest unconditionally) drops
    the station entirely.
    """
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
        # Deepest-viable ring selection: walk shallow-ward from the bottom
        chosen_col = None
        for c in reversed(cols):
            s = df[c].dropna()
            n_cal = int(((s.index >= CAL_START) & (s.index <= CAL_END)).sum())
            n_val = int(((s.index >= VAL_START) & (s.index <= VAL_END)).sum())
            if n_cal >= MLCW_MIN_CAL_OBS and n_val >= MLCW_MIN_VAL_OBS:
                chosen_col = c
                break
        if chosen_col is None:
            return pd.Series(dtype=float)
        # Write per-layer CSV only after a viable ring is confirmed; otherwise
        # a skipped station leaves a stale layer file behind.
        per_layer_path = Path(f"workspace/results_sub/{run_id}/per_station/{sub_id}_mlcw_layer.csv")
        per_layer_path.parent.mkdir(parents=True, exist_ok=True)
        per_layer.to_csv(per_layer_path)
        s = df[chosen_col].dropna()
        return s.iloc[0] - df[chosen_col]
    # GNSS / DBM single-value series
    s = raw["value"] if (isinstance(raw, pd.DataFrame) and "value" in raw.columns) else raw
    s = s.dropna()
    if s.empty:
        return pd.Series(dtype=float)
    return s.iloc[0] - s


def _per_station_bounds(h: np.ndarray, sub_dataset: str = "") -> dict:
    out = dict(DEFAULT_BOUNDS)
    h_finite = h[np.isfinite(h)]
    if h_finite.size > 0:
        out["h_ref"] = (float(np.min(h_finite)) - 5.0, float(np.max(h_finite)) + 5.0)
    if sub_dataset in V_TECT_BOUNDS_BY_DATASET:
        out["v_tect"] = V_TECT_BOUNDS_BY_DATASET[sub_dataset]
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
    obs_path = Path(f"data/ls_cache/clean/{sub_dataset}__{sid_encoded}.parquet")
    if not obs_path.exists():
        return f"  {sub_id}: no cleaned observations; run 03b_clean_ls.py first"
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
    # Re-anchor ζ_obs at the first cal-window observation so it lines up with
    # the simulator's ζ[0]=0 anchoring at h_driver start (= cal_start).  When
    # the obs cache extends pre-cal (MLCW now covers 2010+ after the
    # 2010-start fetch), raw ζ_obs is anchored at 2010 — feeding that to the
    # cal residual against a sim anchored at 2020 produces a 10+ cm offset
    # the optimizer can only absorb pathologically.  Subtracting the first
    # finite at-or-after CAL_START cancels that offset.
    cal_obs = zeta.loc[(zeta.index >= CAL_START) & (zeta.index <= CAL_END)].dropna()
    if cal_obs.empty:
        return f"  {sub_id}: zero finite cal obs; skip"
    zeta = zeta - float(cal_obs.iloc[0])

    t_years = (idx - idx[0]).days.values / 365.25
    cal_mask = (idx >= CAL_START) & (idx <= CAL_END)
    val_mask = (idx >= VAL_START) & (idx <= VAL_END)
    cal_idx = np.where(cal_mask)[0]
    val_idx = np.where(val_mask)[0]
    if cal_idx.size == 0 or val_idx.size == 0:
        return f"  {sub_id}: cal/val period has no data; skip"
    n_obs_cal = int((~np.isnan(zeta.values[cal_idx])).sum())
    n_obs_val = int((~np.isnan(zeta.values[val_idx])).sum())
    if n_obs_val == 0:
        return f"  {sub_id}: zero finite val obs; skip"
    form3_ok = sub_dataset != "ls-wra-mlcw-obs" or n_obs_cal >= MIN_FORM3_OBS

    bounds = _per_station_bounds(h, sub_dataset)
    try:
        fit = sub_shell.fit_station(h=h, zeta_obs=zeta.values, t_years=t_years,
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
    # gpr_fill runs INSIDE this block so a sklearn/kernel failure cannot
    # nullify a successful fit. Plot-only is the operational boundary —
    # sub_shell.fit_station above already received the raw NaN-bearing
    # array, so the per-station CSV is already on disk by this point.
    try:
        from subsidence.sub_plotting import (
            plot_per_variant,
            plot_comparison,
            plot_full_subplots,
            plot_mlcw_layer_profile,
        )
        zeta_filled, zeta_sigma, imputed_mask, render_mask = gpr_fill(idx, zeta.values)
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
                zeta_filled=zeta_filled,
                zeta_sigma=zeta_sigma,
                imputed_mask=imputed_mask,
                render_mask=render_mask,
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
            zeta_filled=zeta_filled,
            zeta_sigma=zeta_sigma,
            imputed_mask=imputed_mask,
            render_mask=render_mask,
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
            cal_idx=cal_idx,
            val_idx=val_idx,
            zeta_filled=zeta_filled,
            zeta_sigma=zeta_sigma,
            imputed_mask=imputed_mask,
            render_mask=render_mask,
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
