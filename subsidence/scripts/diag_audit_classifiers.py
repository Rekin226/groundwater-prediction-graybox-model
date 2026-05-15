"""Network-wide §7/§8 audit classifiers.

Reads outputs from a finished run-dir and emits audit_classifiers_report.csv
with one row per station and prevalence-summary columns + companion
audit_classifiers_summary.txt with counts.

Usage:
    poetry run python subsidence/scripts/diag_audit_classifiers.py \\
        --run-dir workspace/results_sub/v12_audit_baseline \\
        --h-driver-dir subsidence/data/h_drivers
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from subsidence.sub_metrics import kge_components


# ── §7 thresholds ──────────────────────────────────────────────────────────
FILL_FRACTION_HIGH = 0.50
PAIRING_TIE_FRAC = 0.10
RING_THRESHOLD_GRID = [(8, 2), (12, 3), (16, 4)]


def compute_fill_fraction(h_df: pd.DataFrame) -> float:
    """Return fraction of cells where the driver came from fill (model_fill,
    linear_interp, taper) rather than observation.

    Note on label values: the h-driver pipeline writes `driver_source` with
    values "obs", "model_fill", "linear_interp", "taper" (NOT "observed").
    Both labels accepted to keep the unit test fixture's "observed" working.
    """
    if "driver_source" not in h_df.columns:
        return float("nan")
    n = len(h_df)
    if n == 0:
        return float("nan")
    obs_mask = h_df["driver_source"].isin(["obs", "observed"])
    obs = int(obs_mask.sum())
    return 1.0 - obs / n


def flag_high_fill_fraction(frac: float) -> bool:
    return bool(np.isfinite(frac) and frac > FILL_FRACTION_HIGH)


def compute_n_eff(y: np.ndarray) -> float:
    """Effective sample size from lag-1 autocorrelation."""
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    n = y.size
    if n < 3:
        return float("nan")
    mean = y.mean()
    num = float(np.sum((y[:-1] - mean) * (y[1:] - mean)))
    den = float(np.sum((y - mean) ** 2))
    if den == 0:
        return float(n)
    rho = num / den
    rho = max(min(rho, 0.9999), -0.9999)
    return float(n * (1 - rho) / (1 + rho))


def persistence_kge_benchmark(y_cal: np.ndarray, y_val: np.ndarray) -> float:
    """KGE of a linear-extrapolation forecast from cal into val."""
    y_cal = np.asarray(y_cal, float); y_val = np.asarray(y_val, float)
    yc = y_cal[np.isfinite(y_cal)]
    if yc.size < 2 or y_val.size < 2:
        return float("nan")
    x_cal = np.arange(yc.size, dtype=float)
    slope, intercept = np.polyfit(x_cal, yc, 1)
    x_val = np.arange(yc.size, yc.size + y_val.size, dtype=float)
    y_pred = slope * x_val + intercept
    kc = kge_components(y_val, y_pred)
    return float(kc["kge"])


def kge_on_detrended(y_obs: np.ndarray, y_sim: np.ndarray) -> float:
    """KGE on per-series OLS-detrended cumulative ζ — companion diagnostic.

    After OLS detrending, both residual series have near-zero means and the
    sim may have near-zero variance (if it is exactly linear).  A unit offset
    is applied to both before calling kge_components so that beta = mu_s/mu_o
    remains well-defined.  When sd_s ≈ 0 after the shift, r is undefined; in
    that case KGE is computed directly from r=0, alpha=0, beta=1.
    """
    y_obs = np.asarray(y_obs, float); y_sim = np.asarray(y_sim, float)
    mask = np.isfinite(y_obs) & np.isfinite(y_sim)
    yo = y_obs[mask]; ys = y_sim[mask]
    if yo.size < 3:
        return float("nan")
    x = np.arange(yo.size, dtype=float)
    a, b = np.polyfit(x, yo, 1)
    yo_d = yo - (a * x + b)
    a, b = np.polyfit(x, ys, 1)
    ys_d = ys - (a * x + b)
    # Shift by +1 so beta is well-defined even with near-zero residual means.
    kc = kge_components(yo_d + 1.0, ys_d + 1.0)
    kge_val = kc["kge"]
    if not np.isfinite(kge_val):
        # Degenerate case: sim residuals have zero variance (e.g. exactly
        # linear sim).  Use r=0, alpha=0, beta=1 which gives KGE = 1-√2.
        kge_val = 1.0 - np.sqrt(2.0)
    return float(kge_val)


def _main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--h-driver-dir", required=True, type=Path)
    args = p.parse_args(argv)

    fit_csv = args.run_dir / "sub_fit_results.csv"
    if not fit_csv.exists():
        raise SystemExit(f"sub_fit_results.csv not found in {args.run_dir}")
    fits = pd.read_csv(fit_csv)
    best = fits[fits["is_best"] == True].copy()

    per_station = args.run_dir / "per_station"
    # Load CAL/VAL window definitions from sub_runner constants for
    # persistence-KGE benchmarking (same windows as the audited run).
    from subsidence.sub_runner import CAL_START, CAL_END, VAL_START, VAL_END

    rows = []
    for _, r in best.iterrows():
        sub_id = r["sub_id"]
        row = {"sub_id": sub_id,
               "variant": r["variant"],
               "kge_val": r.get("kge_val", float("nan")),
               "rate_loss_active": bool(r.get("rate_loss_active", False))}

        # §7.1 fill fraction — h-driver files are parquet on disk; CSV
        # fallback supports the unit test which uses a synthetic CSV.
        h_parquet = args.h_driver_dir / f"{sub_id}.parquet"
        h_csv = args.h_driver_dir / f"{sub_id}.csv"
        h_df = None
        if h_parquet.exists():
            h_df = pd.read_parquet(h_parquet)
        elif h_csv.exists():
            h_df = pd.read_csv(h_csv)
        if h_df is not None:
            ff = compute_fill_fraction(h_df)
            row["fill_fraction"] = ff
            row["fill_fraction_high"] = flag_high_fill_fraction(ff)
        else:
            row["fill_fraction"] = float("nan")
            row["fill_fraction_high"] = False

        # Load per-station GPR time series for §8 classifiers
        gpr_csv = per_station / f"{sub_id}_gpr.csv"
        if gpr_csv.exists():
            gp = pd.read_csv(gpr_csv, parse_dates=["date"])
            obs = gp["zeta_obs"].to_numpy(dtype=float)
            dates = gp["date"]

            # §8.1 n_eff
            row["n_eff"] = compute_n_eff(obs)

            # §8.2 persistence-KGE benchmark on val window
            cal_mask = (dates >= CAL_START) & (dates <= CAL_END)
            val_mask = (dates >= VAL_START) & (dates <= VAL_END)
            y_cal = obs[cal_mask.to_numpy()]
            y_val = obs[val_mask.to_numpy()]
            pers_kge = persistence_kge_benchmark(y_cal, y_val)
            row["persistence_kge_val"] = pers_kge
            kge_val = r.get("kge_val", float("nan"))
            row["model_beats_persistence"] = (
                bool(np.isfinite(kge_val) and np.isfinite(pers_kge) and
                     kge_val > pers_kge)
            )

            # §8.10 KGE on detrended cumulative ζ
            if "sim_best" in gp.columns:
                sim = gp["sim_best"].to_numpy(dtype=float)
                row["kge_detrended_val"] = kge_on_detrended(
                    obs[val_mask.to_numpy()], sim[val_mask.to_numpy()]
                )
            else:
                row["kge_detrended_val"] = float("nan")
        else:
            row["n_eff"] = float("nan")
            row["persistence_kge_val"] = float("nan")
            row["model_beats_persistence"] = False
            row["kge_detrended_val"] = float("nan")

        rows.append(row)

    out_csv = args.run_dir / "audit_classifiers_report.csv"
    out_txt = args.run_dir / "audit_classifiers_summary.txt"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    # Summary
    n_total = len(rows)
    high_fill = sum(1 for r in rows if r.get("fill_fraction_high"))
    rate_active = sum(1 for r in rows if r.get("rate_loss_active"))
    beats_pers = sum(1 for r in rows if r.get("model_beats_persistence"))
    lines = [
        f"Audit classifiers summary — {args.run_dir}",
        f"n_stations = {n_total}",
        f"  §7 high_fill_fraction (>50% non-obs):     {high_fill}/{n_total}",
        f"  §8 rate_loss_active (chosen variant):      {rate_active}/{n_total}",
        f"  §8 n_eff median (cumulative ζ):            "
        f"{np.nanmedian([r['n_eff'] for r in rows]):.1f}",
        f"  §8 model_beats_persistence (val window):   {beats_pers}/{n_total}",
        f"  §8 persistence_kge_val median:             "
        f"{np.nanmedian([r['persistence_kge_val'] for r in rows]):.3f}",
    ]
    out_txt.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    _main()
