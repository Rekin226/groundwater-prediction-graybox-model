"""
Expanding Window Cross-Validation for Negative Validation R² Stations.

Adapted from single_tank/srcs/fit_test.py (TimeSeriesSplit approach) for the
single_tankV2 model (9-12 params, upstream coupling, coastal extensions).

For each of the 17 stations with negative val R²:
  - Uses TimeSeriesSplit(n_splits=4) with train_split_ratio=0.8
  - Retrains the model on each expanding fold via Differential Evolution
  - Records per-split cal/test metrics + parameter values
  - Generates per-station diagnostic plots (observed vs predicted per split)
  - Saves all results as CSV

Usage:
    cd single_tankV2
    python test_neg_val/run_expanding_cv.py
"""

import sys
import time
import os
from pathlib import Path

# --- Path setup ---
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "srcs"))
sys.path.insert(0, str(ROOT.parent))  # rklib lives in parent code_space dir

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

import gw_subroutine as sub
from gw_shell import (
    prepare_data,
    fit_preprocess,
    estimate_initial_params_inland,
    estimate_bounds_inland,
    estimate_bounds_coastal,
    estimate_rain_lag,
    estimate_upstream_lag,
    _ensure_bounds_spread,
)


# ============================================================================
# Configuration
# ============================================================================
N_SPLITS = 4
TRAIN_SPLIT_RATIO = 0.8   # use 80% of each fold's training indices
DE_POPSIZE = 10
DE_MAXITER = 200

# Output directory
OUT_DIR = ROOT / "test_neg_val" / "results"
PLOT_DIR = ROOT / "test_neg_val" / "plots"


# ============================================================================
# Data preparation (one-time per station, expensive STFT computed here)
# ============================================================================

def prepare_station_data(st_id, ups_id, rf_id, group_name, model_name):
    """Prepare full timeseries for one station. Returns dict with all arrays."""
    args = {
        "st_id": st_id,
        "ups_id": ups_id,
        "rf_id": rf_id,
        "group_name": group_name,
    }
    df_merge, no_upstream = prepare_data(args)

    rain_lag_days = estimate_rain_lag(
        df_merge["gwl"].values, df_merge["rf"].values
    )
    up_lag_days = estimate_upstream_lag(
        df_merge["gwl"].values, df_merge["ups_gwl"].values
    )

    t, rainfall, amp, amt, h_up, h_obs, time_index, doy = fit_preprocess(
        df_merge, rain_lag_days=rain_lag_days, up_lag_days=up_lag_days
    )

    is_coastal = group_name.lower() == "coastal"

    # --- Bounds and initial guess ---
    a0, z0, b0, c0, k0 = estimate_initial_params_inland(
        h_obs, rainfall, amp, h_up, no_upstream=no_upstream
    )
    tau_rain0, tau_up0 = 5.0, 5.0
    d_sin0, d_cos0 = 0.0, 0.0
    seas_lower = [-2.0, -2.0]
    seas_upper = [2.0, 2.0]

    if model_name == "base":
        model_func = sub.gw_model_wrapper
        if is_coastal:
            lower, upper = estimate_bounds_coastal(
                h_obs, rainfall, amp, amt, h_up, no_upstream=no_upstream
            )
            param_names = ["a", "z", "b", "c", "k_link", "k_sgd", "gamma", "h_sea",
                           "tau_rain", "tau_up", "d_sin", "d_cos"]
            h_mean = float(np.mean(h_obs))
            base_p0 = np.array([a0, z0, b0, c0, k0, 0.1, 0.1, h_mean,
                                tau_rain0, tau_up0, d_sin0, d_cos0], dtype=float)
        else:
            lower, upper = estimate_bounds_inland(
                h_obs, rainfall, amp, h_up, no_upstream=no_upstream
            )
            param_names = ["a", "z", "b", "c", "k_link",
                           "tau_rain", "tau_up", "d_sin", "d_cos"]
            base_p0 = np.array([a0, z0, b0, c0, k0,
                                tau_rain0, tau_up0, d_sin0, d_cos0], dtype=float)
        lower = list(lower) + seas_lower
        upper = list(upper) + seas_upper

    elif model_name == "filtered":
        model_func = sub.gw_model_wrapper_filtered
        lambda_min, lambda_max = _ensure_bounds_spread(0.01, 0.8, min_width=0.05)
        if is_coastal:
            lower, upper = estimate_bounds_coastal(
                h_obs, rainfall, amp, amt, h_up, no_upstream=no_upstream
            )
            param_names = ["a", "z", "b", "c", "k_link", "k_sgd", "gamma", "h_sea",
                           "lambda", "tau_rain", "tau_up", "d_sin", "d_cos"]
            h_mean = float(np.mean(h_obs))
            base_p0 = np.array([a0, z0, b0, c0, k0, 0.1, 0.1, h_mean,
                                0.2, tau_rain0, tau_up0, d_sin0, d_cos0], dtype=float)
            lower = list(lower) + [lambda_min] + seas_lower
            upper = list(upper) + [lambda_max] + seas_upper
        else:
            lower, upper = estimate_bounds_inland(
                h_obs, rainfall, amp, h_up, no_upstream=no_upstream
            )
            param_names = ["a", "z", "b", "c", "k_link",
                           "lambda", "tau_rain", "tau_up", "d_sin", "d_cos"]
            base_p0 = np.array([a0, z0, b0, c0, k0, 0.2,
                                tau_rain0, tau_up0, d_sin0, d_cos0], dtype=float)
            lower = list(lower) + [lambda_min] + seas_lower
            upper = list(upper) + [lambda_max] + seas_upper

    return {
        "t": t, "rainfall": rainfall, "amp": amp, "amt": amt,
        "h_up": h_up, "h_obs": h_obs, "time_index": time_index, "doy": doy,
        "is_coastal": is_coastal, "no_upstream": no_upstream,
        "lower": lower, "upper": upper, "param_names": param_names,
        "base_p0": base_p0, "model_func": model_func,
        "df_merge": df_merge,
    }


# ============================================================================
# DE Optimization (standard, matching production pipeline)
# ============================================================================

def run_de_optimization(model_func, xdata, ydata, bounds, base_p0=None,
                        popsize=DE_POPSIZE, maxiter=DE_MAXITER, **model_kwargs):
    """Differential Evolution optimization. Returns (best_params, best_rmse)."""
    lower = np.array(bounds[0], dtype=float)
    upper = np.array(bounds[1], dtype=float)
    de_bounds = list(zip(lower, upper))
    n_params = len(lower)

    rng = np.random.default_rng(42)
    pop = rng.uniform(lower, upper, size=(popsize * n_params, n_params))
    if base_p0 is not None:
        pop[0] = np.clip(np.asarray(base_p0, dtype=float), lower, upper)

    def objective(params):
        try:
            y_pred = model_func(xdata, *params, **model_kwargs)
            if not np.all(np.isfinite(y_pred)):
                return 1e10
            return float(np.sqrt(np.mean((ydata - y_pred) ** 2)))
        except Exception:
            return 1e10

    result = differential_evolution(
        objective,
        bounds=de_bounds,
        init=pop,
        maxiter=maxiter,
        tol=1e-5,
        seed=42,
        workers=1,
        polish=False,
    )

    return result.x, float(result.fun)


# ============================================================================
# Expanding Window CV for one station
# ============================================================================

def run_expanding_cv(station_data, st_id):
    """Run TimeSeriesSplit expanding window CV on one station.

    Returns list of per-split result dicts.
    """
    d = station_data
    n_total = len(d["t"])
    time_array = np.arange(n_total)  # 0..N-1

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    split_results = []

    for fold_i, (train_index, test_index) in enumerate(tscv.split(time_array)):
        # Apply train_split_ratio (use 80% of training indices)
        train_split_length = int(len(train_index) * TRAIN_SPLIT_RATIO)
        train_index = train_index[:train_split_length]

        if len(train_index) < 60 or len(test_index) < 30:
            print(f"    Fold {fold_i+1}: SKIPPED (train={len(train_index)}, test={len(test_index)})")
            continue

        # --- Slice arrays for train ---
        t_train = d["t"][train_index]
        h_obs_train = d["h_obs"][train_index]
        rain_train = d["rainfall"][train_index]
        amp_train = d["amp"][train_index]
        amt_train = d["amt"][train_index]
        h_up_train = d["h_up"][train_index]
        doy_train = d["doy"][train_index]
        h0_train = float(h_obs_train[0])

        # --- Slice arrays for test ---
        t_test = d["t"][test_index]
        h_obs_test = d["h_obs"][test_index]
        rain_test = d["rainfall"][test_index]
        amp_test = d["amp"][test_index]
        amt_test = d["amt"][test_index]
        h_up_test = d["h_up"][test_index]
        doy_test = d["doy"][test_index]
        h0_test = float(h_obs_test[0])

        # Date ranges for logging
        time_idx = d["time_index"]
        train_start = time_idx[train_index[0]]
        train_end = time_idx[train_index[-1]]
        test_start = time_idx[test_index[0]]
        test_end = time_idx[test_index[-1]]

        is_coastal = d["is_coastal"]

        # --- Train model on this fold ---
        model_kwargs_train = dict(
            rainfall=rain_train, amp=amp_train,
            amt=amt_train if is_coastal else None,
            h_up=h_up_train, h0=h0_train, is_coastal=is_coastal, doy=doy_train,
        )

        popt, rmse_fit = run_de_optimization(
            model_func=d["model_func"],
            xdata=t_train,
            ydata=h_obs_train,
            bounds=(d["lower"], d["upper"]),
            base_p0=d["base_p0"],
            **model_kwargs_train,
        )

        # --- Calibration metrics ---
        y_pred_train = d["model_func"](t_train, *popt, **model_kwargs_train)
        rmse_train = float(np.sqrt(mean_squared_error(h_obs_train, y_pred_train)))
        r2_train = float(r2_score(h_obs_train, y_pred_train))

        # --- Test metrics ---
        model_kwargs_test = dict(
            rainfall=rain_test, amp=amp_test,
            amt=amt_test if is_coastal else None,
            h_up=h_up_test, h0=h0_test, is_coastal=is_coastal, doy=doy_test,
        )

        try:
            y_pred_test = d["model_func"](
                t_test - t_test[0], *popt, **model_kwargs_test
            )
            rmse_test = float(np.sqrt(mean_squared_error(h_obs_test, y_pred_test)))
            r2_test = float(r2_score(h_obs_test, y_pred_test))
        except Exception as e:
            print(f"    Fold {fold_i+1}: test prediction FAILED ({e})")
            y_pred_test = None
            rmse_test = np.nan
            r2_test = np.nan

        params_dict = dict(zip(d["param_names"], popt))

        print(f"    Fold {fold_i+1}: train [{train_start.date()} → {train_end.date()}] "
              f"test [{test_start.date()} → {test_end.date()}] | "
              f"R²_train={r2_train:.3f} R²_test={r2_test:.3f}")

        split_results.append({
            "st_id": st_id,
            "fold": fold_i + 1,
            "train_start": str(train_start.date()),
            "train_end": str(train_end.date()),
            "test_start": str(test_start.date()),
            "test_end": str(test_end.date()),
            "n_train": len(train_index),
            "n_test": len(test_index),
            "r2_train": r2_train,
            "r2_test": r2_test,
            "rmse_train": rmse_train,
            "rmse_test": rmse_test,
            "gap": r2_train - r2_test,
            "mean_gwl_train": float(np.mean(h_obs_train)),
            "mean_gwl_test": float(np.mean(h_obs_test)),
            "gwl_shift": float(np.mean(h_obs_test) - np.mean(h_obs_train)),
            "params": params_dict,
            # Store for plotting
            "_time_train": time_idx[train_index],
            "_time_test": time_idx[test_index],
            "_obs_train": h_obs_train,
            "_obs_test": h_obs_test,
            "_pred_train": y_pred_train,
            "_pred_test": y_pred_test,
        })

    return split_results


# ============================================================================
# Per-station diagnostic plot
# ============================================================================

def plot_station_splits(split_results, st_id, model_name, group_name, orig_r2_val):
    """Generate per-split observed vs predicted plot for one station."""
    n_folds = len(split_results)
    if n_folds == 0:
        return

    fig, axes = plt.subplots(n_folds, 1, figsize=(14, 4.5 * n_folds), squeeze=False)

    for i, res in enumerate(split_results):
        ax = axes[i, 0]

        # Plot training
        ax.plot(res["_time_train"], res["_obs_train"],
                color="black", linewidth=0.7, label="Observed")
        ax.plot(res["_time_train"], res["_pred_train"],
                color="blue", linewidth=0.7, alpha=0.8, label="Predicted (train)")

        # Plot test
        if res["_pred_test"] is not None:
            ax.plot(res["_time_test"], res["_obs_test"],
                    color="black", linewidth=0.7)
            ax.plot(res["_time_test"], res["_pred_test"],
                    color="red", linewidth=0.7, alpha=0.8, label="Predicted (test)")

        # Vertical line at train/test boundary
        ax.axvline(x=res["_time_test"][0], color="gray", linestyle="--", alpha=0.5)

        ax.set_ylabel("GWL (m)")
        ax.set_title(
            f"Fold {res['fold']}:  train [{res['train_start']} to {res['train_end']}]  "
            f"test [{res['test_start']} to {res['test_end']}]  |  "
            f"R\u00b2_train={res['r2_train']:.3f}  R\u00b2_test={res['r2_test']:.3f}"
        )
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)

    axes[-1, 0].set_xlabel("Date")
    fig.suptitle(
        f"Station {st_id} | model={model_name} | group={group_name} | "
        f"original val R\u00b2={orig_r2_val:.3f}",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout()

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_DIR / f"{st_id}_expanding_cv.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================

def main():
    t_start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # Load original results to identify the 17 problem stations
    results_path = ROOT / "workspace" / "results" / "initial" / "gw_fit_results.csv"
    df_results = pd.read_csv(results_path)
    neg_stations = df_results[df_results.r2_val < 0].copy()

    # Load input metadata
    input_path = ROOT / "data" / "gray_box_input.csv"
    df_input = pd.read_csv(input_path)

    print("=" * 80)
    print("EXPANDING WINDOW CV (TimeSeriesSplit) — NEGATIVE VAL R² STATIONS")
    print(f"n_splits={N_SPLITS}, train_ratio={TRAIN_SPLIT_RATIO}")
    print(f"DE: popsize={DE_POPSIZE}, maxiter={DE_MAXITER}")
    print(f"Stations: {len(neg_stations)}")
    print("=" * 80)

    all_results = []

    for idx, (_, row) in enumerate(neg_stations.iterrows()):
        st_id = row["st_id"]
        model_name = row["model"]
        gw_st_row = df_input[df_input.st_id == st_id]
        if gw_st_row.empty:
            print(f"Skipping {st_id}: not found in input metadata")
            continue
        ups_id = str(gw_st_row.iloc[0]["ups_id"])
        rf_id = str(gw_st_row.iloc[0]["rf_id"])
        group_name = str(gw_st_row.iloc[0]["group"])

        orig_r2_cal = row["r2"]
        orig_r2_val = row["r2_val"]

        print(f"\n{'='*70}")
        print(f"[{idx+1}/{len(neg_stations)}] Station {st_id} | model={model_name} | group={group_name}")
        print(f"  Original (fixed 2019 split): cal_R²={orig_r2_cal:.3f}, val_R²={orig_r2_val:.3f}")
        print(f"{'='*70}")

        # Prepare data once
        try:
            station_data = prepare_station_data(
                st_id, ups_id, rf_id, group_name, model_name
            )
        except Exception as e:
            print(f"  DATA PREP FAILED: {e}")
            continue

        # Run expanding CV
        split_results = run_expanding_cv(station_data, st_id)

        # Attach metadata
        for res in split_results:
            res["model"] = model_name
            res["group"] = group_name
            res["orig_r2_cal"] = orig_r2_cal
            res["orig_r2_val"] = orig_r2_val

        all_results.extend(split_results)

        # Generate plot
        plot_station_splits(split_results, st_id, model_name, group_name, orig_r2_val)

    # ========================================================================
    # Summary tables
    # ========================================================================
    if not all_results:
        print("\nNo results collected!")
        return

    total_time = time.time() - t_start
    print(f"\n\nTotal runtime: {total_time/60:.1f} minutes")

    # Build flat DataFrame (exclude plot data)
    rows_flat = []
    for r in all_results:
        row = {k: v for k, v in r.items() if not k.startswith("_") and k != "params"}
        # Flatten key params
        for pname in ["a", "z", "tau_rain", "tau_up", "k_link", "c", "lambda"]:
            if pname in r["params"]:
                row[f"param_{pname}"] = r["params"][pname]
        rows_flat.append(row)
    df_all = pd.DataFrame(rows_flat)

    # --- Table 1: Per-fold summary across all stations ---
    print("\n" + "=" * 80)
    print("TABLE 1: MEAN METRICS PER FOLD (across all stations)")
    print("=" * 80)
    print(f"{'Fold':>5s} | {'mean R² train':>13s} | {'mean R² test':>12s} | "
          f"{'mean gap':>8s} | {'neg test':>8s} | {'mean shift':>10s}")
    print("-" * 70)

    for fold in sorted(df_all.fold.unique()):
        subset = df_all[df_all.fold == fold]
        print(f"{fold:>5d} | {subset.r2_train.mean():>13.3f} | {subset.r2_test.mean():>12.3f} | "
              f"{subset.gap.mean():>8.3f} | {(subset.r2_test < 0).sum():>4d}/{len(subset):<3d} | "
              f"{subset.gwl_shift.mean():>+10.3f}")

    # --- Table 2: Per-station summary (average across folds) ---
    print("\n" + "=" * 80)
    print("TABLE 2: PER-STATION SUMMARY (mean across folds)")
    print("=" * 80)
    print(f"{'Station':>8s} | {'model':>8s} | {'orig val R²':>11s} | "
          f"{'mean R² train':>13s} | {'mean R² test':>12s} | {'best fold R² test':>17s}")
    print("-" * 85)

    station_summary = []
    for st_id in df_all.st_id.unique():
        st_data = df_all[df_all.st_id == st_id]
        model = st_data.model.iloc[0]
        orig_val = st_data.orig_r2_val.iloc[0]
        best_test_r2 = st_data.r2_test.max()
        station_summary.append({
            "st_id": st_id, "model": model, "orig_r2_val": orig_val,
            "mean_r2_train": st_data.r2_train.mean(),
            "mean_r2_test": st_data.r2_test.mean(),
            "best_test_r2": best_test_r2,
            "n_positive_folds": (st_data.r2_test > 0).sum(),
            "n_folds": len(st_data),
        })
        print(f"{st_id:>8s} | {model:>8s} | {orig_val:>11.3f} | "
              f"{st_data.r2_train.mean():>13.3f} | {st_data.r2_test.mean():>12.3f} | "
              f"{best_test_r2:>17.3f}")

    df_station_summary = pd.DataFrame(station_summary)

    # --- Table 3: Parameter drift across folds (z is the key suspect) ---
    print("\n" + "=" * 80)
    print("TABLE 3: PARAMETER z DRIFT ACROSS FOLDS")
    print("=" * 80)
    if "param_z" in df_all.columns:
        header = f"{'Station':>8s} |"
        for fold in sorted(df_all.fold.unique()):
            header += f" Fold {fold:>1d} z |"
        header += "  drift"
        print(header)
        print("-" * len(header))

        for st_id in df_all.st_id.unique():
            st_data = df_all[df_all.st_id == st_id].sort_values("fold")
            line = f"{st_id:>8s} |"
            z_vals = []
            for _, r in st_data.iterrows():
                z = r.get("param_z", np.nan)
                z_vals.append(z)
                line += f" {z:>8.2f} |"
            if len(z_vals) >= 2:
                drift = z_vals[-1] - z_vals[0]
                line += f" {drift:>+7.2f}"
            print(line)

    # --- Aggregate ---
    print("\n" + "=" * 80)
    print("AGGREGATE SUMMARY")
    print("=" * 80)
    n_any_positive = (df_station_summary.n_positive_folds > 0).sum()
    n_all_negative = (df_station_summary.n_positive_folds == 0).sum()
    print(f"Stations with at least 1 positive test R² fold: {n_any_positive}/{len(df_station_summary)}")
    print(f"Stations with ALL negative test R² folds:       {n_all_negative}/{len(df_station_summary)}")
    print(f"Mean test R² across all folds/stations:         {df_all.r2_test.mean():.3f}")
    print(f"Mean train R² across all folds/stations:        {df_all.r2_train.mean():.3f}")
    print(f"Mean gap (train - test):                        {df_all.gap.mean():.3f}")

    # Save
    df_all.to_csv(OUT_DIR / "expanding_cv_all_folds.csv", index=False)
    df_station_summary = pd.DataFrame(station_summary)
    df_station_summary.to_csv(OUT_DIR / "expanding_cv_station_summary.csv", index=False)
    print(f"\nResults saved to {OUT_DIR}/")
    print(f"Plots saved to {PLOT_DIR}/")


if __name__ == "__main__":
    main()
