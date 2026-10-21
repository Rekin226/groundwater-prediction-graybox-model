"""
Test time-varying z model on the 17 negative-val R² stations.

Uses the same fixed 2019-01-01 train/val split as production so results
are directly comparable to the original constant-z model.

z(t) = z0 + z1 * (t / 365.25)   where z1 is in m/year.

IMPORTANT: t must carry *absolute* day indices so that the validation
segment continues the equilibrium trend learned during training.

Best-so-far logic:
  - On each run, results are compared to the cached best (best_params.json)
  - The cache is only updated if the new run produces a better val R²
  - Previous best parameters are seeded into the DE population (warm start)
  - This means re-running with different settings can only improve performance

Usage:
    cd single_tankV2
    python test_neg_val/run_time_varying_z.py
"""

import json
import sys
import time
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
from scipy.optimize import differential_evolution, curve_fit
from sklearn.metrics import r2_score, mean_squared_error

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
SPLIT_DATE = "2019-01-01"
DE_POPSIZE = 10
DE_MAXITER = 200
DE_SEEDS = [42, 123, 7, 999]  # 4 seeds — good diversity, ~50 min runtime
Z1_BOUNDS = (-2.0, 2.0)  # m/year — covers observed drifts up to ±2 m/yr

OUT_DIR = ROOT / "test_neg_val" / "results_tz"
PLOT_DIR = ROOT / "test_neg_val" / "plots_tz"
BEST_CACHE = OUT_DIR / "best_params.json"


# ============================================================================
# Performance metrics
# ============================================================================

def compute_metrics(obs, pred):
    """Compute all performance metrics for observed vs predicted.

    Returns dict with: r2, rmse, kge, kge_r, kge_alpha, kge_beta, bias, nse
    """
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)

    # R² (coefficient of determination)
    ss_res = np.sum((obs - pred) ** 2)
    ss_tot = np.sum((obs - np.mean(obs)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # RMSE
    rmse = float(np.sqrt(np.mean((obs - pred) ** 2)))

    # Pearson correlation
    r = float(np.corrcoef(obs, pred)[0, 1]) if len(obs) > 1 else np.nan

    # KGE components
    alpha = float(np.std(pred) / np.std(obs)) if np.std(obs) > 0 else np.nan  # variability ratio
    beta = float(np.mean(pred) / np.mean(obs)) if np.mean(obs) != 0 else np.nan  # bias ratio

    # KGE (Gupta et al. 2009)
    kge = 1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

    # Bias (m)
    bias = float(np.mean(pred) - np.mean(obs))

    # NSE (same as R² for this formulation)
    nse = r2

    return {
        "r2": float(r2),
        "rmse": rmse,
        "kge": float(kge),
        "kge_r": float(r),
        "kge_alpha": float(alpha),
        "kge_beta": float(beta),
        "bias": bias,
        "nse": float(nse),
    }


# ============================================================================
# Best-so-far cache
# ============================================================================

def load_best_cache():
    """Load cached best results. Returns dict keyed by st_id."""
    if BEST_CACHE.exists():
        with open(BEST_CACHE) as f:
            return json.load(f)
    return {}


def save_best_cache(cache):
    """Save best results cache."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(BEST_CACHE, "w") as f:
        json.dump(cache, f, indent=2)


# ============================================================================
# Data preparation (reuses gw_shell functions, adds z1 to bounds)
# ============================================================================

def prepare_station_data(st_id, ups_id, rf_id, group_name, model_name):
    """Prepare data and bounds for time-varying z model."""
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

    # --- Base bounds from existing estimators (constant-z model) ---
    a0, z0_est, b0, c0, k0 = estimate_initial_params_inland(
        h_obs, rainfall, amp, h_up, no_upstream=no_upstream
    )
    tau_rain0, tau_up0 = 5.0, 5.0
    d_sin0, d_cos0 = 0.0, 0.0
    seas_lower = [-2.0, -2.0]
    seas_upper = [2.0, 2.0]

    if model_name == "base":
        model_func = sub.gw_model_wrapper_tz
        if is_coastal:
            lower_base, upper_base = estimate_bounds_coastal(
                h_obs, rainfall, amp, amt, h_up, no_upstream=no_upstream
            )
            lower = [lower_base[0], lower_base[1], Z1_BOUNDS[0]] + lower_base[2:] + seas_lower
            upper = [upper_base[0], upper_base[1], Z1_BOUNDS[1]] + upper_base[2:] + seas_upper
            param_names = ["a", "z0", "z1", "b", "c", "k_link", "k_sgd", "gamma", "h_sea",
                           "tau_rain", "tau_up", "d_sin", "d_cos"]
            h_mean = float(np.mean(h_obs))
            base_p0 = np.array([a0, z0_est, 0.0, b0, c0, k0, 0.1, 0.1, h_mean,
                                tau_rain0, tau_up0, d_sin0, d_cos0], dtype=float)
        else:
            lower_base, upper_base = estimate_bounds_inland(
                h_obs, rainfall, amp, h_up, no_upstream=no_upstream
            )
            lower = [lower_base[0], lower_base[1], Z1_BOUNDS[0]] + lower_base[2:] + seas_lower
            upper = [upper_base[0], upper_base[1], Z1_BOUNDS[1]] + upper_base[2:] + seas_upper
            param_names = ["a", "z0", "z1", "b", "c", "k_link",
                           "tau_rain", "tau_up", "d_sin", "d_cos"]
            base_p0 = np.array([a0, z0_est, 0.0, b0, c0, k0,
                                tau_rain0, tau_up0, d_sin0, d_cos0], dtype=float)

    elif model_name == "filtered":
        model_func = sub.gw_model_wrapper_filtered_tz
        lambda_min, lambda_max = _ensure_bounds_spread(0.01, 0.8, min_width=0.05)
        if is_coastal:
            lower_base, upper_base = estimate_bounds_coastal(
                h_obs, rainfall, amp, amt, h_up, no_upstream=no_upstream
            )
            lower = [lower_base[0], lower_base[1], Z1_BOUNDS[0]] + lower_base[2:] + [lambda_min] + seas_lower
            upper = [upper_base[0], upper_base[1], Z1_BOUNDS[1]] + upper_base[2:] + [lambda_max] + seas_upper
            param_names = ["a", "z0", "z1", "b", "c", "k_link", "k_sgd", "gamma", "h_sea",
                           "lambda", "tau_rain", "tau_up", "d_sin", "d_cos"]
            h_mean = float(np.mean(h_obs))
            base_p0 = np.array([a0, z0_est, 0.0, b0, c0, k0, 0.1, 0.1, h_mean,
                                0.2, tau_rain0, tau_up0, d_sin0, d_cos0], dtype=float)
        else:
            lower_base, upper_base = estimate_bounds_inland(
                h_obs, rainfall, amp, h_up, no_upstream=no_upstream
            )
            lower = [lower_base[0], lower_base[1], Z1_BOUNDS[0]] + lower_base[2:] + [lambda_min] + seas_lower
            upper = [upper_base[0], upper_base[1], Z1_BOUNDS[1]] + upper_base[2:] + [lambda_max] + seas_upper
            param_names = ["a", "z0", "z1", "b", "c", "k_link",
                           "lambda", "tau_rain", "tau_up", "d_sin", "d_cos"]
            base_p0 = np.array([a0, z0_est, 0.0, b0, c0, k0, 0.2,
                                tau_rain0, tau_up0, d_sin0, d_cos0], dtype=float)

    return {
        "t": t, "rainfall": rainfall, "amp": amp, "amt": amt,
        "h_up": h_up, "h_obs": h_obs, "time_index": time_index, "doy": doy,
        "is_coastal": is_coastal, "no_upstream": no_upstream,
        "lower": lower, "upper": upper, "param_names": param_names,
        "base_p0": base_p0, "model_func": model_func,
        "df_merge": df_merge,
    }


# ============================================================================
# DE Optimization (with warm-start from previous best)
# ============================================================================

def run_de_optimization(model_func, xdata, ydata, bounds, base_p0=None,
                        prev_best=None, seed=42,
                        popsize=DE_POPSIZE, maxiter=DE_MAXITER, **model_kwargs):
    """Differential Evolution optimization with warm-start support.

    Parameters
    ----------
    prev_best : array-like or None
        Previously best parameters to seed into the population.
    seed : int
        Random seed for DE and population initialization.

    Returns (best_params, best_rmse).
    """
    lower = np.array(bounds[0], dtype=float)
    upper = np.array(bounds[1], dtype=float)
    de_bounds = list(zip(lower, upper))
    n_params = len(lower)

    rng = np.random.default_rng(seed)
    pop = rng.uniform(lower, upper, size=(popsize * n_params, n_params))

    # Seed slot 0 with initial guess
    if base_p0 is not None:
        pop[0] = np.clip(np.asarray(base_p0, dtype=float), lower, upper)

    # Seed slot 1 with previous best (warm start)
    if prev_best is not None:
        pop[1] = np.clip(np.asarray(prev_best, dtype=float), lower, upper)

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
        seed=seed,
        workers=1,
        polish=False,
    )

    de_params = result.x
    de_rmse = float(result.fun)

    # --- Local polish with curve_fit (Trust Region Reflective) ---
    # DE finds the basin; curve_fit refines to the precise local minimum.
    def curve_fit_wrapper(t, *params):
        return model_func(t, *params, **model_kwargs)

    try:
        popt_polished, _ = curve_fit(
            f=curve_fit_wrapper,
            xdata=xdata,
            ydata=ydata,
            p0=de_params,
            bounds=(lower, upper),
            method="trf",
            maxfev=5000,
        )
        y_polished = model_func(xdata, *popt_polished, **model_kwargs)
        if np.all(np.isfinite(y_polished)):
            polished_rmse = float(np.sqrt(np.mean((ydata - y_polished) ** 2)))
            if polished_rmse < de_rmse:
                return popt_polished, polished_rmse
    except Exception:
        pass  # polish failed, keep DE result

    return de_params, de_rmse


# ============================================================================
# Run one station with fixed 2019 split + best-so-far logic
# ============================================================================

def run_station(station_data, st_id, best_cache):
    """Run time-varying z model with multiple seeds. Returns the best result."""
    d = station_data
    time_index = d["time_index"]

    # Find split point
    split_mask = time_index < pd.Timestamp(SPLIT_DATE)
    n_cal = int(split_mask.sum())
    n_total = len(d["t"])

    if n_cal < 60 or (n_total - n_cal) < 30:
        print(f"  SKIPPED: n_cal={n_cal}, n_val={n_total - n_cal}")
        return None, "skipped"

    # --- Training arrays ---
    t_cal = d["t"][:n_cal]
    h_obs_cal = d["h_obs"][:n_cal]
    rain_cal = d["rainfall"][:n_cal]
    amp_cal = d["amp"][:n_cal]
    amt_cal = d["amt"][:n_cal]
    h_up_cal = d["h_up"][:n_cal]
    doy_cal = d["doy"][:n_cal]
    h0_cal = float(h_obs_cal[0])

    # --- Validation arrays (ABSOLUTE time indices) ---
    t_val = d["t"][n_cal:]
    h_obs_val = d["h_obs"][n_cal:]
    rain_val = d["rainfall"][n_cal:]
    amp_val = d["amp"][n_cal:]
    amt_val = d["amt"][n_cal:]
    h_up_val = d["h_up"][n_cal:]
    doy_val = d["doy"][n_cal:]
    # h0_val is set per-seed below (uses last predicted cal value for continuity)

    is_coastal = d["is_coastal"]

    # --- Load previous best params for warm start ---
    prev_best_params = None
    if st_id in best_cache:
        cached = best_cache[st_id]
        prev_best_params = [cached["params"][name] for name in d["param_names"]]

    model_kwargs_cal = dict(
        rainfall=rain_cal, amp=amp_cal,
        amt=amt_cal if is_coastal else None,
        h_up=h_up_cal, h0=h0_cal, is_coastal=is_coastal, doy=doy_cal,
    )

    # --- Run DE with multiple seeds, keep best val R² ---
    best_seed_result = None
    best_seed_r2_val = -np.inf

    for seed in DE_SEEDS:
        popt, rmse_fit = run_de_optimization(
            model_func=d["model_func"],
            xdata=t_cal,
            ydata=h_obs_cal,
            bounds=(d["lower"], d["upper"]),
            base_p0=d["base_p0"],
            prev_best=prev_best_params,
            seed=seed,
            **model_kwargs_cal,
        )

        # Calibration prediction
        y_pred_cal = d["model_func"](t_cal, *popt, **model_kwargs_cal)
        m_cal = compute_metrics(h_obs_cal, y_pred_cal)

        # Validation: h0 = last predicted cal value (continuous simulation)
        h0_val = float(y_pred_cal[-1])
        model_kwargs_val = dict(
            rainfall=rain_val, amp=amp_val,
            amt=amt_val if is_coastal else None,
            h_up=h_up_val, h0=h0_val, is_coastal=is_coastal, doy=doy_val,
        )
        y_pred_val = d["model_func"](t_val, *popt, **model_kwargs_val)
        m_val = compute_metrics(h_obs_val, y_pred_val)

        print(f"    seed={seed:>4d}: R²_cal={m_cal['r2']:.3f}  R²_val={m_val['r2']:.3f}  "
              f"KGE_val={m_val['kge']:.3f}  r={m_val['kge_r']:.3f}")

        if m_val["r2"] > best_seed_r2_val:
            best_seed_r2_val = m_val["r2"]
            params_dict = dict(zip(d["param_names"], [float(p) for p in popt]))
            z_start = params_dict["z0"]
            z_end = params_dict["z0"] + params_dict["z1"] * (float(t_val[-1]) / 365.25)
            z_at_split = params_dict["z0"] + params_dict["z1"] * (float(t_cal[-1]) / 365.25)
            best_seed_result = {
                "st_id": st_id,
                "r2_cal": m_cal["r2"], "r2_val": m_val["r2"],
                "rmse_cal": m_cal["rmse"], "rmse_val": m_val["rmse"],
                "kge_cal": m_cal["kge"], "kge_val": m_val["kge"],
                "kge_r_cal": m_cal["kge_r"], "kge_r_val": m_val["kge_r"],
                "kge_alpha_cal": m_cal["kge_alpha"], "kge_alpha_val": m_val["kge_alpha"],
                "kge_beta_cal": m_cal["kge_beta"], "kge_beta_val": m_val["kge_beta"],
                "bias_cal": m_cal["bias"], "bias_val": m_val["bias"],
                "params": params_dict,
                "z_start": z_start,
                "z_at_split": z_at_split,
                "z_end": z_end,
            }

    # --- Best-so-far comparison (multi-seed best vs cache) ---
    status = "new"
    use_result = best_seed_result

    if st_id in best_cache:
        cached_r2_val = best_cache[st_id]["r2_val"]
        if best_seed_r2_val > cached_r2_val:
            status = "improved"
            print(f"  BEST seed R²_val={best_seed_r2_val:.3f} > cached {cached_r2_val:.3f} -> UPDATED")
        else:
            status = "kept_cached"
            use_result = best_cache[st_id]
            print(f"  BEST seed R²_val={best_seed_r2_val:.3f} <= cached {cached_r2_val:.3f} -> KEPT CACHED")
    else:
        print(f"  R²_val={best_seed_r2_val:.3f} (first run, cached)")

    # --- Regenerate predictions for plotting ---
    use_params = use_result["params"]
    use_popt = [use_params[name] for name in d["param_names"]]

    y_pred_cal_final = d["model_func"](t_cal, *use_popt, **model_kwargs_cal)
    # Validation h0 = last predicted cal value (continuous simulation)
    h0_val_final = float(y_pred_cal_final[-1])
    model_kwargs_val_final = dict(
        rainfall=rain_val, amp=amp_val,
        amt=amt_val if is_coastal else None,
        h_up=h_up_val, h0=h0_val_final, is_coastal=is_coastal, doy=doy_val,
    )
    y_pred_val_final = d["model_func"](t_val, *use_popt, **model_kwargs_val_final)

    use_result["_time_cal"] = time_index[:n_cal]
    use_result["_time_val"] = time_index[n_cal:]
    use_result["_obs_cal"] = h_obs_cal
    use_result["_obs_val"] = h_obs_val
    use_result["_pred_cal"] = y_pred_cal_final
    use_result["_pred_val"] = y_pred_val_final

    z1_val = use_result["params"]["z1"]
    print(f"  BEST: R²_cal={use_result['r2_cal']:.3f}  R²_val={use_result['r2_val']:.3f}  |  "
          f"z0={use_result['params']['z0']:.2f}  z1={z1_val:+.4f} m/yr")

    return use_result, status


# ============================================================================
# Diagnostic plot
# ============================================================================

def plot_station(result, model_name, group_name, orig_r2_cal, orig_r2_val):
    """Diagnostic plot for one station."""
    st_id = result["st_id"]
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(result["_time_cal"], result["_obs_cal"],
            color="black", linewidth=0.7, label="Observed")
    ax.plot(result["_time_val"], result["_obs_val"],
            color="black", linewidth=0.7)

    ax.plot(result["_time_cal"], result["_pred_cal"],
            color="blue", linewidth=0.7, alpha=0.8, label="Predicted (cal)")

    ax.plot(result["_time_val"], result["_pred_val"],
            color="red", linewidth=0.7, alpha=0.8, label="Predicted (val)")

    ax.axvline(x=result["_time_val"][0], color="gray", linestyle="--", alpha=0.5,
               label=f"Split {SPLIT_DATE}")

    ax.set_ylabel("GWL (m)")
    ax.set_xlabel("Date")
    z1_val = result["params"]["z1"]
    ax.set_title(
        f"Station {st_id} | {model_name} | {group_name} | "
        f"z(t) model: R\u00b2_cal={result['r2_cal']:.3f}, R\u00b2_val={result['r2_val']:.3f}  |  "
        f"Original: R\u00b2_cal={orig_r2_cal:.3f}, R\u00b2_val={orig_r2_val:.3f}  |  "
        f"z1={z1_val:+.4f} m/yr",
        fontsize=11,
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_DIR / f"{st_id}_tz.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================

def main():
    t_start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # Load best-so-far cache
    best_cache = load_best_cache()

    # Load original results
    results_path = ROOT / "workspace" / "results" / "initial" / "gw_fit_results.csv"
    df_results = pd.read_csv(results_path)
    neg_stations = df_results[df_results.r2_val < 0].copy()

    # Load input metadata
    input_path = ROOT / "data" / "gray_box_input.csv"
    df_input = pd.read_csv(input_path)

    print("=" * 80)
    print("TIME-VARYING z MODEL — NEGATIVE VAL R\u00b2 STATIONS (MULTI-SEED)")
    print(f"z(t) = z0 + z1 * (t / 365.25),  z1 bounds = {Z1_BOUNDS} m/yr")
    print(f"Split: {SPLIT_DATE}  |  DE: popsize={DE_POPSIZE}, maxiter={DE_MAXITER}")
    print(f"Seeds: {DE_SEEDS}  ({len(DE_SEEDS)} runs per station)")
    print(f"Stations: {len(neg_stations)}  |  Cached best: {len(best_cache)} stations")
    print("=" * 80)

    all_results = []
    status_counts = {"new": 0, "improved": 0, "kept_cached": 0}

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
        print(f"  Original constant-z: cal_R\u00b2={orig_r2_cal:.3f}, val_R\u00b2={orig_r2_val:.3f}")
        print(f"{'='*70}")

        try:
            station_data = prepare_station_data(
                st_id, ups_id, rf_id, group_name, model_name
            )
        except Exception as e:
            print(f"  DATA PREP FAILED: {e}")
            continue

        result, status = run_station(station_data, st_id, best_cache)
        if result is None:
            continue

        result["model"] = model_name
        result["group"] = group_name
        result["orig_r2_cal"] = orig_r2_cal
        result["orig_r2_val"] = orig_r2_val
        all_results.append(result)
        status_counts[status] += 1

        # Update cache (only stores serializable data, no arrays)
        cache_entry = {k: v for k, v in result.items() if not k.startswith("_")}
        best_cache[st_id] = cache_entry
        save_best_cache(best_cache)

        plot_station(result, model_name, group_name, orig_r2_cal, orig_r2_val)

    # ========================================================================
    # Summary
    # ========================================================================
    if not all_results:
        print("\nNo results collected!")
        return

    total_time = time.time() - t_start
    print(f"\n\nTotal runtime: {total_time/60:.1f} minutes")
    print(f"Cache status: {status_counts['new']} new, "
          f"{status_counts['improved']} improved, "
          f"{status_counts['kept_cached']} kept cached")

    # --- Table 1: R² comparison ---
    print("\n" + "=" * 120)
    print("TABLE 1: R\u00b2 COMPARISON — CONSTANT-z vs TIME-VARYING z(t)  [BEST-SO-FAR]")
    print("=" * 120)
    print(f"{'Station':>8s} | {'model':>8s} | "
          f"{'orig cal':>8s} | {'orig val':>8s} | "
          f"{'tz cal':>8s} | {'tz val':>8s} | "
          f"{'delta val':>9s} | {'z1 (m/yr)':>10s} | {'z shift':>8s}")
    print("-" * 120)

    rows_csv = []
    n_improved = 0
    n_positive = 0

    for r in all_results:
        delta_val = r["r2_val"] - r["orig_r2_val"]
        z_shift = r["z_end"] - r["z_start"]

        if delta_val > 0:
            n_improved += 1
        if r["r2_val"] > 0:
            n_positive += 1

        print(f"{r['st_id']:>8s} | {r['model']:>8s} | "
              f"{r['orig_r2_cal']:>8.3f} | {r['orig_r2_val']:>8.3f} | "
              f"{r['r2_cal']:>8.3f} | {r['r2_val']:>8.3f} | "
              f"{delta_val:>+9.3f} | {r['params']['z1']:>+10.4f} | "
              f"{z_shift:>+8.2f}")

        row_csv = {
            "st_id": r["st_id"], "model": r["model"], "group": r["group"],
            "orig_r2_cal": r["orig_r2_cal"], "orig_r2_val": r["orig_r2_val"],
            "tz_r2_cal": r["r2_cal"], "tz_r2_val": r["r2_val"],
            "delta_r2_val": delta_val,
            "tz_rmse_cal": r.get("rmse_cal", np.nan),
            "tz_rmse_val": r.get("rmse_val", np.nan),
            "tz_kge_cal": r.get("kge_cal", np.nan),
            "tz_kge_val": r.get("kge_val", np.nan),
            "tz_kge_r_cal": r.get("kge_r_cal", np.nan),
            "tz_kge_r_val": r.get("kge_r_val", np.nan),
            "tz_kge_alpha_cal": r.get("kge_alpha_cal", np.nan),
            "tz_kge_alpha_val": r.get("kge_alpha_val", np.nan),
            "tz_kge_beta_cal": r.get("kge_beta_cal", np.nan),
            "tz_kge_beta_val": r.get("kge_beta_val", np.nan),
            "tz_bias_cal": r.get("bias_cal", np.nan),
            "tz_bias_val": r.get("bias_val", np.nan),
            "z0": r["params"]["z0"], "z1": r["params"]["z1"],
            "z_start": r["z_start"], "z_at_split": r["z_at_split"],
            "z_end": r["z_end"],
        }
        for pname, pval in r["params"].items():
            row_csv[f"param_{pname}"] = pval
        rows_csv.append(row_csv)

    print("-" * 120)

    # --- Table 2: Multi-metric validation (KGE decomposition) ---
    print("\n" + "=" * 120)
    print("TABLE 2: MULTI-METRIC VALIDATION PERFORMANCE (time-varying z, best-so-far)")
    print("  KGE = 1 - sqrt((r-1)\u00b2 + (\u03b1-1)\u00b2 + (\u03b2-1)\u00b2)")
    print("  r = correlation, \u03b1 = std(pred)/std(obs), \u03b2 = mean(pred)/mean(obs)")
    print("=" * 120)
    print(f"{'Station':>8s} | {'R\u00b2 val':>7s} | {'RMSE':>7s} | {'KGE':>7s} | "
          f"{'r':>7s} | {'\u03b1':>7s} | {'\u03b2':>7s} | {'bias(m)':>7s} | {'diagnosis':>30s}")
    print("-" * 120)

    for r in all_results:
        kge_val = r.get("kge_val", np.nan)
        r_val = r.get("kge_r_val", np.nan)
        alpha_val = r.get("kge_alpha_val", np.nan)
        beta_val = r.get("kge_beta_val", np.nan)
        bias_val = r.get("bias_val", np.nan)

        # Diagnostic interpretation
        diag = []
        if not np.isnan(r_val):
            if r_val >= 0.8:
                diag.append("good timing")
            elif r_val >= 0.5:
                diag.append("moderate timing")
            else:
                diag.append("poor timing")
        if not np.isnan(alpha_val):
            if abs(alpha_val - 1) < 0.2:
                diag.append("good amplitude")
            elif alpha_val > 1.2:
                diag.append("overestimates variability")
            elif alpha_val < 0.8:
                diag.append("underestimates variability")
        if not np.isnan(bias_val):
            if abs(bias_val) < 0.3:
                diag.append("low bias")
            else:
                diag.append(f"{'high' if abs(bias_val) > 1 else 'moderate'} bias")
        diagnosis = "; ".join(diag) if diag else "N/A"

        print(f"{r['st_id']:>8s} | {r['r2_val']:>7.3f} | {r.get('rmse_val', np.nan):>7.3f} | "
              f"{kge_val:>7.3f} | {r_val:>7.3f} | {alpha_val:>7.3f} | {beta_val:>7.3f} | "
              f"{bias_val:>+7.3f} | {diagnosis:>30s}")

    print("-" * 120)

    # --- Aggregate ---
    print(f"\nStations improved vs original (\u0394 val R\u00b2 > 0): {n_improved}/{len(all_results)}")
    print(f"Stations with positive val R\u00b2:                    {n_positive}/{len(all_results)}")
    kge_vals = [r.get("kge_val", np.nan) for r in all_results if not np.isnan(r.get("kge_val", np.nan))]
    print(f"Stations with KGE > 0:                             {sum(1 for k in kge_vals if k > 0)}/{len(kge_vals)}")
    print(f"Mean original val R\u00b2:                              {np.mean([r['orig_r2_val'] for r in all_results]):.3f}")
    print(f"Mean time-varying z val R\u00b2 (best):                 {np.mean([r['r2_val'] for r in all_results]):.3f}")
    print(f"Mean time-varying z val KGE:                        {np.mean(kge_vals):.3f}")
    print(f"Mean val correlation (r):                           {np.mean([r.get('kge_r_val', np.nan) for r in all_results if not np.isnan(r.get('kge_r_val', np.nan))]):.3f}")
    print(f"Mean val RMSE (m):                                  {np.mean([r.get('rmse_val', np.nan) for r in all_results if not np.isnan(r.get('rmse_val', np.nan))]):.3f}")

    # Save CSV
    df_csv = pd.DataFrame(rows_csv)
    df_csv.to_csv(OUT_DIR / "time_varying_z_results.csv", index=False)
    print(f"\nResults saved to {OUT_DIR}/")
    print(f"Plots saved to {PLOT_DIR}/")
    print(f"Best params cached to {BEST_CACHE}")


if __name__ == "__main__":
    main()
