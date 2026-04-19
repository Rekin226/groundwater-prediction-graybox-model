"""
Production pipeline with time-varying z model: z(t) = z0 + z1 * (t/365.25).

Drop-in replacement for gw_shell.py with three key changes:
  1. z → (z0, z1) — one extra parameter per station
  2. Validation uses predicted h0 (continuous simulation, no state reset)
  3. Multi-seed DE + curve_fit polish for robust optimization
  4. Multi-metric output (R², RMSE, KGE decomposition)

Usage:
    python 03_run_model_tz.py --run-id tz
"""

import sys
import json
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, differential_evolution
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
    _compute_aic,
)


# ============================================================================
# Configuration
# ============================================================================
SPLIT_DATE = "2019-01-01"
DE_POPSIZE = 10
DE_MAXITER = 200
DE_SEEDS = [42, 123, 7, 999]
Z1_BOUNDS = (-2.0, 2.0)  # m/year


# ============================================================================
# Performance metrics
# ============================================================================

def compute_metrics(obs, pred):
    """R², RMSE, KGE and decomposition."""
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)

    ss_res = np.sum((obs - pred) ** 2)
    ss_tot = np.sum((obs - np.mean(obs)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rmse = float(np.sqrt(np.mean((obs - pred) ** 2)))
    r = float(np.corrcoef(obs, pred)[0, 1]) if len(obs) > 1 else np.nan
    alpha = float(np.std(pred) / np.std(obs)) if np.std(obs) > 0 else np.nan
    beta = float(np.mean(pred) / np.mean(obs)) if np.mean(obs) != 0 else np.nan
    kge = 1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    bias = float(np.mean(pred) - np.mean(obs))

    return {
        "r2": float(r2), "rmse": rmse, "kge": float(kge),
        "kge_r": float(r), "kge_alpha": float(alpha),
        "kge_beta": float(beta), "bias": bias,
    }


# ============================================================================
# Best-so-far cache
# ============================================================================

def load_best_cache(cache_path: Path) -> dict:
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)
    return {}


def save_best_cache(cache: dict, cache_path: Path):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)


# ============================================================================
# DE + curve_fit polish (multi-seed)
# ============================================================================

def run_de_optimization(model_func, xdata, ydata, bounds, base_p0=None,
                        prev_best=None, seed=42,
                        popsize=DE_POPSIZE, maxiter=DE_MAXITER, **model_kwargs):
    lower = np.array(bounds[0], dtype=float)
    upper = np.array(bounds[1], dtype=float)
    de_bounds = list(zip(lower, upper))
    n_params = len(lower)

    rng = np.random.default_rng(seed)
    pop = rng.uniform(lower, upper, size=(popsize * n_params, n_params))
    if base_p0 is not None:
        pop[0] = np.clip(np.asarray(base_p0, dtype=float), lower, upper)
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
        objective, bounds=de_bounds, init=pop,
        maxiter=maxiter, tol=1e-5, seed=seed, workers=1, polish=False,
    )

    de_params = result.x
    de_rmse = float(result.fun)

    # Local polish with curve_fit (TRF)
    def cf_wrapper(t, *params):
        return model_func(t, *params, **model_kwargs)

    try:
        popt_p, _ = curve_fit(
            f=cf_wrapper, xdata=xdata, ydata=ydata,
            p0=de_params, bounds=(lower, upper), method="trf", maxfev=5000,
        )
        y_p = model_func(xdata, *popt_p, **model_kwargs)
        if np.all(np.isfinite(y_p)):
            rmse_p = float(np.sqrt(np.mean((ydata - y_p) ** 2)))
            if rmse_p < de_rmse:
                return popt_p, rmse_p
    except Exception:
        pass

    return de_params, de_rmse


# ============================================================================
# Per-station calibration (time-varying z)
# ============================================================================

def _fit_model_tz(
    df_merge: pd.DataFrame,
    group_name: str,
    rain_lag_days: int,
    up_lag_days: int,
    no_upstream: bool,
    model_name: str,
    best_cache: dict,
    st_id: str,
):
    t, rainfall, amp, amt, h_up, h_obs, time_index, doy = fit_preprocess(
        df_merge, rain_lag_days=rain_lag_days, up_lag_days=up_lag_days,
    )

    # --- Train / validation split ---
    split_mask = time_index < pd.Timestamp(SPLIT_DATE)
    n_cal = int(split_mask.sum())
    has_val = (len(time_index) - n_cal) >= 30
    if n_cal < 60:
        n_cal = len(time_index)
        has_val = False

    t_cal = t[:n_cal]
    h_obs_cal = h_obs[:n_cal]
    rain_cal = rainfall[:n_cal]
    amp_cal = amp[:n_cal]
    amt_cal = amt[:n_cal]
    h_up_cal = h_up[:n_cal]
    doy_cal = doy[:n_cal]
    h0 = float(h_obs_cal[0])
    is_coastal = group_name.lower() == "coastal"

    # --- Bounds (insert z1 after z0) ---
    a0, z0_est, b0, c0, k0 = estimate_initial_params_inland(
        h_obs_cal, rain_cal, amp_cal, h_up_cal, no_upstream=no_upstream,
    )
    tau_rain0, tau_up0 = 5.0, 5.0
    d_sin0, d_cos0 = 0.0, 0.0
    seas_lower = [-2.0, -2.0]
    seas_upper = [2.0, 2.0]

    if model_name == "base":
        model_func = sub.gw_model_wrapper_tz
        if is_coastal:
            lb, ub = estimate_bounds_coastal(h_obs_cal, rain_cal, amp_cal, amt_cal, h_up_cal, no_upstream=no_upstream)
            lower = [lb[0], lb[1], Z1_BOUNDS[0]] + lb[2:] + seas_lower
            upper = [ub[0], ub[1], Z1_BOUNDS[1]] + ub[2:] + seas_upper
            param_names = ["a", "z0", "z1", "b", "c", "k_link", "k_sgd", "gamma", "h_sea",
                           "tau_rain", "tau_up", "d_sin", "d_cos"]
            h_mean = float(np.mean(h_obs_cal))
            base_p0 = np.array([a0, z0_est, 0.0, b0, c0, k0, 0.1, 0.1, h_mean,
                                tau_rain0, tau_up0, d_sin0, d_cos0], dtype=float)
        else:
            lb, ub = estimate_bounds_inland(h_obs_cal, rain_cal, amp_cal, h_up_cal, no_upstream=no_upstream)
            lower = [lb[0], lb[1], Z1_BOUNDS[0]] + lb[2:] + seas_lower
            upper = [ub[0], ub[1], Z1_BOUNDS[1]] + ub[2:] + seas_upper
            param_names = ["a", "z0", "z1", "b", "c", "k_link",
                           "tau_rain", "tau_up", "d_sin", "d_cos"]
            base_p0 = np.array([a0, z0_est, 0.0, b0, c0, k0,
                                tau_rain0, tau_up0, d_sin0, d_cos0], dtype=float)

    elif model_name == "filtered":
        model_func = sub.gw_model_wrapper_filtered_tz
        lambda_min, lambda_max = _ensure_bounds_spread(0.01, 0.8, min_width=0.05)
        if is_coastal:
            lb, ub = estimate_bounds_coastal(h_obs_cal, rain_cal, amp_cal, amt_cal, h_up_cal, no_upstream=no_upstream)
            lower = [lb[0], lb[1], Z1_BOUNDS[0]] + lb[2:] + [lambda_min] + seas_lower
            upper = [ub[0], ub[1], Z1_BOUNDS[1]] + ub[2:] + [lambda_max] + seas_upper
            param_names = ["a", "z0", "z1", "b", "c", "k_link", "k_sgd", "gamma", "h_sea",
                           "lambda", "tau_rain", "tau_up", "d_sin", "d_cos"]
            h_mean = float(np.mean(h_obs_cal))
            base_p0 = np.array([a0, z0_est, 0.0, b0, c0, k0, 0.1, 0.1, h_mean,
                                0.2, tau_rain0, tau_up0, d_sin0, d_cos0], dtype=float)
        else:
            lb, ub = estimate_bounds_inland(h_obs_cal, rain_cal, amp_cal, h_up_cal, no_upstream=no_upstream)
            lower = [lb[0], lb[1], Z1_BOUNDS[0]] + lb[2:] + [lambda_min] + seas_lower
            upper = [ub[0], ub[1], Z1_BOUNDS[1]] + ub[2:] + [lambda_max] + seas_upper
            param_names = ["a", "z0", "z1", "b", "c", "k_link",
                           "lambda", "tau_rain", "tau_up", "d_sin", "d_cos"]
            base_p0 = np.array([a0, z0_est, 0.0, b0, c0, k0, 0.2,
                                tau_rain0, tau_up0, d_sin0, d_cos0], dtype=float)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    model_kwargs_cal = dict(
        rainfall=rain_cal, amp=amp_cal,
        amt=amt_cal if is_coastal else None,
        h_up=h_up_cal, h0=h0, is_coastal=is_coastal, doy=doy_cal,
    )

    # --- Load previous best for warm start ---
    cache_key = f"{st_id}_{model_name}"
    prev_best_params = None
    if cache_key in best_cache:
        cached = best_cache[cache_key]
        try:
            prev_best_params = [cached["params"][name] for name in param_names]
        except KeyError:
            prev_best_params = None

    # --- Multi-seed DE + polish ---
    best_popt = None
    best_r2_val = -np.inf
    best_result_dict = None

    for seed in DE_SEEDS:
        popt, rmse_fit = run_de_optimization(
            model_func=model_func, xdata=t_cal, ydata=h_obs_cal,
            bounds=(lower, upper), base_p0=base_p0,
            prev_best=prev_best_params, seed=seed, **model_kwargs_cal,
        )

        # Calibration metrics
        y_fit_cal = model_func(t_cal, *popt, **model_kwargs_cal)
        m_cal = compute_metrics(h_obs_cal, y_fit_cal)

        # Validation: continuous simulation (predicted h0, absolute time)
        m_val = {"r2": np.nan, "rmse": np.nan, "kge": np.nan,
                 "kge_r": np.nan, "kge_alpha": np.nan, "kge_beta": np.nan, "bias": np.nan}
        y_fit_val = None

        if has_val:
            try:
                h0_val = float(y_fit_cal[-1])  # predicted h0 (continuous simulation)
                val_kwargs = dict(
                    rainfall=rainfall[n_cal:], amp=amp[n_cal:],
                    amt=amt[n_cal:] if is_coastal else None,
                    h_up=h_up[n_cal:], h0=h0_val,
                    is_coastal=is_coastal, doy=doy[n_cal:],
                )
                y_fit_val = model_func(t[n_cal:], *popt, **val_kwargs)
                m_val = compute_metrics(h_obs[n_cal:], y_fit_val)
            except Exception:
                pass

        r2_val = m_val["r2"] if not np.isnan(m_val["r2"]) else -1e10

        if r2_val > best_r2_val:
            best_r2_val = r2_val
            params_dict = dict(zip(param_names, [float(p) for p in popt]))
            best_result_dict = {
                "model": model_name,
                "params": popt,
                "params_dict": params_dict,
                "param_names": param_names,
                "m_cal": m_cal,
                "m_val": m_val,
                "aic": _compute_aic(n_cal, m_cal["rmse"], len(popt)),
                "y_fit_cal": y_fit_cal,
                "y_fit_val": y_fit_val,
                "t": t,
                "rainfall": rainfall,
                "amp": amp,
                "amt": amt,
                "h_up": h_up,
                "h_obs": h_obs,
                "doy": doy,
                "is_coastal": is_coastal,
                "time_index": time_index,
                "n_cal": n_cal,
                "has_val": has_val,
            }

    if best_result_dict is None:
        return None

    # --- Best-so-far comparison ---
    if cache_key in best_cache:
        cached_r2_val = best_cache[cache_key].get("r2_val", -1e10)
        if best_r2_val > cached_r2_val:
            print(f"  {model_name}: R²_val={best_r2_val:.3f} > cached {cached_r2_val:.3f} -> UPDATED")
        else:
            print(f"  {model_name}: R²_val={best_r2_val:.3f} <= cached {cached_r2_val:.3f} -> KEPT CACHED")
            # Reconstruct from cache
            cached = best_cache[cache_key]
            cached_popt = [cached["params"][name] for name in param_names]

            y_fit_cal = model_func(t_cal, *cached_popt, **model_kwargs_cal)
            m_cal = compute_metrics(h_obs_cal, y_fit_cal)

            y_fit_val = None
            m_val = {"r2": np.nan, "rmse": np.nan, "kge": np.nan,
                     "kge_r": np.nan, "kge_alpha": np.nan, "kge_beta": np.nan, "bias": np.nan}
            if has_val:
                try:
                    h0_val = float(y_fit_cal[-1])
                    val_kwargs = dict(
                        rainfall=rainfall[n_cal:], amp=amp[n_cal:],
                        amt=amt[n_cal:] if is_coastal else None,
                        h_up=h_up[n_cal:], h0=h0_val,
                        is_coastal=is_coastal, doy=doy[n_cal:],
                    )
                    y_fit_val = model_func(t[n_cal:], *cached_popt, **val_kwargs)
                    m_val = compute_metrics(h_obs[n_cal:], y_fit_val)
                except Exception:
                    pass

            best_result_dict = {
                "model": model_name,
                "params": np.array(cached_popt),
                "params_dict": cached["params"],
                "param_names": param_names,
                "m_cal": m_cal,
                "m_val": m_val,
                "aic": _compute_aic(n_cal, m_cal["rmse"], len(cached_popt)),
                "y_fit_cal": y_fit_cal,
                "y_fit_val": y_fit_val,
                "t": t, "rainfall": rainfall, "amp": amp, "amt": amt,
                "h_up": h_up, "h_obs": h_obs, "doy": doy,
                "is_coastal": is_coastal,
                "time_index": time_index, "n_cal": n_cal, "has_val": has_val,
            }
    else:
        print(f"  {model_name}: R²_val={best_r2_val:.3f} (first run, cached)")

    # Update cache
    cache_entry = {
        "params": best_result_dict["params_dict"],
        "r2_val": float(best_result_dict["m_val"]["r2"]),
    }
    best_cache[cache_key] = cache_entry

    # Print summary
    mc, mv = best_result_dict["m_cal"], best_result_dict["m_val"]
    z1_val = best_result_dict["params_dict"].get("z1", 0.0)
    print(f"  {model_name}: R²_cal={mc['r2']:.3f} R²_val={mv['r2']:.3f} "
          f"KGE_val={mv['kge']:.3f} RMSE_val={mv['rmse']:.3f} z1={z1_val:+.4f} m/yr")

    return best_result_dict


# ============================================================================
# Per-station pipeline
# ============================================================================

def run_station_tz(args_params: dict, best_cache: dict, cache_path: Path) -> dict:
    """Run time-varying z model for one station. Returns results dict."""
    args = {k.lower(): v for k, v in args_params.items()}
    output_root = Path(args.get("output_root", Path(__file__).resolve().parents[1] / "workspace" / "results" / "tz"))

    df_merge, no_upstream = prepare_data(args)
    group_name = args.get('group_name', 'inland')
    lag_hint = int(args.get('lag_days', 0))
    max_lag = max(45, lag_hint)

    rain_lag_arg = args.get('rain_lag_days')
    up_lag_arg = args.get('ups_lag_days')

    if rain_lag_arg is not None:
        rain_lag_days = int(rain_lag_arg)
    else:
        rain_lag_days = estimate_rain_lag(df_merge['gwl'].values, df_merge['rf'].values, max_lag=max_lag)

    if no_upstream:
        up_lag_days = 0
    elif up_lag_arg is not None:
        up_lag_days = int(up_lag_arg)
    else:
        up_lag_days = estimate_upstream_lag(df_merge['gwl'].values, df_merge['ups_gwl'].values, max_lag=max_lag)

    st_id = args.get('st_id', args.get('gw_st', 'unknown'))

    model_results = []
    for model_name in ["base", "filtered"]:
        result = _fit_model_tz(
            df_merge, group_name=group_name,
            rain_lag_days=rain_lag_days, up_lag_days=up_lag_days,
            no_upstream=no_upstream, model_name=model_name,
            best_cache=best_cache, st_id=st_id,
        )
        if result is not None:
            model_results.append(result)

    if not model_results:
        print(f"  No successful fit for {st_id}")
        return None

    # Save cache after each station
    save_best_cache(best_cache, cache_path)

    # Select best model by AIC
    best_model = min(model_results, key=lambda item: item["aic"])

    mc = best_model["m_cal"]
    mv = best_model["m_val"]
    n_cal = best_model["n_cal"]

    # Build results dict for CSV output
    results = {
        'st_id': st_id,
        'gw_st': args.get('gw_st'),
        'ups_id': args.get('ups_id'),
        'rf_id': args.get('rf_id'),
        'group_name': group_name,
        'rain_lag_days': rain_lag_days,
        'up_lag_days': up_lag_days,
        'model': best_model['model'],
        'r2': mc['r2'], 'rmse': mc['rmse'],
        'kge': mc['kge'], 'kge_r': mc['kge_r'],
        'kge_alpha': mc['kge_alpha'], 'kge_beta': mc['kge_beta'],
        'bias': mc['bias'],
        'r2_val': mv['r2'], 'rmse_val': mv['rmse'],
        'kge_val': mv['kge'], 'kge_r_val': mv['kge_r'],
        'kge_alpha_val': mv['kge_alpha'], 'kge_beta_val': mv['kge_beta'],
        'bias_val': mv['bias'],
        'aic': best_model['aic'],
    }
    for name, val in best_model['params_dict'].items():
        results[name] = float(val)

    # Save per-station CSV
    per_station_dir = output_root / "per_station"
    per_station_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([results]).to_csv(per_station_dir / f"{st_id}.csv", index=False)

    return results
