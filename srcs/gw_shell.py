import sys
import os
from pathlib import Path
from typing import List, Tuple, Optional
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error
import jfft
import matplotlib.pyplot as plt
import gw_subroutine as sub


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
GW_DATA_PATH = DATA_DIR / "gw_data2.csv"
RF_DATA_PATH = DATA_DIR / "rf_data.csv"


def argv_phrase(argv: List):
    assert isinstance(argv, list), "argv must be a list"
    args_params = {}
    for content in argv:
        print('==============================', content)
        try:
            sepline = content.split('=')
            flag = sepline[0].lower()
            args_params[flag] = sepline[1]
        except IndexError:
            args_params[flag] = flag
    return args_params

def _compute_amp_amt(hourly_series: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
    fs = 24
    clean_series = hourly_series.astype(float).interpolate(limit_direction='both')
    y = clean_series.to_numpy().reshape(-1, 1)
    rng = clean_series.index.to_pydatetime()

    mystft = jfft.STFT_Obj(y, fs=fs, framesz=30, hop=5, dt_list=rng)

    xfindex_amp = mystft.find_xf_index(1)
    stft_result = mystft.get_yf()
    stft_timeval = mystft.get_timeval()
    df_amp = pd.DataFrame(stft_result[:, xfindex_amp], index=stft_timeval, columns=['amp'])
    df_amp = df_amp.resample('D').mean()

    xfindex_amt = mystft.find_xf_index(1.93)
    stft_result = mystft.get_yf()
    df_amt = pd.DataFrame(stft_result[:, xfindex_amt], index=stft_timeval, columns=['amt'])
    df_amt = df_amt.resample('D').mean()

    return df_amp, df_amt


# Data preparation
def prepare_data(args_params):
    st_id = args_params.get('st_id')
    if st_id is None:
        raise KeyError("Missing 'st_id' argument")

    ups_id = args_params.get('ups_id', 'none')
    rf_id = args_params.get('rf_id')
    if rf_id is None:
        raise KeyError("Missing 'rf_id' argument")

    lag_days = int(args_params.get('lag_days', 0))

    df_gw_hourly = pd.read_csv(
        GW_DATA_PATH,
        parse_dates=['date time']
    )
    df_gw_hourly = df_gw_hourly.rename(columns=str)
    df_gw_hourly = df_gw_hourly.set_index('date time')
    if not isinstance(df_gw_hourly.index, pd.DatetimeIndex):
        raise ValueError("gw_data2.csv must have a datetime index")

    df_gw_daily = df_gw_hourly.resample('D').mean()

    if st_id not in df_gw_daily.columns:
        raise KeyError(f"Station id {st_id} not found in daily groundwater data")

    df_gw = df_gw_daily[[st_id]].rename(columns={st_id: 'gwl'})

    no_upstream = False
    if pd.isna(ups_id) or str(ups_id).lower() == 'none' or str(ups_id) not in df_gw_daily.columns:
        no_upstream = True
        const_value = float(df_gw['gwl'].iloc[0]) if not df_gw.empty else 0.0
        df_ups_gw = pd.DataFrame({'ups_gwl': const_value}, index=df_gw.index)
    else:
        ups_id = str(ups_id)
        df_ups_gw = df_gw_daily[[ups_id]].rename(columns={ups_id: 'ups_gwl'})

    df_rf_daily = pd.read_csv(
        RF_DATA_PATH,
        parse_dates=['date time']
    )
    df_rf_daily = df_rf_daily.rename(columns=str)
    df_rf_daily = df_rf_daily.set_index('date time')
    if rf_id not in df_rf_daily.columns:
        raise KeyError(f"Rainfall id {rf_id} not found in rf_data.csv")
    df_rf = df_rf_daily[[rf_id]].rename(columns={rf_id: 'rf'})
    if lag_days != 0:
        df_rf = df_rf.shift(lag_days)

    if st_id not in df_gw_hourly.columns:
        raise KeyError(f"Station id {st_id} not found in hourly groundwater data")
    hourly_series = df_gw_hourly[st_id]
    df_amp, df_amt = _compute_amp_amt(hourly_series)
    df_amp = df_amp.reindex(df_gw.index)
    df_amp = df_amp.interpolate(method='linear', limit_direction='both')
    df_amp = df_amp.fillna(method='ffill').fillna(method='bfill')

    df_amt = df_amt.reindex(df_gw.index)
    df_amt = df_amt.interpolate(method='linear', limit_direction='both')
    df_amt = df_amt.fillna(method='ffill').fillna(method='bfill')

    df_merge = (
        df_gw
        .join(df_ups_gw, how='inner')
        .join(df_rf, how='inner')
        .join(df_amp, how='inner')
        .join(df_amt, how='inner')
    )

    df_merge = df_merge.dropna()
    if df_merge.empty:
        raise ValueError("No overlapping daily records after merging inputs")

    return df_merge, no_upstream

def fit_preprocess(df_merge: pd.DataFrame, lag_days: int = 0) -> Tuple:
    if lag_days < 0:
        raise ValueError("lag_days must be non-negative")
    if lag_days >= len(df_merge):
        raise ValueError("lag_days is too large for available data length")

    rainfall = df_merge['rf'].values
    amp = df_merge['amp'].values
    amt = df_merge['amt'].values
    h_up = df_merge['ups_gwl'].values
    h_obs = df_merge['gwl'].values

    if lag_days == 0:
        t = np.arange(len(df_merge))
        return t, rainfall, amp, amt, h_up, h_obs

    # Truncate calibration window to start at t = lag_days and align lagged upstream
    h_obs_trunc = h_obs[lag_days:]
    rainfall_trunc = rainfall[lag_days:]
    amp_trunc = amp[lag_days:]
    amt_trunc = amt[lag_days:]
    h_up_lag = h_up[: len(h_obs_trunc)]
    t = np.arange(len(h_obs_trunc))

    return t, rainfall_trunc, amp_trunc, amt_trunc, h_up_lag, h_obs_trunc

###############################################################################
# Parameter bounds estimation
###############################################################################

def estimate_initial_params_inland(
    h_obs: np.ndarray,
    rainfall: np.ndarray,
    amp: np.ndarray,
    h_up: np.ndarray,
    no_upstream: bool = False,
) -> Tuple[float, float, float, float, float]:
    """Estimate (a, z, b, c, k_link) using a simple linear regression.

    We regress daily head change dh on rainfall, AMP, upstream head (if available),
    and current head h. This provides a reasonable starting guess for the
    physical-equilibrium model.
    """
    if len(h_obs) < 3:
        z0 = float(np.mean(h_obs)) if len(h_obs) > 0 else 0.0
        return 0.1, z0, 0.1, 0.1, 0.0

    dh = h_obs[1:] - h_obs[:-1]
    R_short = rainfall[:-1]
    AMP_short = amp[:-1]
    H_short = h_obs[:-1]

    # Normal equations solution with small ridge for stability
    if no_upstream:
        X = np.column_stack([R_short, AMP_short, H_short, np.ones_like(H_short)])
    else:
        Hup_short = h_up[:-1]
        X = np.column_stack([R_short, AMP_short, Hup_short, H_short, np.ones_like(H_short)])

    XtX = X.T @ X
    ridge = 1e-6 * np.eye(XtX.shape[0])
    try:
        coef = np.linalg.solve(XtX + ridge, X.T @ dh)
    except np.linalg.LinAlgError:
        z0 = float(np.mean(h_obs))
        return 0.1, z0, 0.1, 0.1, 0.0

    if no_upstream:
        b_est, amp_coef, h_coef, intercept = coef
        k_est = 0.0
    else:
        b_est, amp_coef, k_est, h_coef, intercept = coef

    # Map linear coefficients to physical parameters
    c_est = -amp_coef
    lambda_est = -h_coef
    a_est = max(0.0, lambda_est - k_est)

    if a_est > 1e-6:
        z_est = float(intercept / a_est)
    else:
        z_est = float(np.mean(h_obs))

    # Clip to reasonable ranges
    h_min = float(np.min(h_obs))
    h_max = float(np.max(h_obs))
    pad = max(1.0, 0.1 * (h_max - h_min))
    min_z = h_min - pad
    max_z = h_max + pad

    a_est = float(np.clip(a_est, 0.0, 3.0))
    b_est = float(np.clip(b_est, 0.0, 3.0))
    c_est = float(np.clip(c_est, 0.0, 10.0))
    k_est = float(np.clip(k_est, 0.0, 5.0))
    z_est = float(np.clip(z_est, min_z, max_z))

    print("\nInitial inland parameter guess from regression:")
    print(
        f"  a ~ {a_est:.3f}, z ~ {z_est:.3f}, b ~ {b_est:.3f}, "
        f"c ~ {c_est:.3f}, k_link ~ {k_est:.3f}"
    )

    return a_est, z_est, b_est, c_est, k_est

def _ensure_bounds_spread(lower: float, upper: float, min_width: float) -> Tuple[float, float]:
    if not np.isfinite(lower) or not np.isfinite(upper):
        return lower, upper
    if upper - lower < min_width:
        center = 0.5 * (lower + upper)
        half = 0.5 * min_width
        return center - half, center + half
    return lower, upper


def estimate_bounds_inland(
    h_obs: np.ndarray,
    rainfall: np.ndarray,
    amp: np.ndarray,
    h_up: np.ndarray,
    no_upstream: bool = False,
) -> Tuple[List[float], List[float]]:
    """Estimate parameter bounds for the inland model.

    Parameters correspond to (a, z, b, c, k_link).
    """
    h_min = float(np.min(h_obs))
    h_max = float(np.max(h_obs))
    pad = max(1.0, 0.1 * (h_max - h_min))
    z_min = h_min - pad
    z_max = h_max + pad

    # a bounds: recession strength
    a_min, a_max = 0.0, 3.0

    # Prepare increments and truncated inputs
    dh = h_obs[1:] - h_obs[:-1]
    R_short = rainfall[:-1]
    AMP_short = amp[:-1]

    # b from rainfall correlation
    valid_r = R_short != 0
    if np.any(valid_r) and np.var(R_short[valid_r]) > 0:
        b_est = np.cov(dh[valid_r], R_short[valid_r])[0, 1] / np.var(R_short[valid_r])
        width_b = max(0.5, abs(b_est))
        b_min = min(0.0, b_est - width_b)
        b_max = b_est + width_b
    else:
        b_est = 0.1
        b_min, b_max = 0.0, 3.0
    b_min, b_max = _ensure_bounds_spread(b_min, b_max, min_width=0.1)

    # c from AMP correlation
    valid_amp = AMP_short != 0
    if np.any(valid_amp) and np.var(AMP_short[valid_amp]) > 0:
        c_est = -np.cov(dh[valid_amp], AMP_short[valid_amp])[0, 1] / np.var(AMP_short[valid_amp])
        width_c = max(0.5, abs(c_est))
        c_min = 0.0
        c_max = max(c_min + 0.1, c_est + width_c)
    else:
        c_est = 0.1
        c_min, c_max = 0.0, 10.0
    c_min, c_max = _ensure_bounds_spread(c_min, c_max, min_width=0.1)

    if no_upstream:
        k_min, k_max = 0.0, 0.05
    elif np.var(h_up) > 0 and np.var(h_obs) > 0:
        corr = np.corrcoef(h_up, h_obs)[0, 1]
        k_est = max(0.0, corr)
        width_k = max(0.2, 2.0 * k_est)
        k_min = max(0.0, k_est - width_k)
        k_max = k_est + width_k
    else:
        k_min, k_max = 0.0, 2.0
    k_min, k_max = _ensure_bounds_spread(k_min, k_max, min_width=0.01)

    z_min, z_max = _ensure_bounds_spread(z_min, z_max, min_width=0.5)
    param_lower_bounds = [a_min, z_min, b_min, c_min, k_min]
    param_upper_bounds = [a_max, z_max, b_max, c_max, k_max]

    print("\nInland parameter bounds (a, z, b, c, k_link):")
    print(f"  a in [{a_min}, {a_max}]")
    print(f"  z in [{z_min}, {z_max}]")
    print(f"  b in [{b_min}, {b_max}] (est={b_est:.3f})")
    print(f"  c in [{c_min}, {c_max}] (est={c_est:.3f})")
    print(f"  k_link in [{k_min}, {k_max}]")

    return param_lower_bounds, param_upper_bounds


def estimate_bounds_coastal(
    h_obs: np.ndarray,
    rainfall: np.ndarray,
    amp: np.ndarray,
    amt: np.ndarray,
    h_up: np.ndarray,
    no_upstream: bool = False,
) -> Tuple[List[float], List[float]]:
    """Estimate parameter bounds for the coastal model.

    Parameters correspond to (a, z, b, c, k_link, k_sgd, gamma, h_sea).
    """
    h_min = float(np.min(h_obs))
    h_max = float(np.max(h_obs))
    h_mean = float(np.mean(h_obs))
    pad = max(1.0, 0.1 * (h_max - h_min))
    z_min = h_min - pad
    z_max = h_max + pad

    # a bounds
    a_min, a_max = 0.0, 3.0

    # Prepare increments and truncated inputs
    dh = h_obs[1:] - h_obs[:-1]
    R_short = rainfall[:-1]
    AMP_short = amp[:-1]
    AMT_short = amt[:-1]

    # b from rainfall correlation
    valid_r = R_short != 0
    if np.any(valid_r) and np.var(R_short[valid_r]) > 0:
        b_est = np.cov(dh[valid_r], R_short[valid_r])[0, 1] / np.var(R_short[valid_r])
        width_b = max(0.5, abs(b_est))
        b_min = min(0.0, b_est - width_b)
        b_max = b_est + width_b
    else:
        b_est = 0.1
        b_min, b_max = 0.0, 3.0
    b_min, b_max = _ensure_bounds_spread(b_min, b_max, min_width=0.1)

    # c from AMP correlation
    valid_amp = AMP_short != 0
    if np.any(valid_amp) and np.var(AMP_short[valid_amp]) > 0:
        c_est = -np.cov(dh[valid_amp], AMP_short[valid_amp])[0, 1] / np.var(AMP_short[valid_amp])
        width_c = max(0.5, abs(c_est))
        c_min = 0.0
        c_max = max(c_min + 0.1, c_est + width_c)
    else:
        c_est = 0.1
        c_min, c_max = 0.0, 10.0
    c_min, c_max = _ensure_bounds_spread(c_min, c_max, min_width=0.1)

    # gamma from AMT correlation
    valid_amt = AMT_short != 0
    if np.any(valid_amt) and np.var(AMT_short[valid_amt]) > 0:
        g_est = np.cov(dh[valid_amt], AMT_short[valid_amt])[0, 1] / np.var(AMT_short[valid_amt])
        width_g = max(0.5, abs(g_est))
        g_min = min(0.0, g_est - width_g)
        g_max = g_est + width_g
    else:
        g_est = 0.1
        g_min, g_max = 0.0, 10.0
    g_min, g_max = _ensure_bounds_spread(g_min, g_max, min_width=0.1)

    # k_link from correlation between h_up and h_obs
    if no_upstream:
        k_min, k_max = 0.0, 0.05
    elif np.var(h_up) > 0 and np.var(h_obs) > 0:
        corr = np.corrcoef(h_up, h_obs)[0, 1]
        k_est = max(0.0, corr)
        width_k = max(0.2, 2.0 * k_est)
        k_min = max(0.0, k_est - width_k)
        k_max = k_est + width_k
    else:
        k_min, k_max = 0.0, 2.0
    k_min, k_max = _ensure_bounds_spread(k_min, k_max, min_width=0.01)

    # k_sgd: submarine groundwater discharge coefficient
    k_sgd_min = 0.0
    k_sgd_max = 0.5
    k_sgd_min, k_sgd_max = _ensure_bounds_spread(k_sgd_min, k_sgd_max, min_width=0.01)

    # h_sea: effective sea level around mean groundwater
    h_sea_min = h_mean - 10.0
    h_sea_max = h_mean + 10.0
    h_sea_min, h_sea_max = _ensure_bounds_spread(h_sea_min, h_sea_max, min_width=0.5)

    z_min, z_max = _ensure_bounds_spread(z_min, z_max, min_width=0.5)
    param_lower_bounds = [a_min, z_min, b_min, c_min, k_min, k_sgd_min, g_min, h_sea_min]
    param_upper_bounds = [a_max, z_max, b_max, c_max, k_max, k_sgd_max, g_max, h_sea_max]

    print("\nCoastal parameter bounds (a, z, b, c, k_link, k_sgd, gamma, h_sea):")
    print(f"  a in [{a_min}, {a_max}]")
    print(f"  z in [{z_min}, {z_max}]")
    print(f"  b in [{b_min}, {b_max}] (est={b_est:.3f})")
    print(f"  c in [{c_min}, {c_max}] (est={c_est:.3f})")
    print(f"  k_link in [{k_min}, {k_max}]")
    print(f"  k_sgd in [{k_sgd_min}, {k_sgd_max}]")
    print(f"  gamma in [{g_min}, {g_max}] (est={g_est:.3f})")
    print(f"  h_sea in [{h_sea_min}, {h_sea_max}] around mean gw={h_mean:.2f}")

    return param_lower_bounds, param_upper_bounds


###############################################################################
# Random multi-start curve fitting
###############################################################################

def random_multi_start(
    model_func,
    xdata: np.ndarray,
    ydata: np.ndarray,
    bounds: Tuple[List[float], List[float]],
    n_starts: int = 10,
    base_p0: np.ndarray = None,
    **model_kwargs,
):
    """Perform multiple random starts of curve_fit within the given bounds.

    Returns the best parameters (lowest RMSE) and covariance.
    """
    lower, upper = bounds
    lower_arr = np.array(lower, dtype=float)
    upper_arr = np.array(upper, dtype=float)

    if base_p0 is not None:
        base_p0 = np.asarray(base_p0, dtype=float)
        # Ensure base_p0 lies within bounds
        base_p0 = np.minimum(np.maximum(base_p0, lower_arr), upper_arr)

    best_rmse = np.inf
    best_popt = None
    best_pcov = None

    for i in range(n_starts):
        # Use regression-based guess for the first start if provided
        if i == 0 and base_p0 is not None:
            guess = base_p0
            print(f"Start {i+1}: using base_p0={guess}, bounds=({lower_arr}, {upper_arr})")
        else:
            # Sample a random initial guess in [lower, upper]
            guess = np.random.uniform(lower_arr, upper_arr)
            print(f"Start {i+1}: guess={guess}, bounds=({lower_arr}, {upper_arr})")

        try:
            popt_i, pcov_i = curve_fit(
                f=lambda tt, *pp: model_func(tt, *pp, **model_kwargs),
                xdata=xdata,
                ydata=ydata,
                p0=guess,
                bounds=(lower_arr, upper_arr),
            )
            # Evaluate the fit
            y_fit_i = model_func(xdata, *popt_i, **model_kwargs)
            rmse_i = float(np.sqrt(mean_squared_error(ydata, y_fit_i)))
            print(f"Start {i+1}: popt={popt_i}, RMSE={rmse_i:.3f}")

            if rmse_i < best_rmse:
                best_rmse = rmse_i
                best_popt = popt_i
                best_pcov = pcov_i
        except (RuntimeError, ValueError) as e:
            print(f"Start {i+1}: FAILED due to {e}")
            continue

    return best_popt, best_pcov, best_rmse


###############################################################################
# Per-station calibration
###############################################################################

def calibrate_station(
    df_merge: pd.DataFrame,
    group_name: str,
    lag_days: int = 0,
    no_upstream: bool = False,
    n_starts: int = 10,
):
    t, rainfall, amp, amt, h_up, h_obs = fit_preprocess(df_merge, lag_days=lag_days)
    h0 = float(h_obs[0])
    is_coastal = group_name.lower() == "coastal"

    # Regression-based initial guess (shared inland structure)
    a0, z0, b0, c0, k0 = estimate_initial_params_inland(
        h_obs,
        rainfall,
        amp,
        h_up,
        no_upstream=no_upstream,
    )

    if is_coastal:
        lower, upper = estimate_bounds_coastal(h_obs, rainfall, amp, amt, h_up, no_upstream=no_upstream)
        param_names = ["a", "z", "b", "c", "k_link", "k_sgd", "gamma", "h_sea"]
        # Extend inland guess with coastal-specific parameters
        h_mean = float(np.mean(h_obs))
        k_sgd0 = 0.1
        gamma0 = 0.1
        h_sea0 = h_mean
        base_p0 = np.array([a0, z0, b0, c0, k0, k_sgd0, gamma0, h_sea0], dtype=float)
    else:
        lower, upper = estimate_bounds_inland(h_obs, rainfall, amp, h_up, no_upstream=no_upstream)
        param_names = ["a", "z", "b", "c", "k_link"]
        base_p0 = np.array([a0, z0, b0, c0, k0], dtype=float)

    best_popt, best_pcov, best_rmse = random_multi_start(
        model_func=sub.gw_model_wrapper,
        xdata=t,
        ydata=h_obs,
        bounds=(lower, upper),
        n_starts=n_starts,
        base_p0=base_p0,
        rainfall=rainfall,
        amp=amp,
        amt=amt if is_coastal else None,
        h_up=h_up,
        h0=h0,
        is_coastal=is_coastal,
    )

    if best_popt is None:
        print("\nNo successful fit found in random_multi_start.")
        return None, None, None, None, param_names

    # Evaluate best fit
    y_fit_best = sub.gw_model_wrapper(
        t,
        *best_popt,
        rainfall=rainfall,
        amp=amp,
        amt=amt if is_coastal else None,
        h_up=h_up,
        h0=h0,
        is_coastal=is_coastal,
    )
    rmse = float(np.sqrt(mean_squared_error(h_obs, y_fit_best)))
    r2 = float(r2_score(h_obs, y_fit_best))

    print("\n========================================================")
    print(f"Group: {group_name} (coastal={is_coastal})")
    for name, val in zip(param_names, best_popt):
        print(f"  {name} = {val:.4f}")
    print(f"  RMSE = {rmse:.4f}")
    print(f"  R^2  = {r2:.4f}")
    print("========================================================\n")

    return best_popt, best_pcov, rmse, r2, param_names


if __name__ == "__main__":
    args = argv_phrase(sys.argv[1:])
    print('Parsed arguments:', args)
    df_merge, no_upstream = prepare_data(args)
    print('Prepared data (head):')
    print(df_merge.head())
    group_name = args.get('group_name', 'inland')
    lag_days = int(args.get('lag_days', 0))

    best_popt, best_pcov, rmse, r2, param_names = calibrate_station(
        df_merge,
        group_name=group_name,
        lag_days=lag_days,
        no_upstream=no_upstream,
        n_starts=10,
    )

    # Plot observed vs predicted groundwater levels for this station
    if best_popt is not None:
        t, rainfall, amp, amt, h_up, h_obs = fit_preprocess(df_merge, lag_days=lag_days)
        is_coastal = group_name.lower() == "coastal"
        y_fit_best = sub.gw_model_wrapper(
            t,
            *best_popt,
            rainfall=rainfall,
            amp=amp,
            amt=amt if is_coastal else None,
            h_up=h_up,
            h0=float(h_obs[0]),
            is_coastal=is_coastal,
        )

        station_label = args.get('st_id', args.get('gw_st', 'unknown'))

        # Create subplots: (1) GW obs vs pred, (2) rainfall, (3) amp
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        plot_index = df_merge.index[lag_days:]

        # Top: groundwater levels
        axes[0].plot(plot_index, h_obs, label="Observed", color="k", linewidth=1.5)
        axes[0].plot(plot_index, y_fit_best, label="Predicted", color="C1", linewidth=1.2)
        axes[0].set_ylabel("GW level")
        axes[0].set_title(
            f"GW {station_label} ({group_name})  "
            f"RMSE={rmse:.3f}, R^2={r2:.3f}"
        )
        axes[0].legend()

        # add upstream gwl
        axes[0].plot(plot_index, h_up, label="Upstream GW", color="C3", linewidth=1.0, linestyle='--')
        axes[0].legend()

        # Middle: rainfall
        axes[1].bar(plot_index, rainfall, width=1.0, color="C0")
        axes[1].set_ylabel("Rainfall")

        # Bottom: amp
        axes[2].plot(plot_index, amp, color="C2", linewidth=1.0)
        axes[2].set_ylabel("AMP")
        axes[2].set_xlabel("Date")

        fig.tight_layout()

        fig_dir = '../workspace/figures'
        os.makedirs(fig_dir, exist_ok=True)
        fig_path = os.path.join(fig_dir, f"gw_fit_{station_label}.png")
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)

    # Optionally, save results to workspace
    if best_popt is not None:
        results = {
            'st_id': args.get('st_id'),
            'gw_st': args.get('gw_st'),
            'ups_id': args.get('ups_id'),
            'rf_id': args.get('rf_id'),
            'group_name': group_name,
            'lag_days': int(args.get('lag_days', 0)),
            'rmse': rmse,
            'r2': r2,
        }
        for name, val in zip(param_names, best_popt):
            results[name] = val
        os.makedirs('../workspace', exist_ok=True)
        out_path = '../workspace/gw_fit_results.csv'
        new_row = pd.DataFrame([results])

        if os.path.exists(out_path):
            df_prev = pd.read_csv(out_path)

            if 'st_id' not in df_prev.columns:
                df_prev['st_id'] = ''

            # Ensure rmse is numeric
            df_prev['rmse'] = pd.to_numeric(df_prev['rmse'], errors='coerce')
            new_row['rmse'] = pd.to_numeric(new_row['rmse'], errors='coerce')

            st_curr = str(results['st_id'])

            if st_curr in df_prev['st_id'].astype(str).values:
                mask = df_prev['st_id'].astype(str) == st_curr
                prev_best = df_prev.loc[mask, :]
                prev_best_rmse = prev_best['rmse'].min()

                # If current is better (smaller RMSE), replace previous rows for this gw_no
                if results['rmse'] < prev_best_rmse:
                    df_prev = df_prev.loc[~mask, :]
                    df_all = pd.concat([df_prev, new_row], ignore_index=True)
                else:
                    df_all = df_prev
            else:
                df_all = pd.concat([df_prev, new_row], ignore_index=True)
        else:
            df_all = new_row

        df_all.to_csv(out_path, index=False)