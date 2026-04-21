import sys
import os
import json
from pathlib import Path
from typing import List, Tuple, Optional
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit, differential_evolution
from sklearn.metrics import r2_score, mean_squared_error
import jfft
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import gw_subroutine as sub

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rklib import TimeSeriesModelFig, setup_font, savefig as rklib_savefig, add_panel_label


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
GW_DATA_PATH = DATA_DIR / "gw_timeseries.csv"
RF_DATA_PATH = DATA_DIR / "rf_timeseries.csv"

# Temporal split: calibration uses data before this date, validation uses data from this date onward.
SPLIT_DATE = "2019-01-01"


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


    df_gw_hourly = pd.read_csv(
        GW_DATA_PATH,
        parse_dates=['date time']
    )
    df_gw_hourly = df_gw_hourly.rename(columns=str)
    df_gw_hourly = df_gw_hourly.set_index('date time')
    if not isinstance(df_gw_hourly.index, pd.DatetimeIndex):
        raise ValueError("gw_timeseries.csv must have a datetime index")

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
        raise KeyError(f"Rainfall id {rf_id} not found in rf_timeseries.csv")
    df_rf = df_rf_daily[[rf_id]].rename(columns={rf_id: 'rf'})

    if st_id not in df_gw_hourly.columns:
        raise KeyError(f"Station id {st_id} not found in hourly groundwater data")
    hourly_series = df_gw_hourly[st_id]
    df_amp, df_amt = _compute_amp_amt(hourly_series)
    df_amp = df_amp.reindex(df_gw.index)
    df_amp = df_amp.interpolate(method='linear', limit_direction='both')
    df_amp = df_amp.ffill().bfill()

    df_amt = df_amt.reindex(df_gw.index)
    df_amt = df_amt.interpolate(method='linear', limit_direction='both')
    df_amt = df_amt.ffill().bfill()

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

def fit_preprocess(df_merge: pd.DataFrame, rain_lag_days: int = 0, up_lag_days: int = 0) -> Tuple:
    if rain_lag_days < 0 or up_lag_days < 0:
        raise ValueError("lag days must be non-negative")
    if max(rain_lag_days, up_lag_days) >= len(df_merge):
        raise ValueError("lag days is too large for available data length")

    df = df_merge.copy()
    if rain_lag_days != 0:
        df['rf'] = df['rf'].shift(rain_lag_days)
    if up_lag_days != 0:
        df['ups_gwl'] = df['ups_gwl'].shift(up_lag_days)

    df = df.dropna()
    if df.empty:
        raise ValueError("No overlapping records after applying lags")

    rainfall = df['rf'].values
    amp = df['amp'].values
    amt = df['amt'].values
    h_up = df['ups_gwl'].values
    h_obs = df['gwl'].values
    t = np.arange(len(df))
    time_index = df.index
    doy = time_index.dayofyear.values.astype(float)

    return t, rainfall, amp, amt, h_up, h_obs, time_index, doy


def estimate_upstream_lag(h_obs: np.ndarray, h_up: np.ndarray, max_lag: int = 45) -> int:
    if len(h_obs) < 3 or len(h_up) < 3:
        return 0
    dh_obs = h_obs[1:] - h_obs[:-1]
    dh_up = h_up[1:] - h_up[:-1]
    best_lag = 0
    best_score = -np.inf
    for lag in range(0, max_lag + 1):
        if lag == 0:
            x = dh_up
            y = dh_obs
        else:
            if lag >= len(dh_obs):
                break
            x = dh_up[:-lag]
            y = dh_obs[lag:]
        if len(x) < 10:
            continue
        if np.var(x) == 0 or np.var(y) == 0:
            continue
        corr = np.corrcoef(x, y)[0, 1]
        if np.isnan(corr):
            continue
        score = abs(corr)
        if score > best_score:
            best_score = score
            best_lag = lag
    return best_lag


def estimate_rain_lag(h_obs: np.ndarray, rainfall: np.ndarray, max_lag: int = 45) -> int:
    if len(h_obs) < 3 or len(rainfall) < 3:
        return 0
    dh_obs = h_obs[1:] - h_obs[:-1]
    best_lag = 0
    best_score = -np.inf
    for lag in range(0, max_lag + 1):
        if lag == 0:
            x = rainfall[:-1]
            y = dh_obs
        else:
            if lag >= len(dh_obs):
                break
            x = rainfall[:-1 - lag]
            y = dh_obs[lag:]
        if len(x) < 10:
            continue
        if np.var(x) == 0 or np.var(y) == 0:
            continue
        corr = np.corrcoef(x, y)[0, 1]
        if np.isnan(corr):
            continue
        score = abs(corr)
        if score > best_score:
            best_score = score
            best_lag = lag
    return best_lag

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

    Parameters correspond to (a, z, b, c, k_link, tau_rain, tau_up).
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

    # b from rainfall correlation (enforce non-negative recharge).
    # Floor scaled to observed b range across 61 stations (max ~0.015) —
    # a wide floor (0.5) left 99% of search space physically implausible.
    valid_r = R_short != 0
    if np.any(valid_r) and np.var(R_short[valid_r]) > 0:
        b_est = np.cov(dh[valid_r], R_short[valid_r])[0, 1] / np.var(R_short[valid_r])
        width_b = max(0.05, 5.0 * abs(b_est))
        b_min = 0.0
        b_max = b_est + width_b
    else:
        b_est = 0.01
        b_min, b_max = 0.0, 0.1
    b_min, b_max = _ensure_bounds_spread(b_min, b_max, min_width=0.01)

    # c from AMP correlation
    valid_amp = AMP_short != 0
    if np.any(valid_amp) and np.var(AMP_short[valid_amp]) > 0:
        c_est = -np.cov(dh[valid_amp], AMP_short[valid_amp])[0, 1] / np.var(AMP_short[valid_amp])
        width_c = max(0.5, abs(c_est))
        c_min = 0.0
        c_max = max(c_min + 0.1, c_est + width_c)
    else:
        c_est = 0.1
        c_min, c_max = 0.0, 3.0
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
    tau_rain_min, tau_rain_max = 0.1, 30.0
    tau_up_min, tau_up_max = 0.1, 30.0
    param_lower_bounds = [a_min, z_min, b_min, c_min, k_min, tau_rain_min, tau_up_min]
    param_upper_bounds = [a_max, z_max, b_max, c_max, k_max, tau_rain_max, tau_up_max]

    print("\nInland parameter bounds (a, z, b, c, k_link, tau_rain, tau_up):")
    print(f"  a in [{a_min}, {a_max}]")
    print(f"  z in [{z_min}, {z_max}]")
    print(f"  b in [{b_min}, {b_max}] (est={b_est:.3f})")
    print(f"  c in [{c_min}, {c_max}] (est={c_est:.3f})")
    print(f"  k_link in [{k_min}, {k_max}]")
    print(f"  tau_rain in [{tau_rain_min}, {tau_rain_max}]")
    print(f"  tau_up in [{tau_up_min}, {tau_up_max}]")

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

    Parameters correspond to (a, z, b, c, k_link, k_sgd, gamma, h_sea, tau_rain, tau_up).
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

    # b from rainfall correlation (enforce non-negative recharge).
    # Floor scaled to observed b range across 61 stations (max ~0.015) —
    # a wide floor (0.5) left 99% of search space physically implausible.
    valid_r = R_short != 0
    if np.any(valid_r) and np.var(R_short[valid_r]) > 0:
        b_est = np.cov(dh[valid_r], R_short[valid_r])[0, 1] / np.var(R_short[valid_r])
        width_b = max(0.05, 5.0 * abs(b_est))
        b_min = 0.0
        b_max = b_est + width_b
    else:
        b_est = 0.01
        b_min, b_max = 0.0, 0.1
    b_min, b_max = _ensure_bounds_spread(b_min, b_max, min_width=0.01)

    # c from AMP correlation
    valid_amp = AMP_short != 0
    if np.any(valid_amp) and np.var(AMP_short[valid_amp]) > 0:
        c_est = -np.cov(dh[valid_amp], AMP_short[valid_amp])[0, 1] / np.var(AMP_short[valid_amp])
        width_c = max(0.5, abs(c_est))
        c_min = 0.0
        c_max = max(c_min + 0.1, c_est + width_c)
    else:
        c_est = 0.1
        c_min, c_max = 0.0, 3.0
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
    tau_rain_min, tau_rain_max = 0.1, 30.0
    tau_up_min, tau_up_max = 0.1, 30.0
    param_lower_bounds = [
        a_min,
        z_min,
        b_min,
        c_min,
        k_min,
        k_sgd_min,
        g_min,
        h_sea_min,
        tau_rain_min,
        tau_up_min,
    ]
    param_upper_bounds = [
        a_max,
        z_max,
        b_max,
        c_max,
        k_max,
        k_sgd_max,
        g_max,
        h_sea_max,
        tau_rain_max,
        tau_up_max,
    ]

    print("\nCoastal parameter bounds (a, z, b, c, k_link, k_sgd, gamma, h_sea, tau_rain, tau_up):")
    print(f"  a in [{a_min}, {a_max}]")
    print(f"  z in [{z_min}, {z_max}]")
    print(f"  b in [{b_min}, {b_max}] (est={b_est:.3f})")
    print(f"  c in [{c_min}, {c_max}] (est={c_est:.3f})")
    print(f"  k_link in [{k_min}, {k_max}]")
    print(f"  k_sgd in [{k_sgd_min}, {k_sgd_max}]")
    print(f"  gamma in [{g_min}, {g_max}] (est={g_est:.3f})")
    print(f"  h_sea in [{h_sea_min}, {h_sea_max}] around mean gw={h_mean:.2f}")
    print(f"  tau_rain in [{tau_rain_min}, {tau_rain_max}]")
    print(f"  tau_up in [{tau_up_min}, {tau_up_max}]")

    return param_lower_bounds, param_upper_bounds


###############################################################################
# Random multi-start curve fitting
###############################################################################

def compute_metrics(obs: np.ndarray, pred: np.ndarray) -> dict:
    """Compute R², RMSE, KGE and its decomposition (r, alpha, beta), bias."""
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


def run_global_optimization(
    model_func,
    xdata: np.ndarray,
    ydata: np.ndarray,
    bounds: Tuple[List[float], List[float]],
    base_p0: np.ndarray = None,
    popsize: int = 15,
    maxiter: int = 500,
    **model_kwargs,
):
    """Global optimisation with differential_evolution, then local polish with curve_fit for pcov.

    Returns (best_popt, best_pcov, best_rmse).
    pcov is None if the polish step fails.
    """
    lower, upper = bounds
    lower_arr = np.array(lower, dtype=float)
    upper_arr = np.array(upper, dtype=float)
    de_bounds = list(zip(lower_arr, upper_arr))
    n_params = len(lower_arr)

    # Seed the population: first row is the regression-based guess (if available),
    # rest is Latin-hypercube sampled across bounds.
    rng = np.random.default_rng(42)
    pop = rng.uniform(lower_arr, upper_arr, size=(popsize * n_params, n_params))
    if base_p0 is not None:
        pop[0] = np.clip(np.asarray(base_p0, dtype=float), lower_arr, upper_arr)

    def objective(params):
        try:
            y_pred = model_func(xdata, *params, **model_kwargs)
            if not np.all(np.isfinite(y_pred)):
                return 1e10
            return float(np.sqrt(mean_squared_error(ydata, y_pred)))
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

    best_popt = result.x
    best_rmse = float(result.fun)
    print(f"DE converged: RMSE={best_rmse:.4f}, nfev={result.nfev}")

    # Polish with curve_fit from the DE solution to get parameter covariance
    best_pcov = None
    try:
        popt_p, pcov_p = curve_fit(
            f=lambda tt, *pp: model_func(tt, *pp, **model_kwargs),
            xdata=xdata,
            ydata=ydata,
            p0=best_popt,
            bounds=(lower_arr, upper_arr),
            maxfev=5000,
        )
        y_fit_p = model_func(xdata, *popt_p, **model_kwargs)
        rmse_p = float(np.sqrt(mean_squared_error(ydata, y_fit_p)))
        if rmse_p <= best_rmse:
            best_popt = popt_p
            best_rmse = rmse_p
        best_pcov = pcov_p
        print(f"Polish step: RMSE={rmse_p:.4f}")
    except Exception as e:
        print(f"Polish step failed (pcov unavailable): {e}")

    return best_popt, best_pcov, best_rmse


###############################################################################
# Per-station calibration
###############################################################################

def _fit_model(
    df_merge: pd.DataFrame,
    group_name: str,
    rain_lag_days: int,
    up_lag_days: int,
    no_upstream: bool,
    model_name: str,
    n_starts: int = 15,
):
    t, rainfall, amp, amt, h_up, h_obs, time_index, doy = fit_preprocess(
        df_merge,
        rain_lag_days=rain_lag_days,
        up_lag_days=up_lag_days,
    )

    # --- Train / validation split ---
    split_mask = time_index < pd.Timestamp(SPLIT_DATE)
    n_cal = int(split_mask.sum())
    has_val = (len(time_index) - n_cal) >= 30  # need at least 30 val days
    if n_cal < 60:
        # Not enough calibration data; use all data for fitting, no validation
        n_cal = len(time_index)
        has_val = False

    t_cal      = t[:n_cal]
    h_obs_cal  = h_obs[:n_cal]
    rain_cal   = rainfall[:n_cal]
    amp_cal    = amp[:n_cal]
    amt_cal    = amt[:n_cal]
    h_up_cal   = h_up[:n_cal]
    doy_cal    = doy[:n_cal]

    h0 = float(h_obs_cal[0])
    is_coastal = group_name.lower() == "coastal"

    # --- Regression-based initial guess ---
    a0, z0, b0, c0, k0 = estimate_initial_params_inland(
        h_obs_cal, rain_cal, amp_cal, h_up_cal, no_upstream=no_upstream,
    )
    tau_rain0 = 5.0
    tau_up0   = 5.0
    d_sin0    = 0.0   # seasonal: flat prior
    d_cos0    = 0.0
    seas_lower = [-2.0, -2.0]
    seas_upper = [ 2.0,  2.0]

    if model_name == "base":
        model_func = sub.gw_model_wrapper
        if is_coastal:
            lower, upper = estimate_bounds_coastal(h_obs_cal, rain_cal, amp_cal, amt_cal, h_up_cal, no_upstream=no_upstream)
            param_names = ["a", "z", "b", "c", "k_link", "k_sgd", "gamma", "h_sea", "tau_rain", "tau_up", "d_sin", "d_cos"]
            h_mean = float(np.mean(h_obs_cal))
            base_p0 = np.array([a0, z0, b0, c0, k0, 0.1, 0.1, h_mean, tau_rain0, tau_up0, d_sin0, d_cos0], dtype=float)
        else:
            lower, upper = estimate_bounds_inland(h_obs_cal, rain_cal, amp_cal, h_up_cal, no_upstream=no_upstream)
            param_names = ["a", "z", "b", "c", "k_link", "tau_rain", "tau_up", "d_sin", "d_cos"]
            base_p0 = np.array([a0, z0, b0, c0, k0, tau_rain0, tau_up0, d_sin0, d_cos0], dtype=float)
        lower = list(lower) + seas_lower
        upper = list(upper) + seas_upper

    elif model_name == "filtered":
        model_func = sub.gw_model_wrapper_filtered
        lambda_min, lambda_max = _ensure_bounds_spread(0.01, 0.8, min_width=0.05)
        if is_coastal:
            lower, upper = estimate_bounds_coastal(h_obs_cal, rain_cal, amp_cal, amt_cal, h_up_cal, no_upstream=no_upstream)
            param_names = ["a", "z", "b", "c", "k_link", "k_sgd", "gamma", "h_sea", "lambda", "tau_rain", "tau_up", "d_sin", "d_cos"]
            h_mean = float(np.mean(h_obs_cal))
            base_p0 = np.array([a0, z0, b0, c0, k0, 0.1, 0.1, h_mean, 0.2, tau_rain0, tau_up0, d_sin0, d_cos0], dtype=float)
            lower = list(lower) + [lambda_min] + seas_lower
            upper = list(upper) + [lambda_max] + seas_upper
        else:
            lower, upper = estimate_bounds_inland(h_obs_cal, rain_cal, amp_cal, h_up_cal, no_upstream=no_upstream)
            param_names = ["a", "z", "b", "c", "k_link", "lambda", "tau_rain", "tau_up", "d_sin", "d_cos"]
            base_p0 = np.array([a0, z0, b0, c0, k0, 0.2, tau_rain0, tau_up0, d_sin0, d_cos0], dtype=float)
            lower = list(lower) + [lambda_min] + seas_lower
            upper = list(upper) + [lambda_max] + seas_upper
    elif model_name == "base_tz":
        model_func = sub.gw_model_wrapper_tz
        z1_lo, z1_hi = -2.0, 2.0
        if is_coastal:
            lower, upper = estimate_bounds_coastal(h_obs_cal, rain_cal, amp_cal, amt_cal, h_up_cal, no_upstream=no_upstream)
            param_names = ["a", "z0", "z1", "b", "c", "k_link", "k_sgd", "gamma", "h_sea", "tau_rain", "tau_up", "d_sin", "d_cos"]
            h_mean = float(np.mean(h_obs_cal))
            base_p0 = np.array([a0, z0, 0.0, b0, c0, k0, 0.1, 0.1, h_mean, tau_rain0, tau_up0, d_sin0, d_cos0], dtype=float)
            lower = [lower[0], lower[1], z1_lo] + list(lower[2:]) + seas_lower
            upper = [upper[0], upper[1], z1_hi] + list(upper[2:]) + seas_upper
        else:
            lower, upper = estimate_bounds_inland(h_obs_cal, rain_cal, amp_cal, h_up_cal, no_upstream=no_upstream)
            param_names = ["a", "z0", "z1", "b", "c", "k_link", "tau_rain", "tau_up", "d_sin", "d_cos"]
            base_p0 = np.array([a0, z0, 0.0, b0, c0, k0, tau_rain0, tau_up0, d_sin0, d_cos0], dtype=float)
            lower = [lower[0], lower[1], z1_lo] + list(lower[2:]) + seas_lower
            upper = [upper[0], upper[1], z1_hi] + list(upper[2:]) + seas_upper

    elif model_name == "filtered_tz":
        model_func = sub.gw_model_wrapper_filtered_tz
        z1_lo, z1_hi = -2.0, 2.0
        lambda_min, lambda_max = _ensure_bounds_spread(0.01, 0.8, min_width=0.05)
        if is_coastal:
            lower, upper = estimate_bounds_coastal(h_obs_cal, rain_cal, amp_cal, amt_cal, h_up_cal, no_upstream=no_upstream)
            param_names = ["a", "z0", "z1", "b", "c", "k_link", "k_sgd", "gamma", "h_sea", "lambda", "tau_rain", "tau_up", "d_sin", "d_cos"]
            h_mean = float(np.mean(h_obs_cal))
            base_p0 = np.array([a0, z0, 0.0, b0, c0, k0, 0.1, 0.1, h_mean, 0.2, tau_rain0, tau_up0, d_sin0, d_cos0], dtype=float)
            lower = [lower[0], lower[1], z1_lo] + list(lower[2:]) + [lambda_min] + seas_lower
            upper = [upper[0], upper[1], z1_hi] + list(upper[2:]) + [lambda_max] + seas_upper
        else:
            lower, upper = estimate_bounds_inland(h_obs_cal, rain_cal, amp_cal, h_up_cal, no_upstream=no_upstream)
            param_names = ["a", "z0", "z1", "b", "c", "k_link", "lambda", "tau_rain", "tau_up", "d_sin", "d_cos"]
            base_p0 = np.array([a0, z0, 0.0, b0, c0, k0, 0.2, tau_rain0, tau_up0, d_sin0, d_cos0], dtype=float)
            lower = [lower[0], lower[1], z1_lo] + list(lower[2:]) + [lambda_min] + seas_lower
            upper = [upper[0], upper[1], z1_hi] + list(upper[2:]) + [lambda_max] + seas_upper

    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    # --- For z(t) models, use absolute time indices ---
    use_absolute_t = model_name.endswith("_tz")

    model_kwargs = dict(
        rainfall=rain_cal, amp=amp_cal,
        amt=amt_cal if is_coastal else None,
        h_up=h_up_cal, h0=h0, is_coastal=is_coastal, doy=doy_cal,
    )

    best_popt, best_pcov, best_rmse = run_global_optimization(
        model_func=model_func,
        xdata=t_cal,
        ydata=h_obs_cal,
        bounds=(lower, upper),
        base_p0=base_p0,
        popsize=n_starts,
        **model_kwargs,
    )

    if best_popt is None:
        print(f"\nNo successful fit found for model '{model_name}'.")
        return None

    # --- Calibration metrics ---
    y_fit_cal = model_func(t_cal, *best_popt, **model_kwargs)
    rmse_cal = float(np.sqrt(mean_squared_error(h_obs_cal, y_fit_cal)))
    r2_cal   = float(r2_score(h_obs_cal, y_fit_cal))

    # --- Validation metrics (out-of-sample) ---
    rmse_val = np.nan
    r2_val   = np.nan
    y_fit_val = None
    if has_val:
        try:
            # Continuous hindcast across cal/val for ALL variants.
            # Val metrics reflect model propagation from the cal-optimized end state;
            # this removes noise-driven discontinuities present when h_obs[n_cal]
            # happens to be a local anomaly (see spec §2, D1).
            h0_val = float(y_fit_cal[-1])
            t_val = t[n_cal:] if use_absolute_t else t[n_cal:] - t[n_cal]
            val_kwargs = dict(
                rainfall=rainfall[n_cal:], amp=amp[n_cal:],
                amt=amt[n_cal:] if is_coastal else None,
                h_up=h_up[n_cal:], h0=h0_val,
                is_coastal=is_coastal, doy=doy[n_cal:],
            )
            y_fit_val = model_func(t_val, *best_popt, **val_kwargs)
            rmse_val = float(np.sqrt(mean_squared_error(h_obs[n_cal:], y_fit_val)))
            r2_val   = float(r2_score(h_obs[n_cal:], y_fit_val))
        except Exception as e:
            print(f"Validation evaluation failed: {e}")

    # --- Parameter uncertainty from pcov ---
    param_std = {}
    if best_pcov is not None:
        diag = np.diag(best_pcov)
        for name, var in zip(param_names, diag):
            param_std[f"{name}_std"] = float(np.sqrt(max(var, 0.0)))

    print("\n========================================================")
    print(f"Model: {model_name} | Group: {group_name} (coastal={is_coastal})")
    for name, val in zip(param_names, best_popt):
        std_str = f" ± {param_std.get(name+'_std', float('nan')):.4f}" if param_std else ""
        print(f"  {name} = {val:.4f}{std_str}")
    print(f"  RMSE_cal={rmse_cal:.4f}  R²_cal={r2_cal:.4f}")
    if has_val:
        print(f"  RMSE_val={rmse_val:.4f}  R²_val={r2_val:.4f}")
    print("========================================================\n")

    # Pre-compute KGE for cal & val using the existing helper so downstream
    # CSV writers and plotting share one source of truth.
    m_cal = compute_metrics(h_obs_cal, y_fit_cal)
    kge_cal = m_cal["kge"]
    if has_val and y_fit_val is not None:
        m_val = compute_metrics(h_obs[n_cal:], y_fit_val)
        kge_val = m_val["kge"]
    else:
        kge_val = float("nan")

    # KGE-beta conditioning diagnostic: |mean(obs_val)|/std(obs_val).
    # When this ratio is near zero, KGE beta (= mean_pred/mean_obs) explodes on
    # tiny absolute biases — a known pathology (Kling 2012, Pool 2018). Flag
    # stations below 0.05 so downstream reporting can exclude them from
    # aggregate KGE summaries without distorting the multi-station median.
    if has_val and len(h_obs) > n_cal:
        obs_val = np.asarray(h_obs[n_cal:], dtype=float)
        ov_mean = float(np.mean(obs_val))
        ov_std  = float(np.std(obs_val))
        mean_obs_val_ratio = abs(ov_mean) / ov_std if ov_std > 0 else float("nan")
    else:
        mean_obs_val_ratio = float("nan")
    kge_ill_conditioned = bool(np.isfinite(mean_obs_val_ratio) and mean_obs_val_ratio < 0.05)

    # Build full-period y_fit for plotting (calibration only — validation plotted separately)
    return {
        "model": model_name,
        "params": best_popt,
        "param_names": param_names,
        "param_std": param_std,
        "rmse": rmse_cal,
        "r2": r2_cal,
        "rmse_val": rmse_val,
        "r2_val": r2_val,
        "kge": kge_cal,
        "kge_val": kge_val,
        "mean_obs_val_ratio": mean_obs_val_ratio,
        "kge_ill_conditioned": kge_ill_conditioned,
        "y_fit": y_fit_cal,
        "y_fit_val": y_fit_val,
        "t": t_cal,
        "rainfall": rain_cal,
        "amp": amp_cal,
        "amt": amt_cal,
        "h_up": h_up_cal,
        "h_obs": h_obs_cal,
        "is_coastal": is_coastal,
        "time_index": time_index[:n_cal],
        "time_index_val": time_index[n_cal:] if has_val else None,
        "h_obs_full": h_obs,
        "rainfall_full": rainfall,
        "amp_full": amp,
        "amt_full": amt,
        "n_cal": n_cal,
        "time_index_full": time_index,
    }


def run_station(args_params: dict) -> None:
    """Run the full per-station pipeline (data prep → fit → plot → save).

    Call this directly from a multiprocessing Pool worker instead of spawning
    a subprocess, to avoid per-station library-import overhead.

    args_params may include 'output_root' (str or Path) pointing to the
    run-specific results directory (e.g. workspace/results/initial).
    Defaults to workspace/results/initial relative to the project root.
    """
    args = {k.lower(): v for k, v in args_params.items()}
    _default_output_root = Path(__file__).resolve().parents[1] / "workspace" / "results" / "initial"
    output_root = Path(args.get("output_root", _default_output_root))
    print('Parsed arguments:', args)
    df_merge, no_upstream = prepare_data(args)
    print('Prepared data (head):')
    print(df_merge.head())
    group_name = args.get('group_name', 'inland')
    lag_hint = int(args.get('lag_days', 0))
    max_lag = max(45, lag_hint)
    rain_lag_arg = args.get('rain_lag_days')
    up_lag_arg = args.get('ups_lag_days')

    if rain_lag_arg is not None:
        rain_lag_days = int(rain_lag_arg)
    else:
        rain_lag_days = estimate_rain_lag(
            df_merge['gwl'].values,
            df_merge['rf'].values,
            max_lag=max_lag,
        )

    if no_upstream:
        up_lag_days = 0
    elif up_lag_arg is not None:
        up_lag_days = int(up_lag_arg)
    else:
        up_lag_days = estimate_upstream_lag(
            df_merge['gwl'].values,
            df_merge['ups_gwl'].values,
            max_lag=max_lag,
        )

    # Configurable model variants (default: all 4)
    models_arg = args.get("models", "base,filtered,base_tz,filtered_tz")
    model_list = [m.strip() for m in models_arg.split(",")]

    model_results = []
    for model_name in model_list:
        result = _fit_model(
            df_merge,
            group_name=group_name,
            rain_lag_days=rain_lag_days,
            up_lag_days=up_lag_days,
            no_upstream=no_upstream,
            model_name=model_name,
        )
        if result is not None:
            model_results.append(result)

    if not model_results:
        print("No successful fit found for any model.")
        sys.exit(1)

    # Select best model by validation KGE (highest = best out-of-sample prediction).
    best_model = max(model_results, key=lambda item: item.get("kge_val", -np.inf))

    station_label = args.get('st_id', args.get('gw_st', 'unknown'))
    plot_index = best_model["time_index"]

    # --- Per-variant and comparison plots (rklib style) ---
    _variant_colors = {
        'base': '#2980b9',        # blue
        'filtered': '#27ae60',    # green
        'base_tz': '#e74c3c',     # red
        'filtered_tz': '#f39c12', # orange
    }
    MODEL_IDS = {
        "base":        "M1",
        "filtered":    "M2",
        "base_tz":     "M3",
        "filtered_tz": "M4",
    }
    MODEL_LABELS = {
        "base":        "M1: Direct",
        "filtered":    "M2: Filtered",
        "base_tz":     "M3: Direct, z(t)",
        "filtered_tz": "M4: Filtered, z(t)",
    }
    setup_font()

    # Per-variant plots (publication-quality)
    CAL_COLOR  = "#1a6faf"   # blue
    VAL_COLOR  = "#c94040"   # red
    SHADE_CAL  = "#2166ac"
    SHADE_VAL  = "#d6604d"
    METRIC_BOX = dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor="#4d4d4d", linewidth=0.8, alpha=0.92)
    SPLIT = pd.Timestamp("2019-01-01")

    for result in model_results:
        variant_dir = output_root / "figures" / result['model']
        variant_dir.mkdir(parents=True, exist_ok=True)

        time_cal = result['time_index']
        time_val = result.get('time_index_val')
        time_full = result.get('time_index_full', time_cal)
        h_obs_full = result.get('h_obs_full', result['h_obs'])

        fig_v, ax_v = plt.subplots(figsize=(11, 5))

        # Background shading cal/val
        t_start = time_full[0]
        t_end = time_full[-1] + pd.Timedelta(days=1)
        ax_v.axvspan(t_start, SPLIT, color=SHADE_CAL, alpha=0.08, zorder=0)
        ax_v.axvspan(SPLIT, t_end, color=SHADE_VAL, alpha=0.08, zorder=0)

        # Observed across full period
        ax_v.plot(time_full, h_obs_full, color="black", lw=1.0,
                  alpha=0.9, label="Observed")
        # Calibration simulation
        ax_v.plot(time_cal, result['y_fit'], color=CAL_COLOR, lw=1.5,
                  alpha=0.95, label="Calibration")
        # Validation simulation
        if result.get('y_fit_val') is not None and time_val is not None:
            ax_v.plot(time_val, result['y_fit_val'], color=VAL_COLOR, lw=1.5,
                      alpha=0.95, label="Validation")

        # Split line
        ax_v.axvline(SPLIT, color="black", ls="--", lw=1.5, alpha=0.9)

        # Annotation boxes (KGE + RMSE)
        kge_cal = result.get('kge', float('nan'))
        rmse_cal = result['rmse']
        ax_v.text(0.45, 0.05,
                  f"Calibration\nKGE = {kge_cal:.2f}   RMSE = {rmse_cal:.2f} m",
                  transform=ax_v.transAxes, fontsize=9, va="bottom", ha="center",
                  color="black", bbox=METRIC_BOX)
        if result.get('y_fit_val') is not None:
            kge_val = result.get('kge_val', float('nan'))
            rmse_val = result['rmse_val']
            ax_v.text(0.9, 0.05,
                      f"Validation\nKGE = {kge_val:.2f}   RMSE = {rmse_val:.2f} m",
                      transform=ax_v.transAxes, fontsize=9, va="bottom", ha="center",
                      color="black", bbox=METRIC_BOX)

        # Axes, title, legend
        ax_v.set_title(
            f"{station_label} ({group_name}) — Model Type: {MODEL_IDS[result['model']]}",
            fontsize=14, fontweight="bold", pad=6,
        )
        ax_v.set_ylabel("Groundwater level (m)", fontsize=12, fontweight="bold")
        ax_v.set_xlabel("Year", fontsize=12, fontweight="bold")
        ax_v.set_xlim(t_start, t_end)
        ax_v.margins(x=0)
        ax_v.xaxis.set_major_locator(mdates.YearLocator())
        ax_v.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax_v.tick_params(axis="both", labelsize=9)
        ax_v.grid(True, lw=0.3, alpha=0.4)
        ax_v.legend(fontsize=9, loc="upper right", framealpha=0.85)

        fig_v.tight_layout()
        rklib_savefig(fig_v, variant_dir / f"gw_fit_{station_label}.png")
        plt.close(fig_v)

    # Comparison overlay plot: cal (solid) + val (dashed) per variant,
    # metrics table as a subplot below (spec §4.4).
    compare_dir = output_root / "figures" / "comparison"
    compare_dir.mkdir(parents=True, exist_ok=True)

    fig_c, (ax_ts, ax_tbl) = plt.subplots(
        2, 1, figsize=(11, 7),
        gridspec_kw={"height_ratios": [4, 1], "hspace": 0.35},
    )

    time_full = best_model.get('time_index_full', best_model['time_index'])
    h_obs_full = best_model.get('h_obs_full', best_model['h_obs'])
    t_start = time_full[0]
    t_end = time_full[-1] + pd.Timedelta(days=1)

    # Shading + observed
    ax_ts.axvspan(t_start, SPLIT, color=SHADE_CAL, alpha=0.08, zorder=0)
    ax_ts.axvspan(SPLIT, t_end, color=SHADE_VAL, alpha=0.08, zorder=0)
    ax_ts.plot(time_full, h_obs_full, color="black", lw=1.3,
               alpha=0.95, label="Observed", zorder=3)

    # Variant lines (cal solid, val dashed)
    variant_order = ["base", "filtered", "base_tz", "filtered_tz"]
    for model_id in variant_order:
        match = [r for r in model_results if r['model'] == model_id]
        if not match:
            continue
        result = match[0]
        is_best = (result is best_model)
        lw = 1.9 if is_best else 1.3
        alpha = 1.0 if is_best else 0.75
        clr = _variant_colors[model_id]
        ax_ts.plot(result['time_index'], result['y_fit'],
                   color=clr, lw=lw, alpha=alpha, ls="-",
                   label=MODEL_IDS[model_id])
        if result.get('y_fit_val') is not None and result.get('time_index_val') is not None:
            ax_ts.plot(result['time_index_val'], result['y_fit_val'],
                       color=clr, lw=lw, alpha=alpha, ls="--")

    ax_ts.axvline(SPLIT, color="black", ls="--", lw=1.5, alpha=0.9)
    ax_ts.set_title(
        f"{station_label} ({group_name}) — Model comparison",
        fontsize=14, fontweight="bold", pad=6,
    )
    ax_ts.set_ylabel("Groundwater level (m)", fontsize=12, fontweight="bold")
    ax_ts.set_xlabel("Year", fontsize=12, fontweight="bold")
    ax_ts.set_xlim(t_start, t_end)
    ax_ts.margins(x=0)
    ax_ts.xaxis.set_major_locator(mdates.YearLocator())
    ax_ts.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_ts.tick_params(axis="both", labelsize=9)
    ax_ts.grid(True, lw=0.3, alpha=0.4)
    ax_ts.legend(fontsize=9, loc="upper right", framealpha=0.85, ncol=1)

    # Metrics table
    ax_tbl.axis("off")
    col_labels = ["Model", "KGE_cal", "KGE_val", "RMSE_cal", "RMSE_val"]
    table_rows = []
    for model_id in variant_order:
        match = [r for r in model_results if r['model'] == model_id]
        if not match:
            continue
        result = match[0]
        is_best = (result is best_model)
        label = MODEL_LABELS[model_id] + (" *" if is_best else "")
        rmse_val_str = (f"{result['rmse_val']:.2f}"
                        if not np.isnan(result.get('rmse_val', np.nan)) else "—")
        table_rows.append([
            label,
            f"{result.get('kge', float('nan')):.2f}",
            f"{result.get('kge_val', float('nan')):.2f}",
            f"{result['rmse']:.2f}",
            rmse_val_str,
        ])
    tbl = ax_tbl.table(
        cellText=table_rows, colLabels=col_labels,
        loc="center", cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.3)
    for j in range(len(col_labels)):
        tbl[(0, j)].set_facecolor("#e8e8e8")
        tbl[(0, j)].set_text_props(weight="bold")

    fig_c.tight_layout()
    rklib_savefig(fig_c, compare_dir / f"gw_compare_{station_label}.png")
    plt.close(fig_c)

    # --- Best-model 3-panel figure: GWL (cal/val styled), rainfall, tidal amp ---
    full_dir = output_root / "figures" / "full_subplots"
    full_dir.mkdir(parents=True, exist_ok=True)

    time_best_full = best_model.get('time_index_full', best_model['time_index'])
    h_obs_best_full = best_model.get('h_obs_full', best_model['h_obs'])
    rainfall_best_full = best_model.get('rainfall_full', best_model['rainfall'])
    amp_best_full = best_model.get('amp_full', best_model['amp'])

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True,
                             gridspec_kw={"height_ratios": [2.5, 1, 1]})

    # Panel (a): GWL — best model with cal/val convention
    ax0 = axes[0]
    t_start = time_best_full[0]
    t_end = time_best_full[-1] + pd.Timedelta(days=1)
    ax0.axvspan(t_start, SPLIT, color=SHADE_CAL, alpha=0.08, zorder=0)
    ax0.axvspan(SPLIT, t_end, color=SHADE_VAL, alpha=0.08, zorder=0)
    ax0.plot(time_best_full, h_obs_best_full, color="black", lw=1.0,
             alpha=0.9, label="Observed")
    ax0.plot(best_model['time_index'], best_model['y_fit'], color=CAL_COLOR,
             lw=1.5, alpha=0.95, label="Calibration")
    if best_model.get('y_fit_val') is not None and best_model.get('time_index_val') is not None:
        ax0.plot(best_model['time_index_val'], best_model['y_fit_val'],
                 color=VAL_COLOR, lw=1.5, alpha=0.95, label="Validation")
    ax0.axvline(SPLIT, color="black", ls="--", lw=1.5, alpha=0.9)
    ax0.text(0.45, 0.05,
             f"Calibration\nKGE = {best_model.get('kge', float('nan')):.2f}   "
             f"RMSE = {best_model['rmse']:.2f} m",
             transform=ax0.transAxes, fontsize=9, va="bottom", ha="center",
             color="black", bbox=METRIC_BOX)
    if best_model.get('y_fit_val') is not None:
        ax0.text(0.9, 0.05,
                 f"Validation\nKGE = {best_model.get('kge_val', float('nan')):.2f}   "
                 f"RMSE = {best_model['rmse_val']:.2f} m",
                 transform=ax0.transAxes, fontsize=9, va="bottom", ha="center",
                 color="black", bbox=METRIC_BOX)
    ax0.set_title(
        f"{station_label} ({group_name}) — Model Type: {MODEL_IDS[best_model['model']]} (best)",
        fontsize=14, fontweight="bold", pad=6,
    )
    ax0.set_ylabel("Groundwater level (m)", fontsize=12, fontweight="bold")
    ax0.set_xlim(t_start, t_end)
    ax0.margins(x=0)
    ax0.legend(fontsize=9, loc="upper right", framealpha=0.85)
    ax0.grid(True, lw=0.3, alpha=0.4)

    # Panel (b): Rainfall bars (full period — cal + val forcing)
    ax1 = axes[1]
    ax1.bar(time_best_full, rainfall_best_full, width=1.0, color="C0", alpha=0.8)
    ax1.set_title(f"Rainfall — {args.get('rf_id', '')}", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Rainfall (mm/day)", fontsize=11, fontweight="bold")
    ax1.set_xlim(t_start, t_end)
    ax1.margins(x=0)
    ax1.grid(True, lw=0.3, alpha=0.4)

    # Panel (c): Tidal amplitude (full period)
    ax2 = axes[2]
    ax2.plot(time_best_full, amp_best_full, color="C2", lw=1.0, alpha=0.85)
    amp_label_id = args.get('gw_st', args.get('st_id', ''))
    ax2.set_title(f"Tidal amplitude — {amp_label_id}", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Amplitude (m)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Year", fontsize=12, fontweight="bold")
    ax2.set_xlim(t_start, t_end)
    ax2.margins(x=0)
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.grid(True, lw=0.3, alpha=0.4)

    for ax, lbl in zip(axes, ['a', 'b', 'c']):
        add_panel_label(ax, lbl, x=0.01, y=1.02)

    fig.tight_layout(h_pad=1.5)
    rklib_savefig(fig, full_dir / f"gw_fit_{station_label}.png")
    plt.close(fig)

    # --- Save results with KGE metrics ---
    station_label_key = args.get('st_id', args.get('gw_st', 'unknown'))

    # Compute KGE for best model (val arrays now surfaced by _fit_model)
    m_cal = compute_metrics(best_model['h_obs'], best_model['y_fit'])
    m_val = {"kge": np.nan, "kge_r": np.nan, "kge_alpha": np.nan, "kge_beta": np.nan, "bias": np.nan}
    if best_model.get('y_fit_val') is not None and best_model.get('h_obs_full') is not None:
        n_cal_idx = best_model['n_cal']
        m_val = compute_metrics(best_model['h_obs_full'][n_cal_idx:], best_model['y_fit_val'])

    # Best-variant results (main output)
    results = {
        'st_id': station_label_key,
        'gw_st': args.get('gw_st'),
        'ups_id': args.get('ups_id'),
        'rf_id': args.get('rf_id'),
        'group_name': group_name,
        'rain_lag_days': rain_lag_days,
        'up_lag_days': up_lag_days,
        'model': best_model['model'],
        'rmse': best_model['rmse'],
        'r2': best_model['r2'],
        'kge': m_cal['kge'],
        'rmse_val': best_model['rmse_val'],
        'r2_val': best_model['r2_val'],
        'kge_val': m_val['kge'],
        'kge_r_val': m_val['kge_r'],
        'kge_alpha_val': m_val['kge_alpha'],
        'kge_beta_val': m_val['kge_beta'],
        'bias_val': m_val['bias'],
        'mean_obs_val_ratio': best_model.get('mean_obs_val_ratio', float('nan')),
        'kge_ill_conditioned': bool(best_model.get('kge_ill_conditioned', False)),
    }
    for name, val in zip(best_model['param_names'], best_model['params']):
        results[name] = float(val)
    for name, std in best_model['param_std'].items():
        results[name] = std

    per_station_dir = output_root / "per_station"
    per_station_dir.mkdir(parents=True, exist_ok=True)
    per_station_path = per_station_dir / f"{station_label_key}.csv"
    pd.DataFrame([results]).to_csv(per_station_path, index=False)

    # All-variants results with KGE (for analysis/comparison)
    all_variant_rows = []
    for result in model_results:
        mc = compute_metrics(result['h_obs'], result['y_fit'])
        if result.get('y_fit_val') is not None and result.get('h_obs_full') is not None:
            mv = compute_metrics(result['h_obs_full'][result['n_cal']:], result['y_fit_val'])
        else:
            mv = {"kge": np.nan}
        row = {
            'st_id': station_label_key, 'model': result['model'],
            'r2': result['r2'], 'rmse': result['rmse'],
            'kge': mc['kge'], 'kge_r': mc['kge_r'],
            'r2_val': result['r2_val'], 'rmse_val': result['rmse_val'],
            'kge_val': mv['kge'],
            'mean_obs_val_ratio': result.get('mean_obs_val_ratio', float('nan')),
            'kge_ill_conditioned': bool(result.get('kge_ill_conditioned', False)),
        }
        for name, val in zip(result['param_names'], result['params']):
            row[name] = float(val)
        all_variant_rows.append(row)
    all_variants_dir = output_root / "all_variants"
    all_variants_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_variant_rows).to_csv(all_variants_dir / f"{station_label_key}.csv", index=False)

    print(f"Per-station results saved → {per_station_path}")


if __name__ == "__main__":
    run_station(dict(argv_phrase(sys.argv[1:])))