import sys
import os
import json
from pathlib import Path
from typing import List, Tuple, Optional
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit, differential_evolution
from scipy.stats import qmc
from sklearn.metrics import mean_squared_error
import jfft
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import gw_subroutine as sub

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rklib import TimeSeriesModelFig, setup_font, savefig as rklib_savefig, add_panel_label


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
GW_DATA_PATH = DATA_DIR / "gw_timeseries.csv"
RF_DATA_PATH = DATA_DIR / "rf_timeseries.csv"

# Three-way temporal split (inclusive end dates):
#   calibration  start .. CAL_END   parameters, lags, pairing and classification
#   selection    CAL_END .. SEL_END variant choice and pairing refinement only
#   test         SEL_END .. end     scored once; never used for any decision
CAL_END = "2018-12-31"
SEL_END = "2020-12-31"
PERIODS = ("cal", "sel", "test")
MIN_PERIOD_DAYS = 30  # minimum records for a period's metrics to be reported

# Timestamp placement of the 30-day STFT window used for AMP/AMT:
#   "centre"   value at the window midpoint (uses 15 days after the timestamp)
#   "trailing" value at the window end (causal: uses only past records)
STFT_ALIGN = os.environ.get("GW_STFT_ALIGN", "centre")
STFT_FRAME_DAYS = 30
STFT_HOP_DAYS = 5


def period_bounds(time_index: pd.DatetimeIndex) -> Tuple[int, int]:
    """Return (n_cal, n_sel_end): row counts up to CAL_END and up to SEL_END."""
    n_cal = int((time_index <= pd.Timestamp(CAL_END)).sum())
    n_sel_end = int((time_index <= pd.Timestamp(SEL_END)).sum())
    return n_cal, n_sel_end


def calibration_slice(df: pd.DataFrame) -> pd.DataFrame:
    """Rows of a date-indexed frame that fall in the calibration period."""
    return df.loc[df.index <= pd.Timestamp(CAL_END)]


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

    mystft = jfft.STFT_Obj(y, fs=fs, framesz=STFT_FRAME_DAYS, hop=STFT_HOP_DAYS, dt_list=rng)

    xfindex_amp = mystft.find_xf_index(1)
    stft_result = mystft.get_yf()
    # get_timeval stamps each window at its midpoint; shifting by half a frame
    # moves the stamp to the window end so each value depends on past data only.
    stft_timeval = pd.DatetimeIndex(mystft.get_timeval())
    if STFT_ALIGN == "trailing":
        stft_timeval = stft_timeval + pd.Timedelta(days=STFT_FRAME_DAYS / 2)
    elif STFT_ALIGN != "centre":
        raise ValueError(f"Unknown STFT_ALIGN: {STFT_ALIGN!r}")
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

TAU_MIN, TAU_MAX = 0.1, 30.0
KERNEL_MAX_LAG = 90

# Bounds without a physical limit, set wide enough that they rarely bind:
# - z (equilibrium/drainage base head) may lie well below the observed range;
#   lower bound at h_min - max(5 m, 2 x observed range).
# - c (diurnal-amplitude coefficient): upper bound c_est + max(5, 5|c_est|).
# - h_sea (effective coastal boundary head): mean head +/- 30 m.
# - lambda (filter rate): up to 1, where the filtered variant equals the direct one.
Z_LOWER_PAD_MIN = 5.0
Z_LOWER_RANGE_FACTOR = 2.0
C_WIDTH_MIN = 5.0
C_WIDTH_FACTOR = 5.0
H_SEA_HALF_RANGE = 30.0
LAMBDA_MIN, LAMBDA_MAX = 0.01, 1.0

# The b estimate above regresses daily head change on same-day rainfall. With a
# unit-sum rainfall kernel, a response spread over tau days needs a gain up to
# the unnormalised kernel mass at TAU_MAX larger than that same-day estimate, so
# the upper bound is widened by that factor (~29) to keep the gain range searched.
RAIN_GAIN_SCALE = float(np.exp(-np.arange(KERNEL_MAX_LAG + 1) / TAU_MAX).sum())


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
    z_min = h_min - max(Z_LOWER_PAD_MIN, Z_LOWER_RANGE_FACTOR * (h_max - h_min))
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
    b_max *= RAIN_GAIN_SCALE

    # c from AMP correlation
    valid_amp = AMP_short != 0
    if np.any(valid_amp) and np.var(AMP_short[valid_amp]) > 0:
        c_est = -np.cov(dh[valid_amp], AMP_short[valid_amp])[0, 1] / np.var(AMP_short[valid_amp])
        width_c = max(C_WIDTH_MIN, C_WIDTH_FACTOR * abs(c_est))
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
    tau_rain_min, tau_rain_max = TAU_MIN, TAU_MAX
    tau_up_min, tau_up_max = TAU_MIN, TAU_MAX
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
    z_min = h_min - max(Z_LOWER_PAD_MIN, Z_LOWER_RANGE_FACTOR * (h_max - h_min))
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
    b_max *= RAIN_GAIN_SCALE

    # c from AMP correlation
    valid_amp = AMP_short != 0
    if np.any(valid_amp) and np.var(AMP_short[valid_amp]) > 0:
        c_est = -np.cov(dh[valid_amp], AMP_short[valid_amp])[0, 1] / np.var(AMP_short[valid_amp])
        width_c = max(C_WIDTH_MIN, C_WIDTH_FACTOR * abs(c_est))
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
    h_sea_min = h_mean - H_SEA_HALF_RANGE
    h_sea_max = h_mean + H_SEA_HALF_RANGE
    h_sea_min, h_sea_max = _ensure_bounds_spread(h_sea_min, h_sea_max, min_width=0.5)

    z_min, z_max = _ensure_bounds_spread(z_min, z_max, min_width=0.5)
    tau_rain_min, tau_rain_max = TAU_MIN, TAU_MAX
    tau_up_min, tau_up_max = TAU_MIN, TAU_MAX
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

METRIC_KEYS = (
    "r2", "nse", "rmse", "mae", "bias",
    "kge", "kge_r", "kge_alpha", "kge_beta",
    "kge_ab", "kge_ab_beta",
)


def compute_metrics(obs: np.ndarray, pred: np.ndarray) -> dict:
    """Goodness-of-fit metrics for simulated vs observed heads.

    kge     classic KGE, bias ratio beta = mu_sim / mu_obs. Depends on the
            vertical datum and is singular when mu_obs is near zero.
    kge_ab  KGE with an additive, scale-free bias term
            beta_ab = (mu_sim - mu_obs) / sigma_obs, i.e.
            1 - sqrt((r - 1)^2 + (alpha - 1)^2 + beta_ab^2).
            Invariant to adding a constant to both series (datum shift).
    nse     Nash-Sutcliffe efficiency (identical to r2 as computed here).
    """
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)
    if len(obs) < 2:
        return {k: float("nan") for k in METRIC_KEYS}

    resid = pred - obs
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((obs - np.mean(obs)) ** 2)
    nse = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    mae = float(np.mean(np.abs(resid)))
    sd_obs = float(np.std(obs))
    sd_pred = float(np.std(pred))
    r = float(np.corrcoef(obs, pred)[0, 1]) if sd_obs > 0 and sd_pred > 0 else np.nan
    alpha = sd_pred / sd_obs if sd_obs > 0 else np.nan
    beta = float(np.mean(pred) / np.mean(obs)) if np.mean(obs) != 0 else np.nan
    bias = float(np.mean(pred) - np.mean(obs))
    beta_ab = bias / sd_obs if sd_obs > 0 else np.nan
    kge = 1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    kge_ab = 1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + beta_ab ** 2)

    return {
        "r2": float(nse), "nse": float(nse), "rmse": rmse, "mae": mae, "bias": bias,
        "kge": float(kge), "kge_r": float(r), "kge_alpha": float(alpha),
        "kge_beta": float(beta),
        "kge_ab": float(kge_ab), "kge_ab_beta": float(beta_ab),
    }


def period_metrics(obs: np.ndarray, pred: np.ndarray) -> dict:
    """compute_metrics, or all-NaN when the period is shorter than MIN_PERIOD_DAYS."""
    if len(obs) < MIN_PERIOD_DAYS:
        return {k: float("nan") for k in METRIC_KEYS}
    return compute_metrics(obs, pred)


def selection_score(result: dict) -> float:
    """Variant/pairing selection criterion: selection-period kge_ab.

    Falls back to calibration kge_ab when the selection period has too few
    records. The test period is never consulted.
    """
    score = result["metrics"]["sel"]["kge_ab"]
    if not np.isfinite(score):
        score = result["metrics"]["cal"]["kge_ab"]
    return float(score) if np.isfinite(score) else -np.inf


def run_global_optimization(
    model_func,
    xdata: np.ndarray,
    ydata: np.ndarray,
    bounds: Tuple[List[float], List[float]],
    base_p0: np.ndarray = None,
    popsize: int = 15,
    maxiter: int = 500,
    seed: int = 42,
    seed_p0: bool = True,
    **model_kwargs,
):
    """Global optimisation with differential_evolution, then local polish with curve_fit for pcov.

    seed drives both the Latin-hypercube initial population and the DE stream;
    seed_p0 places the regression-based guess in the initial population.

    Returns (best_popt, best_pcov, best_rmse).
    pcov is None if the polish step fails.
    """
    lower, upper = bounds
    lower_arr = np.array(lower, dtype=float)
    upper_arr = np.array(upper, dtype=float)
    de_bounds = list(zip(lower_arr, upper_arr))
    n_params = len(lower_arr)

    # Seed the population: first row is the regression-based guess (if available),
    # the rest is a Latin-hypercube sample scaled across the parameter bounds.
    lhs = qmc.LatinHypercube(d=n_params, seed=seed)
    unit_pop = lhs.random(n=popsize * n_params)          # samples in [0, 1)
    pop = qmc.scale(unit_pop, lower_arr, upper_arr)       # scaled to [lower, upper]
    if base_p0 is not None and seed_p0:
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
        seed=seed,
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
        if rmse_p < best_rmse:               # accept polish only if strictly better
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

def setup_fit(
    df_merge: pd.DataFrame,
    group_name: str,
    rain_lag_days: int,
    up_lag_days: int,
    no_upstream: bool,
    model_name: str,
    fixed: Optional[dict] = None,
) -> Optional[dict]:
    """Inputs, period split, initial guess and bounds for one variant (no fitting).

    Returns None when there are too few calibration records. fixed maps
    parameter names to values held constant (bounds collapsed onto the value).
    """
    t, rainfall, amp, amt, h_up, h_obs, time_index, doy = fit_preprocess(
        df_merge,
        rain_lag_days=rain_lag_days,
        up_lag_days=up_lag_days,
    )

    # --- Calibration / selection / test split ---
    # Parameters are fitted on the calibration rows only. Stations without
    # enough calibration data are skipped rather than fitted on later periods,
    # which would leak selection/test data into the parameters.
    n_cal, n_sel_end = period_bounds(time_index)
    if n_cal < 60:
        print(f"Only {n_cal} calibration records; skipping model '{model_name}'.")
        return None

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
        lambda_min, lambda_max = _ensure_bounds_spread(LAMBDA_MIN, LAMBDA_MAX, min_width=0.05)
        if is_coastal:
            lower, upper = estimate_bounds_coastal(h_obs_cal, rain_cal, amp_cal, amt_cal, h_up_cal, no_upstream=no_upstream)
            param_names = ["a", "z", "b", "c", "k_link", "k_sgd", "gamma", "h_sea", "lambda", "tau_rain", "tau_up", "d_sin", "d_cos"]
            h_mean = float(np.mean(h_obs_cal))
            base_p0 = np.array([a0, z0, b0, c0, k0, 0.1, 0.1, h_mean, 0.2, tau_rain0, tau_up0, d_sin0, d_cos0], dtype=float)
            # estimate_bounds_coastal returns [a, z, b, c, k_link, k_sgd, gamma, h_sea, tau_rain, tau_up];
            # lambda must go at index 8 (between h_sea and tau_rain) to match param_names.
            lower = list(lower[:8]) + [lambda_min] + list(lower[8:]) + seas_lower
            upper = list(upper[:8]) + [lambda_max] + list(upper[8:]) + seas_upper
        else:
            lower, upper = estimate_bounds_inland(h_obs_cal, rain_cal, amp_cal, h_up_cal, no_upstream=no_upstream)
            param_names = ["a", "z", "b", "c", "k_link", "lambda", "tau_rain", "tau_up", "d_sin", "d_cos"]
            base_p0 = np.array([a0, z0, b0, c0, k0, 0.2, tau_rain0, tau_up0, d_sin0, d_cos0], dtype=float)
            # estimate_bounds_inland returns [a, z, b, c, k_link, tau_rain, tau_up];
            # lambda must go at index 5 (between k_link and tau_rain) to match param_names.
            lower = list(lower[:5]) + [lambda_min] + list(lower[5:]) + seas_lower
            upper = list(upper[:5]) + [lambda_max] + list(upper[5:]) + seas_upper
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
        lambda_min, lambda_max = _ensure_bounds_spread(LAMBDA_MIN, LAMBDA_MAX, min_width=0.05)
        if is_coastal:
            lower, upper = estimate_bounds_coastal(h_obs_cal, rain_cal, amp_cal, amt_cal, h_up_cal, no_upstream=no_upstream)
            param_names = ["a", "z0", "z1", "b", "c", "k_link", "k_sgd", "gamma", "h_sea", "lambda", "tau_rain", "tau_up", "d_sin", "d_cos"]
            h_mean = float(np.mean(h_obs_cal))
            base_p0 = np.array([a0, z0, 0.0, b0, c0, k0, 0.1, 0.1, h_mean, 0.2, tau_rain0, tau_up0, d_sin0, d_cos0], dtype=float)
            # estimate_bounds_coastal returns [a, z, b, c, k_link, k_sgd, gamma, h_sea, tau_rain, tau_up];
            # final param order is a, z0, z1, b, c, k_link, k_sgd, gamma, h_sea, lambda, tau_rain, tau_up, d_sin, d_cos.
            lower = [lower[0], lower[1], z1_lo] + list(lower[2:8]) + [lambda_min] + list(lower[8:]) + seas_lower
            upper = [upper[0], upper[1], z1_hi] + list(upper[2:8]) + [lambda_max] + list(upper[8:]) + seas_upper
        else:
            lower, upper = estimate_bounds_inland(h_obs_cal, rain_cal, amp_cal, h_up_cal, no_upstream=no_upstream)
            param_names = ["a", "z0", "z1", "b", "c", "k_link", "lambda", "tau_rain", "tau_up", "d_sin", "d_cos"]
            base_p0 = np.array([a0, z0, 0.0, b0, c0, k0, 0.2, tau_rain0, tau_up0, d_sin0, d_cos0], dtype=float)
            # estimate_bounds_inland returns [a, z, b, c, k_link, tau_rain, tau_up];
            # final param order is a, z0, z1, b, c, k_link, lambda, tau_rain, tau_up, d_sin, d_cos.
            lower = [lower[0], lower[1], z1_lo] + list(lower[2:5]) + [lambda_min] + list(lower[5:]) + seas_lower
            upper = [upper[0], upper[1], z1_hi] + list(upper[2:5]) + [lambda_max] + list(upper[5:]) + seas_upper

    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    # Defensive check: bound arrays must align with simulator parameter order.
    # A silent misalignment caused λ and τ_up to be fitted against the wrong
    # bounds in earlier runs (filtered/filtered_tz variants).
    assert len(lower) == len(param_names) == len(upper) == len(base_p0), (
        f"Bound/param/p0 length mismatch for {model_name}: "
        f"lower={len(lower)}, upper={len(upper)}, "
        f"param_names={len(param_names)}, base_p0={len(base_p0)}"
    )


    # Hold fixed parameters at their value with a negligible bound width
    # (the optimisers need lower < upper).
    for name, value in (fixed or {}).items():
        if name not in param_names:
            raise KeyError(f"Cannot fix '{name}': not a parameter of {model_name} ({param_names})")
        i = param_names.index(name)
        eps = 1e-9 * max(1.0, abs(value))
        lower[i], upper[i] = value - eps, value + eps
        base_p0[i] = value

    model_kwargs = dict(
        rainfall=rain_cal, amp=amp_cal,
        amt=amt_cal if is_coastal else None,
        h_up=h_up_cal, h0=h0, is_coastal=is_coastal, doy=doy_cal,
    )

    return {
        "model_func": model_func, "param_names": param_names,
        "lower": lower, "upper": upper, "base_p0": base_p0,
        "t": t, "rainfall": rainfall, "amp": amp, "amt": amt, "h_up": h_up,
        "h_obs": h_obs, "time_index": time_index, "doy": doy,
        "n_cal": n_cal, "n_sel_end": n_sel_end, "h0": h0, "is_coastal": is_coastal,
        "t_cal": t_cal, "h_obs_cal": h_obs_cal, "model_kwargs": model_kwargs,
    }


def _fit_model(
    df_merge: pd.DataFrame,
    group_name: str,
    rain_lag_days: int,
    up_lag_days: int,
    no_upstream: bool,
    model_name: str,
    n_starts: int = 15,
    seed: int = 42,
    fixed: Optional[dict] = None,
    maxiter: int = 500,
    seed_p0: bool = True,
):
    """Fit one variant on the calibration period and score it on all periods.

    fixed maps parameter names to values held constant during the fit (used for
    profile likelihoods and reduced benchmark models).
    """
    setup = setup_fit(df_merge, group_name, rain_lag_days, up_lag_days,
                      no_upstream, model_name, fixed=fixed)
    if setup is None:
        return None
    model_func, param_names = setup["model_func"], setup["param_names"]
    lower, upper, base_p0 = setup["lower"], setup["upper"], setup["base_p0"]
    t, rainfall, amp, amt, h_up = setup["t"], setup["rainfall"], setup["amp"], setup["amt"], setup["h_up"]
    h_obs, time_index, doy = setup["h_obs"], setup["time_index"], setup["doy"]
    n_cal, n_sel_end, h0, is_coastal = setup["n_cal"], setup["n_sel_end"], setup["h0"], setup["is_coastal"]
    t_cal, h_obs_cal, model_kwargs = setup["t_cal"], setup["h_obs_cal"], setup["model_kwargs"]


    best_popt, best_pcov, best_rmse = run_global_optimization(
        model_func=model_func,
        xdata=t_cal,
        ydata=h_obs_cal,
        bounds=(lower, upper),
        base_p0=base_p0,
        popsize=n_starts,
        maxiter=maxiter,
        seed=seed,
        seed_p0=seed_p0,
        **model_kwargs,
    )

    if best_popt is None:
        print(f"\nNo successful fit found for model '{model_name}'.")
        return None

    # --- Continuous simulation over the whole record with calibrated parameters ---
    # The model is causal, so the first n_cal values equal the calibration fit;
    # running straight through avoids restarting the state or the memory
    # convolutions at the period boundaries.
    full_kwargs = dict(
        rainfall=rainfall, amp=amp,
        amt=amt if is_coastal else None,
        h_up=h_up, h0=h0, is_coastal=is_coastal, doy=doy,
    )
    y_full = model_func(t, *best_popt, **full_kwargs)
    bounds_idx = {"cal": (0, n_cal), "sel": (n_cal, n_sel_end), "test": (n_sel_end, len(t))}
    metrics = {
        p: period_metrics(h_obs[i0:i1], y_full[i0:i1]) for p, (i0, i1) in bounds_idx.items()
    }

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
    for p in PERIODS:
        m = metrics[p]
        print(f"  {p:>4}: RMSE={m['rmse']:.4f}  NSE={m['nse']:.4f}  KGE_ab={m['kge_ab']:.4f}")
    print("========================================================\n")

    return {
        "model": model_name,
        "params": best_popt,
        "param_names": param_names,
        "param_std": param_std,
        "lower": list(lower),
        "upper": list(upper),
        "metrics": metrics,
        "y_full": y_full,
        "h_obs_full": h_obs,
        "rainfall_full": rainfall,
        "amp_full": amp,
        "amt_full": amt,
        "time_index_full": time_index,
        "n_cal": n_cal,
        "n_sel_end": n_sel_end,
        "is_coastal": is_coastal,
    }


def _at_bound(value: float, lo: float, hi: float, rel_tol: float = 1e-3) -> bool:
    """True when a fitted value sits within rel_tol of the bound range at either end."""
    tol = rel_tol * max(hi - lo, 1e-12)
    return bool(value <= lo + tol or value >= hi - tol)


def _result_row(result: dict) -> dict:
    """Flat dict of per-period metrics, parameters and bound flags for one fit."""
    row = {"model": result["model"]}
    for p in PERIODS:
        for k in METRIC_KEYS:
            row[f"{k}_{p}"] = result["metrics"][p][k]
    for name, val, lo, hi in zip(result["param_names"], result["params"],
                                 result["lower"], result["upper"]):
        row[name] = float(val)
        row[f"{name}_at_bound"] = _at_bound(float(val), float(lo), float(hi))
    return row


VARIANT_ORDER = ["base", "filtered", "base_tz", "filtered_tz"]
VARIANT_COLORS = {
    "base": "#2980b9",        # blue
    "filtered": "#27ae60",    # green
    "base_tz": "#e74c3c",     # red
    "filtered_tz": "#f39c12", # orange
}
MODEL_IDS = {"base": "M1", "filtered": "M2", "base_tz": "M3", "filtered_tz": "M4"}
MODEL_LABELS = {
    "base":        "M1: Direct",
    "filtered":    "M2: Filtered",
    "base_tz":     "M3: Direct, z(t)",
    "filtered_tz": "M4: Filtered, z(t)",
}
PERIOD_LABELS = {"cal": "Calibration", "sel": "Selection", "test": "Test"}
PERIOD_COLORS = {"cal": "#1a6faf", "sel": "#e08214", "test": "#c94040"}
PERIOD_SHADES = {"cal": "#2166ac", "sel": "#fdb863", "test": "#d6604d"}
METRIC_BOX = dict(boxstyle="round,pad=0.35", facecolor="white",
                  edgecolor="#4d4d4d", linewidth=0.8, alpha=0.92)


def _period_spans(result: dict) -> dict:
    """Index ranges of each period within the full record."""
    n = len(result["time_index_full"])
    return {"cal": (0, result["n_cal"]),
            "sel": (result["n_cal"], result["n_sel_end"]),
            "test": (result["n_sel_end"], n)}


def _shade_periods(ax, result: dict) -> None:
    """Shade the three periods and mark the two boundaries."""
    time_full = result["time_index_full"]
    t_end = time_full[-1] + pd.Timedelta(days=1)
    edges = [time_full[0], pd.Timestamp(CAL_END) + pd.Timedelta(days=1),
             pd.Timestamp(SEL_END) + pd.Timedelta(days=1), t_end]
    for p, x0, x1 in zip(PERIODS, edges[:-1], edges[1:]):
        if x1 > x0:
            ax.axvspan(x0, x1, color=PERIOD_SHADES[p], alpha=0.08, zorder=0)
    for x in edges[1:3]:
        ax.axvline(x, color="black", ls="--", lw=1.2, alpha=0.9)
    ax.set_xlim(edges[0], t_end)
    ax.margins(x=0)


def _annotate_metrics(ax, result: dict) -> None:
    """One KGE_ab / RMSE box per period along the bottom of the axes."""
    for p, x in zip(PERIODS, (0.30, 0.62, 0.90)):
        m = result["metrics"][p]
        if not np.isfinite(m["rmse"]):
            continue
        ax.text(x, 0.05, f"{PERIOD_LABELS[p]}\nKGE = {m['kge_ab']:.2f}   RMSE = {m['rmse']:.2f} m",
                transform=ax.transAxes, fontsize=9, va="bottom", ha="center",
                color="black", bbox=METRIC_BOX)


def _format_time_axis(ax) -> None:
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(True, lw=0.3, alpha=0.4)


def _plot_station(model_results: list, best_model: dict, station_label: str,
                  group_name: str, rf_label: str, amp_label: str, output_root: Path) -> None:
    """Per-variant, comparison and best-model figures for one station."""
    setup_font()

    for result in model_results:
        variant_dir = output_root / "figures" / result["model"]
        variant_dir.mkdir(parents=True, exist_ok=True)
        fig_v, ax_v = plt.subplots(figsize=(11, 5))
        _shade_periods(ax_v, result)
        ax_v.plot(result["time_index_full"], result["h_obs_full"], color="black", lw=1.0,
                  alpha=0.9, label="Observed")
        for p, (i0, i1) in _period_spans(result).items():
            if i1 > i0:
                ax_v.plot(result["time_index_full"][i0:i1], result["y_full"][i0:i1],
                          color=PERIOD_COLORS[p], lw=1.5, alpha=0.95, label=PERIOD_LABELS[p])
        _annotate_metrics(ax_v, result)
        ax_v.set_title(f"{station_label} ({group_name}) — Model Type: {MODEL_IDS[result['model']]}",
                       fontsize=14, fontweight="bold", pad=6)
        ax_v.set_ylabel("Groundwater level (m)", fontsize=12, fontweight="bold")
        ax_v.set_xlabel("Year", fontsize=12, fontweight="bold")
        _format_time_axis(ax_v)
        ax_v.legend(fontsize=9, loc="upper right", framealpha=0.85)
        fig_v.tight_layout()
        rklib_savefig(fig_v, variant_dir / f"gw_fit_{station_label}.png")
        plt.close(fig_v)

    # Comparison overlay: every variant over the full record, metrics table below.
    compare_dir = output_root / "figures" / "comparison"
    compare_dir.mkdir(parents=True, exist_ok=True)
    fig_c, (ax_ts, ax_tbl) = plt.subplots(
        2, 1, figsize=(11, 7), gridspec_kw={"height_ratios": [4, 1], "hspace": 0.35},
    )
    _shade_periods(ax_ts, best_model)
    ax_ts.plot(best_model["time_index_full"], best_model["h_obs_full"], color="black", lw=1.3,
               alpha=0.95, label="Observed", zorder=3)
    by_model = {r["model"]: r for r in model_results}
    for model_id in VARIANT_ORDER:
        result = by_model.get(model_id)
        if result is None:
            continue
        is_best = result is best_model
        ax_ts.plot(result["time_index_full"], result["y_full"], color=VARIANT_COLORS[model_id],
                   lw=1.9 if is_best else 1.3, alpha=1.0 if is_best else 0.75,
                   label=MODEL_IDS[model_id])
    ax_ts.set_title(f"{station_label} ({group_name}) — Model comparison",
                    fontsize=14, fontweight="bold", pad=6)
    ax_ts.set_ylabel("Groundwater level (m)", fontsize=12, fontweight="bold")
    ax_ts.set_xlabel("Year", fontsize=12, fontweight="bold")
    _format_time_axis(ax_ts)
    ax_ts.legend(fontsize=9, loc="upper right", framealpha=0.85, ncol=1)

    ax_tbl.axis("off")
    col_labels = ["Model", "KGE cal", "KGE sel", "KGE test", "RMSE cal", "RMSE sel", "RMSE test"]
    table_rows = []
    for model_id in VARIANT_ORDER:
        result = by_model.get(model_id)
        if result is None:
            continue
        m = result["metrics"]
        table_rows.append(
            [MODEL_LABELS[model_id] + (" *" if result is best_model else "")]
            + [f"{m[p]['kge_ab']:.2f}" if np.isfinite(m[p]["kge_ab"]) else "—" for p in PERIODS]
            + [f"{m[p]['rmse']:.2f}" if np.isfinite(m[p]["rmse"]) else "—" for p in PERIODS]
        )
    tbl = ax_tbl.table(cellText=table_rows, colLabels=col_labels, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.3)
    for j in range(len(col_labels)):
        tbl[(0, j)].set_facecolor("#e8e8e8")
        tbl[(0, j)].set_text_props(weight="bold")
    fig_c.tight_layout()
    rklib_savefig(fig_c, compare_dir / f"gw_compare_{station_label}.png")
    plt.close(fig_c)

    # Best-model 3-panel figure: heads, rainfall, diurnal amplitude.
    full_dir = output_root / "figures" / "full_subplots"
    full_dir.mkdir(parents=True, exist_ok=True)
    time_full = best_model["time_index_full"]
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True,
                             gridspec_kw={"height_ratios": [2.5, 1, 1]})
    ax0 = axes[0]
    _shade_periods(ax0, best_model)
    ax0.plot(time_full, best_model["h_obs_full"], color="black", lw=1.0, alpha=0.9, label="Observed")
    for p, (i0, i1) in _period_spans(best_model).items():
        if i1 > i0:
            ax0.plot(time_full[i0:i1], best_model["y_full"][i0:i1], color=PERIOD_COLORS[p],
                     lw=1.5, alpha=0.95, label=PERIOD_LABELS[p])
    _annotate_metrics(ax0, best_model)
    ax0.set_title(f"{station_label} ({group_name}) — Model Type: {MODEL_IDS[best_model['model']]} (best)",
                  fontsize=14, fontweight="bold", pad=6)
    ax0.set_ylabel("Groundwater level (m)", fontsize=12, fontweight="bold")
    ax0.legend(fontsize=9, loc="upper right", framealpha=0.85)
    ax0.grid(True, lw=0.3, alpha=0.4)

    ax1 = axes[1]
    ax1.bar(time_full, best_model["rainfall_full"], width=1.0, color="C0", alpha=0.8)
    ax1.set_title(f"Rainfall — {rf_label}", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Rainfall (mm/day)", fontsize=11, fontweight="bold")
    ax1.grid(True, lw=0.3, alpha=0.4)

    ax2 = axes[2]
    ax2.plot(time_full, best_model["amp_full"], color="C2", lw=1.0, alpha=0.85)
    ax2.set_title(f"Diurnal amplitude — {amp_label}", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Amplitude (m)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Year", fontsize=12, fontweight="bold")
    _format_time_axis(ax2)

    for ax, lbl in zip(axes, ["a", "b", "c"]):
        add_panel_label(ax, lbl, x=0.01, y=1.02)
    fig.tight_layout(h_pad=1.5)
    rklib_savefig(fig, full_dir / f"gw_fit_{station_label}.png")
    plt.close(fig)


def parse_fixed(spec) -> dict:
    """Parse 'name=value;name=value' into {name: float}; empty/NaN gives {}."""
    if spec is None or (isinstance(spec, float) and np.isnan(spec)) or not str(spec).strip():
        return {}
    out = {}
    for item in str(spec).split(";"):
        if item.strip():
            name, value = item.split("=")
            out[name.strip()] = float(value)
    return out


def estimate_lags_calibration(df_merge: pd.DataFrame, no_upstream: bool, max_lag: int) -> Tuple[int, int]:
    """Rainfall and upstream lags from the calibration period only."""
    df_cal = calibration_slice(df_merge)
    rain_lag = estimate_rain_lag(df_cal["gwl"].values, df_cal["rf"].values, max_lag=max_lag)
    if no_upstream:
        return rain_lag, 0
    up_lag = estimate_upstream_lag(df_cal["gwl"].values, df_cal["ups_gwl"].values, max_lag=max_lag)
    return rain_lag, up_lag


def run_station(args_params: dict) -> None:
    """Run the full per-station pipeline (data prep → fit → select → plot → save).

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
    max_lag = max(45, int(args.get('lag_days', 0)))

    est_rain_lag, est_up_lag = estimate_lags_calibration(df_merge, no_upstream, max_lag)
    rain_lag_days = int(args['rain_lag_days']) if args.get('rain_lag_days') is not None else est_rain_lag
    up_lag_days = 0 if no_upstream else (
        int(args['ups_lag_days']) if args.get('ups_lag_days') is not None else est_up_lag
    )

    model_list = [m.strip() for m in args.get("models", ",".join(VARIANT_ORDER)).split(",")]
    seed = int(args.get("seed", 42))
    fixed = parse_fixed(args.get("fixed", ""))
    model_results = []
    for model_name in model_list:
        result = _fit_model(
            df_merge,
            group_name=group_name,
            rain_lag_days=rain_lag_days,
            up_lag_days=up_lag_days,
            no_upstream=no_upstream,
            model_name=model_name,
            seed=seed,
            fixed=fixed,
        )
        if result is not None:
            model_results.append(result)

    if not model_results:
        print("No successful fit found for any model.")
        sys.exit(1)

    # Variant choice uses the selection period only (see selection_score).
    best_model = max(model_results, key=selection_score)
    selection_basis = "sel" if np.isfinite(best_model["metrics"]["sel"]["kge_ab"]) else "cal"

    station_label = args.get('st_id', args.get('gw_st', 'unknown'))
    if os.environ.get("GW_SKIP_PLOTS", "") != "1":
        _plot_station(model_results, best_model, station_label, group_name,
                      rf_label=args.get('rf_id', ''),
                      amp_label=args.get('gw_st', args.get('st_id', '')),
                      output_root=output_root)

    # --- Save results ---
    n_full = len(best_model["time_index_full"])
    meta = {
        'st_id': station_label,
        'gw_st': args.get('gw_st'),
        'ups_id': args.get('ups_id'),
        'rf_id': args.get('rf_id'),
        'group_name': group_name,
        'rain_lag_days': rain_lag_days,
        'up_lag_days': up_lag_days,
        'selection_basis': selection_basis,
        'n_cal': best_model["n_cal"],
        'n_sel': best_model["n_sel_end"] - best_model["n_cal"],
        'n_test': n_full - best_model["n_sel_end"],
        'stft_align': STFT_ALIGN,
        'seed': seed,
        'fixed': args.get("fixed", "") or "",
    }

    results = {**meta, **_result_row(best_model)}
    for name, std in best_model['param_std'].items():
        results[name] = std
    per_station_dir = output_root / "per_station"
    per_station_dir.mkdir(parents=True, exist_ok=True)
    per_station_path = per_station_dir / f"{station_label}.csv"
    pd.DataFrame([results]).to_csv(per_station_path, index=False)

    all_variant_rows = [
        {'st_id': station_label, 'selected': r is best_model, **_result_row(r)}
        for r in model_results
    ]
    all_variants_dir = output_root / "all_variants"
    all_variants_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_variant_rows).to_csv(all_variants_dir / f"{station_label}.csv", index=False)

    # Daily observed and simulated heads for every variant, for post-hoc analyses.
    period_label = np.empty(n_full, dtype=object)
    for p, (i0, i1) in _period_spans(best_model).items():
        period_label[i0:i1] = p
    series = pd.DataFrame({
        "date": best_model["time_index_full"],
        "period": period_label,
        "obs": best_model["h_obs_full"],
        "rf": best_model["rainfall_full"],
    })
    for r in model_results:
        series[f"sim_{r['model']}"] = r["y_full"]
    series_dir = output_root / "series"
    series_dir.mkdir(parents=True, exist_ok=True)
    series.to_csv(series_dir / f"{station_label}.csv", index=False)

    print(f"Per-station results saved → {per_station_path}")


if __name__ == "__main__":
    run_station(dict(argv_phrase(sys.argv[1:])))