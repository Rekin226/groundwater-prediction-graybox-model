"""Resampling and sentinel helpers for LS API time series."""
from __future__ import annotations
import pandas as pd
from subsidence.api_constants import RAINFALL_NEG_REPLACE


def clip_negative_to_zero(s: pd.Series) -> pd.Series:
    """Replace any negative value (including sentinel ``-998``) with 0.0."""
    return s.where(s >= 0, RAINFALL_NEG_REPLACE)


def mask_sentinels(s: pd.Series, threshold: float = -50.0) -> pd.Series:
    """Replace values below *threshold* with NaN.

    GW levels in the Zhuoshui Alluvial Fan range roughly from -30 m (deep
    coastal aquifers) to +50 m (inland highlands).  Any value below -50 m is
    a WiseEnvr API sentinel (e.g. -999998, -9999, -998) and should be treated
    as missing rather than a real measurement.  We use NaN so that
    ``daily_mean`` silently skips the bad readings via ``resample().mean()``.
    """
    return s.where(s >= threshold, other=float("nan"))


def daily_mean(s: pd.Series) -> pd.Series:
    """Aggregate a higher-frequency series to daily mean (NaN-aware)."""
    return s.resample("1D").mean()


def daily_sum(s: pd.Series) -> pd.Series:
    """Aggregate a higher-frequency series to daily sum (treat NaN as 0)."""
    return s.fillna(0.0).resample("1D").sum()


def monthly_to_native(s: pd.Series) -> pd.Series:
    """Pass-through for native-cadence (monthly) MLCW data; documents intent."""
    return s
