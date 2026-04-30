"""Resampling and sentinel helpers for LS API time series."""
from __future__ import annotations
import pandas as pd
from subsidence.api_constants import RAINFALL_NEG_REPLACE


def clip_negative_to_zero(s: pd.Series) -> pd.Series:
    """Replace any negative value (including sentinel ``-998``) with 0.0."""
    return s.where(s >= 0, RAINFALL_NEG_REPLACE)


def daily_mean(s: pd.Series) -> pd.Series:
    """Aggregate a higher-frequency series to daily mean (NaN-aware)."""
    return s.resample("1D").mean()


def daily_sum(s: pd.Series) -> pd.Series:
    """Aggregate a higher-frequency series to daily sum (treat NaN as 0)."""
    return s.fillna(0.0).resample("1D").sum()


def monthly_to_native(s: pd.Series) -> pd.Series:
    """Pass-through for native-cadence (monthly) MLCW data; documents intent."""
    return s
