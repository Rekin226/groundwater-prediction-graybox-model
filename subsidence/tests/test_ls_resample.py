import numpy as np
import pandas as pd
from subsidence.ls_resample import (
    clip_negative_to_zero,
    daily_mean,
    daily_sum,
    mask_sentinels,
    monthly_to_native,
)


def test_clip_negative_replaces_only_negatives():
    s = pd.Series([1.0, -998.0, 2.0, -1.0, 0.0])
    out = clip_negative_to_zero(s)
    assert list(out) == [1.0, 0.0, 2.0, 0.0, 0.0]


def test_daily_mean_resamples_10min_to_daily():
    idx = pd.date_range("2020-01-01", "2020-01-03 00:00", freq="10min")
    s = pd.Series(np.arange(len(idx), dtype=float), index=idx)
    out = daily_mean(s)
    assert len(out) == 3  # Jan 1, Jan 2, Jan 3 (last bucket may have only 1 sample)
    # First day mean is mean of 144 samples 0..143 = 71.5
    assert abs(out.iloc[0] - 71.5) < 1e-6


def test_daily_sum_resamples_rainfall():
    idx = pd.date_range("2020-01-01", "2020-01-01 23:50", freq="10min")
    s = pd.Series(np.ones(len(idx)), index=idx)
    out = daily_sum(s)
    assert out.iloc[0] == 144  # 144 ten-minute buckets


def test_mask_sentinels_replaces_below_threshold_with_nan():
    """Sentinel values (e.g. -999998) become NaN; legitimate negatives survive."""
    s = pd.Series([5.0, -10.0, -999998.0, -9999.0, -998.0, 0.0, 20.0])
    out = mask_sentinels(s, threshold=-50.0)
    # Values ≥ -50 stay; those below become NaN
    assert out[0] == 5.0
    assert out[1] == -10.0   # legitimate negative (e.g. -10 m below sea level)
    assert np.isnan(out[2])  # -999998 → NaN
    assert np.isnan(out[3])  # -9999   → NaN
    assert np.isnan(out[4])  # -998    → NaN
    assert out[5] == 0.0
    assert out[6] == 20.0


def test_mask_sentinels_default_threshold():
    """Default threshold of -50 filters WiseEnvr sentinels, not deep aquifer values."""
    s = pd.Series([-30.0, -50.0, -50.1, -999998.0])
    out = mask_sentinels(s)
    assert out[0] == -30.0   # valid: -30 m is within physical range
    assert out[1] == -50.0   # boundary: exactly at threshold, kept
    assert np.isnan(out[2])  # just below threshold → NaN
    assert np.isnan(out[3])  # classic sentinel → NaN
