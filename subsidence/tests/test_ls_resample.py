import numpy as np
import pandas as pd
from subsidence.ls_resample import (
    clip_negative_to_zero,
    daily_mean,
    daily_sum,
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
