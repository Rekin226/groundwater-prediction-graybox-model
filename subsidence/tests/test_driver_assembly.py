import numpy as np
import pandas as pd
from subsidence.driver_assembly import assemble_h


def _series(values, start="2020-01-01"):
    idx = pd.date_range(start, periods=len(values), freq="1D")
    return pd.Series(values, index=idx, name="h_obs_m")


def test_observation_used_when_present():
    obs = _series([1.0, 2.0, np.nan, np.nan, 5.0])
    sim = _series([10, 11, 12, 13, 14])
    out = assemble_h(obs, sim, max_interp_gap_days=7, taper_days=0)
    # First two days = obs
    assert out["h_driver"].iloc[0] == 1.0
    assert out["h_driver"].iloc[1] == 2.0
    # Last day = obs
    assert out["h_driver"].iloc[-1] == 5.0


def test_short_gap_linear_interpolated():
    # 2-day NaN gap inside the obs series, threshold = 7 days
    obs = _series([1.0, np.nan, np.nan, 4.0])
    sim = _series([10, 11, 12, 13])
    out = assemble_h(obs, sim, max_interp_gap_days=7, taper_days=0)
    assert abs(out["h_driver"].iloc[1] - 2.0) < 0.01  # linear interp
    assert (out["driver_source"] == "linear_interp").iloc[1]


def test_long_gap_uses_simulated():
    obs = _series([1.0] + [np.nan]*15 + [5.0])
    sim = _series(np.linspace(10, 25, 17))
    out = assemble_h(obs, sim, max_interp_gap_days=7, taper_days=0)
    # Middle days should be model_fill
    assert (out["driver_source"].iloc[5:12] == "model_fill").all()


def test_qc_bias_at_edges_recorded():
    obs = _series([1.0] + [np.nan]*15 + [5.0])
    sim = _series([10.0]*17)
    out = assemble_h(obs, sim, max_interp_gap_days=7, taper_days=0)
    # bias = obs - sim at the boundary days = -9 and -5
    assert "edge_bias_mean" in out.attrs
