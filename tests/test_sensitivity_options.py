"""Optional integrator and input transforms used by sensitivity runs."""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "srcs"))
sys.path.insert(0, str(REPO.parent))

import gw_shell  # noqa: E402
import gw_subroutine as sub  # noqa: E402

N = 200
T = np.arange(N, dtype=float)
ZEROS = np.zeros(N)


def _recession(monkeypatch, integrator, a=1.5):
    monkeypatch.setenv("GW_INTEGRATOR", integrator)
    params = (a, 0.0, 0.0, 0.0, 0.0, 5.0, 5.0, 0.0, 0.0)  # a, z, b, c, k_link, taus, seasonal
    return sub.simulate_inland(params, T, ZEROS, ZEROS, ZEROS, h0=1.0, doy=np.full(N, 180.0))


def test_explicit_is_default_and_overshoots(monkeypatch):
    monkeypatch.delenv("GW_INTEGRATOR", raising=False)
    params = (1.5, 0.0, 0.0, 0.0, 0.0, 5.0, 5.0, 0.0, 0.0)
    h = sub.simulate_inland(params, T, ZEROS, ZEROS, ZEROS, h0=1.0, doy=np.full(N, 180.0))
    assert h[1] == -0.5  # 1 - 1.5 * 1: crosses the equilibrium


def test_implicit_is_monotone(monkeypatch):
    h = _recession(monkeypatch, "implicit")
    assert np.isclose(h[1], 1.0 / 2.5)
    assert np.all(np.diff(h) <= 0) and np.all(h >= 0)


def test_ups_offset_removes_calibration_mean_difference(monkeypatch):
    monkeypatch.setenv("GW_UPS_OFFSET", "cal")
    rng = np.random.default_rng(0)
    h_obs = 10 + rng.normal(size=N)
    h_up = 50 + rng.normal(size=N)
    _, h_up2 = gw_shell.transform_inputs(ZEROS, h_up, h_obs, 100, no_upstream=False)
    assert np.isclose(h_up2[:100].mean(), h_obs[:100].mean())
    assert np.allclose(np.diff(h_up2), np.diff(h_up))


def test_dryness_scaling_is_identity_in_first_year_and_damps_dry_spells(monkeypatch):
    monkeypatch.setenv("GW_RAIN_DRYNESS_BETA", "1")
    n = 1200
    rain = np.full(n, 4.0)
    rain[800:] = 1.0  # dry spell after the calibration window
    r2, _ = gw_shell.transform_inputs(rain, np.zeros(n), np.zeros(n), 700, no_upstream=True)
    assert np.allclose(r2[:365], rain[:365])
    assert r2[-1] < rain[-1]
