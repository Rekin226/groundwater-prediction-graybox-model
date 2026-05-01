import numpy as np
from subsidence.gw_simulator_adapter import select_simulator, simulate


def test_select_simulator_returns_callable():
    fn = select_simulator("base", group_name="inland")
    assert callable(fn)


def test_simulate_runs_for_inland_base(monkeypatch):
    # Smoke: build minimal inputs and check result shape.
    n = 30
    rain = np.ones(n) * 5.0
    h_up = np.zeros(n) + 10.0
    amp = np.zeros(n)
    amt = np.zeros(n)
    h_sea = np.zeros(n)
    doy = np.arange(n) % 365
    params = dict(a=0.05, z=10.0, b=0.001, c=0.0, k_link=0.1,
                  tau_rain=10.0, tau_up=10.0, d_sin=0.0, d_cos=0.0)
    out = simulate(model_name="base", group_name="inland",
                   h0=10.0, params=params,
                   rain=rain, h_up=h_up, amp=amp, amt=amt, h_sea=h_sea, doy=doy,
                   t_abs=np.arange(n))
    assert out.shape == (n,)
    assert np.isfinite(out).all()
