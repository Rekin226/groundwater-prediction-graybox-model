import numpy as np
from subsidence.sub_subroutine import simulate_form2


def _step_h(n=1000, h_high=5.0, h_low=2.0, drop_at=200):
    h = np.full(n, h_high, dtype=float)
    h[drop_at:] = h_low
    return h


def test_form2_zero_when_h_above_href():
    h = np.full(100, 10.0)
    z = simulate_form2(h, t_years=np.arange(100)/365.25,
                       Sk_e=1e-3, Sk_v=1e-2, h_ref=5.0, v_tect=0.0,
                       smooth_max=False)
    assert np.allclose(z, 0.0)


def test_form2_elastic_recovers_when_h_returns():
    """Drop h, then return: cumulative ζ should partially recover (Sk_e component)."""
    n = 1000
    h = np.full(n, 5.0)
    h[200:600] = 2.0  # drop, then recover
    t = np.arange(n) / 365.25
    z = simulate_form2(h, t_years=t, Sk_e=1e-2, Sk_v=0.0,
                       h_ref=5.0, v_tect=0.0, smooth_max=False)
    # During the drop, ζ > 0; after recovery to 5.0, ζ returns to 0
    assert z[400] > 0
    assert abs(z[-1]) < 1e-9


def test_form2_inelastic_does_not_recover():
    n = 1000
    h = np.full(n, 5.0)
    h[200:600] = 2.0
    z = simulate_form2(h, t_years=np.arange(n)/365.25,
                       Sk_e=0.0, Sk_v=1e-2, h_ref=5.0, v_tect=0.0,
                       smooth_max=False)
    # After head recovers, inelastic component remains
    assert z[-1] > 0


def test_form2_tectonic_linear_in_time():
    n = 365
    h = np.full(n, 10.0)
    z = simulate_form2(h, t_years=np.arange(n)/365.25,
                       Sk_e=0.0, Sk_v=0.0, h_ref=5.0, v_tect=-0.01,
                       smooth_max=False)
    # v_tect = -0.01 m/yr * 1 yr = -0.01 m at t=1
    assert abs(z[-1] - (-0.01)) < 5e-4


def test_form2_smooth_max_close_to_hard_max():
    n = 200
    h = np.linspace(8, 2, n)
    t = np.arange(n) / 365.25
    args = dict(t_years=t, Sk_e=1e-3, Sk_v=1e-2, h_ref=5.0, v_tect=0.0)
    z_hard = simulate_form2(h, smooth_max=False, **args)
    z_soft = simulate_form2(h, smooth_max=True, **args)
    assert np.max(np.abs(z_hard - z_soft)) < 0.02  # 2 cm tolerance for β=50
