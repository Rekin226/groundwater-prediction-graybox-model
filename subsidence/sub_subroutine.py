"""Subsidence ODE simulators — Form 2 (Riley/IBS) and Form 3 (with τ delay).

All simulators consume daily-indexed arrays and return cumulative ζ(t).
Time is integrated in years for rate parameters; integrator step is 1 day.
"""
from __future__ import annotations
import numpy as np

# Smooth-max parameter (softplus stiffness); fixed, not fit
SMOOTH_BETA = 50.0


def _smooth_max_zero(x: np.ndarray, beta: float = SMOOTH_BETA) -> np.ndarray:
    """smooth approximation of max(x, 0): (1/β) log(1 + exp(β x))."""
    z = beta * x
    # numerically stable: log1p(exp(z)) = max(z,0) + log1p(exp(-|z|))
    return (np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(z)))) / beta


def _hard_max_zero(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def simulate_form2(h: np.ndarray, t_years: np.ndarray,
                   Sk_e: float, Sk_v: float, h_ref: float, v_tect: float,
                   *, smooth_max: bool = False,
                   v_tect_linear: tuple[float, float] | None = None) -> np.ndarray:
    """Riley/IBS Form 2 (no aquitard delay).

    Parameters
    ----------
    h         : daily head time series (m)
    t_years   : daily t in years from t_0 (same length as h)
    Sk_e      : elastic skeletal storage (1/m)
    Sk_v      : inelastic skeletal storage (1/m)
    h_ref     : reference / preconsolidation head (m)
    v_tect    : tectonic linear trend (m/yr); used when v_tect_linear is None
    v_tect_linear : (v0, v1) tuple for v_tect(t) = v0 + v1·t (m/yr); overrides v_tect
    smooth_max : True → softplus; False → hard max (DE step)

    Returns
    -------
    zeta : cumulative ζ(t) (m), shape (len(h),), zeta[0] = 0.
    """
    mfn = _smooth_max_zero if smooth_max else _hard_max_zero
    n = len(h)

    # Build h_min_hist by running min
    h_min_hist = np.minimum.accumulate(h)

    b_e = Sk_e * mfn(h_ref - h)
    b_i = Sk_v * mfn(h_ref - h_min_hist)
    if v_tect_linear is not None:
        v0, v1 = v_tect_linear
        v_tect_arr = v0 + v1 * t_years
    else:
        v_tect_arr = v_tect * np.ones_like(t_years)
    # cumulative integral of v_tect over t (years): trapezoid on dt
    dt_years = np.diff(t_years, prepend=t_years[0])
    tect_cum = np.cumsum(v_tect_arr * dt_years)

    zeta = b_e + b_i + tect_cum
    # Anchor ζ(t_0) = 0
    zeta = zeta - zeta[0]
    return zeta
