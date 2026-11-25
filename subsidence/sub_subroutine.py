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
    # Inelastic (irreversible): tracks how far the historical minimum head
    # has dropped below the preconsolidation reference. h_min_hist is monotone
    # non-increasing, so b_i is monotone non-decreasing — once accrued, locked in.
    # See Hoffmann et al. (2003) eq. 2; Galloway & Burbey (2011) eq. 4.
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


def simulate_form3(h: np.ndarray, t_years: np.ndarray, *,
                   Sk_e: float, Sk_v: float, h_ref: float, v_tect: float,
                   tau_days: float,
                   smooth_max: bool = False,
                   v_tect_linear: tuple[float, float] | None = None) -> np.ndarray:
    """Form 3: Form 2 with aquitard hydrodynamic delay τ (days).

    dζ/dt = (1/τ) · [b_e + b_i + v_tect·t − ζ]   (Euler, dt = 1 day)

    Parameters
    ----------
    h, t_years, Sk_e, Sk_v, h_ref, v_tect, smooth_max, v_tect_linear
        See simulate_form2.
    tau_days : float
        Aquitard delay time constant (days). Must be ≥ 0.
        tau_days < 1 is treated as sub-timestep (instantaneous response,
        equivalent to Form 2). The Euler gain (1/τ) is clamped to ≤ 1
        for stability.

    Returns
    -------
    zeta : np.ndarray
        Cumulative compaction (m), same shape as h, anchored ζ(t_0) = 0.
    """
    target = simulate_form2(h, t_years=t_years, Sk_e=Sk_e, Sk_v=Sk_v,
                            h_ref=h_ref, v_tect=v_tect,
                            smooth_max=smooth_max,
                            v_tect_linear=v_tect_linear)
    n = len(h)
    zeta = np.zeros(n)
    # Clamp gain to [0, 1]: Euler with dt=1 day is stable only when inv_tau ≤ 1.
    # τ < 1 day is sub-timestep and treated as instantaneous (gain = 1).
    inv_tau = min(1.0 / max(tau_days, 1e-6), 1.0)
    for k in range(1, n):
        zeta[k] = zeta[k-1] + inv_tau * (target[k-1] - zeta[k-1])  # dt = 1 day
    return zeta
