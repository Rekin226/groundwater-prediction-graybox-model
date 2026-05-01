"""Thin wrapper around srcs.gw_subroutine.

Selects the right simulator function based on (model_name, group_name) and
forwards the parameter dict in the order each underlying simulator expects.
The wrapper exists so the subsidence pipeline never has to know the exact
ordering of arguments inside srcs.gw_subroutine.

The wrapper does NOT modify srcs.gw_subroutine in any way.

ARCHITECTURE NOTE
-----------------
All 8 simulators in srcs.gw_subroutine accept:
  fn(params: tuple, t, rainfall, amp[, amt], h_up, h0[, doy])

where ``params`` is a *positional tuple* whose order is variant-specific.
This adapter unpacks the caller-supplied ``params`` dict into the correct
tuple order for each variant.

COASTAL h_sea DEVIATION
-----------------------
For coastal variants the underlying simulator uses a scalar ``h_sea``
(the mean sea level constant) embedded inside the ``params`` tuple — it is
NOT the time-varying ``h_sea`` array the pipeline might carry. Callers must
pass the scalar mean sea level as ``params["h_sea"]``.  The ``h_sea``
positional array accepted by :func:`simulate` is unused for coastal variants
(it exists so callers have a uniform interface); pass ``None`` or zeros.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np

# Imports from srcs/ — production GW model functions, never modified.
from srcs.gw_subroutine import (
    simulate_coastal,
    simulate_coastal_filtered,
    simulate_coastal_filtered_tz,
    simulate_coastal_tz,
    simulate_inland,
    simulate_inland_filtered,
    simulate_inland_filtered_tz,
    simulate_inland_tz,
)

_TABLE: Dict[tuple, Callable] = {
    ("base",        "inland"):  simulate_inland,
    ("filtered",    "inland"):  simulate_inland_filtered,
    ("base_tz",     "inland"):  simulate_inland_tz,
    ("filtered_tz", "inland"):  simulate_inland_filtered_tz,
    ("base",        "coastal"): simulate_coastal,
    ("filtered",    "coastal"): simulate_coastal_filtered,
    ("base_tz",     "coastal"): simulate_coastal_tz,
    ("filtered_tz", "coastal"): simulate_coastal_filtered_tz,
}


def select_simulator(model_name: str, group_name: str) -> Callable:
    """Return the underlying srcs.gw_subroutine simulator for the given combo."""
    try:
        return _TABLE[(model_name, group_name)]
    except KeyError:
        raise ValueError(f"unknown simulator: {model_name=} {group_name=}")


def _require(params: dict, key: str, variant: str) -> float:
    """Pull a required key from params, raising a clear error if absent."""
    try:
        return params[key]
    except KeyError:
        raise KeyError(
            f"params is missing '{key}' required for variant '{variant}'. "
            f"Available keys: {sorted(params.keys())}"
        )


def simulate(
    *,
    model_name: str,
    group_name: str,
    h0: float,
    params: dict,
    rain: np.ndarray,
    h_up: np.ndarray,
    amp: np.ndarray,
    amt: np.ndarray,
    h_sea,  # scalar or array; only the scalar mean is used for coastal variants
    doy: np.ndarray,
    t_abs: np.ndarray,
) -> np.ndarray:
    """Forward to the underlying srcs.gw_subroutine.simulate_*.

    Parameters
    ----------
    model_name : str
        One of ``"base"``, ``"filtered"``, ``"base_tz"``, ``"filtered_tz"``.
    group_name : str
        One of ``"inland"``, ``"coastal"``.
    h0 : float
        Initial head value.
    params : dict
        Model parameters keyed by name.  Required keys depend on variant:

        - All variants: ``a``, ``b``, ``c``, ``k_link``, ``tau_rain``,
          ``tau_up``, ``d_sin``, ``d_cos``.
        - ``base`` / ``filtered``: ``z`` (scalar equilibrium head).
        - ``base_tz`` / ``filtered_tz``: ``z0``, ``z1`` (trend parameters).
        - ``filtered`` / ``filtered_tz``: ``lam`` (low-pass filter rate).
        - Coastal only: ``k_sgd``, ``gamma``, ``h_sea`` (scalar mean sea level).

    rain : np.ndarray
        Daily rainfall array, shape (n,).
    h_up : np.ndarray
        Upstream head array, shape (n,).
    amp : np.ndarray
        Amplitude/pumping array, shape (n,).
    amt : np.ndarray
        Tidal amplitude array (coastal only), shape (n,).
    h_sea : scalar or array
        For coastal variants, pass the scalar mean sea level constant (also
        available as ``params["h_sea"]``); the value here is ignored because
        the coastal simulator reads it from the params tuple.  For inland
        variants this argument is not used at all.
    doy : np.ndarray
        Day-of-year array, shape (n,).
    t_abs : np.ndarray
        Absolute day-index array, shape (n,).  Required by ``_tz`` variants
        so that z(t) = z0 + z1*(t/365.25) is evaluated at absolute time.

    Returns
    -------
    np.ndarray
        Simulated head array, shape (n,).
    """
    fn = select_simulator(model_name, group_name)
    variant = f"{model_name}/{group_name}"

    # ------------------------------------------------------------------
    # Shared parameters (all 8 variants)
    # ------------------------------------------------------------------
    a        = _require(params, "a",        variant)
    b        = _require(params, "b",        variant)
    c        = _require(params, "c",        variant)
    k_link   = _require(params, "k_link",   variant)
    tau_rain = _require(params, "tau_rain", variant)
    tau_up   = _require(params, "tau_up",   variant)
    d_sin    = _require(params, "d_sin",    variant)
    d_cos    = _require(params, "d_cos",    variant)

    # ------------------------------------------------------------------
    # Dispatch per (model_name, group_name)
    # Param-tuple order is taken verbatim from srcs/gw_subroutine.py.
    # ------------------------------------------------------------------

    # ---- inland / base -----------------------------------------------
    # params: (a, z, b, c, k_link, tau_rain, tau_up, d_sin, d_cos)
    if model_name == "base" and group_name == "inland":
        z = _require(params, "z", variant)
        p = (a, z, b, c, k_link, tau_rain, tau_up, d_sin, d_cos)
        return fn(params=p, t=t_abs, rainfall=rain, amp=amp,
                  h_up=h_up, h0=h0, doy=doy)

    # ---- inland / filtered -------------------------------------------
    # params: (a, z, b, c, k_link, lam, tau_rain, tau_up, d_sin, d_cos)
    elif model_name == "filtered" and group_name == "inland":
        z   = _require(params, "z",   variant)
        lam = _require(params, "lam", variant)
        p = (a, z, b, c, k_link, lam, tau_rain, tau_up, d_sin, d_cos)
        return fn(params=p, t=t_abs, rainfall=rain, amp=amp,
                  h_up=h_up, h0=h0, doy=doy)

    # ---- inland / base_tz --------------------------------------------
    # params: (a, z0, z1, b, c, k_link, tau_rain, tau_up, d_sin, d_cos)
    elif model_name == "base_tz" and group_name == "inland":
        z0 = _require(params, "z0", variant)
        z1 = _require(params, "z1", variant)
        p = (a, z0, z1, b, c, k_link, tau_rain, tau_up, d_sin, d_cos)
        return fn(params=p, t=t_abs, rainfall=rain, amp=amp,
                  h_up=h_up, h0=h0, doy=doy)

    # ---- inland / filtered_tz ----------------------------------------
    # params: (a, z0, z1, b, c, k_link, lam, tau_rain, tau_up, d_sin, d_cos)
    elif model_name == "filtered_tz" and group_name == "inland":
        z0  = _require(params, "z0",  variant)
        z1  = _require(params, "z1",  variant)
        lam = _require(params, "lam", variant)
        p = (a, z0, z1, b, c, k_link, lam, tau_rain, tau_up, d_sin, d_cos)
        return fn(params=p, t=t_abs, rainfall=rain, amp=amp,
                  h_up=h_up, h0=h0, doy=doy)

    # ---- coastal / base ----------------------------------------------
    # params: (a, z, b, c, k_link, k_sgd, gamma, h_sea, tau_rain, tau_up,
    #          d_sin, d_cos)
    elif model_name == "base" and group_name == "coastal":
        z       = _require(params, "z",     variant)
        k_sgd   = _require(params, "k_sgd", variant)
        gamma   = _require(params, "gamma", variant)
        h_sea_s = _require(params, "h_sea", variant)  # scalar mean sea level
        p = (a, z, b, c, k_link, k_sgd, gamma, h_sea_s,
             tau_rain, tau_up, d_sin, d_cos)
        return fn(params=p, t=t_abs, rainfall=rain, amp=amp, amt=amt,
                  h_up=h_up, h0=h0, doy=doy)

    # ---- coastal / filtered ------------------------------------------
    # params: (a, z, b, c, k_link, k_sgd, gamma, h_sea, lam, tau_rain,
    #          tau_up, d_sin, d_cos)
    elif model_name == "filtered" and group_name == "coastal":
        z       = _require(params, "z",     variant)
        k_sgd   = _require(params, "k_sgd", variant)
        gamma   = _require(params, "gamma", variant)
        h_sea_s = _require(params, "h_sea", variant)
        lam     = _require(params, "lam",   variant)
        p = (a, z, b, c, k_link, k_sgd, gamma, h_sea_s, lam,
             tau_rain, tau_up, d_sin, d_cos)
        return fn(params=p, t=t_abs, rainfall=rain, amp=amp, amt=amt,
                  h_up=h_up, h0=h0, doy=doy)

    # ---- coastal / base_tz -------------------------------------------
    # params: (a, z0, z1, b, c, k_link, k_sgd, gamma, h_sea, tau_rain,
    #          tau_up, d_sin, d_cos)
    elif model_name == "base_tz" and group_name == "coastal":
        z0      = _require(params, "z0",    variant)
        z1      = _require(params, "z1",    variant)
        k_sgd   = _require(params, "k_sgd", variant)
        gamma   = _require(params, "gamma", variant)
        h_sea_s = _require(params, "h_sea", variant)
        p = (a, z0, z1, b, c, k_link, k_sgd, gamma, h_sea_s,
             tau_rain, tau_up, d_sin, d_cos)
        return fn(params=p, t=t_abs, rainfall=rain, amp=amp, amt=amt,
                  h_up=h_up, h0=h0, doy=doy)

    # ---- coastal / filtered_tz ---------------------------------------
    # params: (a, z0, z1, b, c, k_link, k_sgd, gamma, h_sea, lam, tau_rain,
    #          tau_up, d_sin, d_cos)
    elif model_name == "filtered_tz" and group_name == "coastal":
        z0      = _require(params, "z0",    variant)
        z1      = _require(params, "z1",    variant)
        k_sgd   = _require(params, "k_sgd", variant)
        gamma   = _require(params, "gamma", variant)
        h_sea_s = _require(params, "h_sea", variant)
        lam     = _require(params, "lam",   variant)
        p = (a, z0, z1, b, c, k_link, k_sgd, gamma, h_sea_s, lam,
             tau_rain, tau_up, d_sin, d_cos)
        return fn(params=p, t=t_abs, rainfall=rain, amp=amp, amt=amt,
                  h_up=h_up, h0=h0, doy=doy)

    else:
        # Should not reach here — select_simulator would have raised first.
        raise ValueError(f"unhandled variant: {model_name=} {group_name=}")
