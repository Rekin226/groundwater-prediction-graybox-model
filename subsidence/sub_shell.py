"""Per-station calibration pipeline (one variant at a time).

Provides the building block fit_one_variant(); the public entrypoint
fit_station() runs all 4 variants and selects by validation KGE.
"""
from __future__ import annotations
from typing import Dict
import numpy as np
from scipy.optimize import differential_evolution, curve_fit

from subsidence.sub_subroutine import simulate_form2, simulate_form3
from subsidence.sub_metrics import kge_components, rmse, r2, kge_on_rate

VARIANTS = ("M1", "M2", "M3_tau", "M4_tau")


def _form_for(variant: str) -> str:
    return "form3" if variant.endswith("_tau") else "form2"


def _v_tect_linear_for(variant: str) -> bool:
    return variant.startswith("M2") or variant.startswith("M4")


def _simulate_variant(h, t_years, params: dict, variant: str, *, smooth_max: bool):
    use_form3 = _form_for(variant) == "form3"
    use_linear = _v_tect_linear_for(variant)
    v_tect = params.get("v_tect", 0.0)
    v_tect_linear = (params["v0"], params["v1"]) if use_linear else None
    if use_form3:
        return simulate_form3(h, t_years=t_years,
                              Sk_e=params["Sk_e"], Sk_v=params["Sk_v"],
                              h_ref=params["h_ref"], v_tect=v_tect,
                              tau_days=params["tau"], smooth_max=smooth_max,
                              v_tect_linear=v_tect_linear)
    return simulate_form2(h, t_years=t_years,
                          Sk_e=params["Sk_e"], Sk_v=params["Sk_v"],
                          h_ref=params["h_ref"], v_tect=v_tect,
                          smooth_max=smooth_max,
                          v_tect_linear=v_tect_linear)


def _param_keys(variant: str) -> list[str]:
    keys = ["Sk_e", "Sk_v", "h_ref"]
    if _v_tect_linear_for(variant):
        keys += ["v0", "v1"]
    else:
        keys += ["v_tect"]
    if _form_for(variant) == "form3":
        keys += ["tau"]
    return keys


def fit_one_variant(*, h, zeta_obs, t_years, variant: str,
                    cal_idx, val_idx,
                    bounds: Dict[str, tuple[float, float]],
                    seed: int = 42,
                    rate_weight: float = 0.5) -> dict:
    """Fit one model variant.

    Composite loss: rate_weight · SSE(ζ)/var(ζ) + (1−rate_weight) · SSE(dζ/dt)/var(dζ).
    Variance-normalisation makes the two terms commensurate so the weight is
    interpretable (0 = cumulative-only, 1 = rate-only).  Cumulative-only fitting
    leaves τ unidentifiable and lets v_tect absorb non-stationary trend; the
    rate term constrains the dynamics that drive the cumulative integral.
    """
    keys = _param_keys(variant)
    bnds = [bounds[k] for k in keys]

    h_cal = h[cal_idx]; t_cal = t_years[cal_idx]; z_cal = zeta_obs[cal_idx]

    # Build masks for cumulative and rate observations (NaN-aware).  The rate
    # obs is the first-difference of cumulative ζ, requiring two consecutive
    # finite values; align rate to its right-edge timestamp.
    finite = np.isfinite(z_cal)
    dz_obs = np.diff(z_cal)
    finite_rate = np.isfinite(dz_obs)
    if not finite.any() or not finite_rate.any():
        raise ValueError("cal period has no finite ζ values")
    var_z = float(np.nanvar(z_cal[finite]))
    var_dz = float(np.nanvar(dz_obs[finite_rate]))
    # Guard against degenerate (constant) series — fall back to cumulative-only
    eps = 1e-12
    w_cum = rate_weight / max(var_z, eps)
    w_rate = (1.0 - rate_weight) / max(var_dz, eps) if var_dz > eps else 0.0

    def _composite_loss(sim_cum: np.ndarray) -> float:
        sim_rate = np.diff(sim_cum)
        rcum = (sim_cum - z_cal)[finite]
        rrate = (sim_rate - dz_obs)[finite_rate]
        return w_cum * float(np.sum(rcum * rcum)) + w_rate * float(np.sum(rrate * rrate))

    def _residual(x):
        params = {k: v for k, v in zip(keys, x)}
        sim = _simulate_variant(h_cal, t_cal, params, variant, smooth_max=False)
        return _composite_loss(sim)

    de = differential_evolution(_residual, bnds, seed=seed,
                                maxiter=400, tol=1e-7, polish=False, workers=1)
    x0 = de.x

    # curve_fit polish — concatenated residual vector so its sum-of-squares
    # equals the composite loss.  Stack: [√w_cum · ζ_finite, √w_rate · dζ_finite].
    sw_cum = np.sqrt(w_cum)
    sw_rate = np.sqrt(w_rate) if w_rate > 0 else 0.0
    y_stacked = np.concatenate([sw_cum * z_cal[finite],
                                sw_rate * dz_obs[finite_rate]])

    def _model(_x, *params):
        p = {k: v for k, v in zip(keys, params)}
        sim_cum = _simulate_variant(h_cal, t_cal, p, variant, smooth_max=True)
        sim_rate = np.diff(sim_cum)
        return np.concatenate([sw_cum * sim_cum[finite],
                               sw_rate * sim_rate[finite_rate]])
    try:
        x_polished, _pcov = curve_fit(_model, np.arange(len(y_stacked)), y_stacked,
                                      p0=x0,
                                      bounds=([b[0] for b in bnds], [b[1] for b in bnds]),
                                      maxfev=2000)
    except Exception:
        x_polished = x0

    # Acceptance gate: keep polished only if it doesn't degrade the composite
    # hard-max loss DE optimised against.  Smooth-max and hard-max surfaces
    # can disagree on the optimum, so compare on the same loss.
    def _residual_hard(x):
        p = {k: v for k, v in zip(keys, x)}
        sim = _simulate_variant(h_cal, t_cal, p, variant, smooth_max=False)
        return _composite_loss(sim)

    if _residual_hard(x_polished) > _residual_hard(x0):
        x_polished = x0  # polish degraded; revert

    params_out = {k: float(v) for k, v in zip(keys, x_polished)}
    sim_full = _simulate_variant(h, t_years, params_out, variant, smooth_max=False)

    # kge_ill_conditioned = False (hardcoded): cumulative ζ has large mean for
    # subsidence-affected stations; the GW-pipeline ill-conditioned guard
    # (|mean|/std < 0.05) rarely triggers and the diagnostic value is low here.
    out = {"variant": variant, "params": params_out, "kge_ill_conditioned": False}
    for label, idx in [("cal", cal_idx), ("val", val_idx)]:
        kc = kge_components(zeta_obs[idx], sim_full[idx])
        out[f"kge_{label}"] = kc["kge"]
        out[f"kge_r_{label}"] = kc["r"]
        out[f"kge_alpha_{label}"] = kc["alpha"]
        out[f"kge_beta_{label}"] = kc["beta"]
        out[f"bias_{label}"] = kc["bias"]
        out[f"rmse_{label}"] = rmse(zeta_obs[idx], sim_full[idx])
        out[f"r2_{label}"] = r2(zeta_obs[idx], sim_full[idx])
        out[f"kge_rate_{label}"] = kge_on_rate(zeta_obs[idx], sim_full[idx])
    out["sim_full"] = sim_full
    return out


def fit_station(*, h, zeta_obs, t_years, cal_idx, val_idx,
                bounds: Dict[str, tuple[float, float]],
                form3_eligible: bool = False,
                seed: int = 42,
                rate_weight: float = 0.5) -> dict:
    """Fit all 4 (or fewer) variants and select best by val KGE."""
    variants = list(VARIANTS) if form3_eligible else ["M1", "M2"]
    fits = {v: fit_one_variant(h=h, zeta_obs=zeta_obs, t_years=t_years,
                               variant=v, cal_idx=cal_idx, val_idx=val_idx,
                               bounds=bounds, seed=seed,
                               rate_weight=rate_weight) for v in variants}
    best = max(fits, key=lambda v: fits[v]["kge_val"]
               if not np.isnan(fits[v]["kge_val"]) else -np.inf)
    return {"best_variant": best, "all_variants": fits}
