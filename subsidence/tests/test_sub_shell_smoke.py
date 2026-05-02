import numpy as np
import pandas as pd
from subsidence.sub_shell import fit_one_variant


def _synthetic_h(n=1500, h0=10.0, drop=3.0):
    rng = np.random.default_rng(42)
    h = h0 + 0.5 * np.sin(2*np.pi*np.arange(n)/365.25) - drop * (np.arange(n) > 365) + 0.2*rng.standard_normal(n)
    return h


def test_form2_fit_recovers_true_params_on_synthetic():
    from subsidence.sub_subroutine import simulate_form2
    n = 1095
    h = _synthetic_h(n)
    t = np.arange(n) / 365.25
    Sk_e_true, Sk_v_true, h_ref_true, v_tect_true = 1e-3, 5e-3, 9.0, -0.01
    zeta_true = simulate_form2(h, t_years=t, Sk_e=Sk_e_true, Sk_v=Sk_v_true,
                               h_ref=h_ref_true, v_tect=v_tect_true,
                               smooth_max=False)
    zeta_obs = zeta_true + 0.001 * np.random.default_rng(1).standard_normal(n)

    fit = fit_one_variant(
        h=h, zeta_obs=zeta_obs, t_years=t,
        variant="M1",
        cal_idx=np.arange(0, 730), val_idx=np.arange(820, n),  # buffer 730..820
        bounds=dict(Sk_e=(1e-6, 1e-1), Sk_v=(1e-6, 1e-1),
                    h_ref=(5.0, 12.0), v_tect=(-0.05, 0.05)),
        seed=42,
    )
    assert fit["kge_val"] > 0.7
    assert abs(fit["params"]["Sk_v"] - Sk_v_true) / Sk_v_true < 0.5


def test_rate_augmented_loss_recovers_true_params():
    """Composite loss should still recover true params on clean synthetic."""
    from subsidence.sub_subroutine import simulate_form2
    n = 1095
    h = _synthetic_h(n)
    t = np.arange(n) / 365.25
    Sk_e_true, Sk_v_true, h_ref_true, v_tect_true = 1e-3, 5e-3, 9.0, -0.01
    zeta_true = simulate_form2(h, t_years=t, Sk_e=Sk_e_true, Sk_v=Sk_v_true,
                               h_ref=h_ref_true, v_tect=v_tect_true,
                               smooth_max=False)
    zeta_obs = zeta_true + 0.001 * np.random.default_rng(1).standard_normal(n)

    fit = fit_one_variant(
        h=h, zeta_obs=zeta_obs, t_years=t,
        variant="M1",
        cal_idx=np.arange(0, 730), val_idx=np.arange(820, n),
        bounds=dict(Sk_e=(1e-6, 1e-1), Sk_v=(1e-6, 1e-1),
                    h_ref=(5.0, 12.0), v_tect=(-0.05, 0.05)),
        seed=42, rate_weight=0.5,
    )
    assert fit["kge_val"] > 0.7
    assert abs(fit["params"]["Sk_v"] - Sk_v_true) / Sk_v_true < 0.5


def test_rate_loss_improves_tau_identifiability():
    """With τ in the model, rate-augmented loss should land closer to true τ
    than cumulative-only loss.  Cumulative integration smooths out the rate
    response τ controls, so cumulative-only fitting leaves τ unidentifiable."""
    from subsidence.sub_subroutine import simulate_form3
    n = 1095
    h = _synthetic_h(n)
    t = np.arange(n) / 365.25
    Sk_e_true, Sk_v_true, h_ref_true, v_tect_true, tau_true = (
        1e-3, 5e-3, 9.0, -0.01, 60.0,
    )
    zeta_true = simulate_form3(h, t_years=t, Sk_e=Sk_e_true, Sk_v=Sk_v_true,
                               h_ref=h_ref_true, v_tect=v_tect_true,
                               tau_days=tau_true, smooth_max=False)
    zeta_obs = zeta_true + 0.001 * np.random.default_rng(1).standard_normal(n)

    common = dict(
        h=h, zeta_obs=zeta_obs, t_years=t,
        variant="M3_tau",
        cal_idx=np.arange(0, 730), val_idx=np.arange(820, n),
        bounds=dict(Sk_e=(1e-6, 1e-1), Sk_v=(1e-6, 1e-1),
                    h_ref=(5.0, 12.0), v_tect=(-0.05, 0.05),
                    tau=(7.0, 1500.0)),
        seed=42,
    )
    fit_cum = fit_one_variant(**common, rate_weight=0.0)
    fit_rate = fit_one_variant(**common, rate_weight=0.5)
    err_cum = abs(fit_cum["params"]["tau"] - tau_true) / tau_true
    err_rate = abs(fit_rate["params"]["tau"] - tau_true) / tau_true
    assert err_rate < err_cum, (
        f"rate-augmented loss did not improve τ identifiability: "
        f"err_cum={err_cum:.3f}, err_rate={err_rate:.3f}"
    )
