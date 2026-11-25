import numpy as np
import pandas as pd
from subsidence.sub_shell import fit_one_variant, fit_station


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


def test_rate_loss_active_flag_present_in_fit_result():
    """fit_one_variant return dict includes rate_loss_active boolean."""
    import numpy as np
    from subsidence.sub_shell import fit_one_variant
    n = 1500
    rng = np.random.default_rng(0)
    h = -10.0 + np.cumsum(rng.normal(0, 0.01, n))
    t_years = np.arange(n) / 365.25
    zeta = 0.001 * t_years + rng.normal(0, 1e-4, n)
    cal_idx = np.arange(0, 1000)
    val_idx = np.arange(1100, n)
    bnds = {"Sk_e": (1e-6, 1e-3), "Sk_v": (1e-6, 1e-3),
            "h_ref": (h.min(), h.max()), "v_tect": (-0.005, 0.005)}
    out = fit_one_variant(h=h, zeta_obs=zeta, t_years=t_years, variant="M1",
                          cal_idx=cal_idx, val_idx=val_idx, bounds=bnds)
    assert "rate_loss_active" in out
    assert out["rate_loss_active"] is True


def test_rate_loss_inactive_on_extreme_sparse():
    """If only a single finite-rate pair exists, rate_loss_active is False."""
    import numpy as np
    from subsidence.sub_shell import fit_one_variant
    n = 1500
    h = -10.0 + 0.01 * np.arange(n) / 100.0
    t_years = np.arange(n) / 365.25
    zeta = np.full(n, np.nan)
    zeta[::400] = 0.001 * t_years[::400]
    cal_idx = np.arange(0, 1000)
    val_idx = np.arange(1100, n)
    bnds = {"Sk_e": (1e-6, 1e-3), "Sk_v": (1e-6, 1e-3),
            "h_ref": (h.min(), h.max()), "v_tect": (-0.005, 0.005)}
    out = fit_one_variant(h=h, zeta_obs=zeta, t_years=t_years, variant="M1",
                          cal_idx=cal_idx, val_idx=val_idx, bounds=bnds)
    assert out["rate_loss_active"] is False


def test_fit_station_cal_only_selects_by_cal_kge():
    """When the val window is entirely NaN (mlcw_deep_retired), fit_station must
    still return a finite best variant chosen by cal KGE rather than picking
    arbitrarily off NaN val KGE."""
    from subsidence.sub_subroutine import simulate_form2
    n = 1095
    h = _synthetic_h(n)
    t = np.arange(n) / 365.25
    zeta_true = simulate_form2(h, t_years=t, Sk_e=1e-3, Sk_v=5e-3,
                               h_ref=9.0, v_tect=-0.01, smooth_max=False)
    zeta_obs = zeta_true + 0.001 * np.random.default_rng(1).standard_normal(n)
    # Val window present as indices but entirely NaN (deep ring retired).
    zeta_obs[820:] = np.nan
    fit = fit_station(
        h=h, zeta_obs=zeta_obs, t_years=t,
        cal_idx=np.arange(0, 730), val_idx=np.arange(820, n),
        bounds=dict(Sk_e=(1e-6, 1e-1), Sk_v=(1e-6, 1e-1),
                    h_ref=(5.0, 12.0), v_tect=(-0.05, 0.05),
                    v0=(-0.005, 0.005), v1=(-0.002, 0.002)),
        form3_eligible=False, seed=42,
    )
    best = fit["best_variant"]
    assert best in fit["all_variants"]
    assert np.isnan(fit["all_variants"][best]["kge_val"]), "val KGE should be NaN"
    # Selected variant must be the cal-KGE argmax (principled, not arbitrary).
    cal_best = max(fit["all_variants"],
                   key=lambda v: fit["all_variants"][v]["kge_cal"])
    assert best == cal_best
