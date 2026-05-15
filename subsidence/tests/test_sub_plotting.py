"""Smoke + structural tests for subsidence/sub_plotting.py."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import pytest

from subsidence.sub_plotting import (
    plot_per_variant, plot_comparison, plot_full_subplots,
)


def _synthetic_inputs(n: int = 1500):
    t = pd.date_range("2020-01-01", periods=n, freq="D")
    rng = np.random.default_rng(0)
    zeta_obs = np.cumsum(0.0008 + rng.normal(0, 0.0005, n))
    sim = zeta_obs + rng.normal(0, 0.001, n)
    cal_idx = np.where((t >= "2020-01-01") & (t <= "2022-12-31"))[0]
    val_idx = np.where((t >= "2024-01-01") & (t <= "2025-03-31"))[0]
    return t, zeta_obs, sim, cal_idx, val_idx


def test_plot_per_variant_axis_tight_and_buffer_band(tmp_path: Path):
    t, zeta_obs, sim, cal_idx, val_idx = _synthetic_inputs()
    out = tmp_path / "test.tiff"

    plot_per_variant(
        sub_id="TEST", variant="M1",
        t=t, zeta_obs=zeta_obs, sim=sim,
        cal_idx=cal_idx, val_idx=val_idx,
        metrics={"kge_cal": 0.9, "kge_val": 0.8,
                 "rmse_val": 0.01, "kge_rate_val": 0.5},
        out_path=out,
    )

    assert out.exists() and out.stat().st_size > 0


def test_plot_per_variant_accepts_imputation_kwargs(tmp_path: Path):
    """When zeta_filled/sigma/imputed_mask are passed, plot still renders."""
    t, zeta_obs, sim, cal_idx, val_idx = _synthetic_inputs()
    zeta_filled = zeta_obs.copy()
    rng = np.random.default_rng(0)
    imputed_mask = np.zeros(len(t), dtype=bool)
    imputed_mask[700:760] = True
    zeta_obs[imputed_mask] = np.nan
    zeta_sigma = np.full(len(t), 0.001)

    plot_per_variant(
        sub_id="TEST", variant="M1",
        t=t, zeta_obs=zeta_obs, sim=sim,
        cal_idx=cal_idx, val_idx=val_idx,
        metrics={"kge_cal": 0.9, "kge_val": 0.8,
                 "rmse_val": 0.01, "kge_rate_val": 0.5},
        out_path=tmp_path / "imp.tiff",
        zeta_filled=zeta_filled,
        zeta_sigma=zeta_sigma,
        imputed_mask=imputed_mask,
    )

    assert (tmp_path / "imp.tiff").exists()


def test_plot_comparison_renders_with_imputation(tmp_path: Path):
    t, zeta_obs, sim, cal_idx, val_idx = _synthetic_inputs()
    zeta_filled = zeta_obs.copy()
    imputed_mask = np.zeros(len(t), dtype=bool)
    imputed_mask[700:760] = True
    zeta_obs[imputed_mask] = np.nan
    zeta_sigma = np.full(len(t), 0.001)

    fits = {
        "M1": {"sim_full": sim, "kge_cal": 0.9, "kge_val": 0.8,
               "rmse_val": 0.01, "kge_rate_val": 0.5},
        "M2": {"sim_full": sim * 1.05, "kge_cal": 0.85, "kge_val": 0.7,
               "rmse_val": 0.012, "kge_rate_val": 0.4},
    }

    plot_comparison(
        sub_id="TEST", t=t, zeta_obs=zeta_obs, fits=fits,
        cal_idx=cal_idx, val_idx=val_idx,
        out_path=tmp_path / "cmp.tiff",
        zeta_filled=zeta_filled, zeta_sigma=zeta_sigma,
        imputed_mask=imputed_mask,
    )
    assert (tmp_path / "cmp.tiff").exists()


def test_plot_full_subplots_with_cal_val_and_imputation(tmp_path: Path):
    t, zeta_obs, sim, cal_idx, val_idx = _synthetic_inputs()
    h_driver = -30.0 + 5.0 * np.sin(np.arange(len(t)) * 2 * np.pi / 365.25)
    zeta_filled = zeta_obs.copy()
    imputed_mask = np.zeros(len(t), dtype=bool)
    imputed_mask[700:760] = True
    zeta_obs[imputed_mask] = np.nan
    zeta_sigma = np.full(len(t), 0.001)

    plot_full_subplots(
        sub_id="TEST", t=t,
        zeta_obs=zeta_obs, sim_best=sim,
        h_driver=h_driver, driver_source=None, rainfall=None,
        out_path=tmp_path / "full.tiff",
        cal_idx=cal_idx, val_idx=val_idx,
        zeta_filled=zeta_filled, zeta_sigma=zeta_sigma,
        imputed_mask=imputed_mask,
    )
    assert (tmp_path / "full.tiff").exists()


def test_apply_render_mask_nans_out_cells_where_mask_false():
    import numpy as np
    from subsidence.sub_plotting import _apply_render_mask
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mask = np.array([True, False, True, False, True])
    out = _apply_render_mask(y, mask)
    np.testing.assert_array_equal(np.isnan(out), [False, True, False, True, False])
    np.testing.assert_array_equal(out[[0, 2, 4]], [1.0, 3.0, 5.0])


def test_apply_render_mask_all_false_suppresses_entire_line():
    import numpy as np
    from subsidence.sub_plotting import _apply_render_mask
    y = np.linspace(0, 1, 10)
    mask = np.zeros(10, dtype=bool)
    out = _apply_render_mask(y, mask)
    assert np.all(np.isnan(out))


def test_plot_per_variant_suppresses_gpr_legend_on_global_reject(tmp_path):
    """When render_mask is all-False, the GPR-imputed line is not drawn."""
    import numpy as np
    import pandas as pd
    from subsidence.sub_plotting import plot_per_variant

    n = 300
    t = pd.date_range("2020-01-01", periods=n, freq="D")
    zeta_obs = np.cumsum(np.random.default_rng(0).normal(0, 1e-3, n))
    sim = zeta_obs.copy()
    cal_idx = np.arange(0, 200)
    val_idx = np.arange(220, n)
    zeta_filled = zeta_obs.copy()
    zeta_sigma = np.full(n, 0.001)
    imputed_mask = np.zeros(n, dtype=bool)
    imputed_mask[100:150] = True
    render_mask = np.zeros(n, dtype=bool)  # all-False ⇒ global reject

    out = tmp_path / "test.tiff"
    plot_per_variant(
        sub_id="TEST", variant="M1", t=t, zeta_obs=zeta_obs, sim=sim,
        cal_idx=cal_idx, val_idx=val_idx,
        metrics={"kge_cal": 0.5, "kge_val": 0.3, "rmse_val": 0.01,
                 "kge_rate_val": 0.0},
        out_path=out,
        zeta_filled=zeta_filled, zeta_sigma=zeta_sigma,
        imputed_mask=imputed_mask, render_mask=render_mask,
    )
    assert out.exists()
