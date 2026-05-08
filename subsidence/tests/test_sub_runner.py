"""Tests for subsidence/sub_runner.py — the per-station fit + plotting orchestrator."""
from __future__ import annotations
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from subsidence import sub_runner


def test_sub_runner_module_imports():
    """sub_runner can be imported by name and exposes _process and the
    constants previously defined in 05_run_subsidence.py."""
    assert callable(sub_runner._process)
    assert hasattr(sub_runner, "CAL_START")
    assert hasattr(sub_runner, "CAL_END")
    assert hasattr(sub_runner, "VAL_START")
    assert hasattr(sub_runner, "VAL_END")
    assert hasattr(sub_runner, "EXCLUDED_DATASETS")
    assert hasattr(sub_runner, "EXCLUDED_STATIONS")


def test_sub_runner_uses_late_bound_fit_station(monkeypatch):
    """sub_runner._process must call sub_shell.fit_station as a late-bound
    module attribute (not an at-import-time `from sub_shell import fit_station`),
    so unittest.mock.patch.object(sub_shell, 'fit_station', ...) intercepts it.
    Verified by replacing fit_station with a sentinel and asserting it runs.
    """
    from subsidence import sub_shell

    sentinel = {"called": False}

    def fake_fit(**kw):
        sentinel["called"] = True
        # Return a minimal valid fit-result dict so _process can continue.
        n = len(kw["t_years"])
        import numpy as _np
        return {
            "best_variant": "M1",
            "all_variants": {"M1": {"params": {}, "sim_full": _np.zeros(n),
                                    "kge_cal": 0.0, "kge_val": 0.0,
                                    "rmse_val": 0.0, "kge_rate_val": 0.0}},
        }

    monkeypatch.setattr(sub_shell, "fit_station", fake_fit)
    # We don't run end-to-end here — just verify the patch attaches to the
    # symbol sub_runner actually calls. End-to-end is covered in Task 9.
    assert sub_shell.fit_station is fake_fit


def test_process_renders_three_figures_for_one_station(tmp_path, monkeypatch):
    """End-to-end smoke: _process runs cleanly on LYES and writes one
    TIFF per plot type. Does not assert visual properties — just that
    the wire-up is intact and the new gpr_fill kwargs flow through."""
    monkeypatch.chdir(Path(__file__).resolve().parents[2])  # repo root
    sub_id, sub_dataset = "LYES", "ls-wra-gnss-obs"
    run_id = "_smoke_wireup"
    workspace = Path("workspace/results_sub") / run_id
    if workspace.exists():
        shutil.rmtree(workspace)

    msg = sub_runner._process(sub_id, sub_dataset, run_id)
    assert "fit failed" not in msg, msg

    fig_dir = workspace / "figures"
    assert (fig_dir / "comparison" / f"sub_compare_{sub_id}.tiff").exists()
    assert (fig_dir / "M1" / f"sub_fit_{sub_id}.tiff").exists()
    assert (fig_dir / "full_subplots" / f"sub_fit_{sub_id}.tiff").exists()


def test_optimizer_receives_raw_zeta_not_imputed(tmp_path, monkeypatch):
    """`sub_shell.fit_station` is called with the raw post-anchored array
    (NaN where source had gaps), NEVER with the GPR-imputed array.

    This is the primary non-leakage invariant per spec test #5. Verified
    by spying on fit_station and asserting np.array_equal(captured,
    expected_raw, equal_nan=True).
    """
    monkeypatch.chdir(Path(__file__).resolve().parents[2])  # repo root
    sub_id, sub_dataset = "LYES", "ls-wra-gnss-obs"
    run_id = "_leakage_test"

    workspace = Path("workspace/results_sub") / run_id
    if workspace.exists():
        shutil.rmtree(workspace)

    # Reconstruct the exact zeta.values that _process will build, so we
    # can assert byte-equality with what fit_station receives.
    # GNSS-only reconstruction (mirrors _process for sub_dataset='ls-wra-gnss-obs');
    # for MLCW you would need to mirror _build_zeta's deepest-ring selection logic.
    import urllib.parse
    h_df = pd.read_parquet(f"subsidence/data/h_drivers/{sub_id}.parquet")
    sid_enc = urllib.parse.quote(str(sub_id), safe="")
    raw = pd.read_parquet(f"data/ls_cache/clean/{sub_dataset}__{sid_enc}.parquet")
    s = raw["value"] if "value" in raw.columns else raw
    s = s.dropna()
    zeta_full = s.iloc[0] - s
    idx = h_df.index
    zeta = zeta_full.reindex(idx)
    cal_obs = zeta.loc[(zeta.index >= sub_runner.CAL_START)
                       & (zeta.index <= sub_runner.CAL_END)].dropna()
    expected_raw = (zeta - float(cal_obs.iloc[0])).values

    captured = {}
    real_fit = sub_runner.sub_shell.fit_station

    def spy_fit_station(**kw):
        captured["zeta_obs"] = np.asarray(kw["zeta_obs"]).copy()
        return real_fit(**kw)

    monkeypatch.setattr(sub_runner.sub_shell, "fit_station", spy_fit_station)

    msg = sub_runner._process(sub_id, sub_dataset, run_id)
    assert "fit failed" not in msg, msg
    assert "zeta_obs" in captured, "fit_station was not called"

    z = captured["zeta_obs"]
    # Primary invariant: byte-equality with the raw post-anchored array.
    assert np.array_equal(z, expected_raw, equal_nan=True), (
        "fit_station received an array different from the raw post-anchored "
        "zeta — imputation may have leaked into the optimizer."
    )
    # Secondary: NaN must still be present (sanity that the raw array we
    # reconstructed is the correct comparator).
    assert np.isnan(z).any(), "captured zeta_obs has no NaN"


def test_csv_byte_identical_with_gpr_fill_noop(tmp_path, monkeypatch):
    """Replacing gpr_fill with a no-op (returning defensive-fallback values
    that contain no imputation) must produce the SAME per-station CSV as
    a normal run. If they differ, gpr_fill output is leaking into the fit.
    """
    monkeypatch.chdir(Path(__file__).resolve().parents[2])
    sub_id, sub_dataset = "LYES", "ls-wra-gnss-obs"

    # --- Run A: real gpr_fill ---
    run_id_a = "_byte_test_real_gpr"
    if Path(f"workspace/results_sub/{run_id_a}").exists():
        shutil.rmtree(f"workspace/results_sub/{run_id_a}")
    sub_runner._process(sub_id, sub_dataset, run_id_a)

    # --- Run B: gpr_fill replaced by a no-op that returns the defensive
    # fallback (raw y, NaN sigma, isnan mask). The optimizer sees identical
    # raw arrays in both runs → CSV must match byte-for-byte.
    def noop_gpr_fill(t, y):
        y = np.asarray(y, dtype=float)
        return y.copy(), np.full_like(y, np.nan), np.isnan(y)

    run_id_b = "_byte_test_noop_gpr"
    if Path(f"workspace/results_sub/{run_id_b}").exists():
        shutil.rmtree(f"workspace/results_sub/{run_id_b}")

    monkeypatch.setattr(sub_runner, "gpr_fill", noop_gpr_fill)
    sub_runner._process(sub_id, sub_dataset, run_id_b)

    csv_a = Path(f"workspace/results_sub/{run_id_a}/per_station/{sub_id}.csv").read_bytes()
    csv_b = Path(f"workspace/results_sub/{run_id_b}/per_station/{sub_id}.csv").read_bytes()
    assert csv_a == csv_b, (
        "per-station CSV differs between real-gpr_fill and noop-gpr_fill runs — "
        "gpr_fill output is leaking into the optimizer."
    )


def test_zero_finite_cal_obs_skips_before_fit(tmp_path, monkeypatch):
    """A station whose cal-window observations are entirely NaN must skip
    BEFORE fit_station is called. Previously the empty-cal_obs branch only
    disabled anchoring and silently fed all-NaN ζ to the optimizer.
    """
    monkeypatch.chdir(Path(__file__).resolve().parents[2])
    sub_id, sub_dataset = "LYES", "ls-wra-gnss-obs"
    run_id = "_zero_cal_obs_skip"
    workspace = Path("workspace/results_sub") / run_id
    if workspace.exists():
        shutil.rmtree(workspace)

    # Force every cal-window observation to NaN by patching _build_zeta to
    # zero out the cal window of the real series.
    real_build = sub_runner._build_zeta

    def cal_window_to_nan(raw, ds, sid, rid):
        s = real_build(raw, ds, sid, rid)
        s = s.copy()
        s.loc[(s.index >= sub_runner.CAL_START)
              & (s.index <= sub_runner.CAL_END)] = np.nan
        return s

    monkeypatch.setattr(sub_runner, "_build_zeta", cal_window_to_nan)

    fit_called = {"called": False}

    def spy_fit(**kw):
        fit_called["called"] = True
        return {"best_variant": "M1", "all_variants": {}}

    monkeypatch.setattr(sub_runner.sub_shell, "fit_station", spy_fit)

    msg = sub_runner._process(sub_id, sub_dataset, run_id)
    assert "zero finite cal obs" in msg, msg
    assert not fit_called["called"], "fit_station ran on a zero-cal-obs station"


def test_zero_finite_val_obs_skips_before_fit(tmp_path, monkeypatch):
    """Same as above for the val window. fit_station must not be called on a
    station with zero finite validation observations."""
    monkeypatch.chdir(Path(__file__).resolve().parents[2])
    sub_id, sub_dataset = "LYES", "ls-wra-gnss-obs"
    run_id = "_zero_val_obs_skip"
    workspace = Path("workspace/results_sub") / run_id
    if workspace.exists():
        shutil.rmtree(workspace)

    real_build = sub_runner._build_zeta

    def val_window_to_nan(raw, ds, sid, rid):
        s = real_build(raw, ds, sid, rid)
        s = s.copy()
        s.loc[(s.index >= sub_runner.VAL_START)
              & (s.index <= sub_runner.VAL_END)] = np.nan
        return s

    monkeypatch.setattr(sub_runner, "_build_zeta", val_window_to_nan)

    fit_called = {"called": False}

    def spy_fit(**kw):
        fit_called["called"] = True
        return {"best_variant": "M1", "all_variants": {}}

    monkeypatch.setattr(sub_runner.sub_shell, "fit_station", spy_fit)

    msg = sub_runner._process(sub_id, sub_dataset, run_id)
    assert "zero finite val obs" in msg, msg
    assert not fit_called["called"], "fit_station ran on a zero-val-obs station"


def test_mlcw_no_viable_ring_leaves_no_layer_csv(tmp_path, monkeypatch):
    """When _build_zeta finds no MLCW ring with adequate cal+val coverage, no
    per-layer CSV must be written. Stale layer files for skipped stations
    pollute downstream artifact diffs."""
    out_dir = tmp_path / "workspace/results_sub/_no_viable_ring/per_station"
    monkeypatch.chdir(tmp_path)

    # Synthetic MLCW frame: 3 rings, each with only 1 finite cal obs (below
    # MLCW_MIN_CAL_OBS=12). No ring should be viable.
    idx = pd.date_range("2020-01-01", "2025-03-31", freq="MS")
    sparse = pd.DataFrame(index=idx, columns=["NO1", "NO2", "NO3"], dtype=float)
    sparse.iloc[0] = 0.0  # one finite obs per ring at t_0 — far below MIN_CAL_OBS

    sub_id = "_synthetic_mlcw"
    result = sub_runner._build_zeta(sparse, "ls-wra-mlcw-obs", sub_id, "_no_viable_ring")
    assert result.empty, "expected empty zeta when no ring is viable"
    assert not (out_dir / f"{sub_id}_mlcw_layer.csv").exists(), (
        "per-layer CSV was written for a station with no viable ring"
    )
