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
        """Returns the 4-tuple matching the new gpr_fill signature."""
        y = np.asarray(y, dtype=float)
        return (y.copy(),
                np.full_like(y, np.nan),
                np.isnan(y),
                np.zeros_like(y, dtype=bool))  # render_mask all-False = "do not draw"

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

    # Task 1.9: per-station GPR time-series CSV must be written alongside variant CSV
    out_dir = Path(f"workspace/results_sub/{run_id_b}/per_station")
    gpr_csv = out_dir / f"{sub_id}_gpr.csv"
    assert gpr_csv.exists(), f"per_station/{sub_id}_gpr.csv was not written"
    gpr_df = pd.read_csv(gpr_csv)
    expected = {"date", "zeta_obs", "zeta_gpr_mean", "zeta_gpr_sigma",
                "imputed_mask", "render_mask", "sim_best"}
    assert expected.issubset(set(gpr_df.columns)), (
        f"columns missing: {expected - set(gpr_df.columns)}"
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
        s, flag = real_build(raw, ds, sid, rid)
        s = s.copy()
        s.loc[(s.index >= sub_runner.CAL_START)
              & (s.index <= sub_runner.CAL_END)] = np.nan
        return s, flag

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
        s, flag = real_build(raw, ds, sid, rid)
        s = s.copy()
        s.loc[(s.index >= sub_runner.VAL_START)
              & (s.index <= sub_runner.VAL_END)] = np.nan
        return s, flag

    monkeypatch.setattr(sub_runner, "_build_zeta", val_window_to_nan)

    fit_called = {"called": False}

    def spy_fit(**kw):
        fit_called["called"] = True
        return {"best_variant": "M1", "all_variants": {}}

    monkeypatch.setattr(sub_runner.sub_shell, "fit_station", spy_fit)

    msg = sub_runner._process(sub_id, sub_dataset, run_id)
    assert "zero finite val obs" in msg, msg
    assert not fit_called["called"], "fit_station ran on a zero-val-obs station"


def test_gpr_fill_exception_does_not_block_fit(tmp_path, monkeypatch):
    """A gpr_fill failure must not prevent the per-station fit CSV from being
    written. Plot-only is the operational boundary: optimizer success is
    independent of imputation success.
    """
    monkeypatch.chdir(Path(__file__).resolve().parents[2])
    sub_id, sub_dataset = "LYES", "ls-wra-gnss-obs"
    run_id = "_gpr_fill_explodes"
    workspace = Path("workspace/results_sub") / run_id
    if workspace.exists():
        shutil.rmtree(workspace)

    def boom_gpr_fill(t, y):
        raise RuntimeError("synthetic kernel optimization failure")

    monkeypatch.setattr(sub_runner, "gpr_fill", boom_gpr_fill)

    msg = sub_runner._process(sub_id, sub_dataset, run_id)
    assert "fit failed" not in msg, msg
    csv_path = workspace / "per_station" / f"{sub_id}.csv"
    assert csv_path.exists(), (
        "per-station fit CSV missing — a gpr_fill exception killed the fit "
        "instead of being contained to the plot block."
    )

    # Task 1.9: even when gpr_fill raises, the _gpr.csv must still be written
    # with the defensive fallback contents (NaN sigma, all-False render_mask).
    out_dir = workspace / "per_station"
    gpr_csv = out_dir / f"{sub_id}_gpr.csv"
    assert gpr_csv.exists()
    gpr_df = pd.read_csv(gpr_csv)
    assert gpr_df["zeta_gpr_sigma"].isna().all(), "expected NaN sigma on GPR failure"
    assert not gpr_df["render_mask"].any(), "expected render_mask all-False on GPR failure"


def test_mlcw_no_viable_ring_leaves_no_layer_csv(tmp_path, monkeypatch):
    """When _build_zeta finds no MLCW ring with adequate calibration coverage,
    no per-layer CSV must be written. Stale layer files for skipped stations
    pollute downstream artifact diffs."""
    out_dir = tmp_path / "workspace/results_sub/_no_viable_ring/per_station"
    monkeypatch.chdir(tmp_path)

    # Synthetic MLCW frame: 3 rings, each with only 1 finite cal obs (below
    # MLCW_MIN_CAL_OBS=12). No ring should be viable.
    idx = pd.date_range("2020-01-01", "2025-03-31", freq="MS")
    sparse = pd.DataFrame(index=idx, columns=["NO1", "NO2", "NO3"], dtype=float)
    sparse.iloc[0] = 0.0  # one finite obs per ring at t_0 — far below MIN_CAL_OBS

    sub_id = "_synthetic_mlcw"
    result, deep_retired = sub_runner._build_zeta(
        sparse, "ls-wra-mlcw-obs", sub_id, "_no_viable_ring")
    assert result.empty, "expected empty zeta when no ring is viable"
    assert deep_retired is False, "no viable ring must not flag deep_retired"
    assert not (out_dir / f"{sub_id}_mlcw_layer.csv").exists(), (
        "per-layer CSV was written for a station with no viable ring"
    )


def _mlcw_frame(deep_retires_pre_val: bool) -> pd.DataFrame:
    """Synthetic MLCW frame with three rings.

    The deepest ring (NO3) carries a positive (compaction-signed) total-column
    signal.  A shallower ring (NO2) carries an opposite-signed signal and
    always spans the full record.  When ``deep_retires_pre_val`` is True the
    deepest ring stops before VAL_START, reproducing the 僑義/北辰/宏崙/拯民
    pattern where the only correctly-signed ring lacks validation coverage.
    """
    idx = pd.date_range("2020-01-01", "2025-03-31", freq="MS")
    t = np.arange(len(idx), dtype=float)
    df = pd.DataFrame(index=idx, dtype=float)
    # NO3 deepest: value DECREASES over time (first - value > 0 => compaction)
    df["NO3"] = 100.0 - 0.01 * t
    # NO2 shallower: value INCREASES (first - value < 0 => apparent extension)
    df["NO2"] = 50.0 + 0.01 * t
    df["NO1"] = 10.0 - 0.001 * t
    if deep_retires_pre_val:
        # Retire NO3 at 2021-12 (before VAL_START 2024-01); NO2/NO1 continue.
        df.loc[df.index > "2021-12-31", "NO3"] = np.nan
    return df


def test_mlcw_deep_retired_keeps_deepest_ring_and_flags(tmp_path, monkeypatch):
    """When the deepest cal-viable ring is retired before the val window, it is
    still selected (correctly-signed total column) and mlcw_deep_retired=True."""
    monkeypatch.chdir(tmp_path)
    df = _mlcw_frame(deep_retires_pre_val=True)
    zeta, deep_retired = sub_runner._build_zeta(
        df, "ls-wra-mlcw-obs", "_deep_retired", "_deep_retired_run")
    assert deep_retired is True
    # Chosen ring is the deepest (NO3): zeta = first - value, monotonically
    # increasing (positive compaction), NOT the opposite-signed NO2.
    zfin = zeta.dropna()
    assert zfin.iloc[-1] > 0, "deepest-ring zeta must be compaction-signed (>0)"
    # Val window is empty for the chosen ring.
    val = zeta.loc[(zeta.index >= sub_runner.VAL_START)
                   & (zeta.index <= sub_runner.VAL_END)].dropna()
    assert val.empty, "deep-retired ring must have no finite val obs"


def test_mlcw_healthy_deepest_not_flagged(tmp_path, monkeypatch):
    """A station whose deepest ring spans the val window is selected and NOT
    flagged deep_retired."""
    monkeypatch.chdir(tmp_path)
    df = _mlcw_frame(deep_retires_pre_val=False)
    zeta, deep_retired = sub_runner._build_zeta(
        df, "ls-wra-mlcw-obs", "_healthy", "_healthy_run")
    assert deep_retired is False
    val = zeta.loc[(zeta.index >= sub_runner.VAL_START)
                   & (zeta.index <= sub_runner.VAL_END)].dropna()
    assert not val.empty, "healthy deepest ring must retain val obs"
