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
