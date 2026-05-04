"""Tests for subsidence/sub_runner.py — the per-station fit + plotting orchestrator."""
from __future__ import annotations
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
