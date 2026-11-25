"""End-to-end smoke test for hybrid driver assembly.

Gated by LS_USER / LS_PASS environment variables and requires that the
01/02/03 cache-building steps have already been run.
"""
import os
import pytest


@pytest.mark.skipif(
    not os.environ.get("LS_USER") or not os.environ.get("LS_PASS"),
    reason="needs LS_USER/LS_PASS and prior 01/02/03 cache to run",
)
def test_assemble_one_station_end_to_end():
    """Real end-to-end smoke. Expects steps 01/02/03 cached output present."""
    import importlib
    import sys

    # The script file has a leading-digit name; import via importlib.util
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "_04_assemble_h_driver",
        Path(__file__).parent.parent / "04_assemble_h_driver.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    out = mod.assemble_one("YSLL")
    assert "h_driver" in out.columns
    assert "driver_source" in out.columns
    assert (out["driver_source"] != "missing").all()
