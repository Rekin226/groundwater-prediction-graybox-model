"""Integration smoke test for 03b_clean_ls orchestrator.

Runs against real cached data — requires 01_fetch_ls_data has been run.
Skips if cache absent.
"""
from __future__ import annotations
from pathlib import Path
import urllib.parse
import numpy as np
import pandas as pd
import pytest


CACHE_RAW = Path("data/ls_cache")
CACHE_CLEAN = Path("data/ls_cache/clean")


def _has(sid: str, ds: str = "ls-wra-gnss-obs") -> bool:
    return (CACHE_RAW / f"{ds}__{urllib.parse.quote(sid, safe='')}.parquet").exists()


@pytest.mark.skipif(not _has("YWJS"), reason="raw GNSS cache missing")
def test_clean_orchestrator_clean_station_unchanged(tmp_path, monkeypatch):
    """Cleaning a known-clean station produces a parquet with ≤ 1 modification."""
    import importlib
    mod = importlib.import_module("subsidence.03b_clean_ls")

    # Redirect outputs to tmp_path
    monkeypatch.setattr(mod, "OUT_DIR", tmp_path / "clean")
    monkeypatch.setattr(mod, "QC_REPORT", tmp_path / "qc_report.csv")
    monkeypatch.setattr(mod, "QC_SUMMARY", tmp_path / "qc_summary.csv")
    (tmp_path / "clean").mkdir(parents=True)

    mod.main(["--station", "YWJS"])

    out_path = tmp_path / "clean" / "ls-wra-gnss-obs__YWJS.parquet"
    assert out_path.exists()

    raw = pd.read_parquet(CACHE_RAW / "ls-wra-gnss-obs__YWJS.parquet")["value"]
    clean = pd.read_parquet(out_path)["value"]

    # YWJS is a clean baseline — at most 1 NaN modification expected
    diff = (raw != clean).sum()
    nan_introduced = (clean.isna() & ~raw.isna()).sum()
    assert nan_introduced <= 1


@pytest.mark.skipif(not _has("LNJS"), reason="LNJS cache missing")
def test_clean_orchestrator_lnjs_detects_many_jumps(tmp_path, monkeypatch):
    """LNJS has ≥30 jumps (per scan)."""
    import importlib
    mod = importlib.import_module("subsidence.03b_clean_ls")
    monkeypatch.setattr(mod, "OUT_DIR", tmp_path / "clean")
    monkeypatch.setattr(mod, "QC_REPORT", tmp_path / "qc_report.csv")
    monkeypatch.setattr(mod, "QC_SUMMARY", tmp_path / "qc_summary.csv")
    (tmp_path / "clean").mkdir(parents=True)

    mod.main(["--station", "LNJS"])

    qc = pd.read_csv(tmp_path / "qc_report.csv")
    assert len(qc) >= 30


@pytest.mark.skipif(not _has("秀潭國小", ds="ls-wra-mlcw-obs"),
                    reason="MLCW cache missing")
def test_clean_orchestrator_mlcw_verbatim_copy(tmp_path, monkeypatch):
    """MLCW is deferred — orchestrator copies bytes verbatim."""
    import importlib
    mod = importlib.import_module("subsidence.03b_clean_ls")
    monkeypatch.setattr(mod, "OUT_DIR", tmp_path / "clean")
    monkeypatch.setattr(mod, "QC_REPORT", tmp_path / "qc_report.csv")
    monkeypatch.setattr(mod, "QC_SUMMARY", tmp_path / "qc_summary.csv")
    (tmp_path / "clean").mkdir(parents=True)

    mod.main(["--station", "秀潭國小"])

    sid_enc = urllib.parse.quote("秀潭國小", safe="")
    raw = pd.read_parquet(CACHE_RAW / f"ls-wra-mlcw-obs__{sid_enc}.parquet")
    clean = pd.read_parquet(tmp_path / "clean" / f"ls-wra-mlcw-obs__{sid_enc}.parquet")
    pd.testing.assert_frame_equal(raw, clean)
