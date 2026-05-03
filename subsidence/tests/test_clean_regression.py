"""Synthetic-jump regression test — hard gate for cleaning module.

Replays diag_jump_sensitivity.py logic but routes the perturbed series
through clean_station before refitting. The cleaning module should detect
and NaN the injected step so val KGE recovers.
"""
from __future__ import annotations
from pathlib import Path
import urllib.parse
import numpy as np
import pandas as pd
import pytest


CACHE_RAW = Path("data/ls_cache")


def _has(sid: str) -> bool:
    return (CACHE_RAW / f"ls-wra-gnss-obs__{urllib.parse.quote(sid, safe='')}.parquet").exists()


@pytest.mark.skipif(not _has("YWJS"), reason="raw YWJS cache missing")
def test_clean_recovers_injected_step():
    from subsidence.clean_ls import clean_station, compute_robust_sigma
    from subsidence.eq_catalog import load_catalog

    z = pd.read_parquet(CACHE_RAW / "ls-wra-gnss-obs__YWJS.parquet")["value"].dropna()
    z_zeta = (z.iloc[0] - z).rename("zeta_m")

    # Inject 7 cm step on a date inside cal window (> 6σ for any station).
    # 2022-09-11: pre/post slopes are ~parallel (diff ≈ 0.17 cm/yr < 0.5 tol),
    # so classify_jump assigns datum_reset → rebaseline as expected.
    inject_date = pd.Timestamp("2022-09-11")
    z_perturbed = z_zeta.copy()
    z_perturbed.loc[z_perturbed.index >= inject_date] += 0.07

    eq = load_catalog()
    z_clean, qc = clean_station(
        z=z_perturbed, station_lat_lon=(23.78, 120.39),
        eq_catalog=eq, neighbor_series=[],
    )

    # The injection should be detected and acted upon
    assert len(qc) >= 1
    nearest_jump = qc.iloc[(qc["jump_date"] - inject_date).abs().argsort()].iloc[0]
    assert (nearest_jump["jump_date"] - inject_date).days <= 1
    # Action must remove the offset (rebaseline) OR NaN the day
    assert nearest_jump["action"] in {"rebaseline", "nan_event_day", "nan_spike_day"}

    # Post-clean sigma should not be inflated relative to original
    sigma_orig = compute_robust_sigma(z_zeta)
    sigma_clean = compute_robust_sigma(z_clean)
    assert sigma_clean < 1.5 * sigma_orig
