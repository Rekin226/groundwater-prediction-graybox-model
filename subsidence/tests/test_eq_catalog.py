"""Tests for subsidence.eq_catalog."""
from __future__ import annotations
import pytest


def test_haversine_km_zero_distance():
    from subsidence.eq_catalog import haversine_km
    assert haversine_km((23.7, 120.4), (23.7, 120.4)) == pytest.approx(0.0, abs=1e-6)


def test_haversine_km_known_pair():
    # Taipei (25.0330°N, 121.5654°E) to Kaohsiung (22.6273°N, 120.3014°E)
    # ground-truth great-circle ≈ 296 km
    from subsidence.eq_catalog import haversine_km
    d = haversine_km((25.0330, 121.5654), (22.6273, 120.3014))
    assert 290.0 <= d <= 305.0


def test_haversine_km_array_target():
    """haversine_km broadcasts target to multiple points."""
    import numpy as np
    from subsidence.eq_catalog import haversine_km
    src = (23.7, 120.4)
    targets = np.array([[23.7, 120.4], [25.0, 121.5]])
    d = haversine_km(src, targets)
    assert d.shape == (2,)
    assert d[0] == pytest.approx(0.0, abs=1e-6)
    assert 140.0 <= d[1] <= 200.0  # ~150-180 km


from unittest.mock import patch


_USGS_CSV_FIXTURE = b"""time,latitude,longitude,depth,mag,id
2024-04-02T23:58:11.000Z,23.819,121.6624,34.8,7.4,us7000m9g4
2022-09-17T13:41:19.000Z,23.0925,121.1685,7.0,6.5,us7000i5lq
"""


def test_load_catalog_downloads_when_cache_missing(tmp_path):
    from subsidence.eq_catalog import load_catalog
    cache = tmp_path / "eq_catalog.csv"

    class FakeResp:
        def __init__(self, b): self._b = b
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return self._b

    with patch("subsidence.eq_catalog.urllib.request.urlopen",
               return_value=FakeResp(_USGS_CSV_FIXTURE)):
        df = load_catalog(cache_path=cache)

    assert cache.exists()
    assert len(df) == 2
    assert {"time", "latitude", "longitude", "depth", "mag", "id"}.issubset(df.columns)
    assert df.iloc[0]["mag"] == 7.4


def test_load_catalog_uses_cache_when_present(tmp_path):
    from subsidence.eq_catalog import load_catalog
    cache = tmp_path / "eq_catalog.csv"
    cache.write_bytes(_USGS_CSV_FIXTURE)

    with patch("subsidence.eq_catalog.urllib.request.urlopen") as m:
        df = load_catalog(cache_path=cache)
        assert m.call_count == 0   # no network call

    assert len(df) == 2


def test_load_catalog_refresh_overrides_cache(tmp_path):
    from subsidence.eq_catalog import load_catalog
    cache = tmp_path / "eq_catalog.csv"
    cache.write_bytes(b"time,latitude,longitude,depth,mag,id\n")  # empty cache

    class FakeResp:
        def __init__(self, b): self._b = b
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return self._b

    with patch("subsidence.eq_catalog.urllib.request.urlopen",
               return_value=FakeResp(_USGS_CSV_FIXTURE)):
        df = load_catalog(cache_path=cache, refresh=True)

    assert len(df) == 2
