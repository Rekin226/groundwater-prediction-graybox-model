"""Tests for subsidence.eq_catalog."""
from __future__ import annotations
import pandas as pd
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


def _fixture_catalog():
    """Synthetic 4-event catalog covering match cases."""
    return pd.DataFrame({
        "time": pd.to_datetime([
            "2021-04-18T01:11:00",  # near YWJS, M6.1, 30km
            "2021-06-15T12:00:00",  # very near YWJS, M5.5, 5km
            "2022-09-17T13:41:00",  # far from YWJS (south Taiwan), M6.5
            "2023-01-01T00:00:00",  # near YWJS, depth 80km (too deep)
        ]),
        "latitude":  [23.85, 23.78, 23.09, 23.78],
        "longitude": [120.40, 120.39, 121.17, 120.39],
        "depth":     [12.0, 8.0, 7.0, 80.0],
        "mag":       [6.1, 5.5, 6.5, 6.0],
        "id":        ["A", "B", "C", "D"],
    })


def test_match_jump_to_event_returns_match():
    import pandas as pd
    from subsidence.eq_catalog import match_jump_to_event
    cat = _fixture_catalog()
    station_lat_lon = (23.78, 120.39)  # YWJS-ish
    out = match_jump_to_event(station_lat_lon, pd.Timestamp("2021-06-15"), cat)
    assert out is not None
    assert out["id"] == "B"


def test_match_jump_to_event_no_match_too_far():
    import pandas as pd
    from subsidence.eq_catalog import match_jump_to_event
    cat = _fixture_catalog()
    station_lat_lon = (24.50, 121.50)  # northeast — far from event B
    out = match_jump_to_event(station_lat_lon, pd.Timestamp("2021-06-15"), cat)
    assert out is None


def test_match_jump_to_event_excludes_too_deep():
    import pandas as pd
    from subsidence.eq_catalog import match_jump_to_event
    cat = _fixture_catalog()
    station_lat_lon = (23.78, 120.39)
    out = match_jump_to_event(station_lat_lon, pd.Timestamp("2023-01-01"), cat)
    assert out is None  # depth 80km exceeds 30km threshold


def test_match_jump_to_event_picks_closest_in_time():
    import pandas as pd
    from subsidence.eq_catalog import match_jump_to_event
    cat = _fixture_catalog()
    station_lat_lon = (23.78, 120.39)
    # Both events A (Apr 18) and B (Jun 15) within 50km but only B is within ±2d of Jun 15
    out = match_jump_to_event(station_lat_lon, pd.Timestamp("2021-06-15"), cat)
    assert out["id"] == "B"
