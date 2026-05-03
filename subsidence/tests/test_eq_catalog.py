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
