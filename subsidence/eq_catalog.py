"""USGS earthquake catalog loader, cache, and jump-to-event matcher.

Catalog is downloaded once for the cal/val window and cached on disk.
Public API: load_catalog, match_jump_to_event, haversine_km.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union
import math
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0088


def haversine_km(src: Tuple[float, float],
                 tgt: Union[Tuple[float, float], np.ndarray]) -> Union[float, np.ndarray]:
    """Great-circle distance in km between src=(lat,lon) and tgt.

    tgt may be a single (lat,lon) tuple OR an (N,2) array of [lat,lon] rows.
    Returns scalar for tuple input, ndarray of length N for array input.
    """
    lat1 = math.radians(src[0])
    lon1 = math.radians(src[1])
    if isinstance(tgt, tuple):
        lat2 = math.radians(tgt[0])
        lon2 = math.radians(tgt[1])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = (math.sin(dlat / 2) ** 2 +
             math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2)
        return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))
    arr = np.asarray(tgt, dtype=float)
    lat2 = np.radians(arr[:, 0])
    lon2 = np.radians(arr[:, 1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (np.sin(dlat / 2) ** 2 +
         math.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))
