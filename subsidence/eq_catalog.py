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

USGS_QUERY_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
TAIWAN_BBOX = dict(minlatitude=22, maxlatitude=25.5,
                   minlongitude=119.5, maxlongitude=122.5)


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


def _build_usgs_url(starttime: str, endtime: str, min_magnitude: float) -> str:
    params = {
        "format": "csv",
        "starttime": starttime,
        "endtime": endtime,
        "minmagnitude": str(min_magnitude),
        **{k: str(v) for k, v in TAIWAN_BBOX.items()},
    }
    return f"{USGS_QUERY_URL}?{urllib.parse.urlencode(params)}"


def load_catalog(cache_path: Path = Path("data/eq_catalog.csv"),
                 starttime: str = "2019-10-01",
                 endtime: str = "2025-04-01",
                 min_magnitude: float = 5.0,
                 refresh: bool = False,
                 timeout: float = 60.0) -> pd.DataFrame:
    """Load USGS earthquake catalog (M≥5, Taiwan bbox) from cache or download.

    Returns a DataFrame with at minimum: time, latitude, longitude, depth, mag, id.
    `time` is parsed to UTC-naive datetime64.
    """
    cache_path = Path(cache_path)
    if cache_path.exists() and not refresh:
        df = pd.read_csv(cache_path)
    else:
        url = _build_usgs_url(starttime, endtime, min_magnitude)
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(body)
        df = pd.read_csv(cache_path)

    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce").dt.tz_localize(None)
    return df
