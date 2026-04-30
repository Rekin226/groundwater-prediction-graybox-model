"""WiseEnvr Land Subsidence (LS) API client.

Reads credentials from LS_USER / LS_PASS environment variables.
Caches tokens and dataset payloads under data/ls_cache/.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from subsidence.api_constants import (
    DEFAULT_SCOPE, LS_BASE, TOKEN_URL,
)


def split_to_df(split: dict) -> pd.DataFrame:
    """Convert a pandas-split JSON payload to a DataFrame indexed by datetime."""
    if not isinstance(split, dict) or "index" not in split:
        raise ValueError(f"unexpected payload shape: {type(split)}")
    df = pd.DataFrame(split["data"], columns=split["columns"])
    if split["index"]:
        df.index = pd.to_datetime(split["index"], utc=False, errors="coerce")
        df.index.name = "datetime"
    return df


def to_api_id(gw_st) -> str:
    """Convert a 7- or 8-digit GW station number to the API's 8-digit form."""
    s = str(gw_st)
    if not s.isdigit():
        raise ValueError(f"non-numeric station id: {gw_st!r}")
    if len(s) > 8:
        raise ValueError(f"station id too long: {gw_st!r}")
    return s.zfill(8)


def from_api_id(api_id: str) -> int:
    """Convert an API-format 8-digit id back to a canonical integer."""
    s = str(api_id).strip()
    if not s.isdigit():
        raise ValueError(f"non-numeric api id: {api_id!r}")
    return int(s)


class LSClient:
    """Thin OAuth2-password-flow client for the WiseEnvr LS API.

    Caches the bearer token in memory for the lifetime of the instance.
    For long-running processes use ``c.get_token(force_refresh=True)``
    on 401 responses.
    """

    def __init__(self, host_base: str = LS_BASE, token_url: str = TOKEN_URL,
                 scope: str = DEFAULT_SCOPE, timeout: float = 60.0):
        self.host_base = host_base
        self.token_url = token_url
        self.scope = scope
        self.timeout = timeout
        self._token: Optional[str] = None

    def get_token(self, force_refresh: bool = False) -> str:
        if self._token and not force_refresh:
            return self._token
        user = os.environ.get("LS_USER")
        pw = os.environ.get("LS_PASS")
        if not user or not pw:
            raise RuntimeError("LS_USER / LS_PASS env vars are required")
        body = urllib.parse.urlencode({
            "username": user, "password": pw,
            "grant_type": "password", "scope": self.scope,
        }).encode()
        req = urllib.request.Request(
            self.token_url, method="POST", data=body,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            payload = json.loads(r.read())
        self._token = payload["access_token"]
        return self._token

    def get_json(self, path: str, params: Optional[dict] = None,
                 retries: int = 2) -> Any:
        url = f"{self.host_base}{path}"
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"
        attempt = 0
        while True:
            req = urllib.request.Request(
                url, headers={"Authorization": f"Bearer {self.get_token()}"},
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read())
            except urllib.error.HTTPError as e:
                if e.code == 401 and attempt == 0:
                    self.get_token(force_refresh=True)
                    attempt += 1
                    continue
                if e.code in (502, 503, 504) and attempt < retries:
                    time.sleep(1.5 * (attempt + 1))
                    attempt += 1
                    continue
                body = e.read().decode("utf-8", errors="replace")[:500]
                raise RuntimeError(f"HTTP {e.code} {url}\n{body}") from e

    def cached_get_dataframe(self, path: str, params: Optional[dict] = None,
                             cache_dir: Path = Path("data/ls_cache"),
                             cache_key: Optional[str] = None,
                             refresh: bool = False) -> pd.DataFrame:
        cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
        if cache_key is None:
            safe = path.replace("/", "_").strip("_")
            qstr = urllib.parse.urlencode(params or {})
            cache_key = f"{safe}__{qstr}".replace("=", "-").replace("&", "_")
        fpath = cache_dir / f"{cache_key}.parquet"
        if fpath.exists() and not refresh:
            return pd.read_parquet(fpath)
        payload = self.get_json(path, params=params)
        df = split_to_df(payload)
        df.to_parquet(fpath)
        return df
