"""WiseEnvr Land Subsidence (LS) API client.

Reads credentials from LS_USER / LS_PASS environment variables.
Caches tokens and dataset payloads under data/ls_cache/.
"""
from __future__ import annotations


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
