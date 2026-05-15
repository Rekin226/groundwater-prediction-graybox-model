"""Pairing logic: subsidence stations → GW driver stations.

Pure functions, no I/O.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def pair_subsidence_to_gw(sub: pd.DataFrame, gw: pd.DataFrame,
                          co_located_threshold_m: float = 100.0) -> pd.DataFrame:
    """Return one row per subsidence station with the chosen GW pairing.

    Parameters
    ----------
    sub : DataFrame with columns sub_id, sub_dataset, X_3826, Y_3826, zone
    gw  : DataFrame with columns st_id, gw_st, gw_TM_X97, gw_TM_Y97, zone
    co_located_threshold_m : if a GW station is within this distance, treat as co-located

    Returns
    -------
    DataFrame columns: sub_id, sub_dataset, gw_st, st_id, pairing_method, distance_m, gw_zone
    """
    rows = []
    for _, r in sub.iterrows():
        zone = r["zone"]
        gw_in_zone = gw[gw["zone"] == zone].copy()
        if gw_in_zone.empty:
            rows.append({"sub_id": r["sub_id"], "sub_dataset": r["sub_dataset"],
                         "gw_st": None, "st_id": None,
                         "pairing_method": "no-zone-candidate",
                         "distance_m": np.nan, "gw_zone": zone,
                         "pairing_ambiguous": False})
            continue
        d = np.hypot(gw_in_zone["gw_TM_X97"].values - r["X_3826"],
                     gw_in_zone["gw_TM_Y97"].values - r["Y_3826"])
        # Deterministic tie-break: sort by gw_st then take argmin so equal
        # distances resolve to the smaller-id candidate.
        order = np.lexsort((gw_in_zone["gw_st"].values, d))
        idx_sorted = order[0]
        d_min = float(d[idx_sorted])
        # Ambiguity: any other candidate within 10% of d_min (and d_min > 0).
        if d_min > 1e-6:
            others = np.delete(d, idx_sorted)
            ambiguous = bool(np.any(others <= 1.10 * d_min))
        else:
            ambiguous = False  # co-located unique by definition
        method = "co-located" if d_min <= co_located_threshold_m else "nn-within-zone"
        chosen = gw_in_zone.iloc[int(idx_sorted)]
        rows.append({"sub_id": r["sub_id"], "sub_dataset": r["sub_dataset"],
                     "gw_st": int(chosen["gw_st"]), "st_id": chosen["st_id"],
                     "pairing_method": method,
                     "distance_m": 0.0 if method == "co-located" else d_min,
                     "gw_zone": zone,
                     "pairing_ambiguous": ambiguous})
    return pd.DataFrame(rows)
