"""Compute pairings: subsidence stations → GW driver stations.

Reads:
    subsidence/data/sub_station_master.csv     (from step 01)
    data/gray_box_input.csv                    (existing GW model station table)
Writes:
    subsidence/data/sub_pairing.csv

Run:
    poetry run python subsidence/03_pair_sub_gw.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

from subsidence.pairing import pair_subsidence_to_gw

SUB_MASTER = Path("subsidence/data/sub_station_master.csv")
GW_INPUT = Path("data/gray_box_input.csv")
OUT = Path("subsidence/data/sub_pairing.csv")


def main(argv=None):
    sub = pd.read_csv(SUB_MASTER)
    gw = pd.read_csv(GW_INPUT)
    sub = sub[sub["active"] == 1].copy() if "active" in sub.columns else sub
    # Normalise zone columns
    sub["zone"] = sub.get("GroundwaterZoneIdentifier", 50).fillna(50).astype(int)
    gw["zone"] = gw.get("GroundwaterZoneIdentifier", 50)
    if gw["zone"].isna().any():
        # The existing input CSV may not carry the zone column; default to 50 (Zhuoshui)
        gw["zone"] = 50
    # Reproject MLCW coords if they are missing X_3826 — fall back to *_4326 → not implemented;
    # assume sub_station_master.csv carries X_3826/Y_3826 from API metadata.
    paired = pair_subsidence_to_gw(sub, gw)
    paired.to_csv(OUT, index=False)
    print(paired.groupby("pairing_method").size())
    print(f"\nWrote {OUT} — {len(paired)} pairings.")


if __name__ == "__main__":
    sys.exit(main())
