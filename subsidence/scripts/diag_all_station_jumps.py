"""All-station scan of max |Δζ| to identify artefact-class jumps.

Threshold logic:
    > 10 cm/day  → Tier 1 (auto-handle: spike-NaN or exclude)
    5–10 cm/day  → Tier 2 (grey zone, flag for manual review)
    < 5 cm/day   → Tier 3 (noise floor, no action)
"""
from __future__ import annotations
import sys
import urllib.parse
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

CACHE = Path("data/ls_cache")
MASTER = Path("subsidence/data/sub_station_master.csv")
EXCLUDED_DATASETS = ("ls-wra-dbm-obs",)


def _load_zeta_cm(sub_id, sub_dataset):
    sid_encoded = urllib.parse.quote(str(sub_id), safe="")
    p = CACHE / f"{sub_dataset}__{sid_encoded}.parquet"
    if not p.exists():
        return None
    raw = pd.read_parquet(p)
    if sub_dataset == "ls-wra-mlcw-obs":
        cols = sorted([c for c in raw.columns if c.startswith("NO")],
                      key=lambda c: int(c[2:]))
        for c in reversed(cols):
            s = raw[c].dropna()
            if len(s) >= 12:
                return ((s.iloc[0] - raw[c]) * 100.0).dropna()
        return None
    s = raw["value"].dropna() if "value" in raw.columns else raw.dropna()
    return ((s.iloc[0] - s) * 100.0)


def main():
    master = pd.read_csv(MASTER)
    master = master[master["active"] == 1]
    master = master[~master["sub_dataset"].isin(EXCLUDED_DATASETS)]

    rows = []
    for _, r in master.iterrows():
        sub_id, ds = r["sub_id"], r["sub_dataset"]
        z = _load_zeta_cm(sub_id, ds)
        if z is None or len(z) < 5:
            continue
        dz = z.diff().abs()
        rows.append({
            "sub_id": sub_id,
            "ds": ds.replace("ls-wra-", "").replace("-obs", ""),
            "n_obs": len(z),
            "max_dz_cm": float(np.nanmax(dz.values)),
            "n_ge_5": int((dz.values >= 5).sum()),
            "n_ge_10": int((dz.values >= 10).sum()),
            "n_ge_20": int((dz.values >= 20).sum()),
        })
    df = pd.DataFrame(rows).sort_values("max_dz_cm", ascending=False)

    def tier(x):
        if x > 10: return "T1 (auto-handle)"
        if x > 5:  return "T2 (manual review)"
        return     "T3 (noise floor)"
    df["tier"] = df["max_dz_cm"].apply(tier)

    print(f"\n{'sub_id':<14} {'ds':<5} {'n_obs':>5} "
          f"{'max|Δζ|cm':>11} {'≥5cm':>5} {'≥10cm':>6} {'≥20cm':>6}  tier")
    print("-" * 80)
    for _, r in df.iterrows():
        print(f"{r['sub_id']:<14} {r['ds']:<5} {r['n_obs']:>5} "
              f"{r['max_dz_cm']:>11.2f} {r['n_ge_5']:>5} "
              f"{r['n_ge_10']:>6} {r['n_ge_20']:>6}  {r['tier']}")

    print("\n--- Tier summary ---")
    print(df.groupby("tier").size().to_string())
    df.to_csv("workspace/results_sub/v10_anchor_fix/all_station_jumps.csv", index=False)
    print("\nWrote all_station_jumps.csv")


if __name__ == "__main__":
    main()
