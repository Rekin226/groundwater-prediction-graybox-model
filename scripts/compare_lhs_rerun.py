"""Diff old (uniform-init) vs new (LHS-init) calibration results on every
reported summary quantity. Usage:
    python scripts/compare_lhs_rerun.py workspace/results/final workspace/results/lhs_final
"""
import sys
import numpy as np
import pandas as pd

OLD = sys.argv[1] if len(sys.argv) > 1 else "workspace/results/final"
NEW = sys.argv[2] if len(sys.argv) > 2 else "workspace/results/lhs_final"


def load(d):
    df = pd.read_csv(f"{d}/gw_fit_results.csv")
    df = df.set_index("st_id")
    return df


def ill_mask(df):
    return df.kge_ill_conditioned.astype(str).str.lower().isin(["true", "1", "1.0"])


def summarize(df, tag):
    ill = ill_mask(df)
    dfv = df[~ill]
    def tiers(x):
        x = pd.Series(x).dropna()
        return (int((x >= .70).sum()), int(((x >= .50) & (x < .70)).sum()), int((x < .50).sum()))
    out = {
        "n": len(df),
        "inland": int((df.group_name == "inland").sum()),
        "coastal": int((df.group_name == "coastal").sum()),
        "mix": df.model.value_counts().to_dict(),
        "z(t)": int(df.model.astype(str).str.contains("tz").sum()),
        "filtered": int(df.model.astype(str).str.contains("filtered").sum()),
        "cal_R2_med": round(df.r2.median(), 4),
        "cal_R2_mean": round(df.r2.mean(), 4),
        "cal_KGE_med": round(df.kge.median(), 4),
        "cal_RMSE_mean": round(df.rmse.mean(), 4),
        "val_R2_med": round(df.r2_val.median(), 4),
        "val_KGE_med_exclill": round(dfv.kge_val.median(), 4),
        "val_KGE_mean_exclill": round(dfv.kge_val.mean(), 4),
        "val_RMSE_mean": round(df.rmse_val.mean(), 4),
        "cal_tiers_KGE": tiers(df.kge),
        "val_tiers_KGE": tiers(df.kge_val),
        "ill": list(df.index[ill]),
        "low_val_stations": sorted(df.index[df.kge_val < .50].tolist()),
    }
    print(f"\n===== {tag} =====")
    for k, v in out.items():
        print(f"  {k:20s}: {v}")
    return out


def main():
    old, new = load(OLD), load(NEW)
    o, n = summarize(old, f"OLD {OLD}"), summarize(new, f"NEW {NEW}")

    print("\n===== PER-STATION SELECTED-VARIANT CHANGES =====")
    common = old.index.intersection(new.index)
    changed = [(s, old.loc[s, "model"], new.loc[s, "model"]) for s in common
               if old.loc[s, "model"] != new.loc[s, "model"]]
    print(f"  variant changed at {len(changed)} / {len(common)} stations:")
    for s, a, b in changed:
        print(f"    {s}: {a} -> {b}")

    print("\n===== PER-STATION ups_id CHANGES =====")
    ch = [(s, old.loc[s, "ups_id"], new.loc[s, "ups_id"]) for s in common
          if str(old.loc[s, "ups_id"]) != str(new.loc[s, "ups_id"])]
    print(f"  ups_id changed at {len(ch)} stations:")
    for s, a, b in ch:
        print(f"    {s}: {a} -> {b}")

    print("\n===== KEY-METRIC DELTAS (new - old) =====")
    for k in ["cal_R2_med", "cal_KGE_med", "cal_RMSE_mean", "val_R2_med",
              "val_KGE_med_exclill", "val_RMSE_mean"]:
        print(f"  {k:22s}: {o[k]:8.4f} -> {n[k]:8.4f}  (Δ {n[k]-o[k]:+.4f})")
    print(f"  low-val set added : {sorted(set(n['low_val_stations'])-set(o['low_val_stations']))}")
    print(f"  low-val set removed: {sorted(set(o['low_val_stations'])-set(n['low_val_stations']))}")


if __name__ == "__main__":
    main()
