"""Side-by-side summary of two calibration runs.

Handles both result layouts: the older two-period one (unsuffixed calibration
metrics plus *_val) and the current three-period one (*_cal, *_sel, *_test).
For the older layout the single out-of-sample period is reported under "val";
classic KGE is shown for it because kge_ab was not recorded.

Usage:
    python scripts/compare_runs.py OLD_RUN_DIR NEW_RUN_DIR
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load(run_dir: Path) -> pd.DataFrame:
    return pd.read_csv(run_dir / "gw_fit_results.csv").set_index("st_id")


def is_three_period(df: pd.DataFrame) -> bool:
    return "kge_ab_test" in df.columns


def tiers(x: pd.Series) -> str:
    x = x.dropna()
    return f"{(x >= 0.7).sum()}/{((x >= 0.5) & (x < 0.7)).sum()}/{(x < 0.5).sum()}"


def summary(df: pd.DataFrame, tag: str) -> list[dict]:
    rows = []
    if is_three_period(df):
        for p in ("cal", "sel", "test"):
            rows.append({
                "run": tag, "period": p, "n": int(df[f"kge_ab_{p}"].notna().sum()),
                "KGE_ab med": df[f"kge_ab_{p}"].median(),
                "KGE classic med": df[f"kge_{p}"].median(),
                "NSE/R2 med": df[f"nse_{p}"].median(),
                "RMSE mean (m)": df[f"rmse_{p}"].mean(),
                "MAE mean (m)": df[f"mae_{p}"].mean(),
                "tiers g/m/l (KGE_ab)": tiers(df[f"kge_ab_{p}"]),
            })
    else:
        ill = df.get("kge_ill_conditioned", pd.Series(False, index=df.index))
        ill = ill.astype(str).str.lower().isin(["true", "1", "1.0"])
        for p, k, r2, rmse in (("cal", "kge", "r2", "rmse"), ("val", "kge_val", "r2_val", "rmse_val")):
            kv = df.loc[~ill, k] if p == "val" else df[k]
            rows.append({
                "run": tag, "period": p, "n": int(kv.notna().sum()),
                "KGE_ab med": np.nan,
                "KGE classic med": kv.median(),
                "NSE/R2 med": df[r2].median(),
                "RMSE mean (m)": df[rmse].mean(),
                "MAE mean (m)": np.nan,
                "tiers g/m/l (KGE_ab)": f"(classic) {tiers(kv)}",
            })
    return rows


def structure(df: pd.DataFrame) -> dict:
    model = df["model"].astype(str)
    group = df["group_name"].astype(str).str.lower()
    return {
        "variants": model.value_counts().to_dict(),
        "coastal": sorted(df.index[group == "coastal"], key=lambda s: int(s[2:])),
        "z(t) selected": int(model.str.endswith("_tz").sum()),
        "filtered selected": int(model.str.startswith("filtered").sum()),
    }


def main() -> int:
    old_dir, new_dir = Path(sys.argv[1]), Path(sys.argv[2])
    old, new = load(old_dir), load(new_dir)
    pd.set_option("display.width", 160)
    table = pd.DataFrame(summary(old, old_dir.name) + summary(new, new_dir.name))
    print(table.round(3).to_string(index=False))
    print()
    for tag, df in ((old_dir.name, old), (new_dir.name, new)):
        print(tag, structure(df))
    common = old.index.intersection(new.index)
    changed = [s for s in common if old.loc[s, "model"] != new.loc[s, "model"]]
    print(f"\nVariant changed at {len(changed)}/{len(common)} stations")
    for col in ("ups_id", "rf_id", "group_name"):
        diff = [s for s in common if str(old.loc[s, col]) != str(new.loc[s, col])]
        print(f"{col} changed at {len(diff)} stations: {diff}")
    if is_three_period(new):
        flags = [c for c in new.columns if c.endswith("_at_bound")]
        # Flags are NaN where a parameter does not exist for that variant/group.
        hits = {c[:-9]: f"{int((new[c] == True).sum())}/{int(new[c].notna().sum())}"  # noqa: E712
                for c in flags if (new[c] == True).any()}  # noqa: E712
        print(f"\nSelected-variant parameters at a bound: {hits}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
