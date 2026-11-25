"""Network summaries without suspect records, and the z(t) drift extrapolation.

1. Flagged records: stations with more than FLAG_PCT % of hourly values in
   runs held constant for >= 48 h (probable gap filling before release).
   Network test medians with and without them, as targets and as upstream
   inputs, and the drought-window bias counts without them.
2. z(t) extrapolation: for stations that selected a z(t) variant, the drift
   accumulated from the end of calibration to the end of the test period
   (|z1| x 4 yr), and the test KGE_ab of the selected z(t) variant against the
   matching constant-z variant fitted at the same station.

Usage:
    python scripts/analysis/robustness_checks.py [--run ...]
"""
import argparse

import numpy as np
import pandas as pd

from common import analysis_dir, load_results, run_dir

FLAG_PCT = 1.0
METRICS = ("kge_ab_test", "nse_test", "rmse_test")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    args = ap.parse_args()
    run = run_dir(args.run)
    res = load_results(run)
    out = analysis_dir(run)
    comp = pd.read_csv(out / "data_completeness.csv").astype({"st_id": str}).set_index("st_id")
    flagged = set(comp.index[comp["held_constant_pct"] > FLAG_PCT])
    downstream = set(res.index[res["ups_id"].astype(str).isin(flagged)])
    rows = []
    for label, drop in (("all", set()), ("without_flagged_targets", flagged),
                        ("without_flagged_targets_and_their_downstream", flagged | downstream)):
        keep = res.drop(index=list(drop & set(res.index)))
        row = {"subset": label, "n": len(keep)}
        for m in METRICS:
            row[m] = float(keep[m].median())
            for g in ("coastal", "inland"):
                row[f"{m}_{g}"] = float(keep.loc[keep["group_name"] == g, m].median())
        rows.append(row)
    flag_df = pd.DataFrame(rows)
    flag_df.to_csv(out / "flagged_station_sensitivity.csv", index=False)
    print("Flagged:", sorted(flagged), "| downstream of flagged:", sorted(downstream))
    print(flag_df.round(3).to_string(index=False))

    dr = pd.read_csv(out / "drought_event.csv").astype({"st_id": str}).set_index("st_id")
    dr_keep = dr.drop(index=list(flagged & set(dr.index)))
    for w in ("drought_sel", "drought_test"):
        if f"bias_{w}" in dr_keep:
            b = dr_keep[f"bias_{w}"]
            print(f"{w} without flagged: high at {(b > 0).sum()}/{len(b)}, median {b.median():+.2f} m")

    allv = pd.read_csv(run / "all_variants_results.csv").astype({"st_id": str})
    kge = allv.pivot(index="st_id", columns="model", values="kge_ab_test")
    tz = res[res["model"].str.endswith("_tz")].copy()
    tz["drift_4yr_m"] = (tz["z1"] * 4.0).abs()
    tz["const_z_model"] = tz["model"].str.replace("_tz", "", regex=False)
    tz["kge_ab_test_const_z"] = [kge.loc[s, m] for s, m in zip(tz.index, tz["const_z_model"])]
    tz["gain_over_const_z"] = tz["kge_ab_test"] - tz["kge_ab_test_const_z"]
    cols = ["model", "z1", "drift_4yr_m", "kge_ab_test", "kge_ab_test_const_z", "gain_over_const_z"]
    tz[cols].sort_values("drift_4yr_m", ascending=False).to_csv(out / "drift_extrapolation.csv")
    big = tz["drift_4yr_m"] > 1.0
    print(f"\nz(t) selections: {len(tz)}; drift > 1 m over 4 yr at {int(big.sum())}")
    for lab, m in (("drift > 1 m", big), ("drift <= 1 m", ~big)):
        g = tz.loc[m, "gain_over_const_z"]
        print(f"  {lab}: test KGE_ab median {tz.loc[m, 'kge_ab_test'].median():.3f}; "
              f"vs constant-z variant: median gain {g.median():+.3f}, better at {(g > 0).sum()}/{len(g)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
