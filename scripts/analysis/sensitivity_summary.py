"""Compare sensitivity / benchmark runs with the reference run.

For each run: network medians of test KGE_ab, NSE and RMSE, the median
per-station change against the reference, and the number of stations that got
better or worse by more than 0.05 KGE_ab. Runs that cover only some stations
are compared on those stations, and the reference network medians are
recomputed with those stations replaced.

Usage:
    python scripts/analysis/sensitivity_summary.py [--run ...] sens_ups_nearest sens_ups_bestcorr ...
"""
import argparse

import pandas as pd

from common import REPO, analysis_dir, load_results, run_dir


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    ap.add_argument("runs", nargs="+")
    args = ap.parse_args()
    run = run_dir(args.run)
    ref = load_results(run)

    rows = [{"run": "reference", "n": len(ref),
             **{f"{m}_test_med": ref[f"{m}_test"].median() for m in ("kge_ab", "nse", "rmse")}}]
    for name in args.runs:
        d = REPO / "workspace" / "results" / name
        if not (d / "gw_fit_results.csv").exists():
            print(f"skip {name}: no results")
            continue
        s = load_results(d)
        common = s.index.intersection(ref.index)
        delta = s.loc[common, "kge_ab_test"] - ref.loc[common, "kge_ab_test"]
        merged = ref.copy()
        cols = [c for c in s.columns if c in merged.columns
                and (c.endswith(("_cal", "_sel", "_test")) or c in ("group_name", "model"))]
        merged.loc[common, cols] = s.loc[common, cols]
        row = {"run": name, "n": len(common),
               **{f"{m}_test_med": merged[f"{m}_test"].median() for m in ("kge_ab", "nse", "rmse")},
               "delta_kge_ab_med": delta.median(),
               "n_better_0.05": int((delta > 0.05).sum()), "n_worse_0.05": int((delta < -0.05).sum())}
        if "group_name" in s:
            g = merged.groupby("group_name")["kge_ab_test"].median()
            row.update({f"kge_ab_test_med_{k}": v for k, v in g.items()})
            row["n_coastal"] = int((merged.group_name == "coastal").sum())
        if "model" in s:
            row["variant_changed"] = int((s.loc[common, "model"] != ref.loc[common, "model"]).sum())
        rows.append(row)
    out = pd.DataFrame(rows)
    path = analysis_dir(run) / "sensitivity_summary.csv"
    if path.exists():
        prev = pd.read_csv(path)
        out = pd.concat([prev[~prev.run.isin(out.run)], out], ignore_index=True)
    out.to_csv(path, index=False)
    with pd.option_context("display.width", 200):
        print(out.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
