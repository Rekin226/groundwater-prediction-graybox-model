"""Spread of fitted parameters and skill across optimiser seeds.

Combines the reference run (seed 42) with the seed runs (selected variant
only) and reports per station and parameter: median, 10th-90th percentile
range, relative spread (p90 - p10) / |median|, and how often the value sits at
a bound; plus the spread of calibration RMSE and test KGE_ab across seeds.
Network summary: per parameter, the median relative spread and the number of
stations whose spread exceeds 50% of the median.

Usage:
    python scripts/analysis/multistart_summary.py [--run ...] [--seed-glob 'sens_seed_*']
"""
import argparse

import numpy as np
import pandas as pd

from common import REPO, analysis_dir, load_results, run_dir

SKIP = {"st_id", "gw_st", "ups_id", "rf_id", "group_name", "model", "selection_basis", "stft_align",
        "fixed", "seed", "rain_lag_days", "up_lag_days", "n_cal", "n_sel", "n_test"}


def param_columns(df: pd.DataFrame) -> list[str]:
    metric_suffixes = ("_cal", "_sel", "_test", "_std", "_at_bound")
    return [c for c in df.columns if c not in SKIP and not c.endswith(metric_suffixes)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    ap.add_argument("--seed-glob", default="sens_seed_*")
    args = ap.parse_args()
    run = run_dir(args.run)
    ref = load_results(run).assign(seed=42)
    runs = [ref]
    for d in sorted((REPO / "workspace" / "results").glob(args.seed_glob)):
        if (d / "gw_fit_results.csv").exists():
            runs.append(load_results(d))
    allr = pd.concat([r.reset_index() for r in runs], ignore_index=True)
    print(f"{len(runs)} runs (seeds: {sorted(allr.seed.unique().tolist())})")

    params = param_columns(ref)
    rows = []
    for st, g in allr.groupby("st_id"):
        row = {"st_id": st, "group": g.group_name.iloc[0], "model": ref.loc[st, "model"], "n_seeds": len(g),
               "rmse_cal_range": g.rmse_cal.max() - g.rmse_cal.min(),
               "kge_ab_test_range": g.kge_ab_test.max() - g.kge_ab_test.min(),
               "kge_ab_test_min": g.kge_ab_test.min(), "kge_ab_test_max": g.kge_ab_test.max()}
        for p in params:
            v = g[p].dropna().to_numpy(float)
            if len(v) < 2:
                continue
            med = np.median(v)
            p10, p90 = np.percentile(v, [10, 90])
            row[f"{p}_median"] = med
            row[f"{p}_relspread"] = (p90 - p10) / abs(med) if med != 0 else np.nan
            ab = g.get(f"{p}_at_bound")
            if ab is not None:
                row[f"{p}_at_bound_frac"] = float((ab == True).mean())  # noqa: E712
        rows.append(row)
    df = pd.DataFrame(rows)
    out = analysis_dir(run)
    df.to_csv(out / "multistart_per_station.csv", index=False)

    summ = []
    for p in params:
        col = f"{p}_relspread"
        if col not in df:
            continue
        s = df[col].dropna()
        summ.append({"param": p, "n_stations": len(s), "median_relspread": s.median(),
                     "n_relspread_gt_0.5": int((s > 0.5).sum()),
                     "median_at_bound_frac": df[f"{p}_at_bound_frac"].median() if f"{p}_at_bound_frac" in df else np.nan})
    summ = pd.DataFrame(summ)
    summ.to_csv(out / "multistart_summary.csv", index=False)
    print(summ.round(3).to_string(index=False))
    print(f"\nTest KGE_ab range across seeds: median {df.kge_ab_test_range.median():.3f}, "
          f"max {df.kge_ab_test_range.max():.3f}; calibration RMSE range median {df.rmse_cal_range.median():.4f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
