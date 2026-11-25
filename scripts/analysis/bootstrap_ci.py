"""Station-bootstrap confidence intervals for network summary statistics.

For each period and metric: the median over stations with a 95% percentile
interval from resampling stations with replacement. For the coastal - inland
difference in medians, each group is resampled separately; a two-sided
permutation p-value (group labels shuffled) is also reported.

Usage:
    python scripts/analysis/bootstrap_ci.py [--run workspace/results/final_rev1]
"""
import argparse

import numpy as np
import pandas as pd

from common import PERIODS, analysis_dir, load_results, run_dir

METRICS = ("kge_ab", "nse", "kge", "rmse", "mae")
N_BOOT = 10000
N_PERM = 10000


def boot_median(x: np.ndarray, rng) -> tuple[float, float, float]:
    x = x[np.isfinite(x)]
    idx = rng.integers(0, len(x), size=(N_BOOT, len(x)))
    meds = np.median(x[idx], axis=1)
    return float(np.median(x)), float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def boot_diff(a: np.ndarray, b: np.ndarray, rng) -> tuple[float, float, float, float]:
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    da = np.median(a[rng.integers(0, len(a), size=(N_BOOT, len(a)))], axis=1)
    db = np.median(b[rng.integers(0, len(b), size=(N_BOOT, len(b)))], axis=1)
    diff = da - db
    obs = float(np.median(a) - np.median(b))
    pooled = np.concatenate([a, b])
    perm = np.empty(N_PERM)
    for i in range(N_PERM):
        p = rng.permutation(pooled)
        perm[i] = np.median(p[:len(a)]) - np.median(p[len(a):])
    pval = float((np.abs(perm) >= abs(obs)).mean())
    return obs, float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5)), pval


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    args = ap.parse_args()
    run = run_dir(args.run)
    df = load_results(run)
    rng = np.random.default_rng(20261018)

    rows = []
    for p in PERIODS:
        for m in METRICS:
            col = f"{m}_{p}"
            med, lo, hi = boot_median(df[col].to_numpy(float), rng)
            rows.append({"period": p, "metric": m, "group": "all", "n": int(df[col].notna().sum()),
                         "median": med, "ci_lo": lo, "ci_hi": hi})
            for g in ("inland", "coastal"):
                x = df.loc[df.group_name == g, col].to_numpy(float)
                med, lo, hi = boot_median(x, rng)
                rows.append({"period": p, "metric": m, "group": g, "n": int(np.isfinite(x).sum()),
                             "median": med, "ci_lo": lo, "ci_hi": hi})
            a = df.loc[df.group_name == "coastal", col].to_numpy(float)
            b = df.loc[df.group_name == "inland", col].to_numpy(float)
            d, lo, hi, pv = boot_diff(a, b, rng)
            rows.append({"period": p, "metric": m, "group": "coastal-inland", "n": len(a) + len(b),
                         "median": d, "ci_lo": lo, "ci_hi": hi, "perm_p": pv})

    out = pd.DataFrame(rows)
    path = analysis_dir(run) / "bootstrap_ci.csv"
    out.to_csv(path, index=False)
    show = out[(out.metric.isin(["kge_ab", "nse", "rmse"]))]
    print(show.round(3).to_string(index=False))
    print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
