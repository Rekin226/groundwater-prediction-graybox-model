"""
Re-run the 28 stations that degraded with z(t), using tighter z1 bounds (±0.5 m/yr).
Uses multiprocessing for speed.

Usage:
    cd single_tankV2
    python srcs/run_degraded_tight_bounds.py
"""

import sys
from pathlib import Path
from multiprocessing import Pool, cpu_count

_base = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_base / "srcs"))
sys.path.insert(0, str(_base.parent))

import pandas as pd
import gw_shell_tz as tz

# Override z1 bounds to tighter range
tz.Z1_BOUNDS = (-0.5, 0.5)

OUTPUT_ROOT = _base / "workspace" / "results" / "tz_tight"
CACHE_PATH = OUTPUT_ROOT / "best_params.json"

DEGRADED_STATIONS = [
    "st1","st10","st11","st15","st19","st2","st23","st24","st26","st3",
    "st34","st35","st38","st4","st40","st42","st43","st44","st46","st47",
    "st48","st49","st5","st51","st52","st53","st56","st60",
]


def build_args(row):
    return {
        "st_id": str(row["st_id"]),
        "gw_st": str(row["gw_st"]),
        "ups_id": "none" if pd.isna(row.get("ups_id")) else str(row["ups_id"]),
        "rf_id": str(row["rf_id"]),
        "lag_days": str(row["lag_days"]),
        "group_name": str(row["group"]),
        "output_root": str(OUTPUT_ROOT),
    }


def run_one(task):
    args_dict, cache_path_str = task
    st_id = args_dict["st_id"]
    cache_path = Path(cache_path_str)
    # Each worker loads its own cache (read-only in workers, merge after)
    best_cache = tz.load_best_cache(cache_path)
    try:
        result = tz.run_station_tz(args_dict, best_cache, cache_path)
        if result:
            print(f"  {st_id}: model={result['model']} R2_val={result['r2_val']:.3f} KGE_val={result['kge_val']:.3f} z1={result['z1']:.4f}")
            return result
        return None
    except Exception as e:
        print(f"  {st_id}: FAILED - {e}")
        return None


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    df_input = pd.read_csv(_base / "data" / "gray_box_input.csv")
    df_input["active"] = pd.to_numeric(df_input["active"], errors="coerce").fillna(0).astype(int)

    df_degraded = df_input[df_input["st_id"].isin(DEGRADED_STATIONS) & (df_input["active"] == 1)]

    tasks = []
    for _, row in df_degraded.iterrows():
        args = build_args(row)
        tasks.append((args, str(CACHE_PATH)))

    n_proc = min(len(tasks), max(1, cpu_count() - 1))
    print(f"Running {len(tasks)} degraded stations with Z1_BOUNDS=(-0.5, 0.5)")
    print(f"Using {n_proc} processes")
    print(f"Output: {OUTPUT_ROOT}")

    with Pool(processes=n_proc) as pool:
        results = pool.map(run_one, tasks)

    # Merge per-station CSVs
    per_station_dir = OUTPUT_ROOT / "per_station"
    csv_files = sorted(per_station_dir.glob("*.csv")) if per_station_dir.exists() else []
    if csv_files:
        df_all = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
        df_all.to_csv(OUTPUT_ROOT / "degraded_tight_results.csv", index=False)
        print(f"\nMerged {len(csv_files)} results -> {OUTPUT_ROOT / 'degraded_tight_results.csv'}")

    # Compare with original
    df_orig = pd.read_csv(_base / "workspace" / "results" / "initial" / "gw_fit_results.csv")
    df_tz_wide = pd.read_csv(_base / "workspace" / "results" / "tz" / "gw_fit_results_tz.csv")

    if csv_files:
        df_tight = pd.read_csv(OUTPUT_ROOT / "degraded_tight_results.csv")
        print(f"\n{'='*100}")
        print(f"COMPARISON: Original vs z(t) wide bounds vs z(t) tight bounds")
        print(f"{'='*100}")
        print(f"{'Station':>8s} | {'orig R2v':>8s} | {'wide R2v':>8s} | {'tight R2v':>9s} | {'tight KGE':>9s} | {'tight z1':>9s} | {'winner':>8s}")
        print(f"{'-'*100}")

        n_tight_better = 0
        n_orig_better = 0
        for _, row_t in df_tight.iterrows():
            st = row_t["st_id"]
            orig_r2 = df_orig[df_orig.st_id == st].r2_val.iloc[0]
            wide_r2 = df_tz_wide[df_tz_wide.st_id == st].r2_val.iloc[0]
            tight_r2 = row_t["r2_val"]
            tight_kge = row_t["kge_val"]
            tight_z1 = row_t["z1"]

            if tight_r2 >= orig_r2:
                winner = "tight"
                n_tight_better += 1
            else:
                winner = "orig"
                n_orig_better += 1

            print(f"{st:>8s} | {orig_r2:>8.3f} | {wide_r2:>8.3f} | {tight_r2:>9.3f} | {tight_kge:>9.3f} | {tight_z1:>+9.4f} | {winner:>8s}")

        print(f"{'-'*100}")
        print(f"Tight bounds better than original: {n_tight_better}/{len(df_tight)}")
        print(f"Original still better: {n_orig_better}/{len(df_tight)}")


if __name__ == "__main__":
    main()
