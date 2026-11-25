"""
Run time-varying z model for all active stations.

Usage:
    cd single_tankV2
    python srcs/03_run_model_tz.py                    # first run
    python srcs/03_run_model_tz.py --force             # re-run all (cache protects best)
    python srcs/03_run_model_tz.py --run-id tz_v2      # different output tag
"""

import argparse
import sys
from pathlib import Path

# Ensure rklib (in parent code_space dir) and srcs are importable
_base = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_base / "srcs"))
sys.path.insert(0, str(_base.parent))

import pandas as pd

from gw_shell_tz import run_station_tz, load_best_cache, save_best_cache


def load_gray_box_input(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required_cols = ["gw_st", "st_id", "gw_TM_X97", "gw_TM_Y97",
                     "ups_id", "rf_id", "lag_days", "group", "active"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


def build_station_arguments(row: pd.Series) -> dict:
    args = {
        "gw_st": str(row["gw_st"]),
        "st_id": str(row["st_id"]),
        "gw_x": str(row["gw_TM_X97"]),
        "gw_y": str(row["gw_TM_Y97"]),
        "ups_id": "none" if pd.isna(row.get("ups_id")) else str(row["ups_id"]),
        "rf_id": str(row["rf_id"]),
        "lag_days": str(row["lag_days"]),
        "group_name": str(row["group"]),
    }
    return args


def merge_results(output_root: Path) -> None:
    per_station_dir = output_root / "per_station"
    if not per_station_dir.exists():
        print("No per-station results directory found.")
        return
    csv_files = sorted(per_station_dir.glob("*.csv"))
    if not csv_files:
        print("No per-station result files found.")
        return
    frames = [pd.read_csv(f) for f in csv_files]
    df_all = pd.concat(frames, ignore_index=True)
    out_path = output_root / "gw_fit_results_tz.csv"
    df_all.to_csv(out_path, index=False)
    print(f"\nMerged {len(csv_files)} station(s) -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Run z(t) model for all stations.")
    parser.add_argument("--input", default="data/gray_box_input.csv")
    parser.add_argument("--force", action="store_true", help="Re-run all stations")
    parser.add_argument("--run-id", default="tz", help="Output tag (workspace/results/{run-id}/)")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[1]
    input_path = Path(args.input)
    csv_path = input_path if input_path.is_absolute() else base_dir / input_path

    output_root = base_dir / "workspace" / "results" / args.run_id
    output_root.mkdir(parents=True, exist_ok=True)

    cache_path = output_root / "best_params.json"
    best_cache = load_best_cache(cache_path)

    df = load_gray_box_input(csv_path)
    df["active"] = pd.to_numeric(df["active"], errors="coerce").fillna(0).astype(int)
    df_active = df[df["active"] == 1]

    if df_active.empty:
        print("No active stations found.")
        return

    # Skip done stations unless --force
    if not args.force:
        per_station_dir = output_root / "per_station"
        done = {f.stem for f in per_station_dir.glob("*.csv")} if per_station_dir.exists() else set()
        skipped = df_active["st_id"].astype(str).isin(done)
        if skipped.any():
            print(f"Skipping {skipped.sum()} already-done station(s). Use --force to re-run.")
        df_active = df_active[~skipped]

    if df_active.empty:
        print("All stations done. Use --force to re-run.")
        merge_results(output_root)
        return

    n_stations = len(df_active)
    print(f"{'='*70}")
    print(f"TIME-VARYING z MODEL — ALL STATIONS")
    print(f"Stations: {n_stations}  |  Cached: {len(best_cache)}")
    print(f"Output: {output_root}")
    print(f"{'='*70}")

    for idx, (_, row) in enumerate(df_active.iterrows()):
        st_id = str(row["st_id"])
        group = str(row["group"])
        print(f"\n[{idx+1}/{n_stations}] Station {st_id} | group={group}")

        args_dict = build_station_arguments(row)
        args_dict["output_root"] = str(output_root)

        try:
            run_station_tz(args_dict, best_cache, cache_path)
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

    merge_results(output_root)
    print(f"\nBest params cached -> {cache_path}")


if __name__ == "__main__":
    main()
