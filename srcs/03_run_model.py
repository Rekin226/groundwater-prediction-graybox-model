import sys
from pathlib import Path
from multiprocessing import Pool, cpu_count

# Ensure rklib (parent code_space dir) and srcs are importable
_base = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_base / "srcs"))
sys.path.insert(0, str(_base.parent))

import pandas as pd

import gw_shell


def load_gray_box_input(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required_cols = [
        "gw_st",
        "st_id",
        "gw_TM_X97",
        "gw_TM_Y97",
        "ups_id",
        "rf_id",
        "lag_days",
        "group",
        "active",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in gray_box_input.csv: {missing}")
    return df


def filter_active(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["active"] = pd.to_numeric(df["active"], errors="coerce").fillna(0).astype(int)
    df_active = df[df["active"] == 1]
    return df_active


def build_station_arguments(row: pd.Series) -> dict:
    """Build command-line arguments for a single groundwater station."""
    args = {}
    args["gw_st"] = str(row["gw_st"])
    args["st_id"] = str(row["st_id"])
    args["gw_x"] = str(row["gw_TM_X97"])
    args["gw_y"] = str(row["gw_TM_Y97"])

    ups_val = row.get("ups_id")
    args["ups_id"] = "none" if pd.isna(ups_val) else str(ups_val)

    args["rf_id"] = str(row["rf_id"])
    args["lag_days"] = str(row["lag_days"])

    # coastal / inland
    args["group_name"] = str(row["group"])

    return args



def merge_results(output_root: Path) -> None:
    """Merge all per-station CSV files into a single gw_fit_results.csv."""
    per_station_dir = output_root / "per_station"
    if not per_station_dir.exists():
        print("No per-station results directory found.")
        return

    csv_files = sorted(per_station_dir.glob("*.csv"))
    if not csv_files:
        print("No per-station result files found to merge.")
        return

    frames = [pd.read_csv(f) for f in csv_files]
    df_all = pd.concat(frames, ignore_index=True)

    out_path = output_root / "gw_fit_results.csv"
    df_all.to_csv(out_path, index=False)
    print(f"Merged {len(csv_files)} station result(s) → {out_path}")

    # Also merge all-variants CSVs if they exist
    all_variants_dir = output_root / "all_variants"
    if all_variants_dir.exists():
        av_files = sorted(all_variants_dir.glob("*.csv"))
        if av_files:
            av_frames = [pd.read_csv(f) for f in av_files]
            df_av = pd.concat(av_frames, ignore_index=True)
            av_path = output_root / "all_variants_results.csv"
            df_av.to_csv(av_path, index=False)
            print(f"Merged {len(av_files)} all-variants file(s) → {av_path}")


def _run_single_station(task: tuple) -> None:
    """Worker function to run calibration for a single station via direct function call."""
    row_dict, base_dir_str, output_root_str, models_str = task
    row = pd.Series(row_dict)
    args_dict = build_station_arguments(row)
    args_dict["output_root"] = output_root_str
    args_dict["models"] = models_str
    station_label = args_dict.get('st_id', args_dict.get('gw_st', 'unknown'))
    print(f"Running station st_id={station_label} group={args_dict['group_name']} models={models_str}")
    gw_shell.run_station(args_dict)


def run_group(group_name: str, df_group: pd.DataFrame, base_dir: Path, output_root: Path, models: str = "base,filtered,base_tz,filtered_tz") -> None:
    # Ensure we only use rows for this group
    df_group = df_group.copy()
    df_group = df_group[df_group["group"] == group_name]

    if df_group.empty:
        print(f"No active stations for group '{group_name}'")
        return

    tasks = [
        (row.to_dict(), str(base_dir), str(output_root), models)
        for _, row in df_group.iterrows()
    ]

    n_proc = min(len(tasks), cpu_count())
    print(
        f"Running group '{group_name}' with {len(tasks)} stations "
        f"using {n_proc} processes"
    )

    with Pool(processes=n_proc) as pool:
        pool.map(_run_single_station, tasks)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run gray-box model for active stations.")
    parser.add_argument(
        "--input",
        default="data/gray_box_input.csv",
        help="Path to input CSV (relative to project root or absolute). "
             "Use data/gray_box_input_optimized.csv to re-run with optimized pairings.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run stations even if a per-station result file already exists.",
    )
    parser.add_argument(
        "--run-id",
        default="initial",
        help="Tag for this run. Results are written to workspace/results/{run-id}/. "
             "Use 'optimized' when re-running with gray_box_input_optimized.csv.",
    )
    parser.add_argument(
        "--station",
        default=None,
        help="Run a specific station only (e.g. --station st14).",
    )
    parser.add_argument(
        "--models",
        default="base,filtered,base_tz,filtered_tz",
        help="Comma-separated model variants to run (default: all 4). "
             "Options: base, filtered, base_tz, filtered_tz.",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[1]
    input_path = Path(args.input)
    csv_path = input_path if input_path.is_absolute() else base_dir / input_path

    output_root = base_dir / "workspace" / "results" / args.run_id
    output_root.mkdir(parents=True, exist_ok=True)
    print(f"Output root: {output_root}")

    df = load_gray_box_input(csv_path)
    df_active = filter_active(df)

    if df_active.empty:
        print("No active rows found in gray_box_input.csv")
        return

    # Filter to specific station if requested
    if args.station:
        df_active = df_active[df_active["st_id"].astype(str) == args.station]
        if df_active.empty:
            print(f"Station '{args.station}' not found or not active.")
            return
        print(f"Running single station: {args.station}")

    # Skip stations that already have results unless --force is given
    if not args.force:
        per_station_dir = output_root / "per_station"
        done = {f.stem for f in per_station_dir.glob("*.csv")} if per_station_dir.exists() else set()
        skipped = df_active["st_id"].astype(str).isin(done)
        if skipped.any():
            print(f"Skipping {skipped.sum()} already-done station(s). Use --force to re-run.")
        df_active = df_active[~skipped]

    if df_active.empty:
        print("All active stations already have results. Use --force to re-run.")
        merge_results(output_root)
        return

    models = args.models
    print(f"Models: {models}")

    groups_present = sorted(df_active["group"].unique().tolist())
    print(f"Active groups present: {groups_present}")

    for group_name in ["inland", "coastal"]:
        if group_name in groups_present:
            df_group = df_active[df_active["group"] == group_name]
            run_group(group_name, df_group, base_dir, output_root, models=models)

    merge_results(output_root)


if __name__ == "__main__":
    main()
