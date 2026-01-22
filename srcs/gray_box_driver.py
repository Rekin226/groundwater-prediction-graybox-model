import sys
import subprocess
from pathlib import Path
from multiprocessing import Pool, cpu_count

import pandas as pd


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


def build_command(gw_shell_path: Path, args_dict: dict) -> list:
    cmd = [sys.executable, str(gw_shell_path)]
    for key, value in args_dict.items():
        cmd.append(f"{key}={value}")
    return cmd


def _run_single_station(task: tuple) -> None:
    """Worker function to run calibration for a single station in a subprocess."""
    row_dict, base_dir_str = task
    base_dir = Path(base_dir_str)
    gw_shell_path = base_dir / "srcs" / "gw_shell.py"
    row = pd.Series(row_dict)
    args_dict = build_station_arguments(row)
    cmd = build_command(gw_shell_path, args_dict)
    station_label = args_dict.get('st_id', args_dict.get('gw_st', 'unknown'))
    print(
        f"Running station st_id={station_label} "
        f"group={args_dict['group_name']} with command:\n  {' '.join(cmd)}"
    )
    subprocess.run(cmd, check=True, cwd=base_dir / "srcs")


def run_group(group_name: str, df_group: pd.DataFrame, base_dir: Path) -> None:
    # Ensure we only use rows for this group
    df_group = df_group.copy()
    df_group = df_group[df_group["group"] == group_name]

    if df_group.empty:
        print(f"No active stations for group '{group_name}'")
        return

    tasks = [
        (row.to_dict(), str(base_dir))
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
    base_dir = Path(__file__).resolve().parents[1]
    csv_path = base_dir / "data" / "gray_box_input.csv"

    df = load_gray_box_input(csv_path)
    df_active = filter_active(df)

    if df_active.empty:
        print("No active rows found in gray_box_input.csv")
        return

    groups_present = sorted(df_active["group"].unique().tolist())
    print(f"Active groups present: {groups_present}")

    for group_name in ["inland", "coastal"]:
        if group_name in groups_present:
            df_group = df_active[df_active["group"] == group_name]
            run_group(group_name, df_group, base_dir)


if __name__ == "__main__":
    main()
