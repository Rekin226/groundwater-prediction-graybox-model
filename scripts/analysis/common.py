"""Shared helpers for the analysis scripts."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "srcs"))
sys.path.insert(0, str(REPO.parent))  # rklib lives next to the repo

import gw_shell  # noqa: E402,F401

PERIODS = ("cal", "sel", "test")
DEFAULT_RUN = REPO / "workspace" / "results" / "final_rev1"


def run_dir(path: str | None) -> Path:
    p = Path(path) if path else DEFAULT_RUN
    return p if p.is_absolute() else REPO / p


def load_results(run: Path) -> pd.DataFrame:
    df = pd.read_csv(run / "gw_fit_results.csv")
    df["st_id"] = df["st_id"].astype(str)
    df["group_name"] = df["group_name"].str.lower()
    return df.set_index("st_id")


def load_series(run: Path, st_id: str) -> pd.DataFrame:
    s = pd.read_csv(run / "series" / f"{st_id}.csv", parse_dates=["date"])
    return s


def selected_sim(series: pd.DataFrame, model: str) -> np.ndarray:
    return series[f"sim_{model}"].to_numpy()


def analysis_dir(run: Path) -> Path:
    d = run / "analysis"
    d.mkdir(parents=True, exist_ok=True)
    return d


def station_sort_key(st: str) -> int:
    return int(st[2:]) if st[2:].isdigit() else 10**6
