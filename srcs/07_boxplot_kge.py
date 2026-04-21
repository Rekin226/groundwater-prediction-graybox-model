"""Boxplot of KGE across 61 stations, calibration vs validation, split by group.

Layout: 2 rows (Calibration / Validation) x 4 columns (M1-M4 variants).
Each panel shows two boxes — Inland (n=54) vs Coastal (n=7) — of KGE values
across stations, with outliers as dots and a dashed zero-line.

Inputs
------
- workspace/results/<run_id>/all_variants_results.csv  (st_id, model, kge, kge_val)
- workspace/results/<run_id>/gw_fit_results.csv        (st_id -> group_name lookup)

Output
------
workspace/results/<run_id>/figures/boxplot_kge_by_group.tiff

Notes
-----
- Rows flagged ``kge_ill_conditioned`` are excluded from aggregate box statistics
  (same convention the Makefile ``compare`` target applies to KGE summaries).
- KGE values are clipped to [ymin, ymax] for display so extreme outliers stack at
  the axis edge as dots (matches the reference layout). Box quartiles and whiskers
  are unaffected in practice because the clipped points are already far-outliers.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CODE_SPACE = Path(__file__).resolve().parents[2]
RKLIB_DIR = CODE_SPACE / "rklib"

for _module_name in ("fig_base",):
    _module_path = RKLIB_DIR / f"{_module_name}.py"
    _spec = importlib.util.spec_from_file_location(f"rklib.{_module_name}", str(_module_path))
    _module = importlib.util.module_from_spec(_spec)
    sys.modules[f"rklib.{_module_name}"] = _module
    _spec.loader.exec_module(_module)

from rklib.fig_base import add_panel_label, savefig as rklib_savefig, setup_font  # noqa: E402


VARIANT_ORDER = [
    ("base",        "M1: Direct"),
    ("filtered",    "M2: Filtered"),
    ("base_tz",     "M3: Direct, z(t)"),
    ("filtered_tz", "M4: Filtered, z(t)"),
]

GROUP_ORDER = ["inland", "coastal"]
GROUP_COLORS = {"inland": "#3a6f95", "coastal": "#e27575"}
GROUP_DISPLAY = {"inland": "Inland", "coastal": "Coastal"}

PHASE_ORDER = [("kge", "Calibration"), ("kge_val", "Validation")]


def _load_data(variants_csv: Path, fit_csv: Path) -> tuple[pd.DataFrame, int]:
    variants = pd.read_csv(variants_csv)
    fit = pd.read_csv(fit_csv)

    for col in ("st_id", "model", "kge", "kge_val"):
        if col not in variants.columns:
            raise ValueError(f"{variants_csv} missing required column: {col}")
    if "st_id" not in fit.columns or "group_name" not in fit.columns:
        raise ValueError(f"{fit_csv} must contain st_id and group_name")

    group_lookup = (fit[["st_id", "group_name"]]
                    .drop_duplicates(subset=["st_id"])
                    .assign(group_name=lambda d: d["group_name"].str.lower()))

    merged = variants.merge(group_lookup, on="st_id", how="left")
    missing_group = merged["group_name"].isna().sum()
    if missing_group:
        raise ValueError(f"{missing_group} rows in {variants_csv} have no group mapping in {fit_csv}")

    merged["kge"] = pd.to_numeric(merged["kge"], errors="coerce")
    merged["kge_val"] = pd.to_numeric(merged["kge_val"], errors="coerce")

    n_excluded = 0
    if "kge_ill_conditioned" in merged.columns:
        ill = merged["kge_ill_conditioned"].fillna(False).astype(bool)
        n_excluded = int(ill.sum())
        merged = merged.loc[~ill].copy()

    return merged, n_excluded


def _draw_panel(ax, data: pd.DataFrame, metric_col: str, group_counts: dict[str, int],
                ymin: float, ymax: float) -> None:
    box_data = []
    positions = []
    colors = []
    labels = []
    for i, group in enumerate(GROUP_ORDER):
        values = data.loc[data["group_name"] == group, metric_col].dropna().to_numpy()
        values = np.clip(values, ymin, ymax)
        box_data.append(values)
        positions.append(i + 1)
        colors.append(GROUP_COLORS[group])
        labels.append(f"{GROUP_DISPLAY[group]} (n={group_counts[group]})")

    bp = ax.boxplot(
        box_data,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showfliers=True,
        flierprops=dict(marker="o", markerfacecolor="#6b6b6b",
                        markeredgecolor="#6b6b6b", markersize=3.5, alpha=0.8),
        medianprops=dict(color="white", linewidth=1.8),
        whiskerprops=dict(color="#333333", linewidth=1.0),
        capprops=dict(color="#333333", linewidth=1.0),
        boxprops=dict(linewidth=0.8, edgecolor="#333333"),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)

    ax.axhline(0.0, color="#888888", linestyle="--", linewidth=0.9, zorder=0)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=9)
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, color="#bdbdbd", alpha=0.7)
    ax.set_axisbelow(True)


def plot_boxplot(df: pd.DataFrame, out_path: Path, ymin: float, ymax: float) -> None:
    group_counts = {g: int((df.loc[df["model"] == VARIANT_ORDER[0][0], "group_name"] == g).sum())
                    for g in GROUP_ORDER}

    fig, axes = plt.subplots(
        nrows=2, ncols=4,
        figsize=(13.5, 7.0),
        sharey=True,
    )

    panel_idx = 0
    for row_i, (metric_col, phase_label) in enumerate(PHASE_ORDER):
        for col_i, (variant_key, variant_label) in enumerate(VARIANT_ORDER):
            ax = axes[row_i, col_i]
            sub = df[df["model"] == variant_key]
            _draw_panel(ax, sub, metric_col, group_counts, ymin, ymax)

            if row_i == 0:
                ax.set_title(variant_label, fontsize=11, fontweight="bold", pad=8)
            if col_i == 0:
                ax.set_ylabel(f"{phase_label}\nKGE", fontsize=11, fontweight="bold")
            ax.set_ylim(ymin, ymax)

            add_panel_label(ax, "abcdefgh"[panel_idx], fontsize=11, fontweight="bold")
            panel_idx += 1

    fig.suptitle("KGE distribution across 61 stations — calibration vs validation",
                 fontsize=13, fontweight="bold", y=0.995)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    rklib_savefig(fig, str(out_path))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Boxplot of KGE by group (Inland/Coastal) for calibration vs validation, across 4 model variants.")
    parser.add_argument("--run-id", default="final",
                        help="Run tag under workspace/results/ (default: final)")
    parser.add_argument("--variants-results", type=Path, default=None,
                        help="Override path to all_variants_results.csv")
    parser.add_argument("--fit-results", type=Path, default=None,
                        help="Override path to gw_fit_results.csv (used for group_name lookup)")
    parser.add_argument("--outdir", type=Path, default=None,
                        help="Output directory (default: workspace/results/<run_id>/figures)")
    parser.add_argument("--ymin", type=float, default=-1.0,
                        help="Y-axis lower bound; values below are clipped to this edge (default: -1.0)")
    parser.add_argument("--ymax", type=float, default=1.0,
                        help="Y-axis upper bound (default: 1.0)")
    args = parser.parse_args()

    run_root = Path("workspace/results") / args.run_id
    variants_csv = args.variants_results if args.variants_results is not None \
        else run_root / "all_variants_results.csv"
    fit_csv = args.fit_results if args.fit_results is not None \
        else run_root / "gw_fit_results.csv"
    outdir = args.outdir if args.outdir is not None else run_root / "figures"
    outdir.mkdir(parents=True, exist_ok=True)

    setup_font()

    df, n_excluded = _load_data(variants_csv, fit_csv)
    out_path = outdir / "boxplot_kge_by_group.tiff"
    plot_boxplot(df, out_path, args.ymin, args.ymax)

    n_inland = int((df.loc[df["model"] == VARIANT_ORDER[0][0], "group_name"] == "inland").sum())
    n_coastal = int((df.loc[df["model"] == VARIANT_ORDER[0][0], "group_name"] == "coastal").sum())
    print(f"Wrote {out_path}")
    print(f"  Inland n={n_inland}, Coastal n={n_coastal}; "
          f"excluded {n_excluded} ill-conditioned rows; "
          f"display clipped to [{args.ymin}, {args.ymax}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
