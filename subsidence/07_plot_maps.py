"""Subsidence-pipeline spatial maps.

Reads:
    workspace/results_sub/<run_id>/sub_fit_results.csv      (best variant per station + KGE tier)
    subsidence/data/sub_pairing.csv                         (sub-id / gw_st pairing)
    subsidence/data/sub_station_master.csv                  (subsidence station metadata + coordinates)
    data/gray_box_input.csv                                 (GW station table with TM_X97/TM_Y97)

Writes:
    workspace/maps/sub_station_map.tiff       -- KGE-tier colored points
    workspace/maps/sub_variant_map.tiff       -- 2x2 panel by best variant
    workspace/maps/sub_pairing_map.tiff       -- pairing line segments

Run:
    poetry run python subsidence/07_plot_maps.py --run-id initial
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

try:
    from rklib import setup_font, savefig as rklib_savefig
    _USE_RKLIB = True
except Exception:
    _USE_RKLIB = False


# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------

def _setup():
    if _USE_RKLIB:
        setup_font()
    else:
        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "PingFang TC",        # macOS Traditional Chinese
                "Heiti TC",           # macOS fallback
                "Noto Sans CJK TC",   # Linux/CI
                "WenQuanYi Zen Hei",  # Linux fallback
                "DejaVu Serif",
            ],
            "font.size": 10,
        })


def _save(fig, path: Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if _USE_RKLIB:
        rklib_savefig(fig, str(path), dpi=300)
    else:
        fig.savefig(str(path), dpi=300, format="tiff")
    plt.close(fig)


# ---------------------------------------------------------------------------
# KGE tier definitions (matches project convention)
# ---------------------------------------------------------------------------

KGE_TIERS = [
    ("good (>=0.7)",    0.7,      np.inf,   "#2ca25f"),
    ("medium (0.5-0.7)", 0.5,     0.7,      "#ff7f00"),
    ("weak (<0.5)",     -np.inf,  0.5,      "#de2d26"),
]
NAN_COLOR = "lightgray"
NAN_LABEL = "n/a"

# M4_tau (form3 / linear v_tect) retired from production — never selected as
# best across all stations/runs — so the variant map is a 1x3 panel.
VARIANT_COLORS = {
    "M1":     "tab:blue",
    "M2":     "tab:green",
    "M3_tau": "tab:red",
}
VARIANTS = list(VARIANT_COLORS.keys())


def _tier_info(kge):
    """Return (label, color) for a KGE value (or NaN)."""
    if pd.isna(kge):
        return NAN_LABEL, NAN_COLOR
    for label, lo, hi, color in KGE_TIERS:
        if lo <= kge < hi:
            return label, color
    # fallback (kge == inf or exactly at boundary edge)
    return KGE_TIERS[-1][0], KGE_TIERS[-1][3]


# ---------------------------------------------------------------------------
# Map 1: station map colored by KGE tier
# ---------------------------------------------------------------------------

def plot_station_map(merged: pd.DataFrame, out_path: Path) -> None:
    """All subsidence stations colored by validation KGE tier."""
    _setup()
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot each tier group
    tier_handles = []
    for label, lo, hi, color in KGE_TIERS:
        if hi == np.inf:
            sub = merged[merged["kge_val"] >= lo]
        else:
            sub = merged[(merged["kge_val"] >= lo) & (merged["kge_val"] < hi)]
        ax.scatter(
            sub["X_3826"], sub["Y_3826"],
            c=color, s=50, edgecolors="black", linewidths=0.5,
            label=label, zorder=3,
        )
        tier_handles.append(
            Line2D([0], [0], marker="o", color="none", markerfacecolor=color,
                   markeredgecolor="black", markeredgewidth=0.5, markersize=7,
                   label=f"{label} (n={len(sub)})")
        )

    # NaN tier
    nan_sub = merged[merged["kge_val"].isna()]
    ax.scatter(
        nan_sub["X_3826"], nan_sub["Y_3826"],
        c=NAN_COLOR, s=50, edgecolors="black", linewidths=0.5,
        label=NAN_LABEL, zorder=3,
    )
    tier_handles.append(
        Line2D([0], [0], marker="o", color="none", markerfacecolor=NAN_COLOR,
               markeredgecolor="black", markeredgewidth=0.5, markersize=7,
               label=f"{NAN_LABEL} (n={len(nan_sub)})")
    )

    ax.set_xlabel("Easting (TWD97 m)", fontweight="bold")
    ax.set_ylabel("Northing (TWD97 m)", fontweight="bold")
    ax.set_title("Subsidence stations -- validation KGE tier", fontweight="bold")
    ax.legend(handles=tier_handles, loc="best", fontsize=9, framealpha=0.9,
              edgecolor="black")
    ax.set_aspect("equal", adjustable="datalim")
    ax.ticklabel_format(useOffset=False, style="plain")
    ax.grid(True, lw=0.3, alpha=0.4, linestyle="--")
    fig.tight_layout()
    _save(fig, out_path)
    print(f"  Wrote: {out_path}")


# ---------------------------------------------------------------------------
# Map 2: 2x2 variant panel
# ---------------------------------------------------------------------------

def plot_variant_map(merged: pd.DataFrame, out_path: Path) -> None:
    """1x3 panel showing spatial distribution of the selected best variant."""
    _setup()
    fig, axes = plt.subplots(
        1, 3, figsize=(16.5, 6.0),
        sharex=True, sharey=True,
        constrained_layout=True,
        gridspec_kw={"wspace": 0.05},
    )

    for idx, (ax, v) in enumerate(zip(axes.flat, VARIANTS)):
        color = VARIANT_COLORS[v]
        sub_v = merged[merged["variant"] == v]
        sub_other = merged[merged["variant"] != v]

        # Background: stations not selected for this variant
        if not sub_other.empty:
            ax.scatter(
                sub_other["X_3826"], sub_other["Y_3826"],
                c="lightgray", s=20, alpha=0.5, edgecolors="none", zorder=2,
            )

        # Foreground: stations where this variant was selected
        if not sub_v.empty:
            ax.scatter(
                sub_v["X_3826"], sub_v["Y_3826"],
                c=color, s=60, edgecolors="black", linewidths=0.5, zorder=3,
            )

        panel_label = "abcd"[idx]
        ax.set_title(f"({panel_label}) {v} -- n={len(sub_v)}", fontweight="bold", fontsize=11)
        ax.set_aspect("equal", adjustable="box")
        ax.ticklabel_format(useOffset=False, style="plain")
        ax.grid(True, lw=0.3, alpha=0.4, linestyle="--")

        # Legend for this panel
        handles = [
            Line2D([0], [0], marker="o", color="none", markerfacecolor=color,
                   markeredgecolor="black", markeredgewidth=0.5, markersize=7,
                   label=f"{v} selected (n={len(sub_v)})"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="lightgray",
                   markeredgecolor="none", markersize=6,
                   label=f"Other (n={len(sub_other)})"),
        ]
        ax.legend(handles=handles, loc="best", fontsize=8, framealpha=0.85,
                  edgecolor="black")

    # Single row: every panel gets an x-label; only the leftmost gets a y-label.
    for ax in axes:
        ax.set_xlabel("Easting (TWD97 m)", fontweight="bold")
    axes[0].set_ylabel("Northing (TWD97 m)", fontweight="bold")

    fig.suptitle("Selected best variant -- spatial distribution",
                 fontweight="bold", fontsize=13)
    _save(fig, out_path)
    print(f"  Wrote: {out_path}")


# ---------------------------------------------------------------------------
# Map 3: pairing map (sub station <-> GW station line segments)
# ---------------------------------------------------------------------------

def plot_pairing_map(
    pairing: pd.DataFrame,
    sub_master: pd.DataFrame,
    gw: pd.DataFrame,
    out_path: Path,
) -> None:
    """Line segments between subsidence stations and their paired GW stations."""
    _setup()
    fig, ax = plt.subplots(figsize=(9, 9))

    # Build coordinate lookup tables
    # sub_master may have duplicate sub_ids across datasets — keep first occurrence
    sub_xy = (
        sub_master.drop_duplicates(subset=["sub_id"], keep="first")
                  .set_index("sub_id")[["X_3826", "Y_3826"]]
    )

    # GW stations are keyed by gw_st (numeric well code) in gray_box_input
    gw_xy = (
        gw.rename(columns={"gw_TM_X97": "X_3826", "gw_TM_Y97": "Y_3826"})
          .drop_duplicates(subset=["gw_st"])
          .set_index("gw_st")[["X_3826", "Y_3826"]]
    )

    # Draw pairing lines first (below markers)
    n_coloc = 0
    n_nn = 0
    for _, row in pairing.iterrows():
        sid = row["sub_id"]
        gst = row["gw_st"]
        method = row.get("pairing_method", "nn")

        if sid not in sub_xy.index or gst not in gw_xy.index:
            continue

        sx = sub_xy.loc[sid, "X_3826"]
        sy = sub_xy.loc[sid, "Y_3826"]
        gx = gw_xy.loc[gst, "X_3826"]
        gy = gw_xy.loc[gst, "Y_3826"]

        if str(method) == "co-located":
            lc = "#7b2d8b"   # purple
            lw = 1.0
            n_coloc += 1
        else:
            lc = "#888888"   # gray
            lw = 0.6
            n_nn += 1

        ax.plot([sx, gx], [sy, gy], color=lc, lw=lw, alpha=0.7, zorder=2)

    # Plot GW station markers
    gw_plotted = gw_xy.loc[gw_xy.index.isin(pairing["gw_st"].unique())]
    ax.scatter(
        gw_plotted["X_3826"], gw_plotted["Y_3826"],
        s=35, c="tab:blue", marker="^",
        edgecolors="black", linewidths=0.4,
        zorder=4, label=f"GW stations (n={len(gw_plotted)})",
    )

    # Plot subsidence station markers
    sub_plotted = sub_xy.loc[sub_xy.index.isin(pairing["sub_id"].unique())]
    ax.scatter(
        sub_plotted["X_3826"], sub_plotted["Y_3826"],
        s=45, c="tab:red", marker="o",
        edgecolors="black", linewidths=0.4,
        zorder=5, label=f"Subsidence stations (n={len(sub_plotted)})",
    )

    # Legend
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="tab:red",
               markeredgecolor="black", markeredgewidth=0.4, markersize=7,
               label=f"Subsidence stations (n={len(sub_plotted)})"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor="tab:blue",
               markeredgecolor="black", markeredgewidth=0.4, markersize=7,
               label=f"GW stations (n={len(gw_plotted)})"),
        Line2D([0], [0], color="#7b2d8b", lw=1.5,
               label=f"Co-located (n={n_coloc})"),
        Line2D([0], [0], color="#888888", lw=1.0,
               label=f"Nearest-neighbour (n={n_nn})"),
    ]
    ax.legend(handles=handles, loc="best", fontsize=9,
              framealpha=0.9, edgecolor="black")

    ax.set_xlabel("Easting (TWD97 m)", fontweight="bold")
    ax.set_ylabel("Northing (TWD97 m)", fontweight="bold")
    ax.set_title("Subsidence <-> GW pairing", fontweight="bold")
    ax.set_aspect("equal", adjustable="datalim")
    ax.ticklabel_format(useOffset=False, style="plain")
    ax.grid(True, lw=0.3, alpha=0.4, linestyle="--")
    fig.tight_layout()
    _save(fig, out_path)
    print(f"  Wrote: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Generate subsidence spatial maps (KGE-tier, variant, pairing)."
    )
    p.add_argument("--run-id", required=True, help="Run identifier (e.g. initial, smoke)")
    args = p.parse_args(argv)

    ROOT = Path(__file__).resolve().parents[1]

    fit_csv = ROOT / "workspace" / "results_sub" / args.run_id / "sub_fit_results.csv"
    master_csv = ROOT / "subsidence" / "data" / "sub_station_master.csv"
    gw_csv = ROOT / "data" / "gray_box_input.csv"
    pair_csv = ROOT / "subsidence" / "data" / "sub_pairing.csv"

    for f in (fit_csv, master_csv, gw_csv, pair_csv):
        if not f.exists():
            print(f"ERR: required file not found: {f}")
            return 1

    # Load data
    fit = pd.read_csv(fit_csv)
    master = pd.read_csv(master_csv)
    gw = pd.read_csv(gw_csv)
    pair = pd.read_csv(pair_csv)

    # Keep only best-variant rows (is_best == True)
    if "is_best" in fit.columns:
        best = fit[fit["is_best"].astype(str).str.lower() == "true"].copy()
    else:
        best = fit.copy()

    # Merge with spatial coordinates from station master
    # sub_station_master may have duplicate sub_ids across datasets — use first coord
    master_coords = (
        master[["sub_id", "X_3826", "Y_3826"]]
        .drop_duplicates(subset=["sub_id"], keep="first")
    )
    merged = best.merge(master_coords, on="sub_id", how="left")
    merged["kge_val"] = pd.to_numeric(merged["kge_val"], errors="coerce")

    out_dir = ROOT / "workspace" / "maps"

    print(f"Stations in fit results (best only): {len(merged)}")
    print(f"Pairing records: {len(pair)}")

    plot_station_map(merged, out_dir / "sub_station_map.tiff")
    plot_variant_map(merged, out_dir / "sub_variant_map.tiff")
    plot_pairing_map(pair, master, gw, out_dir / "sub_pairing_map.tiff")

    print(f"\nDone. Wrote 3 TIFFs to {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
