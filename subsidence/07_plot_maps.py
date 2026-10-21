"""Subsidence-pipeline spatial maps.

Reads:
    workspace/results_sub/<run_id>/sub_fit_results.csv      (best variant per station + KGE tier)
    subsidence/data/sub_pairing.csv                         (sub-id / gw_st pairing)
    subsidence/data/sub_station_master.csv                  (subsidence station metadata + coordinates)
    data/gray_box_input.csv                                 (GW station table with TM_X97/TM_Y97)

Writes:
    workspace/maps/sub_station_map.tiff       -- KGE-tier colored points
    workspace/maps/sub_network_map.tiff       -- Fig 1: 2-network study-area map
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

# --- Map-kit bootstrap -----------------------------------------------------
# The Figure-1 network map (boundary outline, scale bar, north arrow, locator
# inset) needs geopandas plus rklib's map_decorations.  rklib lives as a sibling
# of the project (code_space/rklib) and is not pip-installed, so its submodules
# are loaded by file path — mirrors srcs/06_plot_maps.py.  If anything is
# missing the network map is skipped gracefully (the other maps still render).
try:
    import importlib.util as _ilu
    import geopandas as gpd  # noqa: F401
    _RKLIB_DIR = Path(__file__).resolve().parents[2] / "rklib"
    for _m in ("fig_base", "map_decorations"):
        _spec = _ilu.spec_from_file_location(f"rklib.{_m}", str(_RKLIB_DIR / f"{_m}.py"))
        _mod = _ilu.module_from_spec(_spec)
        sys.modules[f"rklib.{_m}"] = _mod
        _spec.loader.exec_module(_mod)
    from rklib.fig_base import setup_font as _mapkit_setup_font, savefig as _mapkit_savefig
    from rklib.map_decorations import NorthArrow, ScaleBar, LocatorInset
    _HAS_MAPKIT = True
    _MAPKIT_ERR = ""
except Exception as _e:  # geopandas / rklib / shapefiles unavailable
    _HAS_MAPKIT = False
    _MAPKIT_ERR = str(_e)


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


def _apply_cjk_fallback():
    """Append CJK fonts after the primary (Times New Roman) so Chinese station
    names (e.g. 土庫國中) render instead of tofu boxes.  Required because rklib's
    setup_font sets a Latin-only family; matplotlib's per-glyph fallback only
    works when font.family is a concrete list (mirrors sub_plotting._setup)."""
    primary = plt.rcParams.get("font.family", ["Times New Roman"])
    if isinstance(primary, str):
        primary = [primary]
    plt.rcParams["font.family"] = list(primary) + [
        "Heiti TC",           # macOS Traditional Chinese
        "Hiragino Sans GB",   # macOS Simplified Chinese
        "Songti SC",          # macOS serif Chinese
        "Arial Unicode MS",   # broad Unicode coverage
        "Noto Sans CJK TC",   # Linux/CI fallback
        "WenQuanYi Zen Hei",  # Linux fallback
    ]


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

# Per-network styling for the Figure-1 study-area map (colours mirror the
# cal/val shading hues used in the other figures).
NETWORK_STYLE = {
    "ls-wra-gnss-obs": dict(color="#2166ac", marker="o", label="GNSS"),
    "ls-wra-mlcw-obs": dict(color="#d6604d", marker="^", label="MLCW"),
}

# Cartographic assets (shared with the groundwater pipeline maps).
_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
_BOUNDARY_SHP = _DATA_DIR / "Zhuoshui Alluvial Fan" / "Zhuoshui Alluvial Fan.shp"
_TAIWAN_GPKG = _DATA_DIR / "water" / "Taiwan_countyV02" / "TW_TWN2.gpkg"


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
# Map 2: study-area network map (Figure 1) — numbered markers + lookup table
# ---------------------------------------------------------------------------

def plot_network_map(best: pd.DataFrame, master: pd.DataFrame, out_path: Path) -> None:
    """Figure 1: study-area map of the two subsidence monitoring networks.

    Numbered markers (north→south) over the Zhuoshui Alluvial Fan outline with a
    Taiwan locator inset, scale bar and north arrow, plus a side lookup table
    mapping each number to its station ID.  GNSS = blue circles, MLCW = red
    triangles.  Replaces the retired per-variant panel map.

    Parameters
    ----------
    best : best-variant results table (needs ``sub_id``, ``sub_dataset``).
    master : station master (supplies ``Longitude_4326`` and the network-split
        latitude columns ``Latitude_4326`` / ``latitude_4326``).
    """
    if not _HAS_MAPKIT:
        print(f"  [skip] sub_network_map: map kit unavailable ({_MAPKIT_ERR})")
        return
    from matplotlib.gridspec import GridSpec
    _mapkit_setup_font()
    _apply_cjk_fallback()

    # Authoritative network per station comes from the results table; join on
    # (sub_id, sub_dataset) so STES — present in 4 datasets — is not duplicated.
    keep = [c for c in ("sub_id", "sub_dataset", "Longitude_4326",
                        "Latitude_4326", "latitude_4326") if c in master.columns]
    m = best[["sub_id", "sub_dataset"]].merge(master[keep],
                                              on=["sub_id", "sub_dataset"], how="left")
    # GNSS stores latitude in ``Latitude_4326``; MLCW in lowercase ``latitude_4326``.
    lat_alt = m["latitude_4326"] if "latitude_4326" in m.columns else np.nan
    m["lat"] = m["Latitude_4326"].fillna(lat_alt)
    m["lon"] = m["Longitude_4326"]
    m = m[m["sub_dataset"].isin(NETWORK_STYLE)].dropna(subset=["lon", "lat"])
    m = m.sort_values("lat", ascending=False).reset_index(drop=True)
    m["num"] = np.arange(1, len(m) + 1)

    boundary = gpd.read_file(_BOUNDARY_SHP).to_crs(epsg=4326)
    b = boundary.total_bounds

    fig = plt.figure(figsize=(14, 9))
    gs = GridSpec(1, 2, width_ratios=[1.9, 1.0], wspace=0.03)
    ax = fig.add_subplot(gs[0])
    axt = fig.add_subplot(gs[1])
    axt.axis("off")

    boundary.plot(ax=ax, facecolor="#FFE8BE", edgecolor="none", alpha=0.5, zorder=1)
    boundary.boundary.plot(ax=ax, edgecolor="black", linewidth=1.2, zorder=4)

    handles = []
    for ds, st in NETWORK_STYLE.items():
        g = m[m["sub_dataset"] == ds]
        if g.empty:
            continue
        ax.scatter(g["lon"], g["lat"], c=st["color"], marker=st["marker"], s=150,
                   edgecolors="black", linewidths=0.7, zorder=5)
        handles.append(
            Line2D([0], [0], marker=st["marker"], color="none",
                   markerfacecolor=st["color"], markeredgecolor="black",
                   markersize=12, label=f"{st['label']} (n={len(g)})")
        )
    for r in m.itertuples():
        ax.text(r.lon, r.lat, str(r.num), fontsize=7.5, fontweight="bold",
                ha="center", va="center", color="white", zorder=7)

    px = (b[2] - b[0]) * 0.05
    py = (b[3] - b[1]) * 0.05
    ax.set_xlim(b[0] - px, b[2] + px)
    ax.set_ylim(b[1] - py, b[3] + py)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Longitude (°E)", fontweight="bold")
    ax.set_ylabel("Latitude (°N)", fontweight="bold")
    ax.ticklabel_format(useOffset=False, style="plain")
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.legend(handles=handles, loc="upper right", fontsize=12, framealpha=0.95,
              edgecolor="black", title="Monitoring network")
    NorthArrow(ax, location="lower right").draw()
    ScaleBar(ax, location="lower center", length_km=10).draw()

    # Taiwan locator inset (study area highlighted in red).
    if _TAIWAN_GPKG.exists():
        try:
            tw = gpd.read_file(_TAIWAN_GPKG).to_crs(epsg=4326)
            LocatorInset(
                ax, bounds=[0.015, 0.63, 0.30, 0.35],
                xlim=(119.9, 122.1), ylim=(21.9, 25.4),
                highlight_gdf=boundary,
                highlight_kw={"edgecolor": "red", "linewidth": 1.2},
                overlay_gdfs=[{"gdf": tw, "facecolor": "#ececec",
                               "edgecolor": "grey", "linewidth": 0.4}],
            ).draw()
        except Exception as e:  # inset is decorative — never fail the figure
            print(f"  [warn] locator inset skipped: {e}")

    # Lookup table — two columns, number colour-coded by network.
    axt.set_title("Station index  (N→S)", fontweight="bold", fontsize=13, loc="left")
    half = (len(m) + 1) // 2

    def _block(rows, x0):
        y = 0.965
        for r in rows.itertuples():
            col = NETWORK_STYLE[r.sub_dataset]["color"]
            axt.text(x0, y, f"{r.num:>2}", fontsize=9.5, fontweight="bold",
                     color=col, family="monospace", transform=axt.transAxes)
            axt.text(x0 + 0.085, y, str(r.sub_id), fontsize=9.5,
                     transform=axt.transAxes)
            y -= 0.0315

    _block(m.iloc[:half], 0.02)
    _block(m.iloc[half:], 0.52)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    _mapkit_savefig(fig, str(out_path), dpi=300)
    plt.close(fig)
    print(f"  Wrote: {out_path}")


# ---------------------------------------------------------------------------
# Map 3: pairing map (sub station <-> GW station line segments)
# ---------------------------------------------------------------------------

def plot_pairing_map(
    pairing: pd.DataFrame,
    best: pd.DataFrame,
    sub_master: pd.DataFrame,
    gw: pd.DataFrame,
    out_path: Path,
) -> None:
    """Supplementary: groundwater-driver provenance for the fitted stations.

    Each subsidence station is forced by the (modeled) head of a paired GW well —
    co-located where a well sits on-site, else the nearest well within the same
    hydrogeological zone.  Lines connect each *fitted* station to its driver well;
    nearest-neighbour links are coloured by separation distance so long-range
    pairings (a weak forcing assumption) are visible.  Restricted to the
    canonical fitted set so counts match the rest of the figure suite.

    Plotted in TWD97 / TM2 (EPSG:3826) metres over the Zhuoshui Fan outline.
    """
    if not _HAS_MAPKIT:
        print(f"  [skip] sub_pairing_map: map kit unavailable ({_MAPKIT_ERR})")
        return
    from matplotlib.collections import LineCollection
    _mapkit_setup_font()
    _apply_cjk_fallback()

    # Restrict pairing to the fitted (canonical) station set so n matches Fig 1.
    keys = best[["sub_id", "sub_dataset"]].drop_duplicates()
    pair = pairing.merge(keys, on=["sub_id", "sub_dataset"], how="inner")

    # Coordinate lookups (EPSG:3826 metres).
    sub_xy = (
        sub_master.drop_duplicates(subset=["sub_id", "sub_dataset"])
                  .set_index(["sub_id", "sub_dataset"])[["X_3826", "Y_3826"]]
    )
    gw_xy = (
        gw.rename(columns={"gw_TM_X97": "X_3826", "gw_TM_Y97": "Y_3826"})
          .drop_duplicates(subset=["gw_st"])
          .set_index("gw_st")[["X_3826", "Y_3826"]]
    )

    # Assemble segments + endpoints.
    coloc_seg, nn_seg, nn_dist = [], [], []
    sub_pts, gw_pts = set(), set()
    for _, row in pair.iterrows():
        key = (row["sub_id"], row["sub_dataset"])
        gst = row["gw_st"]
        if key not in sub_xy.index or gst not in gw_xy.index:
            continue
        sx, sy = sub_xy.loc[key, "X_3826"], sub_xy.loc[key, "Y_3826"]
        gx, gy = gw_xy.loc[gst, "X_3826"], gw_xy.loc[gst, "Y_3826"]
        seg = [(sx, sy), (gx, gy)]
        if str(row.get("pairing_method", "")) == "co-located":
            coloc_seg.append(seg)
        else:
            nn_seg.append(seg)
            nn_dist.append(float(row["distance_m"]) / 1000.0)  # → km
        sub_pts.add((sx, sy))
        gw_pts.add((gx, gy))

    n_coloc, n_nn = len(coloc_seg), len(nn_seg)
    n_far = sum(d > 3.0 for d in nn_dist)

    fig, ax = plt.subplots(figsize=(10, 9))

    # Fan outline basemap (reproject to metres CRS).
    boundary4326 = gpd.read_file(_BOUNDARY_SHP).to_crs(epsg=4326)
    boundary = boundary4326.to_crs(epsg=3826)
    boundary.plot(ax=ax, facecolor="#FFE8BE", edgecolor="none", alpha=0.5, zorder=1)
    boundary.boundary.plot(ax=ax, edgecolor="black", linewidth=1.2, zorder=2)

    # Nearest-neighbour links coloured by separation distance.
    sm = None
    if nn_seg:
        import matplotlib.colors as mcolors
        norm = mcolors.Normalize(vmin=0.0, vmax=max(nn_dist))
        lc = LineCollection(nn_seg, cmap="plasma", norm=norm,
                            linewidths=1.6, alpha=0.9, zorder=3)
        lc.set_array(np.asarray(nn_dist))
        ax.add_collection(lc)
        sm = plt.cm.ScalarMappable(cmap="plasma", norm=norm)
        sm.set_array([])
    # Co-located links (≈0 length — drawn as short teal stubs for completeness).
    if coloc_seg:
        ax.add_collection(LineCollection(coloc_seg, colors="#1b9e77",
                                         linewidths=1.6, zorder=3))

    # Endpoint markers.
    if gw_pts:
        gx, gy = zip(*gw_pts)
        ax.scatter(gx, gy, s=55, c="#377eb8", marker="^",
                   edgecolors="black", linewidths=0.5, zorder=5)
    if sub_pts:
        sx, sy = zip(*sub_pts)
        ax.scatter(sx, sy, s=55, c="#e41a1c", marker="o",
                   edgecolors="black", linewidths=0.5, zorder=6)

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#e41a1c",
               markeredgecolor="black", markersize=9,
               label=f"Subsidence station (n={len(sub_pts)})"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor="#377eb8",
               markeredgecolor="black", markersize=9,
               label=f"GW driver well (n={len(gw_pts)})"),
        Line2D([0], [0], color="#1b9e77", lw=2.2,
               label=f"Co-located (n={n_coloc})"),
        Line2D([0], [0], color="#888888", lw=2.2,
               label=f"Nearest-neighbour (n={n_nn})"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=10,
              framealpha=0.95, edgecolor="black", title="Driver provenance")

    if sm is not None:
        cbar = fig.colorbar(sm, ax=ax, fraction=0.040, pad=0.02)
        cbar.set_label("Nearest-neighbour pairing distance (km)", fontweight="bold")

    NorthArrow(ax, location="lower right").draw()
    ScaleBar(ax, location="lower left", length_km=10).draw()

    ax.set_xlabel("Easting (TWD97 / TM2, m)", fontweight="bold")
    ax.set_ylabel("Northing (TWD97 / TM2, m)", fontweight="bold")
    ax.set_title(
        f"Groundwater-driver provenance  "
        f"({n_coloc} co-located, {n_nn} nearest-neighbour; "
        f"{n_far} pairs > 3 km)",
        fontweight="bold", fontsize=12,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.ticklabel_format(useOffset=False, style="plain")
    ax.grid(True, lw=0.3, alpha=0.35, linestyle="--", zorder=0)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    _mapkit_savefig(fig, str(out_path), dpi=300)
    plt.close(fig)
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
    plot_network_map(best, master, out_dir / "sub_network_map.tiff")
    plot_pairing_map(pair, best, master, gw, out_dir / "sub_pairing_map.tiff")

    print(f"\nDone. Wrote 3 TIFFs to {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
