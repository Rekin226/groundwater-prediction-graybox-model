"""Plot groundwater station performance maps (good/medium/low) and the 2x2 variant-selection map.

Styled to match the rklib publication figures (see workspace/manuscripts/figures/fig5_parameter_idw.py,
fig1_station_map.py): WMS shaded-relief basemap, sea/river overlays, fan-boundary clip, Taiwan locator
inset, NorthArrow + ScaleBar decorations, Times New Roman, 300 DPI TIFF via rklib.savefig.

Thresholds follow project convention (KGE in 0-1 scale):
- good:   kge_val >= 0.7
- medium: 0.5 <= kge_val < 0.7
- low:    kge_val < 0.5

Outputs (workspace/results/<run_id>/figures/):
- performance_good.tiff
- performance_medium.tiff
- performance_low.tiff
- variant_selection_map.tiff  (2x2 panels, station-id labels on all panels)
"""

from __future__ import annotations

import argparse
import importlib.util
import ssl
import sys
import warnings
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen

import geopandas as gpd
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image as PILImage
from shapely.geometry import box as shapely_box

try:
    from adjustText import adjust_text
    HAS_ADJUST_TEXT = True
except ImportError:
    HAS_ADJUST_TEXT = False

ROOT = Path(__file__).resolve().parents[1]
CODE_SPACE = Path(__file__).resolve().parents[2]
RKLIB_DIR = CODE_SPACE / "rklib"

for _module_name in ("fig_base", "map_decorations"):
    _module_path = RKLIB_DIR / f"{_module_name}.py"
    _spec = importlib.util.spec_from_file_location(f"rklib.{_module_name}", str(_module_path))
    _module = importlib.util.module_from_spec(_spec)
    sys.modules[f"rklib.{_module_name}"] = _module
    _spec.loader.exec_module(_module)

from rklib.fig_base import add_panel_label, savefig as rklib_savefig, setup_font  # noqa: E402
from rklib.map_decorations import NorthArrow, ScaleBar  # noqa: E402


GOOD_COLOR = "#2ca25f"
MED_COLOR = "#ff7f00"
LOW_COLOR = "#de2d26"
SEA_COLOR = "#c6e2f0"
RIVER_COLOR = "#3a8bbf"

VARIANT_DISPLAY = [
    ("base",        "M1: Direct"),
    ("filtered",    "M2: Filtered"),
    ("base_tz",     "M3: Direct, z(t)"),
    ("filtered_tz", "M4: Filtered, z(t)"),
]


def _read_results(results_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(results_csv)
    if "gw_st" not in df.columns or "kge_val" not in df.columns:
        raise ValueError(f"Expected columns gw_st and kge_val in {results_csv}, got: {list(df.columns)}")

    df["gw_st"] = df["gw_st"].astype(str)
    df["kge_val"] = pd.to_numeric(df["kge_val"], errors="coerce")
    df = df.drop_duplicates(subset=["gw_st"], keep="first")

    cols = ["gw_st", "kge_val"]
    if "st_id" in df.columns:
        cols.append("st_id")
    if "model" in df.columns:
        cols.append("model")
    return df[cols]


def _read_station_meta(gray_box_input_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(gray_box_input_csv)
    required = {"gw_st", "gw_TM_X97", "gw_TM_Y97"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {sorted(missing)} in {gray_box_input_csv}")

    df["gw_st"] = df["gw_st"].astype(str)
    df["gw_TM_X97"] = pd.to_numeric(df["gw_TM_X97"], errors="coerce")
    df["gw_TM_Y97"] = pd.to_numeric(df["gw_TM_Y97"], errors="coerce")
    df = df.dropna(subset=["gw_TM_X97", "gw_TM_Y97"])

    keep = ["gw_st", "gw_TM_X97", "gw_TM_Y97"]
    if "st_id" in df.columns:
        keep.append("st_id")
    if "group" in df.columns:
        keep.append("group")
    return df[keep].drop_duplicates(subset=["gw_st"], keep="first")


def _load_boundary_wgs84(boundary_path: Path) -> gpd.GeoDataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gdf = gpd.read_file(boundary_path)
    if gdf.empty:
        raise ValueError(f"Boundary file has no features: {boundary_path}")

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:3826")
    return gdf.to_crs("EPSG:4326")


def _make_station_gdf(results_df: pd.DataFrame, meta_df: pd.DataFrame) -> gpd.GeoDataFrame:
    merged = results_df.merge(meta_df, on="gw_st", how="left", suffixes=("", "_meta"))
    if "st_id" not in merged.columns and "st_id_meta" in merged.columns:
        merged["st_id"] = merged["st_id_meta"]
    merged = merged.dropna(subset=["gw_TM_X97", "gw_TM_Y97", "kge_val"])

    gdf = gpd.GeoDataFrame(
        merged,
        geometry=gpd.points_from_xy(merged["gw_TM_X97"], merged["gw_TM_Y97"]),
        crs="EPSG:3826",
    ).to_crs("EPSG:4326")
    gdf["lon"] = gdf.geometry.x
    gdf["lat"] = gdf.geometry.y
    return gdf


def _clip_to_boundary(stations: gpd.GeoDataFrame, boundary: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    stations = stations.copy()
    if stations.crs != boundary.crs:
        stations = stations.to_crs(boundary.crs)
    joined = gpd.sjoin(stations, boundary[["geometry"]], how="inner", predicate="within")
    return joined.drop(columns=[c for c in joined.columns if c.startswith("index_")], errors="ignore")


def _fetch_wms(bbox: tuple[float, float, float, float], width: int, height: int) -> tuple[np.ndarray | None, list[float] | None]:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    xmin, ymin, xmax, ymax = bbox
    url = (
        "https://maps.nlsc.gov.tw/S_Maps/wms"
        "?SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap"
        "&LAYERS=MOI_SHADERMAP&STYLES=&SRS=EPSG:4326"
        f"&BBOX={xmin},{ymin},{xmax},{ymax}"
        f"&WIDTH={width}&HEIGHT={height}"
        "&FORMAT=image/png&TRANSPARENT=FALSE"
    )
    try:
        resp = urlopen(url, timeout=30, context=ctx)
        img = np.array(PILImage.open(BytesIO(resp.read())))
        return img, [xmin, xmax, ymin, ymax]
    except Exception as exc:
        print(f"WMS fetch failed ({exc}) — skipping basemap tile")
        return None, None


def _load_water_overlays(root: Path, bounds: np.ndarray) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    plot_box = shapely_box(bounds[0] - 0.02, bounds[1] - 0.03,
                           bounds[2] + 0.02, bounds[3] + 0.02)
    bbox_gdf = gpd.GeoDataFrame(geometry=[plot_box], crs="EPSG:4326")

    sea = (gpd.read_file(root / "data" / "water" / "sea_TWD97.shp")
             .set_crs("EPSG:3826", allow_override=True)
             .to_crs(epsg=4326))
    sea_clip = gpd.clip(sea, bbox_gdf)

    riv = (gpd.read_file(root / "data" / "water" / "river_TWD97.shp")
             .set_crs("EPSG:3826", allow_override=True)
             .to_crs(epsg=4326))
    riv_clip = gpd.clip(riv[riv["TYPE"] <= 2], bbox_gdf)

    return sea, sea_clip, riv_clip


class MapContext:
    """Shared cartographic backdrop reused by every panel."""

    def __init__(self, boundary: gpd.GeoDataFrame, root: Path):
        self.boundary = boundary
        self.bounds = boundary.total_bounds
        geom = boundary.geometry.union_all() if hasattr(boundary.geometry, "union_all") \
            else boundary.geometry.unary_union
        exterior = geom.exterior if hasattr(geom, "exterior") else geom.geoms[0].exterior
        self.clip_path = mpath.Path(np.array(exterior.coords))

        self.sea, self.sea_clip, self.riv_clip = _load_water_overlays(root, self.bounds)

        # East edge extended past the xlim (fig5_parameter_idw convention) so the
        # shaded-relief basemap fills the full plot box — otherwise the strip between
        # bounds[2] and xlim right shows as white.
        wms_bbox = (self.bounds[0] - 0.02, self.bounds[1] - 0.03,
                    self.bounds[2] + 0.38, self.bounds[3] + 0.02)
        self.wms_img, self.wms_extent = _fetch_wms(wms_bbox, 1000, 720)

        self.tw_bbox = (119.7, 21.8, 122.2, 25.5)
        self.tw_img, self.tw_extent = _fetch_wms(self.tw_bbox, 300, 450)

        tw_box = shapely_box(*self.tw_bbox)
        sea_fixed = self.sea.copy()
        sea_fixed["geometry"] = self.sea.geometry.buffer(0)
        self.sea_tw = gpd.clip(sea_fixed, gpd.GeoDataFrame(geometry=[tw_box], crs="EPSG:4326"))

        # Plot extents — match rklib convention (fixed longitude window covering the fan)
        self.xlim = (120.115, 120.8)
        self.ylim = (self.bounds[1] - 0.02, self.bounds[3] + 0.005)

    def draw_backdrop(self, ax: plt.Axes) -> None:
        if self.wms_img is not None:
            ax.imshow(self.wms_img, extent=self.wms_extent, aspect="auto",
                      zorder=-1, interpolation="bilinear")
        self.sea_clip.plot(ax=ax, color=SEA_COLOR, zorder=0)
        self.boundary.boundary.plot(ax=ax, edgecolor="black", linewidth=1.2, zorder=5)
        self.riv_clip.plot(ax=ax, color=RIVER_COLOR, linewidth=0.7, alpha=0.75, zorder=6)

    def style_axes(self, ax: plt.Axes, *, xlabel: bool = True, ylabel: bool = True) -> None:
        if xlabel:
            ax.set_xlabel("Longitude (\u00b0E)", fontsize=9, fontweight="bold")
        if ylabel:
            ax.set_ylabel("Latitude (\u00b0N)", fontsize=9, fontweight="bold")
        ax.ticklabel_format(useOffset=False, style="plain")
        ax.set_xlim(self.xlim)
        ax.set_ylim(self.ylim)
        ax.grid(True, lw=0.3, alpha=0.3, linestyle="--")
        ax.set_aspect("equal", adjustable="box")
        NorthArrow(ax, location="lower right").draw()
        ScaleBar(ax, location="lower center", length_km=10).draw()

    def draw_inset(self, ax: plt.Axes) -> None:
        if self.tw_img is None:
            return
        ax_ins = ax.inset_axes([0.65, 0.70, 0.33, 0.28])
        ax_ins.imshow(self.tw_img, extent=self.tw_extent, aspect="auto",
                      zorder=-1, interpolation="bilinear")
        self.sea_tw.plot(ax=ax_ins, color=SEA_COLOR, zorder=0)
        self.boundary.boundary.plot(ax=ax_ins, edgecolor="black", linewidth=0.8, zorder=5)
        ax_ins.set_xlim(119.7, 122.05)
        ax_ins.set_ylim(21.9, 25.35)
        ax_ins.set_xticks([])
        ax_ins.set_yticks([])
        ax_ins.set_aspect("equal", adjustable="box")
        for spine in ax_ins.spines.values():
            spine.set_linewidth(1.0)


def _annotate_stations(ax: plt.Axes, points: gpd.GeoDataFrame, *, label_col: str,
                       font_size: int, dx_deg: float, dy_deg: float) -> None:
    if points.empty or label_col not in points.columns:
        return

    texts, xs, ys = [], [], []
    for _, row in points.iterrows():
        if row.geometry is None:
            continue
        label = row.get(label_col)
        if pd.isna(label):
            continue
        t = ax.text(
            row.geometry.x + dx_deg,
            row.geometry.y + dy_deg,
            str(label),
            fontsize=font_size,
            color="black",
            ha="left",
            va="bottom",
            zorder=8,
        )
        texts.append(t)
        xs.append(row.geometry.x)
        ys.append(row.geometry.y)

    if HAS_ADJUST_TEXT and texts:
        adjust_text(
            texts, x=xs, y=ys, ax=ax,
            expand_points=(1.6, 1.8),
            expand_text=(1.1, 1.2),
            force_points=(0.15, 0.22),
            force_text=(0.12, 0.18),
            force_pull=(0.35, 0.45),
            max_move=(10, 10),
            only_move={"points": "xy", "text": "xy"},
            min_arrow_len=40,
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.5,
                            alpha=0.7, shrinkA=4, shrinkB=2),
        )


def _tier_legend(sea_label: str = "Sea", study_label: str = "Study area",
                 tier_handles: list | None = None) -> list:
    handles = list(tier_handles or [])
    handles.append(Patch(facecolor=SEA_COLOR, edgecolor="none", label=sea_label))
    handles.append(Line2D([0], [0], color="black", linewidth=1.2, label=study_label))
    return handles


def _plot_tier_map(*, ctx: MapContext, points: gpd.GeoDataFrame, title: str,
                   color: str, tier_label: str, out_path: Path,
                   label_col: str, add_labels: bool, label_font_size: int,
                   dx_deg: float, dy_deg: float) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    ctx.draw_backdrop(ax)

    if not points.empty:
        ax.scatter(points["lon"], points["lat"], c=color, s=55,
                   edgecolors="white", linewidths=0.5, alpha=0.9,
                   zorder=7, label=tier_label)
        if add_labels:
            _annotate_stations(ax, points, label_col=label_col,
                               font_size=label_font_size,
                               dx_deg=dx_deg, dy_deg=dy_deg)

    marker_handle = Line2D([0], [0], marker="o", color=color, linestyle="None",
                           markersize=7, markeredgecolor="white", markeredgewidth=0.5,
                           label=tier_label)
    ax.legend(handles=_tier_legend(tier_handles=[marker_handle]),
              loc="upper left", fontsize=8, framealpha=0.95, edgecolor="black")

    ax.set_title(title, fontsize=11, pad=6, fontweight="bold")
    ctx.style_axes(ax)
    ctx.draw_inset(ax)

    plt.tight_layout()
    rklib_savefig(fig, str(out_path))


def _plot_variant_grid(*, ctx: MapContext, stations: gpd.GeoDataFrame,
                       kge_good: float, kge_med: float, out_path: Path,
                       label_col: str, add_labels: bool, label_font_size: int,
                       dx_deg: float, dy_deg: float) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 13))

    for idx, (ax, (variant, display)) in enumerate(zip(axes.flat, VARIANT_DISPLAY)):
        ctx.draw_backdrop(ax)

        sub = stations[stations["model"] == variant] if "model" in stations.columns else stations.iloc[0:0]
        good = sub[sub["kge_val"] >= kge_good]
        med = sub[(sub["kge_val"] >= kge_med) & (sub["kge_val"] < kge_good)]
        low = sub[sub["kge_val"] < kge_med]

        tier_handles = []
        for tier_df, color, lbl in (
            (good, GOOD_COLOR, f"KGE \u2265 {kge_good:.1f} (n={len(good)})"),
            (med,  MED_COLOR,  f"{kge_med:.1f} \u2264 KGE < {kge_good:.1f} (n={len(med)})"),
            (low,  LOW_COLOR,  f"KGE < {kge_med:.1f} (n={len(low)})"),
        ):
            if not tier_df.empty:
                ax.scatter(tier_df["lon"], tier_df["lat"], c=color, s=55,
                           edgecolors="white", linewidths=0.5, alpha=0.9,
                           zorder=7)
            tier_handles.append(
                Line2D([0], [0], marker="o", color=color, linestyle="None",
                       markersize=7, markeredgecolor="white", markeredgewidth=0.5, label=lbl)
            )

        if add_labels and not sub.empty:
            _annotate_stations(ax, sub, label_col=label_col,
                               font_size=label_font_size,
                               dx_deg=dx_deg, dy_deg=dy_deg)

        ax.legend(handles=_tier_legend(tier_handles=tier_handles),
                  loc="upper left", fontsize=8, framealpha=0.95, edgecolor="black")

        ax.set_title(f"{display} \u2014 n={len(sub)}", fontsize=11, pad=6, fontweight="bold")
        ctx.style_axes(ax,
                       xlabel=(idx >= 2),    # bottom row
                       ylabel=(idx % 2 == 0))  # left column
        ctx.draw_inset(ax)
        add_panel_label(ax, "abcd"[idx], fontsize=11, fontweight="bold")

    plt.tight_layout()
    rklib_savefig(fig, str(out_path))


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot station performance and variant-selection maps in rklib style")
    parser.add_argument("--run-id", default="initial",
                        help="Run tag matching 03_run_model.py (default: initial)")
    parser.add_argument("--results", type=Path, default=None)
    parser.add_argument("--input", type=Path, default=Path("data/gray_box_input.csv"))
    parser.add_argument("--boundary", type=Path,
                        default=Path("data/Zhuoshui Alluvial Fan/Zhuoshui Alluvial Fan.shp"))
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--kge-medium", type=float, default=0.5)
    parser.add_argument("--kge-good", type=float, default=0.7)
    parser.add_argument("--no-clip", action="store_true", help="Do not clip points to the boundary")
    parser.add_argument("--label-col", type=str, default="st_id",
                        help="Column used for station labels (default: st_id)")
    parser.add_argument("--no-label", action="store_true", help="Disable point labels on tier maps")
    parser.add_argument("--label-font-size", type=int, default=8)
    parser.add_argument("--label-dx", type=float, default=180.0,
                        help="Label x-offset in metres (~degrees via 1/111000, default: 180)")
    parser.add_argument("--label-dy", type=float, default=180.0,
                        help="Label y-offset in metres (default: 180)")

    args = parser.parse_args()

    run_root = Path("workspace/results") / args.run_id
    results_path = args.results if args.results is not None else run_root / "gw_fit_results.csv"
    outdir = args.outdir if args.outdir is not None else run_root / "figures"
    outdir.mkdir(parents=True, exist_ok=True)

    setup_font()

    results_df = _read_results(results_path)
    meta_df = _read_station_meta(args.input)
    boundary = _load_boundary_wgs84(args.boundary)
    stations = _make_station_gdf(results_df, meta_df)

    if not args.no_clip:
        stations = _clip_to_boundary(stations, boundary)

    if args.kge_good <= args.kge_medium:
        raise ValueError("--kge-good must be greater than --kge-medium")

    ctx = MapContext(boundary, ROOT)

    dx_deg = args.label_dx / 111_000
    dy_deg = args.label_dy / 111_000
    add_labels = not args.no_label

    # Use a single "best" row per station for tier maps (same as prior behaviour after drop_duplicates)
    tier_stations = stations.drop_duplicates(subset=["gw_st"], keep="first")

    good = tier_stations[tier_stations["kge_val"] >= args.kge_good]
    medium = tier_stations[(tier_stations["kge_val"] >= args.kge_medium)
                           & (tier_stations["kge_val"] < args.kge_good)]
    low = tier_stations[tier_stations["kge_val"] < args.kge_medium]

    _plot_tier_map(
        ctx=ctx, points=good,
        title=f"Good performance (KGE \u2265 {args.kge_good:.2f}) \u2014 n={len(good)}",
        color=GOOD_COLOR,
        tier_label=f"KGE \u2265 {args.kge_good:.2f} (n={len(good)})",
        out_path=outdir / "performance_good.tiff",
        label_col=args.label_col, add_labels=add_labels,
        label_font_size=args.label_font_size, dx_deg=dx_deg, dy_deg=dy_deg,
    )
    _plot_tier_map(
        ctx=ctx, points=medium,
        title=f"Medium performance ({args.kge_medium:.2f} \u2264 KGE < {args.kge_good:.2f}) \u2014 n={len(medium)}",
        color=MED_COLOR,
        tier_label=f"{args.kge_medium:.2f} \u2264 KGE < {args.kge_good:.2f} (n={len(medium)})",
        out_path=outdir / "performance_medium.tiff",
        label_col=args.label_col, add_labels=add_labels,
        label_font_size=args.label_font_size, dx_deg=dx_deg, dy_deg=dy_deg,
    )
    _plot_tier_map(
        ctx=ctx, points=low,
        title=f"Low performance (KGE < {args.kge_medium:.2f}) \u2014 n={len(low)}",
        color=LOW_COLOR,
        tier_label=f"KGE < {args.kge_medium:.2f} (n={len(low)})",
        out_path=outdir / "performance_low.tiff",
        label_col=args.label_col, add_labels=add_labels,
        label_font_size=args.label_font_size, dx_deg=dx_deg, dy_deg=dy_deg,
    )

    # Variant-selection map always shows station-id labels (per user request)
    variant_out = outdir / "variant_selection_map.tiff"
    _plot_variant_grid(
        ctx=ctx, stations=tier_stations,
        kge_good=args.kge_good, kge_med=args.kge_medium,
        out_path=variant_out,
        label_col=args.label_col, add_labels=True,
        label_font_size=args.label_font_size, dx_deg=dx_deg, dy_deg=dy_deg,
    )

    print("Wrote:")
    print(f"  {outdir / 'performance_good.tiff'}")
    print(f"  {outdir / 'performance_medium.tiff'}")
    print(f"  {outdir / 'performance_low.tiff'}")
    print(f"  {variant_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
