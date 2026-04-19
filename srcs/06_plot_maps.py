"""Plot groundwater station performance maps (good/medium/low) within Zhuoshui Alluvial Fan.

Reads best-model performance from workspace/results/gw_fit_results.csv, merges station coordinates
from data/gray_box_input.csv (TWD97), clips to the fan boundary shapefile, and writes three
figures (good/medium/low) to workspace/results/figures/.

Thresholds follow project convention (R² in 0–1 scale):
- good:   r2 >= 0.7
- medium: 0.5 <= r2 < 0.7
- low:    r2 < 0.5

Usage:
  python srcs/plot_performance_maps.py

Optional args:
  python srcs/plot_performance_maps.py --results workspace/results/gw_fit_results.csv \
      --input data/gray_box_input.csv --boundary "data/Zhuoshui Alluvial Fan/Zhuoshui Alluvial Fan.shp" \
      --outdir workspace/results/figures
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rklib import NorthArrow, ScaleBar, setup_font, savefig as rklib_savefig


TWD97_CRS = "+proj=tmerc +lat_0=0 +lon_0=121 +k=0.9999 +x_0=250000 +y_0=0 +ellps=GRS80 +units=m +no_defs"


def _read_results(results_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(results_csv)
    if "gw_st" not in df.columns or "r2" not in df.columns:
        raise ValueError(f"Expected columns gw_st and r2 in {results_csv}, got: {list(df.columns)}")

    df["gw_st"] = df["gw_st"].astype(str)
    df["r2"] = pd.to_numeric(df["r2"], errors="coerce")

    # Best-model CSV should be one row per station; guard anyway.
    df = df.drop_duplicates(subset=["gw_st"], keep="first")
    return df[["gw_st", "r2"]]


def _read_station_xy(gray_box_input_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(gray_box_input_csv)
    required = {"gw_st", "gw_TM_X97", "gw_TM_Y97"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {sorted(missing)} in {gray_box_input_csv}")

    df["gw_st"] = df["gw_st"].astype(str)
    df["gw_TM_X97"] = pd.to_numeric(df["gw_TM_X97"], errors="coerce")
    df["gw_TM_Y97"] = pd.to_numeric(df["gw_TM_Y97"], errors="coerce")
    df = df.dropna(subset=["gw_TM_X97", "gw_TM_Y97"])

    # One coordinate per station.
    return df[["gw_st", "gw_TM_X97", "gw_TM_Y97"]].drop_duplicates(subset=["gw_st"], keep="first")


def _load_boundary(boundary_path: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(boundary_path)
    if gdf.empty:
        raise ValueError(f"Boundary file has no features: {boundary_path}")

    # Normalize CRS to TWD97 for consistent plotting.
    if gdf.crs is None:
        gdf = gdf.set_crs(TWD97_CRS)
    else:
        gdf = gdf.to_crs(TWD97_CRS)

    # Dissolve into a single geometry for clipping.
    gdf["_dissolve"] = 1
    return gdf.dissolve(by="_dissolve").drop(columns=["_dissolve"], errors="ignore")


def _make_station_gdf(results_df: pd.DataFrame, xy_df: pd.DataFrame) -> gpd.GeoDataFrame:
    merged = results_df.merge(xy_df, on="gw_st", how="left")
    merged = merged.dropna(subset=["gw_TM_X97", "gw_TM_Y97", "r2"])

    gdf = gpd.GeoDataFrame(
        merged,
        geometry=gpd.points_from_xy(merged["gw_TM_X97"], merged["gw_TM_Y97"]),
        crs=TWD97_CRS,
    )
    return gdf


def _annotate_points(
    ax: plt.Axes,
    points: gpd.GeoDataFrame,
    *,
    label_col: str,
    font_size: int,
    dx: float,
    dy: float,
) -> None:
    if points.empty:
        return
    if label_col not in points.columns:
        raise ValueError(f"Label column '{label_col}' not found in points columns: {list(points.columns)}")

    for _, row in points.iterrows():
        if row.geometry is None:
            continue
        label = row.get(label_col)
        if pd.isna(label):
            continue
        ax.annotate(
            text=str(label),
            xy=(row.geometry.x, row.geometry.y),
            xytext=(row.geometry.x + dx, row.geometry.y + dy),
            fontsize=font_size,
            color="black",
            alpha=0.9,
        )


def _clip_to_boundary(stations: gpd.GeoDataFrame, boundary: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # Use spatial join to keep points within the boundary polygon.
    stations = stations.copy()
    if stations.crs != boundary.crs:
        stations = stations.to_crs(boundary.crs)

    joined = gpd.sjoin(stations, boundary[["geometry"]], how="inner", predicate="within")
    # Drop join index columns if present.
    joined = joined.drop(columns=[c for c in joined.columns if c.startswith("index_")], errors="ignore")
    return joined


def _plot_group(
    *,
    boundary: gpd.GeoDataFrame,
    points: gpd.GeoDataFrame,
    title: str,
    color: str,
    out_path: Path,
    label_col: str,
    add_labels: bool,
    label_font_size: int,
    label_dx: float,
    label_dy: float,
) -> None:
    # Reproject to WGS84 for rklib decorations (ScaleBar needs lon/lat)
    boundary_wgs84 = boundary.to_crs("EPSG:4326")
    points_wgs84 = points.to_crs("EPSG:4326") if not points.empty else points

    fig, ax = plt.subplots(figsize=(7.5, 7.5), dpi=200)

    boundary_wgs84.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=1.0)
    if not points_wgs84.empty:
        points_wgs84.plot(ax=ax, color=color, markersize=18, alpha=0.85, linewidth=0)
        if add_labels:
            # Convert TWD97-metre offsets to approximate WGS84 degrees
            _annotate_points(
                ax,
                points_wgs84,
                label_col=label_col,
                font_size=label_font_size,
                dx=label_dx / 111_000,
                dy=label_dy / 111_000,
            )

    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25, linestyle='--')

    NorthArrow(ax, location="lower right").draw()
    ScaleBar(ax, location="lower center", length_km=5).draw()

    rklib_savefig(fig, out_path)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot station performance maps (good/medium/low) within Zhuoshui Alluvial Fan")
    parser.add_argument(
        "--run-id",
        default="initial",
        help="Run tag matching the one used in 03_run_model.py (default: initial). "
             "Sets default --results and --outdir to workspace/results/{run-id}/.",
    )
    parser.add_argument("--results", type=Path, default=None)
    parser.add_argument("--input", type=Path, default=Path("data/gray_box_input.csv"))
    parser.add_argument(
        "--boundary",
        type=Path,
        default=Path("data/Zhuoshui Alluvial Fan/Zhuoshui Alluvial Fan.shp"),
    )
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--r2-medium", type=float, default=0.5, help="Medium/low threshold (default: 0.5)")
    parser.add_argument("--r2-good", type=float, default=0.7, help="Good/medium threshold (default: 0.7)")
    parser.add_argument("--no-clip", action="store_true", help="Do not clip points to the boundary")
    parser.add_argument(
        "--label-col",
        type=str,
        default="gw_st",
        help="Which column to label points with (default: gw_st). Common alternatives: st_id",
    )
    parser.add_argument("--no-label", action="store_true", help="Disable point labels")
    parser.add_argument("--label-font-size", type=int, default=8)
    parser.add_argument(
        "--label-dx",
        type=float,
        default=50.0,
        help="Label x-offset in map units (TWD97 meters) (default: 50)",
    )
    parser.add_argument(
        "--label-dy",
        type=float,
        default=50.0,
        help="Label y-offset in map units (TWD97 meters) (default: 50)",
    )

    args = parser.parse_args()

    run_root = Path("workspace/results") / args.run_id
    results_path = args.results if args.results is not None else run_root / "gw_fit_results.csv"
    outdir = args.outdir if args.outdir is not None else run_root / "figures"
    outdir.mkdir(parents=True, exist_ok=True)

    setup_font()
    results_df = _read_results(results_path)
    xy_df = _read_station_xy(args.input)
    boundary = _load_boundary(args.boundary)
    stations = _make_station_gdf(results_df, xy_df)

    if not args.no_clip:
        stations = _clip_to_boundary(stations, boundary)

    r2_good = args.r2_good
    r2_med = args.r2_medium
    if r2_good <= r2_med:
        raise ValueError("--r2-good must be greater than --r2-medium")

    good = stations[stations["r2"] >= r2_good]
    medium = stations[(stations["r2"] >= r2_med) & (stations["r2"] < r2_good)]
    low = stations[stations["r2"] < r2_med]

    add_labels = not args.no_label

    _plot_group(
        boundary=boundary,
        points=good,
        title=f"Good performance (R² ≥ {r2_good:.2f}) — n={len(good)}",
        color="#2ca25f",
        out_path=outdir / "performance_good.tiff",
        label_col=args.label_col,
        add_labels=add_labels,
        label_font_size=args.label_font_size,
        label_dx=args.label_dx,
        label_dy=args.label_dy,
    )
    _plot_group(
        boundary=boundary,
        points=medium,
        title=f"Medium performance ({r2_med:.2f} ≤ R² < {r2_good:.2f}) — n={len(medium)}",
        color="#ff7f00",
        out_path=outdir / "performance_medium.tiff",
        label_col=args.label_col,
        add_labels=add_labels,
        label_font_size=args.label_font_size,
        label_dx=args.label_dx,
        label_dy=args.label_dy,
    )
    _plot_group(
        boundary=boundary,
        points=low,
        title=f"Low performance (R² < {r2_med:.2f}) — n={len(low)}",
        color="#de2d26",
        out_path=outdir / "performance_low.tiff",
        label_col=args.label_col,
        add_labels=add_labels,
        label_font_size=args.label_font_size,
        label_dx=args.label_dx,
        label_dy=args.label_dy,
    )

    print("Wrote:")
    print(f"  {outdir / 'performance_good.tiff'}")
    print(f"  {outdir / 'performance_medium.tiff'}")
    print(f"  {outdir / 'performance_low.tiff'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
