"""rklib-style figures for the poroelastic storage track.

Four figures (TIFF 300 DPI, Times New Roman, via subsidence.sub_plotting._setup/_save):
  1. storativity map     — S_ke across the fan (GNSS circles, MLCW squares)
  2. storativity link     — seasonal surface amp vs seasonal head amp, both networks,
                            with the published S_ke range overlaid
  3. seasonal exemplars   — trend-removed surface vs head at the best GNSS stations
  4. specific storage     — Ss per valid MLCW well (S_ke / column depth)

Reads the run produced by 08_poroelastic_storage.py. Run under poetry (geopandas):
    poetry run python subsidence/poro_plotting.py --run-id poroelastic_v1
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from subsidence.sub_plotting import _setup, _save

try:
    from adjustText import adjust_text
    _HAS_ADJUST = True
except Exception:
    _HAS_ADJUST = False

# Map-kit bootstrap (geopandas + rklib decorations), mirrors 07_plot_maps.py.
try:
    import importlib.util as _ilu
    import geopandas as gpd
    _RKLIB_DIR = Path(__file__).resolve().parents[2] / "rklib"
    for _m in ("fig_base", "map_decorations"):
        _spec = _ilu.spec_from_file_location(f"rklib.{_m}", str(_RKLIB_DIR / f"{_m}.py"))
        _mod = _ilu.module_from_spec(_spec)
        sys.modules[f"rklib.{_m}"] = _mod
        _spec.loader.exec_module(_mod)
    from rklib.map_decorations import NorthArrow, ScaleBar, LocatorInset
    _HAS_MAPKIT = True
    _MAPKIT_ERR = ""
except Exception as _e:
    _HAS_MAPKIT = False
    _MAPKIT_ERR = str(_e)

_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
_BOUNDARY_SHP = _DATA_DIR / "Zhuoshui Alluvial Fan" / "Zhuoshui Alluvial Fan.shp"
_TAIWAN_GPKG = _DATA_DIR / "water" / "Taiwan_countyV02" / "TW_TWN2.gpkg"

# Published elastic skeletal storativity range (Rezaei & Mousavi 2019, Gorgan).
LIT_S_KE = (0.0035, 0.0142)
LIT_LABEL = "Rezaei & Mousavi (2019)"
CMAP = "viridis"
R_VMIN, R_VMAX = 0.4, 0.85
NET_MARKER = {"GNSS": "o", "MLCW": "s"}


def _coupling_size(r) -> np.ndarray:
    """Marker area scaled by coupling strength. Kept small: the map is label-dense
    (co-located GNSS/MLCW pairs sit within a few hundred metres) and oversized
    symbols push the station labels into each other."""
    return 28 + 82 * np.clip((np.asarray(r) - R_VMIN) / (R_VMAX - R_VMIN), 0, 1)


# ---------------------------------------------------------------------------
# Figure 1 — storativity map
# ---------------------------------------------------------------------------
def plot_storativity_map(res: pd.DataFrame, out_path: Path) -> None:
    if not _HAS_MAPKIT:
        print(f"  [skip] storativity map — map-kit unavailable: {_MAPKIT_ERR}")
        return
    _setup()
    m = res.dropna(subset=["Longitude_4326", "Latitude_4326"]).copy()
    boundary = gpd.read_file(_BOUNDARY_SHP).to_crs(epsg=4326)
    b = boundary.total_bounds

    fig, ax = plt.subplots(figsize=(10, 9))
    boundary.plot(ax=ax, facecolor="#FFE8BE", edgecolor="none", alpha=0.5, zorder=1)
    boundary.boundary.plot(ax=ax, edgecolor="black", linewidth=1.2, zorder=4)

    # colour scale from our own data, not an external (non-universal) range
    valid_all = m[m["elastic_valid"]]
    vmin = float(valid_all["s_ke"].min()) if len(valid_all) else 0.0
    vmax = float(valid_all["s_ke"].max()) if len(valid_all) else 1.0

    sc = None
    handles = []
    texts = []
    for net, mk in NET_MARKER.items():
        g = m[m["network"] == net]
        if g.empty:
            continue
        inv = g[~g["elastic_valid"]]
        val = g[g["elastic_valid"]]
        ax.scatter(inv["Longitude_4326"], inv["Latitude_4326"], s=32, marker=mk,
                   facecolors="none", edgecolors="#9e9e9e", linewidths=1.0, zorder=5)
        sc = ax.scatter(val["Longitude_4326"], val["Latitude_4326"], marker=mk,
                        c=val["s_ke"], cmap=CMAP, s=_coupling_size(val["coupling_r"]),
                        edgecolors="black", linewidths=0.7, zorder=6,
                        vmin=vmin, vmax=vmax)
        for r in val.itertuples():
            texts.append(ax.text(r.Longitude_4326, r.Latitude_4326, str(r.station_label),
                                 fontsize=8, zorder=7))
        handles.append(Line2D([0], [0], marker=mk, color="none", markerfacecolor="#4d4d4d",
                              markeredgecolor="black", markersize=7,
                              label=f"{net} valid (n={len(val)})"))
    handles.append(Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
                          markeredgecolor="#9e9e9e", markersize=7, label="no elastic signal"))
    # Symbol area encodes coupling strength, so it needs its own key; without one the
    # size channel is unreadable and the caption promises something the figure omits.
    handles.append(Line2D([0], [0], color="none", label="$\\it{coupling\\ r}$"))
    for r_ref in (0.4, 0.6, 0.8):
        handles.append(Line2D([0], [0], marker="o", color="none", markerfacecolor="#bdbdbd",
                              markeredgecolor="black", markeredgewidth=0.7,
                              markersize=np.sqrt(_coupling_size(r_ref)),
                              label=f"   {r_ref:.1f}"))

    px = (b[2] - b[0]) * 0.05; py = (b[3] - b[1]) * 0.05
    ax.set_xlim(b[0] - px, b[2] + px); ax.set_ylim(b[1] - py, b[3] + py)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Longitude (°E)", fontweight="bold", fontsize=13, color="black")
    ax.set_ylabel("Latitude (°N)", fontweight="bold", fontsize=13, color="black")
    ax.tick_params(axis="both", labelsize=12, labelcolor="black")
    ax.ticklabel_format(useOffset=False, style="plain")
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.legend(handles=handles, loc="upper right", fontsize=9.5, framealpha=0.95,
              edgecolor="black")
    # de-overlap station labels (co-located GNSS+MLCW pairs crowd the centre)
    if _HAS_ADJUST and texts:
        adjust_text(texts, ax=ax, expand=(1.9, 2.2), force_text=(0.6, 0.9),
                    min_arrow_len=6,
                    arrowprops=dict(arrowstyle="-", color="0.45", lw=0.5))
    NorthArrow(ax, location="lower right").draw()
    ScaleBar(ax, location="lower center", length_km=10).draw()
    if sc is not None:
        cb = fig.colorbar(sc, ax=ax, fraction=0.040, pad=0.02)
        cb.set_label("Elastic skeletal storativity  S$_{ke}$  (–)", fontweight="bold")

    if _TAIWAN_GPKG.exists():
        try:
            tw = gpd.read_file(_TAIWAN_GPKG).to_crs(epsg=4326)
            LocatorInset(ax, bounds=[0.015, 0.63, 0.30, 0.35],
                         xlim=(119.9, 122.1), ylim=(21.9, 25.4),
                         highlight_gdf=boundary,
                         highlight_kw={"edgecolor": "red", "linewidth": 1.2},
                         overlay_gdfs=[{"gdf": tw, "facecolor": "#ececec",
                                        "edgecolor": "grey", "linewidth": 0.4}]).draw()
        except Exception as e:
            print(f"  [warn] locator inset skipped: {e}")

    _save(fig, out_path)
    print(f"  Wrote: {out_path}")


# ---------------------------------------------------------------------------
# Figure 2 — storativity link
# ---------------------------------------------------------------------------
def plot_storativity_link(res: pd.DataFrame, out_path: Path) -> None:
    _setup()
    m = res.dropna(subset=["head_seas_amp_m", "disp_seas_amp_m", "coupling_r"]).copy()
    m["disp_amp_cm"] = m["disp_seas_amp_m"] * 100.0
    valid = m[m["elastic_valid"]]

    fig, ax = plt.subplots(figsize=(7.5, 6))
    xmax = m["head_seas_amp_m"].max() * 1.08
    xs = np.linspace(0, xmax, 50)

    sc = None
    for net, mk in NET_MARKER.items():
        g = m[m["network"] == net]
        gi = g[~g["elastic_valid"]]; gv = g[g["elastic_valid"]]
        ax.scatter(gi["head_seas_amp_m"], gi["disp_amp_cm"], s=55, marker=mk,
                   facecolors="none", edgecolors="#9e9e9e", linewidths=1.0, zorder=2)
        if len(gv) and "disp_amp_lo" in gv.columns:
            xe = np.vstack([gv["head_seas_amp_m"] - gv["head_amp_lo"],
                            gv["head_amp_hi"] - gv["head_seas_amp_m"]])
            ye = np.vstack([(gv["disp_seas_amp_m"] - gv["disp_amp_lo"]) * 100,
                            (gv["disp_amp_hi"] - gv["disp_seas_amp_m"]) * 100])
            ax.errorbar(gv["head_seas_amp_m"], gv["disp_amp_cm"], xerr=xe, yerr=ye,
                        fmt="none", ecolor="#9e9e9e", elinewidth=0.7, capsize=2, zorder=2)
        sc = ax.scatter(gv["head_seas_amp_m"], gv["disp_amp_cm"], marker=mk,
                        c=gv["coupling_r"], cmap=CMAP, vmin=R_VMIN, vmax=R_VMAX,
                        s=75, edgecolors="black", linewidths=0.5, zorder=3)

    if len(valid) > 2:
        rr, pp = stats.pearsonr(valid["head_seas_amp_m"], valid["disp_amp_cm"])
        # through-origin reference line at the median per-station storativity
        med_ske = valid["s_ke"].median()
        ax.plot(xs, med_ske * xs * 100, "k--", lw=1.3, zorder=4,
                label=f"constant S$_{{ke}}$ = {med_ske:.4f} (median)")
        ax.text(0.03, 0.97,
                f"r = {rr:.2f},  p = {pp:.3f},  n = {len(valid)}",
                transform=ax.transAxes, va="top", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                          edgecolor="#4d4d4d", linewidth=0.8, alpha=0.92))

    handles = [Line2D([0], [0], marker=mk, color="none", markerfacecolor="#4d4d4d",
                      markeredgecolor="black", markersize=9, label=net)
               for net, mk in NET_MARKER.items()]
    handles.append(Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
                          markeredgecolor="#9e9e9e", markersize=9, label="no elastic signal"))
    if len(valid) > 2:
        handles.append(Line2D([0], [0], color="black", ls="--", lw=1.3,
                              label=f"constant S$_{{ke}}$ = {valid['s_ke'].median():.4f} (median)"))
    if sc is not None:
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label("coupling r", fontweight="bold")
    ax.set_xlabel("Seasonal head amplitude (m)", fontweight="bold", fontsize=13, color="black")
    ax.set_ylabel("Seasonal vertical-displacement amplitude (cm)", fontweight="bold",
                  fontsize=13, color="black")
    ax.tick_params(axis="both", labelsize=12, labelcolor="black")
    xhi = np.nanmax(m["head_amp_hi"]) if "head_amp_hi" in m.columns else m["head_seas_amp_m"].max()
    yhi = np.nanmax(m["disp_amp_hi"]) * 100 if "disp_amp_hi" in m.columns else m["disp_amp_cm"].max()
    ax.set_xlim(0, max(xmax, xhi * 1.02))
    ax.set_ylim(0, max(m["disp_amp_cm"].max(), yhi) * 1.1)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(handles=handles, loc="lower right", fontsize=9, framealpha=0.95,
              edgecolor="black")
    _save(fig, out_path)
    print(f"  Wrote: {out_path}")


# ---------------------------------------------------------------------------
# Figure 3 — seasonal exemplars (GNSS, daily — cleanest visual)
# ---------------------------------------------------------------------------
def plot_seasonal_exemplars(run_dir: Path, res: pd.DataFrame, out_path: Path,
                            exemplars: list[str] | None = None) -> None:
    _setup()
    valid = res[(res["elastic_valid"]) & (res["network"] == "GNSS")] \
        .sort_values("coupling_r", ascending=False)
    if exemplars is None:
        exemplars = list(valid["sub_id"].head(3))
    if not exemplars:
        print("  [skip] exemplars — no elastic-valid GNSS stations")
        return

    fig, axes = plt.subplots(len(exemplars), 1, figsize=(7.2, 3.0 * len(exemplars)),
                             squeeze=False)
    for ax, sid in zip(axes[:, 0], exemplars):
        f = run_dir / "per_station" / f"{sid}_decomp.csv"
        if not f.exists():
            ax.set_visible(False); continue
        d = pd.read_csv(f, index_col=0, parse_dates=True)
        surf_dates = d["disp_resid_m"].dropna().index
        if len(surf_dates):
            d = d.loc[(d.index >= surf_dates.min()) & (d.index <= surf_dates.max())]
        row = res[res["sub_id"] == sid].iloc[0]
        surf = d["disp_resid_m"] * 100
        head = d["head_resid_m"]
        ax.plot(d.index, surf, color="#d6604d", lw=1.1, label="GNSS surface")
        ax.set_ylabel("surface anomaly (cm)", color="black", fontsize=13)
        ax.tick_params(axis="y", labelcolor="black", labelsize=12)
        ax.tick_params(axis="x", labelcolor="black", labelsize=12)
        ax.axhline(0, color="0.7", lw=0.6, zorder=0)
        ax2 = ax.twinx()
        ax2.plot(d.index, head, color="#2166ac", lw=1.1, alpha=0.85, label="head")
        ax2.set_ylabel("head anomaly (m)", color="black", fontsize=13)
        ax2.tick_params(axis="y", labelcolor="black", labelsize=12)

        # The inversion reads amplitude and phase off these fitted harmonics, so show
        # them: without the fit the reader cannot check what S_ke and the lag came from.
        t0 = pd.Timestamp(row["phase_t0"])
        yrs = (d.index - t0).total_seconds().to_numpy() / (365.25 * 86400.0)
        w = 2 * np.pi
        fit_surf = row["disp_seas_amp_m"] * 100 * np.sin(w * yrs + row["disp_phase_rad"])
        fit_head = row["head_seas_amp_m"] * np.sin(w * yrs + row["head_phase_rad"])
        ax.plot(d.index, fit_surf, color="#7f2718", lw=1.7, zorder=4,
                label="fitted annual harmonic")
        ax2.plot(d.index, fit_head, color="#0b3d70", lw=1.7, ls="--", zorder=4,
                 label="fitted annual harmonic")

        # Scale the head axis by the station's own S_ke, so a perfect poroelastic
        # response would superimpose the two curves exactly. Independently scaled
        # twin axes would manufacture agreement whatever the data showed.
        def _pad(lo, hi, top=0.42, bot=0.10):
            span = hi - lo
            return lo - bot * span, hi + top * span
        lo, hi = _pad(np.nanmin(surf), np.nanmax(surf))
        ax.set_ylim(lo, hi)
        ax2.set_ylim(lo / (row["s_ke"] * 100), hi / (row["s_ke"] * 100))
        ax.set_xlim(d.index.min(), d.index.max())
        ax.set_title(str(row["station_label"]), fontsize=11, fontweight="bold")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        l1 = ["GNSS surface", "surface harmonic"]
        l2 = ["head", "head harmonic"]
        ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8, framealpha=0.92,
                  edgecolor="#4d4d4d", ncol=2,
                  title=f"r={row['coupling_r']:.2f},  S$_{{ke}}$={row['s_ke']:.4f},  "
                        f"lag={row['phase_lag_days']:+.0f} d",
                  title_fontsize=8.5)
        ax.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout()
    _save(fig, out_path)
    print(f"  Wrote: {out_path}")


# ---------------------------------------------------------------------------
# Figure 4 — specific storage (MLCW, column-bulk Ss = S_ke / depth)
# ---------------------------------------------------------------------------
def plot_specific_storage(res: pd.DataFrame, out_path: Path) -> None:
    _setup()
    v = res[(res["network"] == "MLCW") & (res["elastic_valid"]) &
            res["ss_per_m"].notna()].copy().sort_values("ss_per_m")
    if v.empty:
        print("  [skip] specific storage — no valid MLCW wells")
        return
    fig, ax = plt.subplots(figsize=(7.5, 0.55 * len(v) + 2.2))
    y = np.arange(len(v))
    ax.barh(y, v["ss_per_m"], color="#4393c3", edgecolor="black", linewidth=0.6, zorder=3)
    if "ss_lo" in v.columns:
        xe = np.vstack([v["ss_per_m"] - v["ss_lo"], v["ss_hi"] - v["ss_per_m"]])
        ax.errorbar(v["ss_per_m"], y, xerr=xe, fmt="none", ecolor="black",
                    elinewidth=0.8, capsize=3, zorder=5)
    labels = [f"{r.station_label}  ({r.thickness_m:.0f} m)"
              for r in v.itertuples()]
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=11, color="black")
    med = v["ss_per_m"].median()
    ax.axvline(med, color="#d6604d", ls="--", lw=1.3, zorder=4,
               label=f"median = {med:.2e} /m")
    for yi, val in zip(y, v["ss_per_m"]):
        ax.text(val, yi, f"  {val:.2e}", va="center", fontsize=8)
    # extend the x-axis by one tick interval so end-of-bar value labels have room
    vmax = v["ss_hi"].max() if "ss_hi" in v.columns else v["ss_per_m"].max()
    ticks = ax.get_xticks()
    step = ticks[1] - ticks[0] if len(ticks) > 1 else vmax * 0.2
    ax.set_xlim(0, vmax + 0.4 * step)
    ax.set_xlabel("Elastic skeletal specific storage  S$_s$  (m$^{-1}$)",
                  fontweight="bold", fontsize=13, color="black")
    ax.tick_params(axis="x", labelsize=12, labelcolor="black")
    ax.grid(True, axis="x", alpha=0.3, linestyle="--")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95, edgecolor="black")
    fig.tight_layout()
    _save(fig, out_path)
    print(f"  Wrote: {out_path}")


ESM_PATH = Path("workspace/manuscripts/poroelastic_gnss-mlcw/submission/"
                "ESM1_poroelastic_per_station_results.csv")


def _attach_labels(res: pd.DataFrame) -> pd.DataFrame:
    """Attach the Latin station label used in the manuscript figures and tables.

    MLCW wells are keyed by their Chinese site name, which cannot go on a figure
    aimed at an English-language journal, so the romanisation lives in the ESM and
    is merged in here rather than duplicated across scripts.
    """
    esm = pd.read_csv(ESM_PATH)[["station", "network", "station_label"]]
    out = res.merge(esm, left_on=["sub_id", "network"],
                    right_on=["station", "network"], how="left")
    out["station_label"] = out["station_label"].fillna(out["sub_id"])
    return out.drop(columns=["station"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="poroelastic_v1")
    args = ap.parse_args()
    run_dir = Path("workspace/results_sub") / args.run_id
    res = pd.read_csv(run_dir / "poroelastic_results.csv")
    res = _attach_labels(res)
    fig_dir = run_dir / "figures"
    print(f"Poroelastic figures -> {fig_dir}")
    plot_storativity_map(res, fig_dir / "figure1.tiff")
    plot_storativity_link(res, fig_dir / "figure2.tiff")
    plot_seasonal_exemplars(run_dir, res, fig_dir / "figure3.tiff")


if __name__ == "__main__":
    main()
