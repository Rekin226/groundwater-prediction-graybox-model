"""rklib-style plotting helpers for the subsidence pipeline.

All figures use Times New Roman, TIFF 300 DPI, mirror the existing
srcs/gw_shell.py plot conventions for visual consistency in the manuscript.

Variant colors:
    M1 (Form 2 const v_tect):     blue
    M2 (Form 2 linear v_tect):    green
    M3_tau (Form 3 const):        red
    M4_tau (Form 3 linear):       orange
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Try to use rklib helpers if available; otherwise fall back to plain matplotlib
try:
    from rklib import setup_font, savefig
    _USE_RKLIB = True
except Exception:
    _USE_RKLIB = False

VARIANT_COLOR = {
    "M1":     "tab:blue",
    "M2":     "tab:green",
    "M3_tau": "tab:red",
    "M4_tau": "tab:orange",
}

METRIC_BOX = dict(
    boxstyle="round,pad=0.35",
    facecolor="white",
    edgecolor="#4d4d4d",
    linewidth=0.8,
    alpha=0.92,
)


def _apply_render_mask(y: "np.ndarray", mask: "np.ndarray") -> "np.ndarray":
    """Return y with cells set to NaN where mask is False.

    Matplotlib renders NaN as a line break, so this is the one-line
    primitive for "don't draw here" used by all three plot helpers.
    """
    out = np.asarray(y, dtype=float).copy()
    out[~np.asarray(mask, dtype=bool)] = np.nan
    return out


def _setup():
    """Apply publication font settings.

    CJK note: matplotlib's per-glyph fallback only works when font.family is a
    direct list of font names — NOT when family is the generic "serif" alias
    with CJK fonts in font.serif (verified empirically with mpl 3.10).  We list
    Times New Roman first for Latin, then concrete CJK fonts so Chinese station
    names (e.g. 土庫國中) render correctly.
    """
    if _USE_RKLIB:
        setup_font()
        # Append CJK fallbacks after rklib's primary serif setup so Chinese
        # glyphs don't render as boxes.  Direct family list (not "serif" generic)
        # is required for per-glyph fallback in matplotlib ≥ 3.6.
        primary = plt.rcParams.get("font.family", ["Times New Roman"])
        if isinstance(primary, str):
            primary = [primary]
        plt.rcParams["font.family"] = list(primary) + [
            "Heiti TC",           # macOS Traditional Chinese (system font)
            "Hiragino Sans GB",   # macOS Simplified Chinese
            "Songti SC",          # macOS serif Chinese
            "Arial Unicode MS",   # broad Unicode coverage
            "Noto Sans CJK TC",   # Linux/CI fallback
            "WenQuanYi Zen Hei",  # Linux fallback
        ]
    else:
        plt.rcParams.update({
            "font.family": [
                "Times New Roman",    # Latin primary
                "Heiti TC",           # macOS Traditional Chinese
                "Hiragino Sans GB",   # macOS Simplified Chinese
                "Songti SC",          # macOS serif Chinese
                "Arial Unicode MS",   # broad Unicode coverage
                "Noto Sans CJK TC",   # Linux/CI
                "WenQuanYi Zen Hei",  # Linux fallback
                "DejaVu Serif",       # ultimate fallback
            ],
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
        })


def _save(fig, path: Path):
    """Save figure as TIFF 300 DPI, creating parent dirs as needed."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if _USE_RKLIB:
        savefig(fig, str(path), dpi=300)
    else:
        fig.savefig(str(path), dpi=300, format="tiff",
                    bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def _fmt_xaxis(ax):
    """Apply year-based x-axis formatting."""
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", labelsize=9)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def plot_per_variant(
    sub_id: str,
    variant: str,
    t: pd.DatetimeIndex,
    zeta_obs: np.ndarray,
    sim: np.ndarray,
    cal_idx: np.ndarray,
    val_idx: np.ndarray,
    metrics: dict,
    out_path: Path,
    *,
    zeta_filled: np.ndarray | None = None,
    zeta_sigma: np.ndarray | None = None,
    imputed_mask: np.ndarray | None = None,
    render_mask: np.ndarray | None = None,
):
    """One plot per variant — observed black, cal+val sim in variant color.

    When zeta_filled/zeta_sigma/imputed_mask are passed, the observed line
    is rendered in two passes (black for real, muted purple #7e57c2 for
    imputed) with a ±1σ envelope over imputed regions.
    """
    _setup()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    color = VARIANT_COLOR.get(variant, "tab:gray")

    # Cal / val / buffer shading
    if cal_idx.size:
        ax.axvspan(t[cal_idx[0]], t[cal_idx[-1]], color="#2166ac", alpha=0.10, zorder=0,
                   label="Cal (training)")
    if val_idx.size:
        ax.axvspan(t[val_idx[0]], t[val_idx[-1]], color="#d6604d", alpha=0.10, zorder=0,
                   label="Val (forecast)")
    if cal_idx.size and val_idx.size:
        buf_lo = t[int(cal_idx[-1])] + pd.Timedelta(days=1)
        buf_hi = t[int(val_idx[0])]  - pd.Timedelta(days=1)
        if buf_lo < buf_hi:
            ax.axvspan(buf_lo, buf_hi, color="#bdbdbd", alpha=0.15,
                       zorder=0, label="Buffer (excluded)")

    # Observed (real + imputed two-pass when imputation provided)
    if zeta_filled is not None and imputed_mask is not None:
        y_obs_only = np.where(imputed_mask, np.nan, zeta_filled)
        y_imp_only = np.where(imputed_mask, zeta_filled, np.nan)
        ax.plot(t, y_obs_only, color="black", lw=1.0, alpha=0.85, label="Observed")
        if render_mask is None:
            render_mask = np.ones_like(y_imp_only, dtype=bool)
        y_imp_only = _apply_render_mask(y_imp_only, render_mask)
        if np.any(render_mask & imputed_mask):
            ax.plot(t, y_imp_only, color="#7e57c2", lw=1.0, alpha=0.95,
                    label="GPR-imputed ±1σ")
        # else: globally rejected → no line OR legend entry
        if zeta_sigma is not None and np.any(np.isfinite(zeta_sigma)):
            ax.fill_between(
                t,
                zeta_filled - zeta_sigma,
                zeta_filled + zeta_sigma,
                where=imputed_mask & render_mask,
                color="#7e57c2", alpha=0.20, linewidth=0, zorder=1,
            )
    else:
        ax.plot(t, zeta_obs, color="black", lw=1.0, alpha=0.85, label="Observed")

    # Buffer-year sim segment (subdued dotted; plotted before cal/val so they overlay)
    if cal_idx.size and val_idx.size:
        buf_lo_i = int(cal_idx[-1]) + 1
        buf_hi_i = int(val_idx[0])
        if buf_lo_i < buf_hi_i:
            ax.plot(t[buf_lo_i:buf_hi_i], sim[buf_lo_i:buf_hi_i], color=color,
                    lw=1.2, alpha=0.5, ls=":", label=f"{variant} – buffer")

    # Sim cal / val
    if cal_idx.size:
        ax.plot(t[cal_idx], sim[cal_idx], color=color, lw=1.8,
                alpha=0.95, ls="-", label=f"{variant} – cal")
    if val_idx.size:
        ax.plot(t[val_idx], sim[val_idx], color=color, lw=1.8,
                alpha=0.95, ls="--", label=f"{variant} – val")

    # Split lines (cal-end and val-start)
    if cal_idx.size and val_idx.size:
        ax.axvline(t[int(cal_idx[-1])], color="black", ls="--", lw=1.2, alpha=0.8)
        ax.axvline(t[int(val_idx[0])], color="black", ls="--", lw=1.2, alpha=0.8)

    # Metrics annotation
    kge_cal  = metrics.get("kge_cal",  float("nan"))
    kge_val  = metrics.get("kge_val",  float("nan"))
    rmse_val = metrics.get("rmse_val", float("nan"))
    txt = (
        f"KGE_cal = {kge_cal:.3f}\n"
        f"KGE_val = {kge_val:.3f}\n"
        f"RMSE_val = {rmse_val:.3f} m"
    )
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", fontsize=8,
            bbox=METRIC_BOX)

    ax.set_xlabel("Year", fontweight="bold")
    ax.set_ylabel(r"Cumulative $\zeta$ (m)", fontweight="bold")
    ax.set_title(f"{sub_id} — {variant}", fontweight="bold", pad=6)
    ax.legend(loc="best", framealpha=0.92, frameon=True, fontsize=8)
    ax.grid(True, lw=0.3, alpha=0.4)
    _fmt_xaxis(ax)
    ax.set_xlim(t[0], t[-1])
    ax.margins(x=0)
    fig.tight_layout()
    _save(fig, out_path)


def plot_comparison(
    sub_id: str,
    t: pd.DatetimeIndex,
    zeta_obs: np.ndarray,
    fits: dict,
    cal_idx: np.ndarray,
    val_idx: np.ndarray,
    out_path: Path,
    *,
    zeta_filled: np.ndarray | None = None,
    zeta_sigma: np.ndarray | None = None,
    imputed_mask: np.ndarray | None = None,
    render_mask: np.ndarray | None = None,
):
    """All fitted variants overlaid; metrics table beneath; best variant asterisked.

    Parameters
    ----------
    fits : dict  variant → fit-dict (must contain sim_full + metric keys)
    """
    _setup()
    fig, (ax, ax_t) = plt.subplots(
        2, 1, figsize=(9, 7),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.38},
    )

    # Shade cal/val
    if cal_idx.size:
        ax.axvspan(t[cal_idx[0]], t[cal_idx[-1]], color="#2166ac", alpha=0.06, zorder=0,
                   label="Cal (training)")
    if val_idx.size:
        ax.axvspan(t[val_idx[0]], t[val_idx[-1]], color="#d6604d", alpha=0.06, zorder=0,
                   label="Val (forecast)")
    if cal_idx.size and val_idx.size:
        buf_lo = t[int(cal_idx[-1])] + pd.Timedelta(days=1)
        buf_hi = t[int(val_idx[0])]  - pd.Timedelta(days=1)
        if buf_lo < buf_hi:
            ax.axvspan(buf_lo, buf_hi, color="#bdbdbd", alpha=0.15,
                       zorder=0, label="Buffer (excluded)")

    # Observed (real + imputed two-pass when imputation provided)
    if zeta_filled is not None and imputed_mask is not None:
        y_obs_only = np.where(imputed_mask, np.nan, zeta_filled)
        y_imp_only = np.where(imputed_mask, zeta_filled, np.nan)
        ax.plot(t, y_obs_only, color="black", lw=1.2, alpha=0.9, label="Observed", zorder=3)
        if render_mask is None:
            render_mask = np.ones_like(y_imp_only, dtype=bool)
        y_imp_only = _apply_render_mask(y_imp_only, render_mask)
        if np.any(render_mask & imputed_mask):
            ax.plot(t, y_imp_only, color="#7e57c2", lw=1.2, alpha=0.95,
                    label="GPR-imputed ±1σ", zorder=3)
        # else: globally rejected → no line OR legend entry
        if zeta_sigma is not None and np.any(np.isfinite(zeta_sigma)):
            ax.fill_between(
                t,
                zeta_filled - zeta_sigma,
                zeta_filled + zeta_sigma,
                where=imputed_mask & render_mask,
                color="#7e57c2", alpha=0.20, linewidth=0, zorder=1,
            )
    else:
        ax.plot(t, zeta_obs, color="black", lw=1.2, alpha=0.9, label="Observed", zorder=3)

    # Identify best variant by kge_val
    best_kge_val = -np.inf
    best_variant = None
    for v, f in fits.items():
        kv = f.get("kge_val", np.nan)
        if not np.isnan(kv) and kv > best_kge_val:
            best_kge_val = kv
            best_variant = v

    # Plot each variant
    buf_lo = int(cal_idx[-1]) + 1 if cal_idx.size else None
    buf_hi = int(val_idx[0]) if val_idx.size else None
    has_buffer = (buf_lo is not None and buf_hi is not None and buf_lo < buf_hi)
    _buf_label_used = False
    for v, f in fits.items():
        sim = f["sim_full"]
        c = VARIANT_COLOR.get(v, "tab:gray")
        is_best = (v == best_variant)
        lw = 2.0 if is_best else 1.2
        alpha = 1.0 if is_best else 0.75
        # Buffer-year segment (subdued dotted; plotted first so cal/val overlay)
        # Only the first variant gets the legend entry; subsequent ones are suppressed.
        if has_buffer:
            buf_label = "Buffer-year sim (excluded)" if not _buf_label_used else "_nolegend_"
            ax.plot(t[buf_lo:buf_hi], sim[buf_lo:buf_hi], color=c, lw=lw*0.7,
                    alpha=alpha*0.5, ls=":", label=buf_label)
            _buf_label_used = True
        if cal_idx.size:
            ax.plot(t[cal_idx], sim[cal_idx], color=c, lw=lw, alpha=alpha, ls="-")
        if val_idx.size:
            ax.plot(t[val_idx], sim[val_idx], color=c, lw=lw, alpha=alpha, ls="--",
                    label=f"{v} (best)" if is_best else v)

    # Split line
    if cal_idx.size and val_idx.size:
        ax.axvline(t[val_idx[0]], color="black", ls="--", lw=1.2, alpha=0.8)

    ax.set_xlabel("Year", fontweight="bold")
    ax.set_ylabel(r"Cumulative $\zeta$ (m)", fontweight="bold")
    ax.set_title(f"{sub_id} — variant comparison", fontweight="bold", pad=6)
    ax.legend(loc="best", framealpha=0.85)
    ax.grid(True, lw=0.3, alpha=0.4)
    _fmt_xaxis(ax)

    # Metrics table
    ax_t.axis("off")
    columns = ["Variant", "KGE_cal", "KGE_val", "RMSE_val (m)"]
    cell_text = []
    for v, f in fits.items():
        label = f"{v} (best)" if v == best_variant else v
        cell_text.append([
            label,
            f"{f.get('kge_cal',  float('nan')):.3f}",
            f"{f.get('kge_val',  float('nan')):.3f}",
            f"{f.get('rmse_val', float('nan')):.3f}",
        ])
    tbl = ax_t.table(cellText=cell_text, colLabels=columns,
                     loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.35)
    # Header row styling
    for j in range(len(columns)):
        tbl[(0, j)].set_facecolor("#e8e8e8")
        tbl[(0, j)].set_text_props(weight="bold")

    ax.set_xlim(t[0], t[-1])
    ax.margins(x=0)
    fig.tight_layout()
    _save(fig, out_path)


def plot_full_subplots(
    sub_id: str,
    t: pd.DatetimeIndex,
    zeta_obs: np.ndarray,
    sim_best: np.ndarray,
    h_driver: np.ndarray,
    driver_source,
    rainfall,
    out_path: Path,
    *,
    cal_idx: np.ndarray | None = None,
    val_idx: np.ndarray | None = None,
    zeta_filled: np.ndarray | None = None,
    zeta_sigma: np.ndarray | None = None,
    imputed_mask: np.ndarray | None = None,
    render_mask: np.ndarray | None = None,
):
    """3-panel overview: ζ (best variant), h_driver shaded by source, rainfall (optional).

    Parameters
    ----------
    driver_source : array-like of strings (same length as t) or None
    rainfall : array-like or None — if None, the rainfall panel is omitted
    cal_idx, val_idx : optional index arrays for cal/val shading
    zeta_filled, zeta_sigma, imputed_mask : optional GPR-imputation arrays
    """
    _setup()
    n_panels = 3 if rainfall is not None else 2
    height_ratios = [2.5, 1, 1][:n_panels]
    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(9, 2.2 * n_panels + 1),
        sharex=True,
        gridspec_kw={"height_ratios": height_ratios, "hspace": 0.25},
    )
    if n_panels == 2:
        axes = list(axes)

    # ---- Panel (a): ζ observed + best sim ----
    ax0 = axes[0]

    # Cal / val / buffer shading on top panel
    if cal_idx is not None and cal_idx.size:
        ax0.axvspan(t[cal_idx[0]], t[cal_idx[-1]], color="#2166ac", alpha=0.06, zorder=0,
                    label="Cal (training)")
    if val_idx is not None and val_idx.size:
        ax0.axvspan(t[val_idx[0]], t[val_idx[-1]], color="#d6604d", alpha=0.06, zorder=0,
                    label="Val (forecast)")
    if cal_idx is not None and cal_idx.size and val_idx is not None and val_idx.size:
        buf_lo = t[int(cal_idx[-1])] + pd.Timedelta(days=1)
        buf_hi = t[int(val_idx[0])]  - pd.Timedelta(days=1)
        if buf_lo < buf_hi:
            ax0.axvspan(buf_lo, buf_hi, color="#bdbdbd", alpha=0.15,
                        zorder=0, label="Buffer (excluded)")

    # Observed (real + imputed two-pass when imputation provided)
    if zeta_filled is not None and imputed_mask is not None:
        y_obs_only = np.where(imputed_mask, np.nan, zeta_filled)
        y_imp_only = np.where(imputed_mask, zeta_filled, np.nan)
        ax0.plot(t, y_obs_only, color="black", lw=1.0, alpha=0.85, label="Observed")
        if render_mask is None:
            render_mask = np.ones_like(y_imp_only, dtype=bool)
        y_imp_only = _apply_render_mask(y_imp_only, render_mask)
        if np.any(render_mask & imputed_mask):
            ax0.plot(t, y_imp_only, color="#7e57c2", lw=1.0, alpha=0.95,
                     label="GPR-imputed ±1σ")
        # else: globally rejected → no line OR legend entry
        if zeta_sigma is not None and np.any(np.isfinite(zeta_sigma)):
            ax0.fill_between(
                t,
                zeta_filled - zeta_sigma,
                zeta_filled + zeta_sigma,
                where=imputed_mask & render_mask,
                color="#7e57c2", alpha=0.20, linewidth=0, zorder=1,
            )
    else:
        ax0.plot(t, zeta_obs, color="black", lw=1.0, alpha=0.85, label="Observed")

    ax0.plot(t, sim_best, color="tab:blue", lw=1.6, alpha=0.95, label="Best variant")
    ax0.set_ylabel(r"$\zeta$ (m)", fontweight="bold")
    ax0.set_title(f"{sub_id} — best fit overview", fontweight="bold", pad=6)
    ax0.legend(fontsize=8, loc="upper right", framealpha=0.85)
    ax0.grid(True, lw=0.3, alpha=0.4)

    # ---- Panel (b): h_driver, shaded by driver_source ----
    ax1 = axes[1]
    ax1.plot(t, h_driver, color="tab:gray", lw=0.8, alpha=0.9)
    if driver_source is not None:
        driver_source = np.asarray(driver_source)
        if len(driver_source) == len(t):
            h_min = float(np.nanmin(h_driver))
            h_max = float(np.nanmax(h_driver))
            _src_colors = [
                ("model_fill",    "lightcoral"),
                ("linear_interp", "khaki"),
                ("taper",         "lightblue"),
            ]
            for src, col in _src_colors:
                mask = (driver_source == src)
                if mask.any():
                    ax1.fill_between(t, h_min, h_max, where=mask,
                                     color=col, alpha=0.30, label=src, zorder=0)
    ax1.set_ylabel(r"$h$ driver (m)", fontweight="bold")
    ax1.legend(fontsize=7, loc="upper right", framealpha=0.85)
    ax1.grid(True, lw=0.3, alpha=0.4)

    # ---- Panel (c): rainfall ----
    if rainfall is not None:
        ax2 = axes[2]
        ax2.bar(t, np.asarray(rainfall), width=1.0, color="tab:blue", alpha=0.55)
        ax2.set_ylabel("Rain (mm/d)", fontweight="bold")
        ax2.grid(True, lw=0.3, alpha=0.4)

    axes[-1].set_xlabel("Year", fontweight="bold")
    _fmt_xaxis(axes[-1])

    # Panel labels
    for ax, lbl in zip(axes, ["a", "b", "c"]):
        ax.text(-0.07, 1.02, f"({lbl})", transform=ax.transAxes,
                fontsize=10, fontweight="bold", va="bottom", ha="left")

    axes[-1].set_xlim(t[0], t[-1])
    for ax in axes:
        ax.margins(x=0)
    fig.tight_layout()
    _save(fig, out_path)


def plot_mlcw_layer_profile(sub_id: str, layer_csv_path: Path, out_path: Path):
    """Per-layer cumulative compaction time series (MLCW only).

    Reads the *_mlcw_layer.csv written by _build_zeta() and draws one line
    per ring (NO* column), coloured by a viridis gradient, with a colorbar.
    """
    _setup()
    df = pd.read_csv(layer_csv_path, index_col=0, parse_dates=True)
    cols = [c for c in df.columns if c.startswith("NO")]
    cols.sort(key=lambda c: int(c[2:]))
    if not cols:
        return  # nothing to plot

    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.cm.viridis
    n = len(cols)
    for i, c in enumerate(cols):
        col_color = cmap(i / max(1, n - 1))
        # Only add to legend for first, middle, last ring to avoid clutter
        show_label = i in (0, n // 2, n - 1)
        ax.plot(df.index, df[c], color=col_color, lw=1.0,
                label=c if show_label else None)

    ax.set_xlabel("Date", fontweight="bold")
    ax.set_ylabel("Cumulative compaction (m)", fontweight="bold")
    ax.set_title(f"{sub_id} — MLCW per-layer compaction", fontweight="bold", pad=6)
    ax.legend(fontsize=7, ncol=2, loc="upper left", framealpha=0.85)
    ax.grid(True, lw=0.3, alpha=0.4)
    _fmt_xaxis(ax)

    # Colorbar (ring number)
    ring_min = int(cols[0][2:])
    ring_max = int(cols[-1][2:])
    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=plt.Normalize(vmin=ring_min, vmax=ring_max),
    )
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Ring NO")

    fig.tight_layout()
    _save(fig, out_path)
