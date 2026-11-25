"""Figures 4 and 5 of the poroelastic manuscript.

Fig. 4  the amplitude scaling is sublinear, so S_ke is not a fan-wide constant
        (a) log-log seasonal displacement vs head amplitude, fitted exponent
            against the beta = 1 line the Terzaghi-Jacob model requires
        (b) the consequence: per-station S_ke falls as head amplitude rises

Fig. 5  per-station S_ke with bootstrap 95% intervals, co-located GNSS/MLCW
        pairs on adjacent rows, which is where the cross-instrument agreement
        and the six-fold spread are both legible

Run:
    poetry run python subsidence/scripts/poro_fig_storativity_variation.py
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

# Figure styling comes from an external `figstyle` module (Times New Roman, journal
# column widths, TIFF export). Point FIGSTYLE_DIR at the directory containing it;
# it is not vendored here because it is shared across manuscripts.
_FIGSTYLE_DIR = os.environ.get(
    "FIGSTYLE_DIR",
    str(Path.home() / ".claude" / "skills" / "paper-figures" / "scripts"),
)
if not (Path(_FIGSTYLE_DIR) / "figstyle.py").exists():
    raise SystemExit(
        f"figstyle.py not found in {_FIGSTYLE_DIR}. Set FIGSTYLE_DIR to the directory "
        "containing it, or substitute any matplotlib style module exposing "
        "use_style(), new_figure(), panel_label(), metric_box() and save()."
    )
sys.path.insert(0, _FIGSTYLE_DIR)
import figstyle as fs  # noqa: E402

ESM = Path("workspace/manuscripts/poroelastic_gnss-mlcw/submission/"
           "ESM1_poroelastic_per_station_results.csv")
OUT = Path("workspace/manuscripts/poroelastic_gnss-mlcw/submission/figures")
C_GNSS, C_MLCW = "#0072B2", "#D55E00"   # Okabe-Ito blue / vermillion


def _decade_ticks(ax, axis: str, ticks) -> None:
    """Label a log axis that spans less than a decade. Matplotlib's default only
    labels powers of ten, which on these ranges leaves a single labelled tick."""
    a = ax.xaxis if axis == "x" else ax.yaxis
    a.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
    a.set_minor_locator(matplotlib.ticker.NullLocator())
    a.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda t, _: f"{t:g}" if t >= 0.01 else f"{t:.3f}".rstrip("0")))


def load():
    v = pd.read_csv(ESM)
    v = v[v.elastic_valid].copy()
    assert len(v) > 0, "no elastic-valid stations in the ESM"
    return v


def fig4(v: pd.DataFrame) -> None:
    x, y, s = (v.head_seasonal_amp_m.to_numpy(), v.disp_seasonal_amp_m.to_numpy() * 100,
               v.S_ke.to_numpy())
    g = (v.network == "GNSS").to_numpy()
    beta, ic, _, _, se = stats.linregress(np.log(x), np.log(y))
    rho, p_rho = stats.spearmanr(x, s)

    fig, axes = fs.new_figure(width="double", height=3.2, ncols=2)
    fig.subplots_adjust(wspace=0.30)
    ax = axes[0]
    xx = np.logspace(np.log10(x.min() * 0.75), np.log10(x.max() * 1.35), 100)
    ax.plot(xx, np.exp(ic) * xx**beta, "-", color="0.25", lw=1.4, zorder=2,
            label=fr"fitted $\beta$ = {beta:.2f} $\pm$ {se:.2f}")
    # beta = 1 reference, anchored on the median S_ke so it passes through the cloud
    ax.plot(xx, np.median(s) * 100 * xx, "--", color="0.25", lw=1.2, zorder=2,
            label=r"$\beta$ = 1 (constant $S_{ke}$)")
    for m, lab, c in ((g, "GNSS", C_GNSS), (~g, "MLCW", C_MLCW)):
        ax.plot(x[m], y[m], "o" if lab == "GNSS" else "s", ms=6, mfc=c, mec="k",
                mew=0.6, ls="none", zorder=3, label=lab)
    ax.set(xscale="log", yscale="log",
           xlabel="Seasonal head amplitude (m)",
           ylabel="Seasonal displacement amplitude (cm)")
    _decade_ticks(ax, "x", [0.5, 1, 2, 5])
    _decade_ticks(ax, "y", [0.25, 0.5, 1, 2])
    ax.legend(loc="upper left", fontsize=7.4)
    fs.panel_label(ax, "(a)")

    ax = axes[1]
    ax.axhline(np.median(s), ls=":", color="0.35", lw=1.2, zorder=1)
    for m, lab, c in ((g, "GNSS", C_GNSS), (~g, "MLCW", C_MLCW)):
        ax.errorbar(x[m], s[m],
                    yerr=[s[m] - v["S_ke_ci2.5"].to_numpy()[m],
                          v["S_ke_ci97.5"].to_numpy()[m] - s[m]],
                    fmt="o" if lab == "GNSS" else "s", ms=6, mfc=c, mec="k",
                    mew=0.6, ecolor="0.55", elinewidth=0.9, capsize=2, zorder=3,
                    label=lab)
    ax.set(xscale="log", yscale="log", xlabel="Seasonal head amplitude (m)",
           ylabel=r"Elastic skeletal storativity $S_{ke}$  ($-$)")
    _decade_ticks(ax, "x", [0.5, 1, 2, 5])
    _decade_ticks(ax, "y", [0.002, 0.005, 0.01])
    # The GNSS/MLCW key is in panel (a); repeating it here would only crowd the
    # long Ciaoyi interval at lower left.
    ax.annotate(f"median {np.median(s):.4f}", xy=(0.985, np.median(s)),
                xycoords=("axes fraction", "data"), ha="right", va="bottom",
                fontsize=7.4, color="0.35")
    fs.metric_box(ax, f"Spearman $\\rho$ = {rho:.2f}\np = {p_rho:.1e}",
                  loc="upper right")
    fs.panel_label(ax, "(b)")

    fs.save(fig, str(OUT / "figure4"), formats=("tiff",))
    print(f"Fig. 4  beta = {beta:.3f} +/- {se:.3f}   rho = {rho:.3f} (p = {p_rho:.2g})")


def fig5(v: pd.DataFrame) -> None:
    # Order by head amplitude so co-located pairs (which share a well) sit adjacent
    # and the S_ke gradient of Fig. 4b reads down the axis.
    v = v.sort_values(["head_seasonal_amp_m", "network"],
                      ascending=[False, True]).reset_index(drop=True)
    shared = v.paired_well_id.duplicated(keep=False)
    ypos = np.arange(len(v))[::-1]

    fig, ax = fs.new_figure(width="onehalf", height=4.6)
    # Shade each co-located pair as a band, which reads as "these two rows are the
    # same site" far better than a connector drawn off at the axis edge.
    for well, d in v[shared].groupby("paired_well_id"):
        rows = [ypos[i] for i in d.index]
        # 0.42 rather than 0.5 leaves a hairline gap, so consecutive pairs read
        # as separate bands instead of merging into one block.
        ax.axhspan(min(rows) - 0.42, max(rows) + 0.42, color="0.90", zorder=0)
    for i, (_, r) in enumerate(v.iterrows()):
        c = C_GNSS if r.network == "GNSS" else C_MLCW
        ax.plot([r["S_ke_ci2.5"], r["S_ke_ci97.5"]], [ypos[i]] * 2,
                "-", color="0.45", lw=1.1, zorder=2)
        ax.plot(r.S_ke, ypos[i], "o" if r.network == "GNSS" else "s",
                ms=6.5, mfc=c, mec="k", mew=0.6, zorder=3)

    med = v.groupby("network").S_ke.median()
    ax.axvline(med["GNSS"], ls="--", color=C_GNSS, lw=1.1, zorder=1)
    ax.axvline(med["MLCW"], ls="--", color=C_MLCW, lw=1.1, zorder=1)
    ax.set_yticks(ypos, v.station_label)
    # Headroom above the top row so the legend sits clear of every interval.
    ax.set(xscale="log", ylim=(-0.8, len(v) + 1.4),
           xlabel=r"Elastic skeletal storativity $S_{ke}$  ($-$)")
    _decade_ticks(ax, "x", [0.002, 0.005, 0.01, 0.02])
    ax.tick_params(axis="y", length=0)
    handles = [
        ax.plot([], [], "o", ms=6.5, mfc=C_GNSS, mec="k", mew=0.6, ls="none",
                label="GNSS")[0],
        ax.plot([], [], "s", ms=6.5, mfc=C_MLCW, mec="k", mew=0.6, ls="none",
                label="MLCW")[0],
        ax.axhspan(np.nan, np.nan, color="0.90", label="co-located pair"),
        ax.plot([], [], "--", color="0.45", lw=1.1, label="network median")[0],
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=7.4, ncol=2)
    fs.save(fig, str(OUT / "figure5"), formats=("tiff",))
    print(f"Fig. 5  {len(v)} stations, {shared.sum() // 2} co-located pairs, "
          f"S_ke {v.S_ke.min():.4f}-{v.S_ke.max():.4f}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fs.use_style()
    v = load()
    fig4(v)
    fig5(v)
