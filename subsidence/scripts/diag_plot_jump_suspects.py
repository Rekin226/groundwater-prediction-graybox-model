"""Plot raw ζ_obs and daily Δζ for jump-suspect stations.

Visual sanity check for the parameter-based signature scan: if the flagged
stations actually show jumps in the raw series, the signature picks up real
artefacts.  If not, the signature is noise.

Run:
    poetry run python subsidence/scripts/diag_plot_jump_suspects.py
"""
from __future__ import annotations
import sys
import urllib.parse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

try:
    from rklib import setup_font
    setup_font()
except Exception:
    plt.rcParams["font.family"] = "Times New Roman"

CAL_START = pd.Timestamp("2020-01-01")
CAL_END   = pd.Timestamp("2022-12-31")
VAL_START = pd.Timestamp("2024-01-01")
VAL_END   = pd.Timestamp("2025-03-31")

# 3 suspects + 1 control (clean baseline) for visual comparison
SUSPECTS = [
    ("STES", "ls-wra-gnss-obs"),
    ("JJES", "ls-wra-gnss-obs"),
    ("NGES", "ls-wra-gnss-obs"),
    ("YWJS", "ls-wra-gnss-obs"),  # control — clean
]


def _load_zeta(sub_id, sub_dataset):
    sid_encoded = urllib.parse.quote(str(sub_id), safe="")
    raw = pd.read_parquet(f"data/ls_cache/{sub_dataset}__{sid_encoded}.parquet")
    s = raw["value"].dropna() if "value" in raw.columns else raw.dropna()
    zeta = s.iloc[0] - s
    return zeta * 100.0  # convert m → cm


def _detect_jumps(dz_cm, threshold_cm=1.0):
    """Mark days where |Δζ| exceeds threshold_cm."""
    return dz_cm.index[np.abs(dz_cm.values) >= threshold_cm]


def main():
    out_dir = Path("workspace/results_sub/v10_anchor_fix/figures/jump_diag")
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(len(SUSPECTS), 2, figsize=(12, 2.5 * len(SUSPECTS)),
                             sharex=False)
    for row, (sub_id, ds) in enumerate(SUSPECTS):
        zeta = _load_zeta(sub_id, ds)
        dz = zeta.diff()  # cm/day
        is_clean = sub_id == "YWJS"

        # Left: raw ζ
        ax = axes[row, 0]
        ax.plot(zeta.index, zeta.values, lw=0.6,
                color="tab:blue" if is_clean else "tab:red")
        ax.axvspan(CAL_START, CAL_END, alpha=0.08, color="gray")
        ax.axvspan(VAL_START, VAL_END, alpha=0.12, color="orange")
        ax.set_ylabel(f"ζ (cm)\n{sub_id}", fontsize=10)
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        title_tag = "control (clean)" if is_clean else "suspect"
        ax.set_title(f"{sub_id} — raw ζ_obs ({title_tag})", fontsize=10)

        # Right: daily Δζ with threshold lines
        ax = axes[row, 1]
        ax.plot(dz.index, dz.values, lw=0.5,
                color="tab:blue" if is_clean else "tab:red")
        ax.axhline(1.0, color="k", lw=0.5, ls="--", alpha=0.6)
        ax.axhline(-1.0, color="k", lw=0.5, ls="--", alpha=0.6)
        ax.axhline(3.0, color="purple", lw=0.5, ls=":", alpha=0.6)
        ax.axhline(-3.0, color="purple", lw=0.5, ls=":", alpha=0.6)

        # Mark large jumps
        big_jumps = _detect_jumps(dz, threshold_cm=1.0)
        if len(big_jumps) > 0:
            ax.scatter(big_jumps, dz.loc[big_jumps].values, s=12,
                       color="black", zorder=3, label=f"|Δζ|≥1cm  (n={len(big_jumps)})")
            ax.legend(fontsize=8, loc="upper right")
        ax.set_ylabel("Δζ (cm/day)", fontsize=10)
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.set_title(f"{sub_id} — daily Δζ (dashed=±1cm, dotted=±3cm)", fontsize=10)

        # Print summary to stdout
        n1 = int((np.abs(dz.values) >= 1.0).sum())
        n3 = int((np.abs(dz.values) >= 3.0).sum())
        n5 = int((np.abs(dz.values) >= 5.0).sum())
        max_jump = float(np.nanmax(np.abs(dz.values)))
        print(f"{sub_id:6s}  n_obs={len(zeta):5d}  "
              f"|Δζ|≥1cm: {n1:4d}   ≥3cm: {n3:3d}   ≥5cm: {n5:3d}   "
              f"max|Δζ|={max_jump:6.2f} cm")

    plt.tight_layout()
    out_path = out_dir / "jump_suspects_overview.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
