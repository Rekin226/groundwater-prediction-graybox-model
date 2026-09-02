"""Manuscript-facing products of the poroelastic run: the ESM 1 per-station table
and the summary statistics quoted in the text.

Covers the three findings raised in the 2026-08-17 audit:
  C1  response lag, recomputed on a shared phase epoch (fixed in poroelastic.py)
  C2  the amplitude scaling exponent beta and the S_ke <-> head-amplitude relation,
      which test whether a single network storativity is supportable
  C3  the GNSS vs MLCW comparison, PAIRED at co-located sites rather than as a
      difference of two network medians over different station sets

Run:
    poetry run python subsidence/scripts/poro_manuscript_stats.py --run-id poroelastic_v2
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Romanised MLCW well names. MLCW stations are keyed by their Chinese site name, so
# they need a Latin label for the figures and tables; GNSS stations already carry an
# unambiguous four-letter code and keep it. The five elastic-valid wells follow the
# spellings already used in the manuscript's Table 1; the remaining six use the same
# Tongyong convention and should be confirmed against the WRA station register.
SITE_EN = {
    "拯民國小": "Jhengmin", "秀潭國小": "Sioutan", "僑義國小": "Ciaoyi",
    "元長國小": "Yuanchang", "宏崙國小": "Honglun", "客厝國小": "Kecuo",
    "光復國小": "Guangfu", "湖南國小": "Hunan", "土庫國中": "Tuku",
    "北辰國小": "Beichen", "嘉興國小": "Jiasing",
}
ESM_COLS = [
    "station", "station_label", "site_name_zh", "network", "paired_well_id",
    "paired_well_distance_km", "deepest_ring_depth_m", "n_obs",
    "disp_seasonal_amp_m", "disp_amp_ci2.5_m", "disp_amp_ci97.5_m",
    "disp_seasonal_R2", "disp_trend_m_per_yr_at_record_start",
    "head_seasonal_amp_m", "head_amp_ci2.5_m", "head_amp_ci97.5_m",
    "coupling_r", "n_common_dates", "phase_lag_days",
    "S_ke", "S_ke_ci2.5", "S_ke_ci97.5",
    "Ss_per_m", "Ss_ci2.5_per_m", "Ss_ci97.5_per_m", "elastic_valid",
]


def build_esm(run_id: str) -> pd.DataFrame:
    src = Path("workspace/results_sub") / run_id / "poroelastic_results.csv"
    r = pd.read_csv(src)
    pair = pd.read_csv("subsidence/data/sub_pairing.csv")
    master = pd.read_csv("subsidence/data/sub_station_master.csv")

    zh = master.set_index(["sub_id", "sub_dataset"])["ObservatoryName"]
    key = list(zip(r["sub_id"], r["sub_dataset"]))
    # MLCW stations are keyed by their Chinese site name; GNSS carry it in ObservatoryName.
    r["site_name_zh"] = [zh.get(k, np.nan) if not pd.isna(zh.get(k, np.nan)) else k[0]
                         for k in key]
    # One Latin label per station for figures and tables: the four-letter code for
    # GNSS, the romanised site name for the Chinese-keyed MLCW wells.
    r["station_label"] = np.where(r["network"] == "GNSS", r["sub_id"],
                                  r["site_name_zh"].map(SITE_EN))
    missing = r.loc[r["station_label"].isna(), "site_name_zh"].tolist()
    if missing:
        raise KeyError(f"no romanisation for MLCW well(s): {missing}; add them to SITE_EN")
    gw = pair.set_index(["sub_id", "sub_dataset"])["gw_st"]
    r["paired_well_id"] = [gw.get(k, np.nan) for k in key]

    out = r.rename(columns={
        "sub_id": "station", "dist_km": "paired_well_distance_km",
        "thickness_m": "deepest_ring_depth_m", "disp_seas_amp_m": "disp_seasonal_amp_m",
        "disp_amp_lo": "disp_amp_ci2.5_m", "disp_amp_hi": "disp_amp_ci97.5_m",
        "disp_seas_r2": "disp_seasonal_R2",
        "disp_trend_m_yr": "disp_trend_m_per_yr_at_record_start",
        "head_seas_amp_m": "head_seasonal_amp_m",
        "head_amp_lo": "head_amp_ci2.5_m", "head_amp_hi": "head_amp_ci97.5_m",
        "coupling_n": "n_common_dates", "s_ke": "S_ke",
        "s_ke_lo": "S_ke_ci2.5", "s_ke_hi": "S_ke_ci97.5",
        "ss_per_m": "Ss_per_m", "ss_lo": "Ss_ci2.5_per_m", "ss_hi": "Ss_ci97.5_per_m",
    })
    for c in ESM_COLS:
        if c not in out.columns:
            out[c] = np.nan
    return out[ESM_COLS].sort_values(["network", "station"]).reset_index(drop=True)


def report(esm: pd.DataFrame) -> None:
    v = esm[esm["elastic_valid"]].copy()
    g = v[v.network == "GNSS"]
    m = v[v.network == "MLCW"]
    say = print

    say("=" * 72)
    say("C1 — RESPONSE LAG on a shared phase epoch")
    say("=" * 72)
    for net, sub in (("GNSS", g), ("MLCW", m)):
        lags = sub["phase_lag_days"].to_numpy()
        # median uncertainty via the bootstrap of the median across stations
        rng = np.random.default_rng(11)
        med_bs = np.array([np.median(rng.choice(lags, len(lags), replace=True))
                           for _ in range(5000)])
        lo, hi = np.percentile(med_bs, [2.5, 97.5])
        say(f"  {net}: median {np.median(lags):+.1f} d  95% CI on the median "
            f"[{lo:+.1f}, {hi:+.1f}]  range {lags.min():+.1f} to {lags.max():+.1f}  n={len(lags)}")

    say("\n" + "=" * 72)
    say("C2 — IS A SINGLE NETWORK STORATIVITY SUPPORTABLE?")
    say("=" * 72)
    x = v.head_seasonal_amp_m.to_numpy()
    y = v.disp_seasonal_amp_m.to_numpy()
    s = v.S_ke.to_numpy()
    b, ic, rr, pp, se = stats.linregress(np.log(x), np.log(y))
    say(f"  D_def ~ D_h^beta :  beta = {b:.3f} +/- {se:.3f}"
        f"   (beta=1 <=> constant S_ke;  t = {(b-1)/se:+.2f},"
        f" p = {2*stats.t.sf(abs(b-1)/se, len(x)-2):.4f})")
    say(f"  S_ke vs head amplitude : Pearson r = {stats.pearsonr(x, s)[0]:+.3f} "
        f"(p = {stats.pearsonr(x, s)[1]:.4f}),  Spearman rho = {stats.spearmanr(x, s)[0]:+.3f} "
        f"(p = {stats.spearmanr(x, s)[1]:.5f})")
    say(f"  S_ke spread : {s.min():.4f} to {s.max():.4f}  ({s.max()/s.min():.1f}x), "
        f"median {np.median(s):.4f}, IQR {np.percentile(s,25):.4f}-{np.percentile(s,75):.4f}")
    say(f"  head amplitude spans {x.max()/x.min():.1f}x ({x.min():.2f}-{x.max():.2f} m); "
        f"displacement amplitude spans {y.max()/y.min():.1f}x ({y.min()*100:.2f}-{y.max()*100:.2f} cm)")
    say(f"  amplitude scaling  : r = {stats.pearsonr(x, y)[0]:.3f}, "
        f"p = {stats.pearsonr(x, y)[1]:.4f}, n = {len(x)}"
        f"  | Spearman {stats.spearmanr(x, y)[0]:.3f}, p = {stats.spearmanr(x, y)[1]:.4f}")
    o = np.argsort(x)
    for k in (2, 3):
        i = o[:-k]
        say(f"  leverage: drop the {k} highest-head stations -> r = {stats.pearsonr(x[i], y[i])[0]:.3f}, "
            f"p = {stats.pearsonr(x[i], y[i])[1]:.4f}, n = {len(i)}")
    a = esm[esm.n_common_dates > 0]
    ax, ay = a.head_seasonal_amp_m.to_numpy(), a.disp_seasonal_amp_m.to_numpy()
    say(f"  ungated (all {len(a)} stations with temporal overlap): "
        f"r = {stats.pearsonr(ax, ay)[0]:.3f}, p = {stats.pearsonr(ax, ay)[1]:.5f}")
    j = np.argsort(ax)[:-2]
    say(f"  ungated minus the 2 highest-head: r = {stats.pearsonr(ax[j], ay[j])[0]:.3f}, "
        f"p = {stats.pearsonr(ax[j], ay[j])[1]:.5f}, n = {len(j)}")

    say("\n" + "=" * 72)
    say("C3 — GNSS vs MLCW, PAIRED AT CO-LOCATED SITES")
    say("=" * 72)
    both = v[v.paired_well_id.notna()].groupby("paired_well_id").filter(
        lambda d: d.network.nunique() == 2)
    rows = []
    for well, d in both.groupby("paired_well_id"):
        a1 = d[d.network == "GNSS"].iloc[0]
        b1 = d[d.network == "MLCW"].iloc[0]
        rows.append(dict(site=b1.station_label, gnss=a1.station,
                         ske_g=a1.S_ke, ske_m=b1.S_ke, ratio=a1.S_ke / b1.S_ke,
                         lag_g=a1.phase_lag_days, lag_m=b1.phase_lag_days))
    p = pd.DataFrame(rows)
    say(p.round(5).to_string(index=False))
    say(f"\n  paired S_ke ratio GNSS/MLCW : median {p.ratio.median():.3f} "
        f"(range {p.ratio.min():.3f}-{p.ratio.max():.3f}); GNSS higher at "
        f"{(p.ratio > 1).sum()} of {len(p)} sites")
    say(f"  -> paired bias {100*(p.ratio.median()-1):+.1f}%, versus "
        f"{100*(g.S_ke.median()/m.S_ke.median()-1):+.1f}% from the unpaired network medians")
    say(f"  paired lags: GNSS {p.lag_g.round(1).tolist()} d vs MLCW {p.lag_m.round(1).tolist()} d")

    say("\n" + "=" * 72)
    say("NETWORK SUMMARY (Table 2)")
    say("=" * 72)
    for net, sub, tot in (("GNSS", g, (esm.network == "GNSS").sum()),
                         ("MLCW", m, (esm.network == "MLCW").sum())):
        hw = ((sub["S_ke_ci97.5"] - sub["S_ke_ci2.5"]) / 2 / sub.S_ke * 100).median()
        say(f"  {net}: valid {len(sub)}/{tot} | S_ke median {sub.S_ke.median():.4f} "
            f"| r median {sub.coupling_r.median():.2f} "
            f"| lag median {sub.phase_lag_days.median():+.0f} d "
            f"| median relative CI half-width {hw:.0f}%")
    say(f"  MLCW Ss: {m.Ss_per_m.min():.2e} to {m.Ss_per_m.max():.2e} /m, "
        f"median {m.Ss_per_m.median():.2e} /m")
    alpha = m.Ss_per_m / (1000.0 * 9.80665)
    say(f"  MLCW skeletal compressibility: {alpha.min():.2e} to {alpha.max():.2e} /Pa")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="poroelastic_v4")
    ap.add_argument("--esm-out", default="workspace/manuscripts/poroelastic_gnss-mlcw/submission/"
                                         "ESM1_poroelastic_per_station_results.csv")
    args = ap.parse_args()
    esm = build_esm(args.run_id)
    Path(args.esm_out).parent.mkdir(parents=True, exist_ok=True)
    esm.to_csv(args.esm_out, index=False)
    print(f"ESM 1 written: {args.esm_out}  ({len(esm)} stations)\n")
    report(esm)


if __name__ == "__main__":
    main()
