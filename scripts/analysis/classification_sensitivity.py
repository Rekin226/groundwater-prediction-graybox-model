"""Sensitivity of the coastal/inland classification to its two thresholds.

Recomputes, from calibration-period hourly heads, each station's spectrum
(high-pass filtered as in 02_pairing) and reports:

  dist_km        distance to the coastline
  m2_rel_top10   operational rule: M2 amplitude / largest peak, using only the
                 10 largest spectral bins (NaN when M2 is not among them)
  m2_rel_band    M2 amplitude (max in 1.93 +/- 0.05 cpd) / largest amplitude
                 in the full spectrum
  m2_snr         M2 amplitude / median amplitude of the surrounding band
                 (1.6-2.4 cpd, excluding the M2 and S2 windows)

and the coastal set for distance thresholds {3, 5, 7, 10} km crossed with
relative-amplitude thresholds {5, 10, 15, 20}%, under both the operational
top-10 rule and the full-spectrum ratio.

Usage:
    python scripts/analysis/classification_sensitivity.py [--run ...]
"""
import argparse
import importlib.util

import numpy as np
import pandas as pd
from numpy.fft import fft, fftfreq

from common import REPO, analysis_dir, gw_shell, load_results, run_dir, station_sort_key

spec = importlib.util.spec_from_file_location("pairing", REPO / "srcs" / "02_pairing.py")
pairing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pairing)

DIST_KM = (3.0, 5.0, 7.0, 10.0)
REL = (0.05, 0.10, 0.15, 0.20)
M2, TOL = pairing.M2_TARGET_CPD, pairing.M2_TOL_CPD


def spectrum(h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    filt = pairing.high_pass_filter(h, pairing.CUTOFF_CPD, pairing.FS_CPD, pairing.FILTER_ORDER)
    n = len(filt)
    amp = 2.0 / n * np.abs(fft(filt)[: n // 2])
    f = fftfreq(n, 1.0)[: n // 2] * 24.0
    return f, amp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    args = ap.parse_args()
    run = run_dir(args.run)
    res = load_results(run)
    cls = pd.read_csv(REPO / "data" / "intermediate" / "gw_coastal_inland_class.csv").set_index("st_id")

    gw = pd.read_csv(gw_shell.GW_DATA_PATH, parse_dates=["date time"]).set_index("date time")
    gw = gw.loc[gw.index <= pd.Timestamp(gw_shell.CAL_END) + pd.Timedelta(hours=23)]

    rows = []
    for st in sorted(res.index, key=station_sort_key):
        h = gw[st].astype(float).interpolate(limit_direction="both").to_numpy()
        f, amp = spectrum(h)
        valid = f > pairing.CUTOFF_CPD
        m2_win = np.abs(f - M2) <= TOL
        s2_win = np.abs(f - 2.0) <= TOL
        bg = (f >= 1.6) & (f <= 2.4) & ~m2_win & ~s2_win
        m2_amp = amp[m2_win].max()
        top10 = pairing.identify_top_dominant_frequencies(pairing.high_pass_filter(
            h, pairing.CUTOFF_CPD, pairing.FS_CPD, pairing.FILTER_ORDER), top_n=10)
        m2_freq, m2_amp_top = pairing.find_m2_peak(top10)
        rows.append({
            "st_id": st, "group": res.loc[st, "group_name"],
            "dist_km": cls.loc[st, "dist_to_coast_m"] / 1000.0,
            "m2_amp_m": m2_amp,
            "m2_rel_top10": (m2_amp_top / top10["Amplitude"].max()) if m2_amp_top is not None else np.nan,
            "m2_rel_band": m2_amp / amp[valid].max(),
            "m2_snr": m2_amp / np.median(amp[bg]),
        })
    df = pd.DataFrame(rows).set_index("st_id")
    out = analysis_dir(run)
    df.to_csv(out / "classification_m2.csv")

    grid = []
    for rule, col in (("top10", "m2_rel_top10"), ("band", "m2_rel_band")):
        for d in DIST_KM:
            for r in REL:
                coastal = sorted(df.index[(df.dist_km <= d) & (df[col].fillna(0) >= r)], key=station_sort_key)
                grid.append({"rule": rule, "dist_km": d, "rel": r, "n_coastal": len(coastal),
                             "coastal": ",".join(coastal)})
    g = pd.DataFrame(grid)
    g.to_csv(out / "classification_grid.csv", index=False)

    ref = set(res.index[res.group_name == "coastal"])
    print("Near-coast stations (<= 10 km):")
    print(df[df.dist_km <= 10].sort_values("dist_km").round(3).to_string())
    print(f"\nReference coastal set ({len(ref)}): {sorted(ref, key=station_sort_key)}")
    for rule in ("top10", "band"):
        print(f"\nCoastal count, rule={rule} (rows: distance km, cols: relative M2 threshold)")
        print(g[g.rule == rule].pivot(index="dist_km", columns="rel", values="n_coastal").to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
