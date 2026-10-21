"""Stability of variant selection under a vertical-datum shift.

The fits minimise RMSE, which is unaffected by adding a constant to the heads,
so a datum shift D moves observed and simulated heads together. Variant
selection is recomputed from the saved selection-period series for several D,
once with classic KGE (bias ratio mu_sim/mu_obs) and once with kge_ab (additive
bias), and compared with the selection at D = 0.

Usage:
    python scripts/analysis/datum_shift_selection.py [--run ...]
"""
import argparse

import pandas as pd

from common import analysis_dir, gw_shell, load_results, load_series, run_dir, station_sort_key

SHIFTS = (-100.0, -50.0, -20.0, 0.0, 20.0, 50.0, 100.0)
VARIANTS = ("base", "filtered", "base_tz", "filtered_tz")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    args = ap.parse_args()
    run = run_dir(args.run)
    res = load_results(run)

    rows = []
    for st in sorted(res.index, key=station_sort_key):
        s = load_series(run, st)
        sel = s[s["period"] == "sel"]
        obs = sel["obs"].to_numpy()
        for d in SHIFTS:
            scores = {}
            for v in VARIANTS:
                col = f"sim_{v}"
                if col not in sel:
                    continue
                m = gw_shell.compute_metrics(obs + d, sel[col].to_numpy() + d)
                scores[v] = (m["kge"], m["kge_ab"])
            pick_classic = max(scores, key=lambda v: scores[v][0])
            pick_ab = max(scores, key=lambda v: scores[v][1])
            rows.append({"st_id": st, "shift_m": d, "pick_classic_kge": pick_classic, "pick_kge_ab": pick_ab})
    df = pd.DataFrame(rows)
    path = analysis_dir(run) / "datum_shift_selection.csv"
    df.to_csv(path, index=False)

    ref = df[df.shift_m == 0].set_index("st_id")
    print("Stations whose selected variant differs from the D = 0 choice:")
    for d in SHIFTS:
        cur = df[df.shift_m == d].set_index("st_id")
        n_c = int((cur.pick_classic_kge != ref.pick_classic_kge).sum())
        n_a = int((cur.pick_kge_ab != ref.pick_kge_ab).sum())
        print(f"  D = {d:+6.0f} m: classic KGE {n_c:2d}/{len(cur)}, kge_ab {n_a:2d}/{len(cur)}")
    n_diff = int((ref.pick_classic_kge != ref.pick_kge_ab).sum())
    print(f"At D = 0, classic KGE and kge_ab pick different variants at {n_diff}/{len(ref)} stations")
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
