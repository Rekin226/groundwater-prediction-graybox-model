"""Window-matched sensitivity for the poroelastic storage inversion.

The production run (08_poroelastic_storage.py) fits the head harmonic over each
well's full record (typically 2010-2025), while GNSS displacement spans 2020-2025
and MLCW spans 2010-2025. If the seasonal head amplitude changed over the decade,
that mismatch propagates into S_ke for the GNSS network and into the GNSS-vs-MLCW
comparison. This script quantifies the effect two ways:

  A. GNSS: refit each paired head harmonic over the station's displacement window
     (head truncated to [disp.min, disp.max]) and recompute S_ke.
  B. MLCW: truncate BOTH displacement and head to 2020-01-01 onward (the GNSS era)
     and recompute S_ke.

Run:
    poetry run python subsidence/scripts/poro_window_sensitivity.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from subsidence.poroelastic import decompose, couple, storativity, is_elastic_valid

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "poro_runner", Path(__file__).resolve().parent.parent / "08_poroelastic_storage.py")
_runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_runner)

GNSS, MLCW = _runner.GNSS_DATASET, _runner.MLCW_DATASET


def main() -> None:
    base = pd.read_csv("workspace/results_sub/poroelastic_v1/poroelastic_results.csv")
    pair = pd.read_csv("subsidence/data/sub_pairing.csv")

    rows = []
    for _, st in base.iterrows():
        sid, dataset = st["sub_id"], st["sub_dataset"]
        disp, _ = _runner.load_displacement(dataset, sid)
        pr = pair[(pair["sub_id"] == sid) & (pair["sub_dataset"] == dataset)]
        if pr.empty:
            pr = pair[pair["sub_id"] == sid]
        head = _runner.load_head(pr.iloc[0]["gw_st"]) if not pr.empty else None
        if disp is None or head is None or len(head) < 24:
            continue

        if dataset == MLCW:  # variant B: both series restricted to the GNSS era
            disp = disp[disp.index >= "2020-01-01"]
        if len(disp) < 24:
            continue
        head_w = head[(head.index >= disp.index.min()) & (head.index <= disp.index.max())]
        if len(head_w) < 24:
            continue

        sd, hd = decompose(disp), decompose(head_w)
        r, n = couple(sd.residual, hd.residual, resample=_runner.NET_RESAMPLE[dataset])
        rows.append({
            "sub_id": sid, "network": _runner.NET_LABEL[dataset],
            "head_win": f"{head_w.index.min().date()}..{head_w.index.max().date()}",
            "s_ke_base": st["s_ke"], "s_ke_matched": storativity(sd.amplitude, hd.amplitude),
            "r_matched": r, "n_common": n,
            "valid_base": bool(st["elastic_valid"]),
            "valid_matched": is_elastic_valid(r, sd.seasonal_r2),
        })

    out = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(out.to_string())
    print()
    for net in ("GNSS", "MLCW"):
        g = out[out["network"] == net]
        vb = g[g["valid_base"]]
        vm = g[g["valid_matched"]]
        print(f"{net}: baseline-valid set n={len(vb)}  "
              f"S_ke median base={vb['s_ke_base'].median():.4f}  "
              f"matched-window={vb['s_ke_matched'].median():.4f}  "
              f"({100*(vb['s_ke_matched'].median()/vb['s_ke_base'].median()-1):+.1f}%)")
        print(f"{net}: matched-window gate keeps n={len(vm)}  "
              f"S_ke median={vm['s_ke_matched'].median():.4f}")


if __name__ == "__main__":
    main()
