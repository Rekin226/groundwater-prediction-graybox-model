"""A/B comparison: KGE_val before/after cleaning per spec §6.2.

Reads:
    workspace/results_sub/v10_anchor_fix/sub_fit_results.csv  (raw baseline)
    workspace/results_sub/<post_clean_run>/sub_fit_results.csv (clean A/B)
Writes:
    workspace/results_sub/clean_ab_comparison.csv
    workspace/results_sub/clean_ab_comparison.txt (human-readable)
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

POOR_PROBE = ["KTES", "ANES", "YSLL", "SNES", "CHSG", "HNES", "SJES", "STES",
              "SLES", "HLES", "GFES", "XPES", "FRES"]
CLEAN_PROBE = ["LYES", "YWJS", "NGES", "JJES", "YCES"]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", default="workspace/results_sub/v10_anchor_fix/sub_fit_results.csv")
    p.add_argument("--post-clean", required=True,
                   help="Path to sub_fit_results.csv from a run on cleaned data")
    args = p.parse_args(argv)

    a = pd.read_csv(args.baseline)
    b = pd.read_csv(args.post_clean)
    a = a[a["is_best"] == True][["sub_id", "kge_val"]].rename(columns={"kge_val": "kge_val_raw"})
    b = b[b["is_best"] == True][["sub_id", "kge_val"]].rename(columns={"kge_val": "kge_val_clean"})
    m = a.merge(b, on="sub_id", how="outer")
    m["delta_kge_val"] = m["kge_val_clean"] - m["kge_val_raw"]

    m_sorted = m.sort_values("delta_kge_val", ascending=False)
    out_csv = Path("workspace/results_sub/clean_ab_comparison.csv")
    m_sorted.to_csv(out_csv, index=False)

    poor = m_sorted[m_sorted["sub_id"].isin(POOR_PROBE)]
    clean = m_sorted[m_sorted["sub_id"].isin(CLEAN_PROBE)]
    lines = []
    lines.append(f"Median \u0394KGE_val on Poor probe ({len(poor)}/{len(POOR_PROBE)}): "
                 f"{poor['delta_kge_val'].median():+.3f}  (gate: \u2265 +0.30)")
    lines.append(f"Max  |\u0394KGE_val| on Clean probe ({len(clean)}/{len(CLEAN_PROBE)}): "
                 f"{clean['delta_kge_val'].abs().max():+.3f}  (gate: < 0.10)")
    lines.append("")
    lines.append("Top 10 improvements:")
    lines.append(m_sorted.head(10).to_string(index=False))
    lines.append("")
    lines.append("Top 10 regressions:")
    lines.append(m_sorted.tail(10).to_string(index=False))
    out_txt = Path("workspace/results_sub/clean_ab_comparison.txt")
    out_txt.write_text("\n".join(lines))

    print("\n".join(lines))
    print(f"\nWrote {out_csv} and {out_txt}")


if __name__ == "__main__":
    sys.exit(main())
