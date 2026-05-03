"""Scan v10 results for the injected-jump fit-signature.

Signature (derived empirically from synthetic 3 cm jump injection on YWJS):
    - Sk_v pinned at lower bound (≈ 1e-6 → log10 ≤ −5)
    - tau ≥ 200 d (only meaningful for tau-eligible variants M3_tau / M4_tau)
    - |h_ref − mean(h)| > 3 m   (optimizer pushes h_ref to absorb step offset)

Stations matching ≥ 2 of 3 are flagged as "jump-suspect".  This is a screening
heuristic, not a diagnosis — but it lets us estimate how widespread silent
artefacts are before designing the cleaning module.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

RESULTS = Path("workspace/results_sub/v10_anchor_fix/sub_fit_results.csv")
H_DIR = Path("subsidence/data/h_drivers")


def main():
    df = pd.read_csv(RESULTS)
    df = df[df["is_best"] == True].copy()

    rows = []
    for _, r in df.iterrows():
        sub_id = r["sub_id"]
        h_path = H_DIR / f"{sub_id}.parquet"
        if not h_path.exists():
            continue
        h = pd.read_parquet(h_path)["h_driver"].values
        h = h[np.isfinite(h)]
        h_mean = float(np.nanmean(h)) if h.size else np.nan
        h_min = float(np.nanmin(h)) if h.size else np.nan
        h_max = float(np.nanmax(h)) if h.size else np.nan

        sk_v = r["Sk_v"]
        tau = r.get("tau", np.nan)
        h_ref = r["h_ref"]
        variant = r["variant"]

        # Signature components
        sig_skv = bool(np.log10(max(sk_v, 1e-12)) <= -5.0) if pd.notna(sk_v) else False
        sig_tau = bool(pd.notna(tau) and tau >= 200) if variant.endswith("_tau") else None
        sig_href = bool(abs(h_ref - h_mean) > 3.0) if pd.notna(h_ref) else False

        n_match = sum([sig_skv, sig_tau is True, sig_href])
        applicable = 3 if sig_tau is not None else 2

        rows.append({
            "sub_id": sub_id,
            "ds": r["sub_dataset"].replace("ls-wra-", "").replace("-obs", ""),
            "variant": variant,
            "kge_val": r["kge_val"],
            "Sk_v": sk_v,
            "tau": tau if pd.notna(tau) else "—",
            "h_ref": h_ref,
            "h_mean": h_mean,
            "|h_ref-h_mean|": abs(h_ref - h_mean),
            "sig_Skv_floor": "Y" if sig_skv else " ",
            "sig_tau_high": "Y" if sig_tau is True else (" " if sig_tau is False else "n/a"),
            "sig_href_shift": "Y" if sig_href else " ",
            "n_match": f"{n_match}/{applicable}",
            "suspect": n_match >= 2,
        })

    out = pd.DataFrame(rows).sort_values("kge_val", ascending=True)
    print(f"\nScanned {len(out)} stations\n")
    print(f"{'sub_id':<14} {'ds':<5} {'var':<7} {'kge_v':>6} "
          f"{'Sk_v':>10} {'tau':>7} {'|Δh_ref|':>9} "
          f"{'Skv':>4} {'tau':>4} {'href':>5} {'match':>6} {'sus':>4}")
    print("-" * 100)
    for _, r in out.iterrows():
        sk_v_str = f"{r['Sk_v']:.2e}" if pd.notna(r["Sk_v"]) else "—"
        tau_str = f"{r['tau']:.0f}" if isinstance(r["tau"], (int, float)) and pd.notna(r["tau"]) else str(r["tau"])
        print(f"{r['sub_id']:<14} {r['ds']:<5} {r['variant']:<7} "
              f"{r['kge_val']:>6.2f} {sk_v_str:>10} {tau_str:>7} "
              f"{r['|h_ref-h_mean|']:>9.2f} "
              f"{r['sig_Skv_floor']:>4} {r['sig_tau_high']:>4} {r['sig_href_shift']:>5} "
              f"{r['n_match']:>6} {'YES' if r['suspect'] else '':>4}")

    # Summary by tier
    print("\n--- Summary by tier ---")
    out["tier"] = pd.cut(out["kge_val"],
                         bins=[-np.inf, 0.0, 0.4, 0.6, np.inf],
                         labels=["Poor(<0)", "Poor(0–0.4)", "Fair(0.4–0.6)", "Good(>0.6)"])
    summary = out.groupby("tier", observed=True)["suspect"].agg(
        ["count", "sum", lambda s: f"{100*s.mean():.0f}%"])
    summary.columns = ["n", "n_suspect", "pct_suspect"]
    print(summary.to_string())

    out.to_csv("workspace/results_sub/v10_anchor_fix/jump_signature_scan.csv", index=False)
    print("\nWrote jump_signature_scan.csv")


if __name__ == "__main__":
    main()
