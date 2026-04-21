import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def near_bound(series: pd.Series, lo: float, hi: float, eps: float) -> pd.Series:
    s = _to_num(series)
    return s.notna() & (((s - lo).abs() <= eps) | ((s - hi).abs() <= eps))


def tiny(series: pd.Series, tol: float) -> pd.Series:
    s = _to_num(series)
    return s.notna() & (s.abs() < tol)


def build_diagnostics(
    fit_results: pd.DataFrame,
    gray_box_input: pd.DataFrame | None,
    kge_threshold: float,
    eps: float,
    tiny_tol: float,
) -> pd.DataFrame:
    df = fit_results.copy()

    # Normalize expected columns
    if "group_name" in df.columns and "group" not in df.columns:
        df["group"] = df["group_name"]

    df["kge_val"] = _to_num(df["kge_val"]) if "kge_val" in df.columns else np.nan
    df_low = df[df["kge_val"].notna() & (df["kge_val"] < kge_threshold)].copy()

    if gray_box_input is not None and "st_id" in gray_box_input.columns and "st_id" in df_low.columns:
        gb = gray_box_input.copy()
        gb["st_id"] = gb["st_id"].astype(str)
        df_low["st_id"] = df_low["st_id"].astype(str)

        keep_cols = [
            c
            for c in [
                "st_id",
                "gw_st",
                "ups_id",
                "rf_id",
                "correlation",
                "quality",
                "spearmanr",
                "max_corr",
                "distance_m",
                "lag_days",
                "group",
                "active",
            ]
            if c in gb.columns
        ]
        gb = gb[keep_cols].drop_duplicates(subset=["st_id"])
        df_low = df_low.merge(gb, on="st_id", how="left", suffixes=("", "_input"))

    # Bound diagnostics (based on project conventions)
    bound_specs: dict[str, tuple[float, float]] = {
        "a": (0.0, 1.0),
        "b": (0.0, 5.0),
        "c": (0.0, 10.0),
        "lambda": (0.01, 1.0),
        "tau_rain": (0.1, 30.0),
        "tau_up": (0.1, 30.0),
        "k_link": (0.001, 1.0),
        "k_sgd": (0.001, 1.0),
    }

    for col, (lo, hi) in bound_specs.items():
        if col in df_low.columns:
            df_low[f"near_bound_{col}"] = near_bound(df_low[col], lo, hi, eps=eps)

    for col in ["b", "c", "k_link", "gamma", "k_sgd"]:
        if col in df_low.columns:
            df_low[f"tiny_{col}"] = tiny(df_low[col], tol=tiny_tol)

    # Heuristic mismatch flags
    if "group_name" in df_low.columns:
        grp = df_low["group_name"].astype(str)
    elif "group" in df_low.columns:
        grp = df_low["group"].astype(str)
    else:
        grp = pd.Series(["" for _ in range(len(df_low))], index=df_low.index)

    if "gamma" in df_low.columns:
        df_low["coastal_gamma_tiny"] = (grp.str.lower() == "coastal") & tiny(df_low["gamma"], tol=tiny_tol)
    else:
        df_low["coastal_gamma_tiny"] = False

    # Recommendations
    recs: list[str] = []
    for _, row in df_low.iterrows():
        suggestions: list[str] = []

        if pd.notna(row.get("k_link")) and abs(float(row.get("k_link"))) < tiny_tol:
            suggestions.append("k_link≈0 → upstream driver not informative; try different ups_id")

        if pd.notna(row.get("b")) and float(row.get("b")) < tiny_tol:
            suggestions.append("b≈0 → rainfall driver weak; try different rf_id / recharge proxy")

        if pd.notna(row.get("c")) and abs(float(row.get("c"))) < tiny_tol:
            suggestions.append("c≈0 → AMP not helping; check AMP extraction or consider seasonal baseline")

        if (str(row.get("group_name", row.get("group", ""))).lower() == "coastal"):
            gamma_val = row.get("gamma")
            if pd.notna(gamma_val) and abs(float(gamma_val)) < tiny_tol:
                suggestions.append("coastal but gamma≈0 → tidal term unused; re-check coastal classification / AMT")

        tau_rain = row.get("tau_rain")
        tau_up = row.get("tau_up")
        if pd.notna(tau_rain) and (abs(float(tau_rain) - 0.1) <= eps or abs(float(tau_rain) - 30.0) <= eps):
            suggestions.append("tau_rain at bound → revisit lag scan / bounds / missing physics")
        if pd.notna(tau_up) and (abs(float(tau_up) - 0.1) <= eps or abs(float(tau_up) - 30.0) <= eps):
            suggestions.append("tau_up at bound → upstream signal may be mis-specified; try other ups_id")

        if not suggestions:
            suggestions.append("likely missing driver (pumping/river/management) or data quality issue")

        recs.append("; ".join(suggestions))

    df_low["recommendation"] = recs

    # A simple severity score for sorting
    flag_cols = [c for c in df_low.columns if c.startswith("near_bound_") or c.startswith("tiny_") or c == "coastal_gamma_tiny"]
    df_low["diagnostic_flags"] = df_low[flag_cols].sum(axis=1).astype(int) if flag_cols else 0

    # Column order (keep stable)
    front = [
        c
        for c in [
            "st_id",
            "gw_st",
            "group_name",
            "group",
            "model",
            "kge",
            "kge_val",
            "rmse",
            "ups_id",
            "rf_id",
            "rain_lag_days",
            "up_lag_days",
            "tau_rain",
            "tau_up",
            "a",
            "z",
            "b",
            "c",
            "k_link",
            "k_sgd",
            "gamma",
            "h_sea",
            "diagnostic_flags",
            "recommendation",
        ]
        if c in df_low.columns
    ]
    rest = [c for c in df_low.columns if c not in front]
    df_low = df_low[front + rest]

    return df_low.sort_values(["diagnostic_flags", "kge_val"], ascending=[False, True])


def main() -> int:
    base_dir = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(description="Generate diagnostics report for low-KGE stations.")
    parser.add_argument(
        "--run-id",
        default="initial",
        help="Run tag matching the one used in 03_run_model.py. Sets default --fit-results path.",
    )
    parser.add_argument(
        "--fit-results",
        default=None,
        help="Path to gw_fit_results.csv. Defaults to workspace/results/{run-id}/gw_fit_results.csv.",
    )
    parser.add_argument(
        "--gray-box-input",
        default=str(base_dir / "data" / "gray_box_input.csv"),
        help="Path to gray_box_input.csv (optional merge for metadata)",
    )
    parser.add_argument(
        "--kge-threshold",
        type=float,
        default=0.5,
        help="Stations with kge_val < threshold are included in the report",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-3,
        help="Tolerance for detecting bound hits",
    )
    parser.add_argument(
        "--tiny-tol",
        type=float,
        default=1e-6,
        help="Absolute tolerance for detecting near-zero coefficients",
    )
    parser.add_argument(
        "--output",
        default=str(base_dir / "workspace" / "diagnostics" / "diagnostics_report_kge_below_0p5.csv"),
        help="Output CSV path",
    )

    args = parser.parse_args()

    fit_results_path = args.fit_results if args.fit_results is not None else \
        str(base_dir / "workspace" / "results" / args.run_id / "gw_fit_results.csv")
    fit_path = Path(fit_results_path)
    if not fit_path.exists():
        raise FileNotFoundError(f"fit results not found: {fit_path}")

    fit_df = pd.read_csv(fit_path)

    gb_path = Path(args.gray_box_input)
    gb_df = pd.read_csv(gb_path) if gb_path.exists() else None

    report = build_diagnostics(
        fit_results=fit_df,
        gray_box_input=gb_df,
        kge_threshold=args.kge_threshold,
        eps=args.eps,
        tiny_tol=args.tiny_tol,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out_path, index=False)

    print(f"Wrote diagnostics report: {out_path}")
    print(f"Stations in report (kge_val < {args.kge_threshold}): {len(report)}")
    if len(report) > 0:
        show_cols = [c for c in ["st_id", "group_name", "model", "kge_val", "diagnostic_flags", "recommendation"] if c in report.columns]
        print(report[show_cols].head(20).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
