import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import gw_shell


@dataclass
class FitOutcome:
    st_id: str
    rf_id: str
    ups_id: str
    group_name: str
    model: str
    rmse: float
    r2: float
    rain_lag_days: int
    up_lag_days: int
    r2_val: float = float("nan")
    rmse_val: float = float("nan")


def _corr_abs(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 10:
        return -np.inf
    if np.var(x) == 0 or np.var(y) == 0:
        return -np.inf
    c = np.corrcoef(x, y)[0, 1]
    if np.isnan(c):
        return -np.inf
    return float(abs(c))


def best_abs_corr_rain(h_obs: np.ndarray, rainfall: np.ndarray, max_lag: int = 45) -> Tuple[int, float]:
    if len(h_obs) < 3 or len(rainfall) < 3:
        return 0, -np.inf
    dh_obs = h_obs[1:] - h_obs[:-1]

    best_lag = 0
    best_score = -np.inf
    for lag in range(0, max_lag + 1):
        if lag == 0:
            x = rainfall[:-1]
            y = dh_obs
        else:
            if lag >= len(dh_obs):
                break
            x = rainfall[:-1 - lag]
            y = dh_obs[lag:]
        score = _corr_abs(x, y)
        if score > best_score:
            best_score = score
            best_lag = lag
    return best_lag, best_score


def best_abs_corr_upstream(h_obs: np.ndarray, h_up: np.ndarray, max_lag: int = 45) -> Tuple[int, float]:
    if len(h_obs) < 3 or len(h_up) < 3:
        return 0, -np.inf
    dh_obs = h_obs[1:] - h_obs[:-1]
    dh_up = h_up[1:] - h_up[:-1]

    best_lag = 0
    best_score = -np.inf
    for lag in range(0, max_lag + 1):
        if lag == 0:
            x = dh_up
            y = dh_obs
        else:
            if lag >= len(dh_obs):
                break
            x = dh_up[:-lag]
            y = dh_obs[lag:]
        score = _corr_abs(x, y)
        if score > best_score:
            best_score = score
            best_lag = lag
    return best_lag, best_score


def build_df_merge(
    st_id: str,
    rf_id: str,
    ups_id: str,
    group_name: str,
    gw_hourly: pd.DataFrame,
    gw_daily: pd.DataFrame,
    rf_daily: pd.DataFrame,
) -> Tuple[pd.DataFrame, bool]:
    if st_id not in gw_daily.columns or st_id not in gw_hourly.columns:
        raise KeyError(f"Missing station {st_id} in groundwater data")

    df_gw = gw_daily[[st_id]].rename(columns={st_id: "gwl"})

    no_upstream = False
    if pd.isna(ups_id) or str(ups_id).lower() == "none" or str(ups_id) not in gw_daily.columns:
        no_upstream = True
        const_value = float(df_gw["gwl"].iloc[0]) if not df_gw.empty else 0.0
        df_up = pd.DataFrame({"ups_gwl": const_value}, index=df_gw.index)
    else:
        df_up = gw_daily[[str(ups_id)]].rename(columns={str(ups_id): "ups_gwl"})

    if rf_id not in rf_daily.columns:
        raise KeyError(f"Missing rainfall {rf_id} in rf data")
    df_rf = rf_daily[[rf_id]].rename(columns={rf_id: "rf"})

    hourly_series = gw_hourly[st_id]
    df_amp, df_amt = gw_shell._compute_amp_amt(hourly_series)
    df_amp = df_amp.reindex(df_gw.index).interpolate(limit_direction="both").ffill().bfill()
    df_amt = df_amt.reindex(df_gw.index).interpolate(limit_direction="both").ffill().bfill()

    df_merge = df_gw.join(df_up, how="inner").join(df_rf, how="inner").join(df_amp, how="inner").join(df_amt, how="inner")
    df_merge = df_merge.dropna()
    if df_merge.empty:
        raise ValueError("No overlapping daily records after merging inputs")

    # Store group_name for later logging
    df_merge.attrs["group_name"] = group_name

    return df_merge, no_upstream


def fit_combo(
    st_id: str,
    rf_id: str,
    ups_id: str,
    group_name: str,
    gw_hourly: pd.DataFrame,
    gw_daily: pd.DataFrame,
    rf_daily: pd.DataFrame,
    max_lag: int,
    n_starts: int,
) -> FitOutcome:
    df_merge, no_upstream = build_df_merge(
        st_id=st_id,
        rf_id=rf_id,
        ups_id=ups_id,
        group_name=group_name,
        gw_hourly=gw_hourly,
        gw_daily=gw_daily,
        rf_daily=rf_daily,
    )

    h_obs = df_merge["gwl"].values
    rainfall = df_merge["rf"].values

    rain_lag = gw_shell.estimate_rain_lag(h_obs, rainfall, max_lag=max_lag)
    if no_upstream:
        up_lag = 0
    else:
        up_lag = gw_shell.estimate_upstream_lag(h_obs, df_merge["ups_gwl"].values, max_lag=max_lag)

    results = []
    for model_name in ["base", "filtered"]:
        r = gw_shell._fit_model(
            df_merge,
            group_name=group_name,
            rain_lag_days=rain_lag,
            up_lag_days=up_lag,
            no_upstream=no_upstream,
            n_starts=n_starts,
            model_name=model_name,
        )
        if r is not None:
            results.append(r)

    if not results:
        raise RuntimeError("No successful fit for any model")

    best = min(results, key=lambda item: item["aic"])
    return FitOutcome(
        st_id=st_id,
        rf_id=str(rf_id),
        ups_id=str(ups_id),
        group_name=group_name,
        model=str(best["model"]),
        rmse=float(best["rmse"]),
        r2=float(best["r2"]),
        rain_lag_days=int(rain_lag),
        up_lag_days=int(up_lag),
        r2_val=float(best.get("r2_val", float("nan"))),
        rmse_val=float(best.get("rmse_val", float("nan"))),
    )


def rank_rf_candidates(
    st_id: str,
    h_obs: np.ndarray,
    rf_daily: pd.DataFrame,
    max_lag: int,
    top_k: int,
    include: Optional[str] = None,
) -> list[str]:
    scores = []
    for col in rf_daily.columns:
        if col == "date time":
            continue
        rain = rf_daily[col].values
        _, sc = best_abs_corr_rain(h_obs, rain, max_lag=max_lag)
        scores.append((col, sc))

    scores.sort(key=lambda x: x[1], reverse=True)
    selected = [c for c, _ in scores[:top_k]]
    if include is not None and include not in selected and include in rf_daily.columns:
        selected = [include] + selected
    return selected


def rank_ups_candidates(
    st_id: str,
    h_obs: np.ndarray,
    gw_daily: pd.DataFrame,
    max_lag: int,
    top_k: int,
    include: Optional[str] = None,
    candidates: Optional[list] = None,
) -> list[str]:
    # If a pre-filtered geographic candidate list is provided, restrict scoring to it.
    # This avoids scanning every station and keeps upstream candidates physically plausible.
    cols_to_score = candidates if candidates is not None else list(gw_daily.columns)
    scores = []
    for col in cols_to_score:
        if col == st_id or col == "date time" or col not in gw_daily.columns:
            continue
        h_up = gw_daily[col].values
        _, sc = best_abs_corr_upstream(h_obs, h_up, max_lag=max_lag)
        scores.append((col, sc))

    scores.sort(key=lambda x: x[1], reverse=True)
    selected = [c for c, _ in scores[:top_k]]

    # Always consider 'none' as a candidate to test "no upstream" hypothesis
    if "none" not in selected:
        selected.append("none")

    if include is not None and include not in selected and include in gw_daily.columns:
        selected = [include] + selected
    return selected


def _select_metric(outcome: FitOutcome) -> float:
    """Return the metric to maximise during search. Prefer r2_val when valid."""
    if np.isfinite(outcome.r2_val):
        return outcome.r2_val
    return outcome.r2


def _apply_improvements(
    input_csv: Path,
    summary_rows: list,
    out_csv: Path,
    min_delta_r2: float = 0.02,
) -> pd.DataFrame:
    """Write gray_box_input_optimized.csv with improved pairings applied for qualifying stations."""
    df = pd.read_csv(input_csv)
    df["st_id"] = df["st_id"].astype(str)

    applied = 0
    for row in summary_rows:
        # Prefer delta_r2_val (validation improvement) as the gate; fall back to delta_r2.
        delta_val = row.get("delta_r2_val")
        delta_train = row.get("delta_r2")
        if delta_val is not None and np.isfinite(float(delta_val)):
            delta = float(delta_val)
        elif delta_train is not None and np.isfinite(float(delta_train)):
            delta = float(delta_train)
        else:
            continue
        if delta < min_delta_r2:
            continue
        mask = df["st_id"] == str(row["st_id"])
        if not mask.any():
            continue
        df.loc[mask, "rf_id"] = row["best_rf"]
        df.loc[mask, "ups_id"] = row["best_ups"]
        rain_lag = row.get("best_rain_lag_days")
        if rain_lag is not None and np.isfinite(float(rain_lag)):
            df.loc[mask, "lag_days"] = int(float(rain_lag))
        applied += 1

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Applied {applied} pairing improvement(s) → {out_csv}")
    return df


def _plot_upstream_map(df_links: pd.DataFrame, title: str, out_path: Path, shp_path: str) -> None:
    """Draw upstream link arrows on a map from a gray_box_input-style DataFrame."""
    try:
        import geopandas as gpd
        from pyproj import CRS, Transformer

        TWD97_CRS = CRS.from_string(
            "+proj=tmerc +lat_0=0 +lon_0=121 +k=0.9999 +x_0=250000 +y_0=0 +ellps=GRS80 +units=m +no_defs"
        )
        transformer = Transformer.from_crs(TWD97_CRS, CRS.from_epsg(4326), always_xy=True)

        fig, ax = plt.subplots(figsize=(8, 10))

        if os.path.exists(shp_path):
            boundary = gpd.read_file(shp_path).to_crs("EPSG:4326")
            boundary.plot(ax=ax, edgecolor="black", facecolor="none")

        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(title)

        coord_lookup = {
            str(r["st_id"]): (float(r["gw_TM_X97"]), float(r["gw_TM_Y97"]))
            for _, r in df_links.iterrows()
        }

        for _, row in df_links.iterrows():
            lon, lat = transformer.transform(row["gw_TM_X97"], row["gw_TM_Y97"])
            ax.scatter(lon, lat, color="blue", s=10, zorder=5)
            ax.annotate(str(row["st_id"]), (lon, lat), fontsize=6,
                        xytext=(2, 2), textcoords="offset points")

        for _, row in df_links.iterrows():
            gw_id = str(row["st_id"])
            ups_id = str(row.get("ups_id", "none"))
            if ups_id in ("none", "nan", "") or ups_id not in coord_lookup:
                continue
            x1, y1 = coord_lookup[gw_id]
            x2, y2 = coord_lookup[ups_id]
            lon1, lat1 = transformer.transform(x1, y1)
            lon2, lat2 = transformer.transform(x2, y2)
            ax.annotate(
                "",
                xy=(lon1, lat1),
                xytext=(lon2, lat2),
                arrowprops=dict(arrowstyle="->", color="red", lw=0.8),
            )

        blue_dot = plt.Line2D([], [], marker="o", color="blue", markersize=5,
                              label="Observation Wells", linestyle="none")
        red_arr = plt.Line2D([], [], color="red", linewidth=1, label="Upstream Links")
        ax.legend(handles=[blue_dot, red_arr], loc="upper left")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Upstream map saved → {out_path}")
    except Exception as exc:
        print(f"Failed to draw upstream map: {exc}")
        import traceback
        traceback.print_exc()


def main() -> int:
    parser = argparse.ArgumentParser(description="Search alternative rf_id/ups_id pairings for low-R² stations.")
    parser.add_argument(
        "--run-id",
        default="initial",
        help="Run tag matching the one used in 03_run_model.py. Sets default --fit-results path.",
    )
    parser.add_argument("--fit-results", default=None,
                        help="Path to gw_fit_results.csv. Defaults to workspace/results/{run-id}/gw_fit_results.csv.")
    parser.add_argument("--gray-box-input", default="data/gray_box_input.csv")
    parser.add_argument("--r2-threshold", type=float, default=0.5)
    parser.add_argument("--max-lag", type=int, default=45)
    parser.add_argument("--top-k-rf", type=int, default=2)
    parser.add_argument("--top-k-ups", type=int, default=2)
    parser.add_argument("--n-starts", type=int, default=2)
    parser.add_argument("--passes", type=int, default=2, help="Coordinate-descent passes (rf then ups).")
    parser.add_argument("--output-summary", default="workspace/diagnostics/pairing_search_summary.csv")
    parser.add_argument("--output-trials", default="workspace/diagnostics/pairing_search_trials.csv")
    parser.add_argument("--output-optimized", default="data/gray_box_input_optimized.csv",
                        help="Path to write improved gray_box_input with optimized pairings applied.")
    parser.add_argument("--min-improvement", type=float, default=0.02,
                        help="Minimum delta_r2 required to apply a new pairing to the optimized CSV.")

    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[1]

    def _resolve(p: str) -> Path:
        pp = Path(p)
        return pp if pp.is_absolute() else (base_dir / pp)

    fit_results_path = args.fit_results if args.fit_results is not None else \
        str(base_dir / "workspace" / "results" / args.run_id / "gw_fit_results.csv")
    fit_df = pd.read_csv(_resolve(fit_results_path))
    input_df = pd.read_csv(_resolve(args.gray_box_input))

    # Load timeseries once
    gw_hourly = pd.read_csv(gw_shell.GW_DATA_PATH, parse_dates=["date time"]).rename(columns=str).set_index("date time")
    gw_daily = gw_hourly.resample("D").mean()

    rf_daily = pd.read_csv(gw_shell.RF_DATA_PATH, parse_dates=["date time"]).rename(columns=str).set_index("date time")

    # Identify low training-R² stations — pairing changes can only fix poor fit,
    # not train/val generalisation gaps (which are a model-structure problem).
    fit_df["r2"] = pd.to_numeric(fit_df.get("r2"), errors="coerce")
    fit_df["r2_val"] = pd.to_numeric(fit_df.get("r2_val"), errors="coerce")
    low = fit_df[fit_df["r2"].notna() & (fit_df["r2"] < args.r2_threshold)].copy()

    # Merge group/inputs
    input_df["st_id"] = input_df["st_id"].astype(str)
    low["st_id"] = low["st_id"].astype(str)

    meta_cols = [c for c in ["st_id", "group", "rf_id", "ups_id"] if c in input_df.columns]
    low = low.merge(input_df[meta_cols], on="st_id", how="left", suffixes=("", "_input"))

    trials = []
    summary_rows = []

    for _, row in low.iterrows():
        st_id = str(row["st_id"])
        baseline_rf = str(row.get("rf_id_input", row.get("rf_id")))
        baseline_ups = str(row.get("ups_id_input", row.get("ups_id")))
        group_name = str(row.get("group", row.get("group_name", "inland")))

        # Build baseline gwl for correlation screening
        if st_id not in gw_daily.columns:
            continue
        h_obs = gw_daily[st_id].values

        rf_candidates = rank_rf_candidates(st_id, h_obs, rf_daily, max_lag=args.max_lag, top_k=args.top_k_rf, include=baseline_rf)

        # Use pre-computed geographic candidates from 02_pairing if available,
        # so the upstream search stays physically constrained rather than scanning all stations.
        geo_pool_str = ""
        if "ups_candidates" in input_df.columns:
            mask_st = input_df["st_id"] == st_id
            if mask_st.any():
                geo_pool_str = str(input_df.loc[mask_st, "ups_candidates"].values[0])
        geo_pool = [c.strip() for c in geo_pool_str.split(",")
                    if c.strip() and c.strip() not in ("nan", "none", "")]
        ups_candidates = rank_ups_candidates(
            st_id, h_obs, gw_daily, max_lag=args.max_lag, top_k=args.top_k_ups,
            include=baseline_ups,
            candidates=geo_pool if geo_pool else None,
        )

        current_rf = baseline_rf
        current_ups = baseline_ups

        # Use stored fit results as baseline — avoids optimizer noise from a fresh re-fit
        # which could corrupt the delta_r2 comparison.
        baseline = FitOutcome(
            st_id=st_id,
            rf_id=current_rf,
            ups_id=current_ups,
            group_name=group_name,
            model=str(row.get("model", "")),
            rmse=float(row.get("rmse", np.nan)),
            r2=float(row.get("r2", np.nan)),
            rain_lag_days=int(row.get("rain_lag_days", 0) if pd.notna(row.get("rain_lag_days", 0)) else 0),
            up_lag_days=int(row.get("up_lag_days", 0) if pd.notna(row.get("up_lag_days", 0)) else 0),
            r2_val=float(row.get("r2_val", np.nan)) if pd.notna(row.get("r2_val", np.nan)) else float("nan"),
            rmse_val=float(row.get("rmse_val", np.nan)) if pd.notna(row.get("rmse_val", np.nan)) else float("nan"),
        )

        best_overall = baseline

        for p in range(args.passes):
            # RF search (ups fixed)
            best_rf_out = best_overall
            for cand_rf in rf_candidates:
                try:
                    out = fit_combo(
                        st_id=st_id,
                        rf_id=cand_rf,
                        ups_id=current_ups,
                        group_name=group_name,
                        gw_hourly=gw_hourly,
                        gw_daily=gw_daily,
                        rf_daily=rf_daily,
                        max_lag=args.max_lag,
                        n_starts=args.n_starts,
                    )
                    trials.append({
                        "st_id": st_id,
                        "stage": f"pass{p+1}_rf",
                        "rf_id": out.rf_id,
                        "ups_id": out.ups_id,
                        "model": out.model,
                        "rmse": out.rmse,
                        "r2": out.r2,
                        "rain_lag_days": out.rain_lag_days,
                        "up_lag_days": out.up_lag_days,
                    })
                    if _select_metric(out) > _select_metric(best_rf_out):
                        best_rf_out = out
                except Exception as e:
                    trials.append({
                        "st_id": st_id,
                        "stage": f"pass{p+1}_rf",
                        "rf_id": str(cand_rf),
                        "ups_id": str(current_ups),
                        "error": str(e),
                    })

            current_rf = best_rf_out.rf_id
            best_overall = best_rf_out

            # UPS search (rf fixed)
            best_ups_out = best_overall
            for cand_ups in ups_candidates:
                try:
                    out = fit_combo(
                        st_id=st_id,
                        rf_id=current_rf,
                        ups_id=cand_ups,
                        group_name=group_name,
                        gw_hourly=gw_hourly,
                        gw_daily=gw_daily,
                        rf_daily=rf_daily,
                        max_lag=args.max_lag,
                        n_starts=args.n_starts,
                    )
                    trials.append({
                        "st_id": st_id,
                        "stage": f"pass{p+1}_ups",
                        "rf_id": out.rf_id,
                        "ups_id": out.ups_id,
                        "model": out.model,
                        "rmse": out.rmse,
                        "r2": out.r2,
                        "rain_lag_days": out.rain_lag_days,
                        "up_lag_days": out.up_lag_days,
                    })
                    if _select_metric(out) > _select_metric(best_ups_out):
                        best_ups_out = out
                except Exception as e:
                    trials.append({
                        "st_id": st_id,
                        "stage": f"pass{p+1}_ups",
                        "rf_id": str(current_rf),
                        "ups_id": str(cand_ups),
                        "error": str(e),
                    })

            current_ups = best_ups_out.ups_id
            best_overall = best_ups_out

        summary_rows.append({
            "st_id": st_id,
            "group": group_name,
            "baseline_rf": baseline.rf_id,
            "baseline_ups": baseline.ups_id,
            "baseline_model": baseline.model,
            "baseline_rmse": baseline.rmse,
            "baseline_r2": baseline.r2,
            "baseline_r2_val": baseline.r2_val,
            "best_rf": best_overall.rf_id,
            "best_ups": best_overall.ups_id,
            "best_model": best_overall.model,
            "best_rmse": best_overall.rmse,
            "best_r2": best_overall.r2,
            "best_r2_val": best_overall.r2_val,
            "delta_r2": (best_overall.r2 - baseline.r2) if np.isfinite(baseline.r2) else np.nan,
            "delta_r2_val": (best_overall.r2_val - baseline.r2_val)
                if np.isfinite(baseline.r2_val) and np.isfinite(best_overall.r2_val) else np.nan,
            "delta_rmse": (baseline.rmse - best_overall.rmse) if np.isfinite(baseline.rmse) else np.nan,
            "best_rain_lag_days": best_overall.rain_lag_days,
            "best_up_lag_days": best_overall.up_lag_days,
            "rf_candidates": ",".join(rf_candidates),
            "ups_candidates": ",".join(ups_candidates),
        })

    out_summary = _resolve(args.output_summary)
    out_trials = _resolve(args.output_trials)
    pd.DataFrame(summary_rows).to_csv(out_summary, index=False)
    pd.DataFrame(trials).to_csv(out_trials, index=False)

    print(f"Wrote summary: {out_summary}")
    print(f"Wrote trials: {out_trials}")
    if summary_rows:
        show = pd.DataFrame(summary_rows).sort_values("delta_r2_val", ascending=False)
        print(show[["st_id", "baseline_r2", "best_r2", "delta_r2", "baseline_r2_val", "best_r2_val", "delta_r2_val", "baseline_rf", "best_rf", "baseline_ups", "best_ups"]].head(20).to_string(index=False))

    # Write optimized input CSV with improved pairings applied
    out_optimized = _resolve(args.output_optimized)
    input_csv = _resolve(args.gray_box_input)
    df_optimized = _apply_improvements(
        input_csv=input_csv,
        summary_rows=summary_rows,
        out_csv=out_optimized,
        min_delta_r2=args.min_improvement,
    )

    # Draw the optimized upstream map for visual comparison with the initial map
    shp_path = str(base_dir / "data" / "Zhuoshui Alluvial Fan" / "Zhuoshui Alluvial Fan.shp")
    map_out = base_dir / "workspace" / "maps" / "upstream_links_optimized.tiff"
    _plot_upstream_map(
        df_links=df_optimized,
        title="Upstream Links — Optimized Pairings",
        out_path=map_out,
        shp_path=shp_path,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
