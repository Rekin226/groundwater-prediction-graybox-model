import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

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

    best = min(results, key=lambda item: item["rmse"])
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
) -> list[str]:
    scores = []
    for col in gw_daily.columns:
        if col == st_id or col == "date time":
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Search alternative rf_id/ups_id pairings for low-R² stations.")
    parser.add_argument("--fit-results", default="workspace/gw_fit_results.csv")
    parser.add_argument("--gray-box-input", default="data/gray_box_input.csv")
    parser.add_argument("--r2-threshold", type=float, default=0.5)
    parser.add_argument("--max-lag", type=int, default=45)
    parser.add_argument("--top-k-rf", type=int, default=3)
    parser.add_argument("--top-k-ups", type=int, default=3)
    parser.add_argument("--n-starts", type=int, default=4)
    parser.add_argument("--passes", type=int, default=2, help="Coordinate-descent passes (rf then ups).")
    parser.add_argument("--output-summary", default="workspace/pairing_search_summary.csv")
    parser.add_argument("--output-trials", default="workspace/pairing_search_trials.csv")

    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[1]

    def _resolve(p: str) -> Path:
        pp = Path(p)
        return pp if pp.is_absolute() else (base_dir / pp)

    fit_df = pd.read_csv(_resolve(args.fit_results))
    input_df = pd.read_csv(_resolve(args.gray_box_input))

    # Load timeseries once
    gw_hourly = pd.read_csv(gw_shell.GW_DATA_PATH, parse_dates=["date time"]).rename(columns=str).set_index("date time")
    gw_daily = gw_hourly.resample("D").mean()

    rf_daily = pd.read_csv(gw_shell.RF_DATA_PATH, parse_dates=["date time"]).rename(columns=str).set_index("date time")

    # Identify low-R² stations
    fit_df["r2"] = pd.to_numeric(fit_df.get("r2"), errors="coerce")
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
        ups_candidates = rank_ups_candidates(st_id, h_obs, gw_daily, max_lag=args.max_lag, top_k=args.top_k_ups, include=baseline_ups)

        current_rf = baseline_rf
        current_ups = baseline_ups

        # Baseline fit (for delta)
        try:
            baseline = fit_combo(
                st_id=st_id,
                rf_id=current_rf,
                ups_id=current_ups,
                group_name=group_name,
                gw_hourly=gw_hourly,
                gw_daily=gw_daily,
                rf_daily=rf_daily,
                max_lag=args.max_lag,
                n_starts=max(2, args.n_starts),
            )
        except Exception:
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
                    if out.rmse < best_rf_out.rmse:
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
                    if out.rmse < best_ups_out.rmse:
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
            "best_rf": best_overall.rf_id,
            "best_ups": best_overall.ups_id,
            "best_model": best_overall.model,
            "best_rmse": best_overall.rmse,
            "best_r2": best_overall.r2,
            "delta_r2": (best_overall.r2 - baseline.r2) if np.isfinite(baseline.r2) else np.nan,
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
        show = pd.DataFrame(summary_rows).sort_values("delta_r2", ascending=False)
        print(show[["st_id", "baseline_r2", "best_r2", "delta_r2", "baseline_rf", "best_rf", "baseline_ups", "best_ups"]].head(20).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
