"""Hybrid driver assembly: observed h with GW-model gap-fill and cosine taper."""
from __future__ import annotations
import numpy as np
import pandas as pd


def assemble_h(obs: pd.Series, sim: pd.Series,
               max_interp_gap_days: int = 7,
               taper_days: int = 5) -> pd.DataFrame:
    """Compose `h_driver` per the hybrid rule (spec §6).

    obs and sim must be daily-indexed and aligned. NaN in obs marks missing days.

    Returns a DataFrame with columns: h_driver, driver_source.
    Sets ``df.attrs['edge_bias_mean']`` = mean of (obs − sim) at obs/sim edges.
    """
    if not obs.index.equals(sim.index):
        # Align and reindex
        idx = obs.index.union(sim.index)
        obs = obs.reindex(idx)
        sim = sim.reindex(idx)
    h = obs.copy()
    src = pd.Series("obs", index=obs.index, dtype=object)
    src[obs.isna()] = "missing"

    # Identify gap runs in the observation series
    is_gap = obs.isna()
    # Run-length encoding by diff of cumulative non-gap
    grp = (is_gap != is_gap.shift()).cumsum()
    edge_biases = []
    for _, run_idx in is_gap[is_gap].groupby(grp):
        gap_len = len(run_idx)
        first, last = run_idx.index[0], run_idx.index[-1]
        if gap_len <= max_interp_gap_days:
            # Mark for linear interp; set NaN to ensure interpolation happens
            h.loc[run_idx.index] = np.nan
            src.loc[run_idx.index] = "linear_interp"
        else:
            # Substitute simulated values
            h.loc[run_idx.index] = sim.loc[run_idx.index].values
            src.loc[run_idx.index] = "model_fill"
            # Record bias at the boundary days (day just before / after the gap)
            for edge in (first - pd.Timedelta(days=1), last + pd.Timedelta(days=1)):
                if edge in obs.index and pd.notna(obs.loc[edge]):
                    edge_biases.append(float(obs.loc[edge] - sim.loc[edge]))

    # After: linear interpolate any remaining NaNs (these are linear_interp marked)
    h = h.interpolate(method="time", limit_direction="both")

    # Cosine taper at obs↔model boundaries
    if taper_days > 0:
        # Build a mask of model_fill spans bounded by obs spans
        is_model = (src == "model_fill").values
        for i in range(len(src)):
            # Left edge of a model_fill span
            if is_model[i] and (i == 0 or not is_model[i - 1]):
                end = min(i + taper_days, len(src))
                t = np.arange(end - i)
                w = 0.5 * (1 - np.cos(np.pi * t / max(1, taper_days)))  # 0→1
                if i > 0 and pd.notna(obs.iloc[i - 1]):
                    base_obs = obs.iloc[i - 1]
                    h_old = h.iloc[i:end].values
                    h.iloc[i:end] = (1 - w) * base_obs + w * h_old
                    for k in range(i, end):
                        if src.iloc[k] == "model_fill":
                            src.iloc[k] = "taper"
            # Right edge of a model_fill span
            if is_model[i] and (i == len(is_model) - 1 or not is_model[i + 1]):
                start = max(i + 1 - taper_days, 0)
                t = np.arange(i + 1 - start)[::-1]
                w = 0.5 * (1 - np.cos(np.pi * t / max(1, taper_days)))
                if i + 1 < len(obs) and pd.notna(obs.iloc[i + 1]):
                    base_obs = obs.iloc[i + 1]
                    h_old = h.iloc[start:i + 1].values
                    h.iloc[start:i + 1] = (1 - w) * base_obs + w * h_old
                    for k in range(start, i + 1):
                        if src.iloc[k] == "model_fill":
                            src.iloc[k] = "taper"

    out = pd.DataFrame({"h_driver": h, "driver_source": src})
    out.attrs["edge_bias_mean"] = float(np.mean(edge_biases)) if edge_biases else 0.0
    out.attrs["n_edge_samples"] = len(edge_biases)
    return out
