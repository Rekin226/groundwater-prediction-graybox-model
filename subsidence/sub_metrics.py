"""KGE / NSE / RMSE / R² metrics for cumulative ζ and its rate."""
from __future__ import annotations
import numpy as np


def kge_components(y_obs: np.ndarray, y_sim: np.ndarray) -> dict:
    y_obs = np.asarray(y_obs, float); y_sim = np.asarray(y_sim, float)
    mask = ~(np.isnan(y_obs) | np.isnan(y_sim))
    yo = y_obs[mask]; ys = y_sim[mask]
    mu_o, mu_s = yo.mean(), ys.mean()
    sd_o, sd_s = yo.std(ddof=0), ys.std(ddof=0)
    r = np.corrcoef(yo, ys)[0, 1] if (sd_o > 0 and sd_s > 0) else np.nan
    alpha = sd_s / sd_o if sd_o > 0 else np.nan
    beta = mu_s / mu_o if abs(mu_o) > 1e-9 else np.nan
    bias = mu_s - mu_o
    if np.isnan(r) or np.isnan(alpha) or np.isnan(beta):
        kge_val = np.nan
    else:
        kge_val = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    return {"kge": kge_val, "r": r, "alpha": alpha, "beta": beta, "bias": bias}


def kge(y_obs, y_sim) -> float:
    return float(kge_components(y_obs, y_sim)["kge"])


def kge_on_rate(y_obs, y_sim) -> float:
    """KGE on first difference of the cumulative series (rate)."""
    return kge(np.diff(y_obs), np.diff(y_sim))


def rmse(y_obs, y_sim) -> float:
    y_obs = np.asarray(y_obs, float); y_sim = np.asarray(y_sim, float)
    return float(np.sqrt(np.nanmean((y_obs - y_sim) ** 2)))


def r2(y_obs, y_sim) -> float:
    y_obs = np.asarray(y_obs, float); y_sim = np.asarray(y_sim, float)
    mask = ~(np.isnan(y_obs) | np.isnan(y_sim))
    yo = y_obs[mask]; ys = y_sim[mask]
    ss_res = np.sum((yo - ys) ** 2)
    ss_tot = np.sum((yo - yo.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
