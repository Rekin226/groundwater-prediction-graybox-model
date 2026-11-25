"""Per-subsidence-station hybrid driver assembly.

Reads:
    subsidence/data/sub_pairing.csv
    workspace/results/final/gw_fit_results.csv  (frozen GW params)
    data/ls_cache/gw__<api_id>.parquet          (observed GW)
    data/ls_cache/rain__<rf_id>.parquet         (rainfall)
Writes:
    subsidence/data/h_drivers/<sub_id>.parquet  (cols: h_driver, driver_source)

Run:
    LS_USER=... LS_PASS=... poetry run python subsidence/04_assemble_h_driver.py
"""
from __future__ import annotations
import sys
from pathlib import Path

# Ensure project root is on sys.path when the script is run directly
# (e.g. `python subsidence/04_assemble_h_driver.py` from the project root)
_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import numpy as np
import pandas as pd

from subsidence.driver_assembly import assemble_h
from subsidence.gw_simulator_adapter import simulate
from subsidence.ls_client import to_api_id

PAIRING = Path("subsidence/data/sub_pairing.csv")
GW_FIT = Path("workspace/results/final/gw_fit_results.csv")
GW_INPUT = Path("data/gray_box_input.csv")
CACHE_DIR = Path("data/ls_cache")
OUT_DIR = Path("subsidence/data/h_drivers")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# β window: calibration start → validation end
CAL_START = pd.Timestamp("2020-01-01")
VAL_END = pd.Timestamp("2025-03-31")


def _build_params_dict(row: pd.Series) -> dict:
    """Build the params dict expected by gw_simulator_adapter.simulate().

    Key action: rename 'lambda' → 'lam' for filtered variants, since
    'lambda' is a Python reserved word and the adapter uses 'lam'.
    Only include keys that are not NaN.
    """
    param_keys = [
        "a", "z", "b", "c", "k_link", "tau_rain", "tau_up",
        "d_sin", "d_cos", "z0", "z1", "k_sgd", "gamma", "h_sea",
    ]
    params = {}
    for k in param_keys:
        v = row.get(k, np.nan)
        if pd.notna(v):
            params[k] = float(v)

    # Rename lambda → lam (filtered / filtered_tz variants need 'lam')
    lam_val = row.get("lambda", np.nan)
    if pd.notna(lam_val):
        params["lam"] = float(lam_val)

    return params


def _resimulate_gw(params: dict, model: str, group_name: str,
                   idx: pd.DatetimeIndex,
                   gw_obs: pd.Series, gw_meta: pd.DataFrame) -> pd.Series:
    """Re-run the GW simulator once over the full β window with fresh forcings."""
    n = len(idx)

    # Upstream head series
    ups_id = params.get("ups_id", "")
    if ups_id and str(ups_id) not in ("none", "nan", "", "None"):
        ups_rows = gw_meta[gw_meta["st_id"] == ups_id]
        if not ups_rows.empty:
            api_id_up = to_api_id(int(ups_rows.iloc[0]["gw_st"]))
            up_path = CACHE_DIR / f"gw__{api_id_up}.parquet"
            if up_path.exists():
                up_df = pd.read_parquet(up_path)
                h_up_raw = up_df["h_obs_m"].reindex(idx)
                h_up = h_up_raw.ffill().bfill().fillna(0.0).values
            else:
                h_up = np.zeros(n)
        else:
            h_up = np.zeros(n)
    else:
        h_up = np.zeros(n)

    # Rainfall
    rf_id = params.get("rf_id", "")
    rain_path = CACHE_DIR / f"rain__{rf_id}.parquet"
    if rain_path.exists():
        rain_df = pd.read_parquet(rain_path)
        # Handle timezone-aware vs naive index
        rain_idx = rain_df.index
        if hasattr(rain_idx, "tz") and rain_idx.tz is not None:
            rain_idx = rain_idx.tz_localize(None)
            rain_df = rain_df.copy()
            rain_df.index = rain_idx
        rain = rain_df["rainfall_mm"].reindex(idx).fillna(0.0).values
    else:
        rain = np.zeros(n)

    # Initial head: first non-NaN observed value, fallback to z/z0
    obs_valid = gw_obs.dropna()
    if not obs_valid.empty:
        h0 = float(obs_valid.iloc[0])
    else:
        h0 = float(params.get("z", params.get("z0", 0.0)))

    # Build simulator params dict (exclude orchestration keys)
    sim_params = {k: v for k, v in params.items()
                  if k not in ("ups_id", "rf_id", "model", "group_name",
                               "st_id", "gw_st")}

    sim_arr = simulate(
        model_name=model,
        group_name=group_name,
        h0=h0,
        params=sim_params,
        rain=rain,
        h_up=h_up,
        amp=np.zeros(n),
        amt=np.zeros(n),
        h_sea=np.zeros(n),
        doy=idx.dayofyear.values - 1,
        t_abs=np.arange(n, dtype=float),
    )
    return pd.Series(sim_arr, index=idx)


def assemble_one(sub_id: str) -> pd.DataFrame:
    """Assemble the hybrid h driver for a single subsidence station.

    Exposed as a public function so smoke tests can call it directly.
    """
    pair = pd.read_csv(PAIRING)
    gw_fit = pd.read_csv(GW_FIT)
    gw_meta = pd.read_csv(GW_INPUT)

    row_pair = pair[pair["sub_id"] == sub_id]
    if row_pair.empty:
        raise RuntimeError(f"{sub_id}: no pairing entry")
    row_pair = row_pair.iloc[0]

    if pd.isna(row_pair.get("gw_st")):
        raise RuntimeError(f"{sub_id}: pairing has NaN gw_st")

    api_id = to_api_id(int(row_pair["gw_st"]))
    gw_cache_path = CACHE_DIR / f"gw__{api_id}.parquet"
    if not gw_cache_path.exists():
        raise RuntimeError(
            f"{sub_id}: cached GW obs not found at {gw_cache_path}"
        )

    # Load observed GW and reindex to full β window
    gw_obs_raw = pd.read_parquet(gw_cache_path)["h_obs_m"]
    idx = pd.date_range(CAL_START, VAL_END, freq="1D")
    # Handle timezone-aware index from cache
    if hasattr(gw_obs_raw.index, "tz") and gw_obs_raw.index.tz is not None:
        gw_obs_raw.index = gw_obs_raw.index.tz_localize(None)
    gw_obs = gw_obs_raw.reindex(idx)

    # Get fitted GW params for this station's paired GW station
    st_id = row_pair["st_id"]
    params_row = gw_fit[gw_fit["st_id"] == st_id]
    if params_row.empty:
        raise RuntimeError(f"{sub_id}: no fitted GW params for {st_id}")
    params_row = params_row.iloc[0]

    model = str(params_row["model"])
    group_name = str(params_row["group_name"])

    # Build params dict with lambda → lam rename
    params = _build_params_dict(params_row)
    # Add orchestration metadata needed by _resimulate_gw
    params["ups_id"] = str(params_row.get("ups_id", ""))
    params["rf_id"] = str(params_row.get("rf_id", ""))

    # Re-simulate GW over the full β window
    sim = _resimulate_gw(params, model, group_name, idx, gw_obs, gw_meta)

    # Assemble hybrid driver
    out = assemble_h(gw_obs, sim, max_interp_gap_days=7, taper_days=5)

    # Write parquet
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_DIR / f"{sub_id}.parquet")
    return out


def main(argv=None):
    pair = pd.read_csv(PAIRING)
    n_ok = 0
    n_err = 0
    for _, r in pair.iterrows():
        try:
            out = assemble_one(r["sub_id"])
            n_ok += 1
            print(f"  {r['sub_id']:25s}  rows={len(out)}  "
                  f"edge_bias_mean={out.attrs['edge_bias_mean']:.4f}")
        except Exception as e:
            n_err += 1
            print(f"  {r['sub_id']:25s}  ERR {str(e)[:120]}")

    print(f"\nDone: {n_ok} OK, {n_err} failed.")


if __name__ == "__main__":
    sys.exit(main())
