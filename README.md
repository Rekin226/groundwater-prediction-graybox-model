# Single Tank V2 — Gray-Box Groundwater Model

A physics-informed, gray-box model for simulating groundwater level (GWL) dynamics at monitoring stations across an alluvial fan. The model fits discrete-time ODEs to observed GWL using rainfall, upstream GWL, and tidal/semi-diurnal signals as drivers.

## Model overview

Two ODE variants are implemented (both integrated with Euler at dt = 1 day):

**Inland stations**
```
h[t+1] = h[t] - a*(h[t] - z) + b*R_eff[t] - c*AMP[t] + k_link*(h_up_eff[t] - h[t])
```

**Coastal stations** (adds submarine groundwater discharge and tidal amplitude terms)
```
h[t+1] = h[t] - a*(h[t] - z) + b*R_eff[t] - c*AMP[t] + k_link*(h_up_eff[t] - h[t])
        - k_sgd*(h[t] - h_sea) + gamma*AMT[t]
```

Where:
- `a` — drainage/recession coefficient
- `z` — baseflow equilibrium level
- `b` — rainfall response coefficient
- `c` — tidal pumping coefficient
- `k_link` — upstream hydraulic connectivity
- `R_eff`, `h_up_eff` — exponentially weighted (convolved) rainfall and upstream GWL with time constants `tau_rain`, `tau_up`
- `AMP` / `AMT` — diurnal (~1 cpd) and semi-diurnal (~1.93 cpd) tidal amplitudes from STFT of hourly GWL

Parameters are fitted using `scipy.optimize.curve_fit` with random multi-start to avoid local minima.

## Project structure

```
single_tankV2/
├── data/
│   ├── gray_box_input.csv   # Station config (active flag, group, paired rf/upstream IDs, lag)
│   ├── gw_data2.csv         # Hourly groundwater level time series (columns = station IDs)
│   ├── rf_data.csv          # Daily rainfall time series (columns = rainfall station IDs)
│   ├── gw_meta.csv          # Groundwater station metadata
│   └── rf_meta.csv          # Rainfall station metadata
├── srcs/
│   ├── gray_box_driver.py   # Entry point: reads config, dispatches stations in parallel
│   ├── gw_shell.py          # Per-station pipeline: data prep, lag estimation, fitting, output
│   ├── gw_subroutine.py     # ODE simulators (inland, coastal, filtered variants)
│   ├── jfft.py              # STFT wrapper for tidal amplitude extraction
│   ├── input_prepar.py      # Input CSV preparation utilities
│   ├── classif_inld_coast.py # Inland/coastal classification
│   ├── corr_up_rf.py        # Upstream/rainfall correlation analysis
│   ├── diagnostics_*.py     # Diagnostic and reporting tools
│   ├── plot_performance_maps.py # Spatial R² performance maps
│   └── makefile             # Build targets
├── pyproject.toml
└── poetry.toml
```

## Setup

Requires Python 3.10–3.14 and [Poetry](https://python-poetry.org/).

```bash
# Install dependencies
poetry install

# Activate the environment
eval $(poetry env activate)
```

## Usage

Run the full model across all active stations:

```bash
cd srcs
make gray_box
```

This calls `gray_box_driver.py`, which reads `data/gray_box_input.csv`, filters rows where `active == 1`, groups stations by `group` (inland/coastal), and dispatches each station to `gw_shell.py` in parallel.

To run a single station manually:

```bash
.venv/bin/python srcs/gw_shell.py \
  gw_st=9200211 st_id=st3 gw_x=169424.217 gw_y=2607200.343 \
  ups_id=st43 rf_id=rf17 lag_days=1 group_name=inland
```

## Station configuration (`gray_box_input.csv`)

| Column | Description |
|---|---|
| `gw_st` | Official groundwater station number |
| `st_id` | Short station ID (matches column name in `gw_data2.csv`) |
| `gw_TM_X97`, `gw_TM_Y97` | TWD97 coordinates |
| `ups_id` | Upstream station ID (`none` if no upstream) |
| `rf_id` | Paired rainfall station ID |
| `lag_days` | Rainfall lag (days) |
| `group` | `inland` or `coastal` |
| `active` | `1` to include in run, `0` to skip |

## Dependencies

| Package | Purpose |
|---|---|
| numpy, scipy | Numerics and curve fitting |
| pandas | Time series handling |
| matplotlib | Plotting |
| scikit-learn | R², RMSE metrics |
| geopandas, shapely, pyproj | Spatial operations |
| numba | JIT acceleration |
| tqdm | Progress bars |
| pyshp | Shapefile I/O |
