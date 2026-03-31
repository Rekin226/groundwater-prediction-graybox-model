# Single Tank V2 — Gray-Box Groundwater Model

A physics-informed, gray-box model for simulating groundwater level (GWL) dynamics at monitoring stations across an alluvial fan. The model fits discrete-time ODEs to observed GWL using rainfall, upstream GWL, tidal/semi-diurnal signals, and a seasonal cycle as drivers.

## Model overview

Two ODE variants are implemented (both integrated with Euler at dt = 1 day):

**Inland stations**
```
h[t+1] = h[t] - a*(h[t] - z) + b*R_eff[t] - c*AMP[t] + k_link*(h_up_eff[t] - h[t])
        + d_sin*sin(2π*DOY/365.25) + d_cos*cos(2π*DOY/365.25)
```

**Coastal stations** (adds submarine groundwater discharge and semi-diurnal tidal terms)
```
h[t+1] = h[t] - a*(h[t] - z) + b*R_eff[t] - c*AMP[t] + k_link*(h_up_eff[t] - h[t])
        - k_sgd*(h[t] - h_sea) + gamma*AMT[t]
        + d_sin*sin(2π*DOY/365.25) + d_cos*cos(2π*DOY/365.25)
```

A **filtered-upstream** variant is also available for both groups, which passes `h_up` through an IIR smoother (`lambda`) before the k_link term.

Where:
- `a` — drainage/recession coefficient
- `z` — baseflow equilibrium level
- `b` — rainfall response coefficient
- `c` — diurnal tidal pumping coefficient
- `k_link` — upstream hydraulic connectivity
- `k_sgd` — submarine groundwater discharge coefficient (coastal only)
- `gamma` — semi-diurnal tidal response (coastal only)
- `h_sea` — effective sea level (coastal only)
- `d_sin`, `d_cos` — seasonal cycle amplitude (annual, decomposed into sine + cosine)
- `R_eff`, `h_up_eff` — exponentially weighted (convolved) rainfall and upstream GWL with time constants `tau_rain`, `tau_up` (convolution window = 90 days)
- `AMP` / `AMT` — diurnal (~1 cpd) and semi-diurnal (~1.93 cpd) tidal amplitudes extracted by STFT of hourly GWL

**Calibration** uses `scipy.optimize.differential_evolution` (global search) followed by a `curve_fit` polish step. The best of the two model variants (base vs filtered) is selected by AIC. Data before `SPLIT_DATE` (default 2019-01-01) is used for calibration; data from that date onward is reserved for out-of-sample validation.

## Pipeline overview

```
01_input_prepar.py
        ↓  data/raw/gw_stations_prefilter.csv
        ↓  data/raw/rf_stations_prefilter.csv
02_pairing.py
        ↓  data/gray_box_input_raw.csv    (permanent original, never overwritten)
        ↓  data/gray_box_input.csv        (working copy used by step 3)
        ↓  workspace/maps/upstream_links_initial.tiff
03_run_model.py                           (set active=1 in gray_box_input.csv first)
        ↓  workspace/results/per_station/{st_id}.csv  (one file per station)
        ↓  workspace/results/gw_fit_results.csv        (merged after pool finishes)
04_diag_pairing_search.py
        ↓  workspace/diagnostics/pairing_search_summary.csv
        ↓  workspace/diagnostics/pairing_search_trials.csv
        ↓  data/gray_box_input_optimized.csv
        ↓  workspace/maps/upstream_links_optimized.tiff
05_diag_report.py
        ↓  workspace/diagnostics/diagnostics_report_r2_below_0p5.csv

(optional re-run with optimized pairings)
03_run_model.py --input data/gray_box_input_optimized.csv
```

## Project structure

```
single_tankV2/
├── data/
│   ├── gray_box_input_raw.csv        # Original pairings from step 2 — never modified
│   ├── gray_box_input.csv            # Working copy used by step 3
│   ├── gray_box_input_optimized.csv  # Improved pairings from step 4
│   ├── gw_stations.csv               # Filtered groundwater station metadata
│   ├── gw_timeseries.csv             # Hourly groundwater level time series
│   ├── rf_stations.csv               # Rainfall station metadata
│   └── rf_timeseries.csv             # Daily rainfall time series
├── srcs/
│   ├── 01_input_prepar.py            # Data preparation and filtering
│   ├── 02_pairing.py                 # Upstream + rainfall pairing, coastal/inland classification
│   ├── 03_run_model.py               # Parallel calibration dispatcher
│   ├── 04_diag_pairing_search.py     # Pairing search for low-R² stations
│   ├── 05_diag_report.py             # Parameter-level diagnostics report
│   ├── 06_plot_maps.py               # Spatial R² performance maps
│   ├── gw_shell.py                   # Per-station pipeline: data prep, lag estimation, fitting, output
│   └── gw_subroutine.py              # ODE simulators (inland, coastal, filtered variants)
├── workspace/
│   ├── results/
│   │   ├── per_station/              # One CSV per station (written by gw_shell, merged by 03)
│   │   ├── gw_fit_results.csv        # Merged results from all stations
│   │   └── figures/                  # Per-station fit plots
│   ├── diagnostics/                  # Diagnostic CSVs and reports
│   └── maps/                         # Generated maps
├── pyproject.toml
└── poetry.toml
```

## Setup

Requires Python 3.10–3.14 and [Poetry](https://python-poetry.org/).

```bash
poetry install
eval $(poetry env activate)
```

## Usage

### Full pipeline

```bash
# Step 1 — prepare raw data
python srcs/01_input_prepar.py

# Step 2 — pair stations, classify coastal/inland
python srcs/02_pairing.py

# Step 3 — calibrate (set active=1 in gray_box_input.csv first)
python srcs/03_run_model.py

# Step 4 — search for better pairings for low-R² stations
python srcs/04_diag_pairing_search.py

# Step 5 — generate diagnostics report
python srcs/05_diag_report.py
```

### Resume an interrupted run

Stations whose `workspace/results/per_station/{st_id}.csv` already exists are skipped automatically:

```bash
python srcs/03_run_model.py          # resumes from where it stopped
python srcs/03_run_model.py --force  # re-runs all active stations regardless
```

### Re-run with optimized pairings

```bash
python srcs/03_run_model.py --input data/gray_box_input_optimized.csv
```

### Run a single station manually

```bash
python srcs/gw_shell.py \
  gw_st=9200211 st_id=st3 gw_x=169424.217 gw_y=2607200.343 \
  ups_id=st43 rf_id=rf17 lag_days=1 group_name=inland
```

### Diagnostic search options

```bash
python srcs/04_diag_pairing_search.py \
  --r2-threshold 0.5 \
  --top-k-rf 3 \
  --top-k-ups 3 \
  --passes 2 \
  --min-improvement 0.02 \
  --output-optimized data/gray_box_input_optimized.csv
```

## Calibration details

| Aspect | Implementation |
|---|---|
| Optimizer | `scipy.optimize.differential_evolution` (global) + `curve_fit` polish |
| Model selection | AIC — lower penalises extra parameters; selects base vs filtered variant |
| Temporal split | Calibration: data before `SPLIT_DATE` (default 2019-01-01); Validation: remainder |
| Outputs | `rmse`, `r2`, `rmse_val`, `r2_val`, `aic` + `{param}_std` from pcov |
| Convolution window | 90 days (covers >95% of exponential kernel for `tau` up to 30 d) |
| Parallelism | `multiprocessing.Pool` — one worker per active station; no subprocess overhead |

## Fit results columns (`gw_fit_results.csv`)

| Column | Description |
|---|---|
| `st_id`, `gw_st` | Station identifiers |
| `group_name` | `inland` or `coastal` |
| `model` | `base` or `filtered` (selected by AIC) |
| `rmse`, `r2` | Calibration-period metrics |
| `rmse_val`, `r2_val` | Validation-period metrics (out-of-sample, data ≥ `SPLIT_DATE`) |
| `aic` | Akaike Information Criterion for the selected model |
| `rain_lag_days`, `up_lag_days` | Estimated input lags |
| `a`, `z`, `b`, `c`, `k_link`, … | Fitted parameters (see model equations) |
| `a_std`, `b_std`, … | Parameter standard errors from `pcov` |

## Station configuration (`gray_box_input.csv`)

| Column | Description |
|---|---|
| `gw_st` | Official groundwater station number |
| `st_id` | Short station ID (matches column in `gw_timeseries.csv`) |
| `gw_TM_X97`, `gw_TM_Y97` | TWD97 coordinates |
| `ups_id` | Primary upstream station ID (`none` if no upstream) |
| `ups_candidates` | Comma-separated top-3 geographically-plausible upstream candidates (used by step 4) |
| `rf_id` | Paired rainfall station ID |
| `lag_days` | Rainfall lag hint (days); actual lag re-estimated during calibration |
| `spearmanr` | Spearman r between station and its upstream |
| `correlation` | Strong / Medium / Low / Insufficient Data / No Upstream |
| `quality` | RF pairing quality: Strong / Medium / Weak / None |
| `group` | `inland` or `coastal` |
| `active` | `1` to include in run, `0` to skip |

## Pairing strategy

**Step 2 — initial pairing (geographic + correlation)**

Upstream candidates are first filtered geographically (next eastward X-band, ~6.6 km wide, plus one additional band), then ranked by Spearman r. The highest-correlated candidate becomes `ups_id`; the top 3 are stored in `ups_candidates`. Nearest-neighbour is the fallback when no time-series overlap exists.

Rainfall stations are paired by scanning all RF stations within 25 km and selecting the one with the highest lagged Pearson correlation against the daily GWL difference series.

**Step 4 — model-driven re-pairing (low-R² stations)**

A coordinate-descent search tests alternative `rf_id` and `ups_id` combinations using actual model RMSE as the objective. The `ups_candidates` list from step 2 constrains the search pool to physically plausible options. Improvements above `--min-improvement` (default Δ R²=0.02) are applied to `gray_box_input_optimized.csv`.

## Output maps

| File | Description |
|---|---|
| `workspace/maps/upstream_links_initial.tiff` | Upstream pairing from step 2 |
| `workspace/maps/upstream_links_optimized.tiff` | Upstream pairing after step 4 (comparison) |
| `workspace/maps/station_map.tiff` | All GW and rainfall stations |
| `workspace/maps/station_map_subplots.tiff` | GW and rainfall stations in separate panels |
| `workspace/maps/coastal_stations_map_wgs84.tiff` | Classified coastal stations |

## Dependencies

| Package | Purpose |
|---|---|
| numpy, scipy | Numerics, ODE integration, global optimisation |
| pandas | Time series handling |
| matplotlib | Plotting |
| scikit-learn | R², RMSE metrics |
| geopandas, shapely, pyproj | Spatial operations |
| numba | JIT acceleration |
| tqdm | Progress bars |
| pyshp | Shapefile I/O |
