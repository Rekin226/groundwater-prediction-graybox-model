# Single Tank V2 — Gray-Box Groundwater Model

A physics-informed, gray-box model for simulating groundwater level (GWL) dynamics at monitoring stations across an alluvial fan. The model fits discrete-time ODEs to observed GWL using rainfall, upstream GWL, tidal/semi-diurnal signals, and a seasonal cycle as drivers. Four model variants are fit per station; the best is selected by validation-period KGE.

## Model overview

### 2×2 factorial design

Four variants are fit per station, spanning two orthogonal model-structure axes:

|  | **Constant z** | **Time-varying z(t)** |
|---|---|---|
| **Raw upstream driver** | `M1`: *Direct* (`base`) | `M3`: *Direct, z(t)* (`base_tz`) |
| **Low-pass filtered upstream** | `M2`: *Filtered* (`filtered`) | `M4`: *Filtered, z(t)* (`filtered_tz`) |

- **Constant-z** variants treat the baseflow equilibrium level `z` as a single scalar parameter.
- **z(t)** variants let it drift linearly in time: `z(t) = z0 + z1 · (t / 365.25)` with `z1` in m/year — captures multi-year non-stationarity (aquifer storage trend).
- **Filtered** variants pass the upstream GWL through a first-order IIR low-pass filter (EMA with rate `lambda`) before the `k_link` coupling term:
  `u[t+1] = (1 − λ)·u[t] + λ·h_up[t]` — damps local high-frequency noise in the upstream signal.
- **Direct** variants feed raw `h_up` into the coupling term.

### ODEs (Euler integration, dt = 1 day)

**Inland stations**
```
h[t+1] = h[t] − a·(h[t] − z_t) + b·R_eff[t] − c·AMP[t] + k_link·(h_up_eff[t] − h[t])
        + d_sin·sin(2π·DOY/365.25) + d_cos·cos(2π·DOY/365.25)
```

**Coastal stations** (adds submarine groundwater discharge and semi-diurnal tidal response)
```
h[t+1] = h[t] − a·(h[t] − z_t) + b·R_eff[t] − c·AMP[t] + k_link·(h_up_eff[t] − h[t])
        − k_sgd·(h[t] − h_sea) + gamma·AMT[t]
        + d_sin·sin(2π·DOY/365.25) + d_cos·cos(2π·DOY/365.25)
```

where `z_t = z` (constant-z) or `z_t = z0 + z1·(t/365.25)` (z(t) variants).

### Parameter glossary

- `a` — drainage / recession coefficient
- `z` (or `z0`, `z1`) — baseflow equilibrium level (constant, or linear-in-time trend)
- `b` — rainfall response coefficient
- `c` — diurnal tidal pumping coefficient
- `k_link` — upstream hydraulic connectivity
- `lambda` — low-pass filter rate on upstream driver (*Filtered* variants only)
- `k_sgd` — submarine groundwater discharge coefficient (coastal only)
- `gamma` — semi-diurnal tidal response (coastal only)
- `h_sea` — effective sea level (coastal only)
- `d_sin`, `d_cos` — seasonal cycle amplitude (annual, decomposed)
- `R_eff`, `h_up_eff` — exponentially weighted (convolved) rainfall and upstream GWL with time constants `tau_rain`, `tau_up` (convolution window = 90 days)
- `AMP` / `AMT` — diurnal (~1 cpd) and semi-diurnal (~1.93 cpd) tidal amplitudes extracted by STFT of hourly GWL

### Calibration

- **Optimizer**: `scipy.optimize.differential_evolution` (global) + `curve_fit` polish for `pcov`.
- **Temporal split**: calibration on data before `SPLIT_DATE` (default 2019-01-01); validation on data from that date onward.
- **Validation hindcast**: continuous across the split — `h0_val = y_fit_cal[-1]` for all variants (no cold-start at the seam).
- **Model selection**: best of the four variants by **`KGE_val`** (highest out-of-sample KGE). Falls back to AIC when validation data is unavailable.

## Pipeline overview

```
01_input_prepar.py
        ↓  data/raw/gw_stations_prefilter.csv
        ↓  data/raw/rf_stations_prefilter.csv
02_pairing.py
        ↓  data/gray_box_input_raw.csv    (permanent original, never overwritten)
        ↓  data/gray_box_input.csv        (working copy used by step 3)
        ↓  workspace/maps/upstream_links_initial.tiff
03_run_model.py                           (default --run_id initial;
                                           set active=1 in gray_box_input.csv first)
        ↓  workspace/results/<run_id>/per_station/{st_id}.csv    (one per station)
        ↓  workspace/results/<run_id>/all_variants/{st_id}.csv   (one row per variant)
        ↓  workspace/results/<run_id>/gw_fit_results.csv         (merged best-variant)
        ↓  workspace/results/<run_id>/all_variants_results.csv   (merged all-variants)
        ↓  workspace/results/<run_id>/figures/
               ├─ base/, filtered/, base_tz/, filtered_tz/       (per-variant plots)
               ├─ comparison/                                    (overlay + metrics table)
               └─ full_subplots/                                 (best variant + rainfall + AMP)
04_diag_pairing_search.py
        ↓  workspace/diagnostics/pairing_search_summary.csv
        ↓  workspace/diagnostics/pairing_search_trials.csv
        ↓  data/gray_box_input_optimized.csv
        ↓  workspace/maps/upstream_links_optimized.tiff
05_diag_report.py
        ↓  workspace/diagnostics/diagnostics_report_r2_below_0p5.csv
06_plot_maps.py
        ↓  workspace/maps/station_map.tiff, coastal_stations_map_wgs84.tiff, etc.

(optional rerun with optimized pairings)
03_run_model.py --input data/gray_box_input_optimized.csv --run_id final
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
│   ├── 03_run_model.py               # Parallel 4-variant calibration dispatcher
│   ├── 04_diag_pairing_search.py     # Pairing search for low-skill stations
│   ├── 05_diag_report.py             # Parameter-level diagnostics report
│   ├── 06_plot_maps.py               # Spatial performance maps
│   ├── gw_shell.py                   # Per-station pipeline: prep, fit 4 variants, plot, write CSVs
│   ├── gw_subroutine.py              # ODE simulators (8 variants: inland/coastal × base/filtered × const-z/z(t))
│   ├── jfft.py                       # STFT helper for tidal amplitude extraction
│   └── archived/                     # Superseded scaffolding (early TZ experiments)
├── scripts/
│   └── verify_cal_val_plots.py       # Two-station smoke test (st3 + st14) for plot changes
├── workspace/
│   ├── results/<run_id>/
│   │   ├── per_station/              # Best-variant CSV per station
│   │   ├── all_variants/             # Four-variant CSV per station (one row per variant)
│   │   ├── gw_fit_results.csv        # Merged best-variant results
│   │   ├── all_variants_results.csv  # Merged all-variants results
│   │   └── figures/                  # Per-variant, comparison, and full-subplot plots
│   ├── diagnostics/                  # Diagnostic CSVs and reports
│   └── maps/                         # Generated spatial maps
├── docs/
│   └── superpowers/
│       ├── specs/                    # Design documents
│       └── plans/                    # Implementation plans
├── manuscripts/                      # Paper drafts and manuscript figure scripts
├── pyproject.toml
└── poetry.toml
```

## Setup

Requires Python 3.10–3.14 and [Poetry](https://python-poetry.org/). The model code depends on the sibling package `rklib` (located at `../rklib/` relative to repo root).

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

# Step 3 — calibrate all four model variants (set active=1 in gray_box_input.csv first)
python srcs/03_run_model.py

# Step 4 — search for better pairings for weak-skill stations
python srcs/04_diag_pairing_search.py

# Step 5 — generate diagnostics report
python srcs/05_diag_report.py

# Step 6 — spatial maps
python srcs/06_plot_maps.py
```

### Resume an interrupted run

Stations whose per-station CSV already exists are skipped automatically:

```bash
python srcs/03_run_model.py            # resumes from where it stopped
python srcs/03_run_model.py --force    # re-runs all active stations regardless
```

### Re-run with optimized pairings

```bash
python srcs/03_run_model.py \
  --input data/gray_box_input_optimized.csv \
  --run_id final
```

### Subset of variants

```bash
# Only direct variants (skip filtered-upstream)
python srcs/03_run_model.py --models base,base_tz
```

### Run a single station manually

```bash
python srcs/gw_shell.py \
  gw_st=9200211 st_id=st3 gw_x=169424.217 gw_y=2607200.343 \
  ups_id=st43 rf_id=rf17 lag_days=1 group_name=inland
```

### Plot-change verification (st3 + st14)

```bash
PYTHONPATH=../:srcs python scripts/verify_cal_val_plots.py
```

Re-runs the pipeline on two representative stations (st3 = bad fit, st14 = good fit) to sanity-check plot changes before a full 61-station rerun.

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
| Variants fit per station | 4 (base, filtered, base_tz, filtered_tz) |
| Model selection | `KGE_val` (highest out-of-sample KGE); AIC fallback when no validation data |
| Temporal split | Calibration before `SPLIT_DATE` (default 2019-01-01); validation onward |
| Validation hindcast | Continuous: `h0_val = y_fit_cal[-1]` across all variants |
| Metrics recorded | `rmse`, `r2`, `kge`, `rmse_val`, `r2_val`, `kge_val` + KGE decomposition (`kge_r`, `kge_alpha`, `kge_beta`, `bias`) |
| Parameter uncertainty | Standard errors from `pcov` → `{param}_std` |
| Convolution window | 90 days (covers >95% of exponential kernel for `tau` up to 30 d) |
| Parallelism | `multiprocessing.Pool` — one worker per active station |

## Fit results columns (`gw_fit_results.csv`, best variant per station)

| Column | Description |
|---|---|
| `st_id`, `gw_st`, `ups_id`, `rf_id` | Station identifiers |
| `group_name` | `inland` or `coastal` |
| `rain_lag_days`, `up_lag_days` | Estimated input lags |
| `model` | Selected variant: `base` / `filtered` / `base_tz` / `filtered_tz` |
| `rmse`, `r2`, `kge` | Calibration-period metrics |
| `rmse_val`, `r2_val`, `kge_val` | Validation-period metrics |
| `kge_r_val`, `kge_alpha_val`, `kge_beta_val`, `bias_val` | KGE decomposition on validation period |
| `aic` | Akaike Information Criterion for the selected model |
| `a`, `z` (or `z0`, `z1`), `b`, `c`, `k_link`, `lambda`, `tau_rain`, `tau_up`, `d_sin`, `d_cos` | Fitted parameters |
| `k_sgd`, `gamma`, `h_sea` | Extra coastal parameters |
| `{param}_std` | Parameter standard errors from `pcov` |

`all_variants_results.csv` has the same columns but one row per (station × variant), for cross-variant comparisons.

## Station configuration (`gray_box_input.csv`)

| Column | Description |
|---|---|
| `gw_st` | Official groundwater station number |
| `st_id` | Short station ID (matches column in `gw_timeseries.csv`) |
| `gw_TM_X97`, `gw_TM_Y97` | TWD97 coordinates |
| `ups_id` | Primary upstream station ID (`none` if no upstream) |
| `ups_candidates` | Comma-separated top-3 geographically-plausible upstream candidates (step 4 search pool) |
| `spearmanr`, `correlation` | Upstream correlation strength + label |
| `ups_lag_days` | Estimated lag between station and upstream (days) |
| `ups_head_diff_m` | Mean head difference upstream − downstream (m) |
| `rf_id` | Paired rainfall station ID |
| `distance_m` | Distance to paired rainfall station (m) |
| `max_corr`, `lag_days` | Best lagged correlation with rainfall and corresponding lag |
| `quality` | RF pairing quality: Strong / Medium / Weak / None |
| `group` | `inland` or `coastal` |
| `active` | `1` to include in run, `0` to skip |

## Pairing strategy

**Step 2 — initial pairing (geographic + correlation)**

Upstream candidates are first filtered geographically (next eastward X-band, ~6.6 km wide, plus one additional band), then ranked by Spearman r. The highest-correlated candidate becomes `ups_id`; the top 3 are stored in `ups_candidates`. Nearest-neighbour is the fallback when no time-series overlap exists.

Rainfall stations are paired by scanning all RF stations within 25 km and selecting the one with the highest lagged Pearson correlation against the daily GWL difference series.

**Step 4 — model-driven re-pairing (weak-skill stations)**

A coordinate-descent search tests alternative `rf_id` and `ups_id` combinations using actual model RMSE as the objective. The `ups_candidates` list from step 2 constrains the search pool to physically plausible options. Improvements above `--min-improvement` (default Δ R² = 0.02) are applied to `gray_box_input_optimized.csv`.

## Plot outputs

For each station (`<run_id>/figures/...`):

| Path | Contents |
|---|---|
| `base/gw_fit_{st}.tiff`, `filtered/…`, `base_tz/…`, `filtered_tz/…` | One plot per variant: observed (black) + calibration (blue) + validation (red), continuous cal→val seam, KGE + RMSE annotation boxes |
| `comparison/gw_compare_{st}.tiff` | Four variants overlaid (cal solid, val dashed), per-variant colors, with a metrics table below listing `KGE_cal`, `KGE_val`, `RMSE_cal`, `RMSE_val` and asterisking the best by `KGE_val` |
| `full_subplots/gw_fit_{st}.tiff` | Three-panel composite: best-variant GWL, rainfall (rf station), tidal amplitude (AMP) — all across the full study period |

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
| geopandas, shapely, pyproj, pyshp | Spatial operations |
| numba | JIT acceleration |
| tqdm | Progress bars |
| adjusttext | Non-overlapping label placement on maps |
| python-docx | Manuscript export |
| rklib *(sibling)* | In-house publication-figure helpers (`setup_font`, `savefig`, `add_panel_label`) |
