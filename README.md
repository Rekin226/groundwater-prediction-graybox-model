# Groundwater Prediction Gray-Box Model

A physics-informed, gray-box model for simulating daily groundwater levels (GWL) at every well of a monitoring network in an alluvial fan. Each well is one storage tank whose head is updated by named terms (drainage toward an equilibrium head, rainfall response, coupling to an upstream well, diurnal and seasonal terms, and, at coastal wells, a coastal-boundary term) with coefficients calibrated from the observed record. Four structural variants are fitted per well and one is selected on an independent period; skill is reported on a held-out test period.

The repository also contains a land-subsidence pipeline that uses the groundwater model as a read-only library (see [Subsidence pipeline](#subsidence-pipeline)).

## Features

- Groundwater and rainfall data preparation, including detection of rainfall-gauge reporting outages
- Upstream-station pairing from Delaunay neighbours, the regional head gradient and lagged correlation
- Rainfall-gauge assignment by lagged correlation with daily head change
- Coastal vs. inland classification from shoreline distance and the M₂ tidal signal (amplitude ratio and signal-to-noise test)
- Four-variant gray-box calibration per station (differential evolution + bounded local refinement), run in parallel
- Three-period evaluation (calibration / selection / test) with leakage control: no test-period data enters any modelling decision
- Datum-invariant variant selection (KGE with a normalised bias term), plus NSE, RMSE and MAE in metres
- Analysis tools: benchmark models, sensitivity runs, multi-start optimisation, profile likelihoods, parameter correlations, bootstrap confidence intervals, interpolation cross-validation and drought-event analysis
- Diagnostics, per-station figures and spatial maps

## Model overview

### 2×2 factorial design

Four variants are fit per station, spanning two orthogonal model-structure axes:

|  | **Constant z** | **Time-varying z(t)** |
|---|---|---|
| **Direct upstream coupling** | `M1`: *Direct* (`base`) | `M3`: *Direct, z(t)* (`base_tz`) |
| **Low-pass filtered upstream** | `M2`: *Filtered* (`filtered`) | `M4`: *Filtered, z(t)* (`filtered_tz`) |

- **Constant-z** variants treat the equilibrium head `z` as a single scalar parameter.
- **z(t)** variants let it drift linearly in time: `z(t) = z0 + z1 · (t / 365.25)` with `z1` in m/year, for multi-year non-stationarity.
- **Filtered** variants pass the convolved upstream head through a first-order low-pass filter before the coupling term: `u[t+1] = (1 − λ)·u[t] + λ·h_up_eff[t]`. At `λ = 1` this equals direct coupling with a one-day delay.
- **Direct** variants feed `h_up_eff` straight into the coupling term.

### State update (Euler, dt = 1 day)

**Inland stations**
```
h[t+1] = h[t] − a·(h[t] − z_t) + b·R_eff[t] − c·AMP[t] + k_link·(s[t] − h[t])
        + d_sin·sin(2π·DOY/365.25) + d_cos·cos(2π·DOY/365.25)
```

**Coastal stations** add an effective coastal-boundary exchange and an M₂-envelope term:
```
        … − k_sgd·(h[t] − h_sea) + gamma·AMT[t]
```

where `z_t = z` (constant-z) or `z0 + z1·(t/365.25)` (z(t) variants), and `s = h_up_eff` (direct) or `u` (filtered). Every term is a head increment per daily step, not a volumetric flux. The exchange term shares its head dependence with the drainage term, so `k_sgd` and `h_sea` are effective boundary parameters rather than a discharge estimate.

### Parameter glossary

- `a`: drainage coefficient (d⁻¹)
- `z` (or `z0`, `z1`): equilibrium head (constant, or linear in time)
- `b`: rainfall-response gain (an empirical head response, not a recharge estimate)
- `c`: diurnal-amplitude coefficient
- `k_link`: effective upstream coupling coefficient
- `lambda`: low-pass filter rate on the upstream driver (*Filtered* variants only)
- `k_sgd`, `h_sea`: effective coastal-boundary exchange coefficient and boundary head (coastal only)
- `gamma`: M₂-envelope coefficient (coastal only)
- `d_sin`, `d_cos`: annual harmonic amplitudes
- `R_eff`, `h_up_eff`: rainfall and upstream head convolved with unit-sum exponential kernels (time constants `tau_rain`, `tau_up`; 90-day window). Because the kernels sum to 1, `tau` sets only the response shape and `b` / `k_link` carry the gain.
- `AMP` / `AMT`: diurnal (1 cpd) and semidiurnal (M₂, ~1.93 cpd) amplitudes from a short-time Fourier transform of the hourly record

### Calibration and evaluation

- **Optimiser**: `scipy.optimize.differential_evolution` (best1bin, Latin-hypercube initialisation, fixed seed) followed by a bounded trust-region-reflective refinement (`curve_fit` with bounds), kept only if it lowers the calibration RMSE.
- **Temporal split** (`CAL_END`, `SEL_END` in `srcs/gw_shell.py`): calibration 2012–2018, selection 2019–2020, test 2021–2022. Pairing, gauge assignment, lags, classification and parameter bounds use calibration data only; variant choice and pairing refinement use the selection period only; the test period is scored once.
- **Simulation**: one continuous run over the whole record with the calibrated parameters (no restart at period boundaries).
- **Variant selection**: highest selection-period `kge_ab`, a KGE whose bias term is normalised, `(μ_sim − μ_obs)/σ_obs`, so it does not depend on the vertical datum.
- **Leakage check**: `tests/test_gw_split_kernels.py` perturbs every test-period input and checks that pairing inputs, lags, parameters, calibration and selection metrics and the selected variant are unchanged.

## Pipeline overview

```
01_input_prepar.py
        ↓  data/raw/gw_stations_prefilter.csv, rf_stations_prefilter.csv
02_pairing.py
        ↓  data/gray_box_input_raw.csv    (permanent original, never overwritten)
        ↓  data/gray_box_input.csv        (working copy used by step 3)
        ↓  workspace/maps/upstream_links_initial.tiff
03_run_model.py                           (default --run_id initial;
                                           set active=1 in gray_box_input.csv first)
        ↓  workspace/results/<run_id>/per_station/{st_id}.csv
        ↓  workspace/results/<run_id>/all_variants/{st_id}.csv
        ↓  workspace/results/<run_id>/gw_fit_results.csv         (best variant per station)
        ↓  workspace/results/<run_id>/all_variants_results.csv
        ↓  workspace/results/<run_id>/series/{st_id}.csv          (daily obs + sim, all variants)
        ↓  workspace/results/<run_id>/figures/
04_diag_pairing_search.py                 (refinement on the selection period)
        ↓  data/gray_box_input_optimized.csv
05_diag_report.py
        ↓  workspace/diagnostics/
06_plot_maps.py, 07_boxplot_kge.py
        ↓  workspace/maps/, performance figures
scripts/analysis/                         (benchmarks, sensitivity, identifiability)
```

## Pairing and classification

**Upstream pairing (step 2, calibration period only).** The regional up-gradient direction comes from a planar trend surface fitted to the mean heads. Candidates are the station's natural neighbours in a Delaunay triangulation that lie up-gradient and whose mean head is at least 0.5 m higher (relaxed to 0.1 m, then the half-plane condition is dropped, where needed). They are ranked by the maximum Pearson correlation of daily heads at lags of 0–90 days. Stations with no admissible neighbour run without an upstream input and a near-zero `k_link` bound.

**Rainfall gauges.** Gauges with reporting outages (long runs of zero rain) are excluded. Each station gets the gauge within 25 km with the largest absolute correlation between daily rainfall and daily head change at lags of 0–45 days.

**Coastal classification.** A station is coastal when it lies within 5 km of the shoreline and its calibration-period hourly record carries an M₂ peak (1.9323 ± 0.05 cpd) of at least 10% of the largest spectral amplitude and at least three times the median background amplitude (1.6–2.4 cpd, M₂ and S₂ windows excluded).

**Refinement (step 4).** Stations with low calibration skill are re-tested against their next-best upstream and rainfall candidates; a change is kept only if it improves the selection-period KGE by at least 0.02.

## Analysis tools (`scripts/analysis/`)

| Script | Purpose |
|---|---|
| `benchmarks.py` | Persistence, climatology, seasonal-naive, AR(1) and a rainfall-only single tank |
| `make_sensitivity_inputs.py`, `run_sensitivity_batch.sh`, `sensitivity_summary.py` | Alternative upstream schemes, no upstream, no AMP/AMT, causal STFT window, coastal-structure runs |
| `classification_sensitivity.py` | Coastal set under alternative distance, amplitude-ratio and SNR thresholds |
| `datum_shift_selection.py` | Variant selection under vertical-datum shifts (original vs normalised-bias KGE) |
| `multistart_summary.py`, `optimizer_check.py` | Seed-to-seed parameter spread; stronger-search check |
| `profile_likelihood.py` | Profile likelihoods of selected parameters |
| `parameter_correlation.py` | Local parameter correlations from the calibration Jacobian |
| `bootstrap_ci.py` | Bootstrap confidence intervals and permutation tests for network medians |
| `rbf_loo.py` | Leave-one-station-out test of parameter interpolation |
| `drought_event.py` | Bias of the selected variants through a drought window |
| `data_completeness.py` | Held-constant (probable gap-filled) segments per station |

## Subsidence pipeline

Extends the GW model with a parallel land-subsidence pipeline that predicts cumulative vertical displacement ζ(t) at GNSS, MLCW and DBM stations in the Zhuoshui Alluvial Fan. Each station is driven by a hybrid h(t) (observed groundwater levels with GW-model gap-fill), so the two pipelines are decoupled: `srcs/` is imported as a read-only library and all subsidence code lives in `subsidence/`.

### Pipeline scripts

| Script | Purpose |
|---|---|
| `01_fetch_ls_data.py` | Fetch GNSS / MLCW / DBM observations from the WiseEnvr LS API |
| `02_fetch_drivers.py` | Fetch GW + rainfall driver data |
| `03_pair_sub_gw.py` | Co-located + nearest-neighbour-within-zone pairing |
| `04_assemble_h_driver.py` | Hybrid driver with cosine-tapered gap-fill |
| `05_run_subsidence.py` | Multiprocessing dispatcher; 4-variant fit per station |
| `06_diag_report.py` | Weak-station diagnostics |
| `07_plot_maps.py` | Spatial maps |

### Calibration protocol

- **Form 2** (Riley/IBS): elastic + inelastic compaction, available to all station types.
- **Form 3**: adds an aquitard drainage time constant τ, MLCW stations only.

|  | **Constant v\_tect** | **Linear v\_tect(t)** |
|---|---|---|
| **Form 2** | M1 | M2 |
| **Form 3 (MLCW only)** | M3 | M4 |

| Period | Dates |
|---|---|
| Calibration | 2020-01-01 → 2022-12-31 |
| Buffer (excluded) | 2023 |
| Validation | 2024-01-01 → 2025-03-31 |

### Required environment variables

Steps 01 and 04 call the live WiseEnvr LS API:

```bash
export LS_USER=your_username
export LS_PASS=your_password
```

### Make targets

| Target | Description |
|---|---|
| `make sub-fetch` | Steps 01 + 02: fetch LS observations and drivers (live API) |
| `make sub-pair` | Step 03: subsidence ↔ GW pairings |
| `make sub-driver` | Step 04: hybrid h_driver per station (live API) |
| `make sub-run` | Step 05: fit all active stations (default `SUB_RUN_ID=initial`) |
| `make sub-run-station SUB_STATION=TKJS` | Single-station fit |
| `make sub-compare` | Step 06: diagnostic report on weak stations |
| `make sub-maps` | Step 07: spatial maps |
| `make sub-help` | Subsidence-pipeline help |
| `make sub-clean` | Remove `workspace/results_sub/<SUB_RUN_ID>` |

## Repository structure

```
.
├── srcs/
│   ├── 01_input_prepar.py            # Data preparation and filtering
│   ├── 02_pairing.py                 # Upstream + rainfall pairing, coastal/inland classification
│   ├── 03_run_model.py               # Parallel 4-variant calibration dispatcher
│   ├── 04_diag_pairing_search.py     # Pairing refinement for low-skill stations
│   ├── 05_diag_report.py             # Parameter-level diagnostics report
│   ├── 06_plot_maps.py               # Spatial performance maps
│   ├── 07_boxplot_kge.py             # KGE distributions by variant and period
│   ├── gw_shell.py                   # Per-station pipeline: split, fit 4 variants, metrics, plots, CSVs
│   ├── gw_subroutine.py              # State-update simulators and kernels
│   ├── jfft.py                       # STFT helper for tidal amplitude extraction
│   └── archived/                     # Superseded scaffolding
├── scripts/
│   ├── analysis/                     # Benchmarks, sensitivity and identifiability analyses
│   ├── compare_runs.py               # Compare two result runs station by station
│   └── verify_cal_val_plots.py       # Two-station smoke test for plot changes
├── subsidence/                       # Land-subsidence pipeline
├── tests/                            # pytest: split, kernels and leakage checks
├── data/                             # Input data (not distributed; see below)
├── workspace/                        # Run outputs (git-ignored)
├── Makefile
├── pyproject.toml, poetry.toml, poetry.lock
└── LICENSE
```

## Setup

Requires Python 3.10–3.14 and [Poetry](https://python-poetry.org/). Figure styling uses the sibling package `rklib` (expected at `../rklib/` relative to the repository root).

```bash
poetry install
eval $(poetry env activate)
pytest tests/
```

## Usage

### Full pipeline

```bash
python srcs/01_input_prepar.py        # prepare raw data
python srcs/02_pairing.py             # pair stations, classify coastal/inland
python srcs/03_run_model.py           # calibrate four variants (set active=1 first)
python srcs/04_diag_pairing_search.py # refine pairings for weak-skill stations
python srcs/05_diag_report.py         # diagnostics report
python srcs/06_plot_maps.py           # spatial maps
```

`make help` lists the equivalent Make targets.

### Resume an interrupted run

Stations whose per-station CSV already exists are skipped:

```bash
python srcs/03_run_model.py            # resumes from where it stopped
python srcs/03_run_model.py --force    # re-runs all active stations
```

### Re-run with refined pairings

```bash
python srcs/03_run_model.py --input data/gray_box_input_optimized.csv --run_id final
```

### Subset of variants

```bash
python srcs/03_run_model.py --models base,base_tz   # direct variants only
```

### Run a single station manually

```bash
python srcs/gw_shell.py \
  gw_st=9200211 st_id=st3 gw_x=169424.217 gw_y=2607200.343 \
  ups_id=st43 rf_id=rf17 lag_days=1 group_name=inland
```

## Fit results columns (`gw_fit_results.csv`, best variant per station)

| Column | Description |
|---|---|
| `st_id`, `gw_st`, `ups_id`, `rf_id` | Station identifiers |
| `group_name` | `inland` or `coastal` |
| `rain_lag_days`, `up_lag_days` | Estimated input lags |
| `model`, `selection_basis` | Selected variant and the period it was chosen on (`sel`, or `cal` as fallback) |
| `n_cal`, `n_sel`, `n_test` | Daily records in each period |
| `{metric}_{cal,sel,test}` | `r2`/`nse`, `rmse`, `mae`, `bias`, classic `kge` (+ `kge_r`, `kge_alpha`, `kge_beta`), `kge_ab` (+ `kge_ab_beta`) |
| `{param}_at_bound` | Fitted value within 0.1% of the bound range at either end |
| `a`, `z` (or `z0`, `z1`), `b`, `c`, `k_link`, `lambda`, `tau_rain`, `tau_up`, `d_sin`, `d_cos` | Fitted parameters |
| `k_sgd`, `gamma`, `h_sea` | Coastal parameters |
| `{param}_std` | Parameter standard errors from `pcov` |

`all_variants_results.csv` has one row per (station × variant) with metrics, parameters and a `selected` flag.

## Station configuration (`gray_box_input.csv`)

| Column | Description |
|---|---|
| `gw_st`, `st_id` | Official station number and short ID (matches `gw_timeseries.csv`) |
| `gw_TM_X97`, `gw_TM_Y97` | TWD97 coordinates |
| `ups_id`, `ups_candidates` | Upstream station (`none` if absent) and ranked candidates for refinement |
| `ups_lag_days`, `ups_head_diff_m` | Upstream lag (days) and mean head difference (m) |
| `rf_id`, `distance_m`, `max_corr`, `lag_days` | Rainfall gauge, distance, best lagged correlation and lag |
| `group` | `inland` or `coastal` |
| `active` | `1` to include in a run, `0` to skip |

## Expected input data

The monitoring data are not distributed with this repository. The pipeline expects, under `data/`:

- `gw_stations.csv`, `gw_timeseries.csv`: groundwater-station metadata and hourly heads
- `rf_stations.csv`, `rf_timeseries.csv`: rainfall-gauge metadata and daily totals
- `Zhuoshui Alluvial Fan/Zhuoshui Alluvial Fan.shp`: study-area boundary (for clipping and maps)

## Dependencies

| Package | Purpose |
|---|---|
| numpy, scipy | Numerics, optimisation |
| pandas | Time-series handling |
| matplotlib | Plotting |
| scikit-learn | Metrics |
| geopandas, shapely, pyproj, pyshp | Spatial operations |
| numba | JIT acceleration |
| tqdm | Progress bars |
| adjusttext | Non-overlapping map labels |
| rklib *(sibling)* | In-house figure helpers (`setup_font`, `savefig`, `add_panel_label`) |

## Research context

This repository is related to published work on groundwater process analysis, reference-level estimation and signal-processing-based hydrogeological interpretation.

## Related publications

1. Hsu, S.M., Ouédraogo, A.R. & Chen, YW. **A data-driven approach to establishing groundwater reference levels through hydrogeological process analysis in central Taiwan.** *Hydrogeology Journal* **34**, 103–124 (2026). DOI: https://doi.org/10.1007/s10040-025-02992-2
2. Ouédraogo, A. R., Hsu, S. M., & Wang, Y. **Estimating the Average Magnitude of Pumping Surrounding Monitoring Wells Using Signal Processing.** *Journal of Hydrologic Engineering* **28**(4): 05023002 (2023). ASCE. DOI: https://doi.org/10.1061/JHYEFF.HEENG-5760

## Contributing

Contributions are welcome, especially improvements to documentation, reproducibility, code structure, testing, and example datasets or tutorials.

1. Fork the repository and create a feature branch.
2. Make your changes and run `pytest tests/`.
3. Open a pull request that says what changed, why, and whether it affects results or workflows.

## Limitations

- The repository contains **code only**; the monitoring datasets are not included.
- Some scripts use relative paths and assume execution from the repository root.
- GIS-dependent scripts require shapefiles and a working geospatial Python stack.
- Figure styling depends on the sibling `rklib` package.
- File naming reflects an active research workflow and may include legacy conventions.

## License

Released under the [MIT License](LICENSE).

## Author

Repository owner: [Rekin226](https://github.com/Rekin226)
