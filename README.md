# Groundwater Prediction Gray-Box Model

A Python-based gray-box modeling workflow for groundwater level prediction, station pairing, and coastal/inland classification in the Zhuoshui Alluvial Fan region.

## Overview

This repository contains a groundwater modeling pipeline that combines:

- **Data preparation** for groundwater and rainfall stations
- **Groundwater–upstream station pairing**
- **Groundwater–rainfall pairing** using lagged correlation search
- **Coastal vs inland station classification** using spatial proximity and frequency-domain tidal signatures
- **Gray-box model calibration** for each groundwater station
- **Diagnostics and reporting** for low-performing fits

The code is written in Python and is organized around standalone scripts under `srcs/`.

## Main workflow

The project appears to support the following high-level workflow:

1. Prepare and clean groundwater and rainfall metadata/time series
2. Pair each groundwater station with:
   - an upstream groundwater station
   - a rainfall station
3. Classify stations into **inland** or **coastal** groups
4. Build `data/gray_box_input.csv`
5. Run the gray-box calibration driver for active stations
6. Save figures and model-fit summaries to `workspace/`
7. Run diagnostics on low-`R²` stations

## Repository structure

```text
.
├── data/                  # Input and generated data files (mostly not committed)
├── srcs/                  # Source scripts for preprocessing, modeling, and diagnostics
│   ├── classif_inld_coast.py
│   ├── corr_up_rf.py
│   ├── diagnostics_pairing_search.py
│   ├── diagnostics_report.py
│   ├── gray_box_driver.py
│   ├── gw_shell.py
│   ├── gw_subroutine.py
│   ├── input_prepar.py
│   ├── jfft.py
│   ├── makefile
│   └── test.py
└── workspace/             # Output plots, summaries, and diagnostics
```

## Key scripts

### `srcs/gray_box_driver.py`
Runs the project-level calibration workflow:

- loads `data/gray_box_input.csv`
- filters rows where `active == 1`
- groups stations by `group` (`inland` / `coastal`)
- launches `gw_shell.py` in parallel for each station

This is the main batch runner for model fitting.

### `srcs/gw_shell.py`
Per-station model fitting and result export.

Responsibilities include:

- loading groundwater and rainfall data
- computing daily groundwater series
- extracting tidal indicators (`amp`, `amt`) from hourly groundwater data using STFT
- estimating rainfall and upstream lags
- fitting two model variants:
  - `base`
  - `filtered`
- comparing model performance using RMSE and `R²`
- exporting:
  - plot images to `workspace/muli_model/`
  - best-fit summaries to `workspace/gw_fit_results.csv`
  - per-model comparisons to `workspace/gw_fit_model_compare.csv`

### `srcs/gw_subroutine.py`
Contains the gray-box groundwater model equations and wrappers used during calibration.

Implemented models include:

- inland model
- coastal model
- inland model with filtered upstream driver
- coastal model with filtered upstream driver

### `srcs/corr_up_rf.py`
Builds the input relationships needed for gray-box modeling.

Main tasks:

- filter groundwater stations inside the study boundary
- prepare groundwater metadata and time series
- load rainfall metadata and daily rainfall data
- identify upstream groundwater station links
- identify best rainfall station by lagged correlation
- classify stations as coastal/inland
- generate `data/gray_box_input.csv`
- create visual outputs in `workspace/`

### `srcs/classif_inld_coast.py`
Standalone coastal/inland classification workflow based on:

- distance to coastline
- spectral evidence of the M2 tidal component

### `srcs/input_prepar.py`
Earlier-stage data preparation utility for:

- groundwater metadata extraction from shapefiles
- groundwater time series filtering
- rainfall metadata preparation
- writing intermediate CSV inputs

### `srcs/diagnostics_report.py`
Generates a diagnostics CSV for stations with low model performance.

It checks for issues such as:

- parameters near optimization bounds
- tiny coefficients
- weak rainfall/upstream/tidal contributions
- suspicious lag values

### `srcs/diagnostics_pairing_search.py`
Searches for improved rainfall/upstream pairings for stations with poor `R²` values by re-evaluating candidate combinations.

### `srcs/test.py`
Simple helper script for filtering model-fit results below an `R²` threshold.

## Modeling concept

This project uses a **gray-box** approach, meaning it blends:

- **physics-inspired process structure** (groundwater balance / response equations)
- **data-driven parameter estimation** using nonlinear optimization

The implemented models account for combinations of:

- groundwater recession toward an equilibrium level
- rainfall recharge effects
- upstream groundwater influence
- tidal amplitude forcing
- for coastal sites:
  - tidal modulation (`AMT`)
  - sea-level interaction / submarine groundwater discharge term

Two model families are tested for each station:

1. **Base model**
2. **Filtered model** with a low-pass filtered upstream signal

The best model is selected using fit error metrics.

## Expected input data

The repository itself currently does not include the actual modeling datasets, but the scripts expect files under `data/` such as:

- `data/gray_box_input.csv`
- `data/gw_data2.csv`
- `data/rf_data.csv`
- `data/gw_meta2.csv`
- GIS/shapefile resources for:
  - groundwater stations
  - rainfall stations
  - coastline / sea polygons
  - Zhuoshui Alluvial Fan boundary

Some scripts also reference intermediate or source files such as:

- `data/input_gw_st.csv`
- `data/input_rf_st.csv`
- `data/gw_data.csv`
- `data/rf_meta.csv`
- `data/GIS/...`

Because the datasets are not committed, you will need to supply them locally before running the full workflow.

## Required columns

### `data/gray_box_input.csv`
The batch driver expects at least these columns:

- `gw_st`
- `st_id`
- `gw_TM_X97`
- `gw_TM_Y97`
- `ups_id`
- `rf_id`
- `lag_days`
- `group`
- `active`

Only rows with `active == 1` are executed by `gray_box_driver.py`.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Rekin226/groundwater-prediction-graybox-model.git
cd groundwater-prediction-graybox-model
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

The repository does not currently include a `requirements.txt` or `pyproject.toml`, so dependencies must be installed manually.

A likely starting set is:

```bash
pip install pandas numpy scipy scikit-learn matplotlib geopandas shapely pyproj pyshp tqdm numba
```

Depending on your environment and GIS stack, you may also need system libraries required by GeoPandas/Fiona/GDAL.

## Usage

### Generate gray-box input relationships

From the repository root:

```bash
python srcs/corr_up_rf.py
```

This step is intended to generate:

- groundwater metadata filtered to the study area
- station pairings
- coastal/inland classification
- `data/gray_box_input.csv`
- visualization outputs in `workspace/`

### Run gray-box calibration for active stations

```bash
python srcs/gray_box_driver.py
```

This will:

- read `data/gray_box_input.csv`
- run active stations in parallel
- call `srcs/gw_shell.py` for each station
- write results and figures into `workspace/`

### Run a single-station fit manually

Example command pattern:

```bash
python srcs/gw_shell.py st_id=st1 gw_st=station_id gw_x=0 gw_y=0 ups_id=none rf_id=RF001 lag_days=5 group_name=inland
```

Arguments are passed in `key=value` format.

### Generate diagnostics report

```bash
python srcs/diagnostics_report.py
```

### Search for better pairings on low-performing stations

```bash
python srcs/diagnostics_pairing_search.py
```

## Outputs

Typical outputs written to `workspace/` include:

- `gw_fit_results.csv`
- `gw_fit_model_compare.csv`
- `gw_fit_results_r2_below_0p5.csv`
- `diagnostics_report_r2_below_0p5.csv`
- `pairing_search_summary.csv`
- `pairing_search_trials.csv`
- station fit figures
- coastal station maps
- upstream-link maps / animations

## Notes and caveats

- This repository currently contains **code only**; core datasets are not included.
- Some scripts use relative paths and assume execution from specific working directories.
- GIS-dependent scripts require shapefiles and a working geospatial Python stack.
- There is no packaged CLI or dependency lockfile yet.
- File naming is partly legacy/in-progress, so some scripts may reflect evolving conventions.

## Recommended next improvements

If you continue developing this repository, useful additions would be:

- a `requirements.txt` or `environment.yml`
- a sample dataset or synthetic example
- a documented end-to-end pipeline example
- clearer separation between raw data, processed data, and outputs
- unit tests for model and preprocessing utilities
- argument parsing for all scripts

## License

No license file is currently present in the repository. If you want others to use or contribute to this project, consider adding a license such as MIT, BSD-3-Clause, or GPL.

## Author

Repository owner: [Rekin226](https://github.com/Rekin226)
