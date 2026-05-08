# Groundwater Prediction Gray-Box Model

A Python-based gray-box modeling workflow for groundwater level prediction, station pairing, coastal/inland classification, and diagnostics in the Zhuoshui Alluvial Fan region of central Taiwan.

> [!NOTE]
> This repository is primarily a **research codebase**. It is designed to document and reproduce a groundwater modeling workflow, rather than provide a polished end-user software package.

## Features

- Groundwater and rainfall data preparation
- Upstream groundwater station pairing
- Rainfall station pairing using lagged correlation
- Coastal vs. inland station classification
- Gray-box groundwater model calibration
- Model comparison using RMSE and `R²`
- Diagnostics for low-performing stations
- Plot and report generation for analysis workflows

## Overview

This repository contains a groundwater modeling pipeline that combines:

- **Data preparation** for groundwater and rainfall stations
- **Groundwater–upstream station pairing**
- **Groundwater–rainfall pairing** using lagged correlation search
- **Coastal vs. inland station classification** using spatial proximity and frequency-domain tidal signatures
- **Gray-box model calibration** for each groundwater station
- **Diagnostics and reporting** for low-performing fits

The project is implemented in Python and organized around standalone scripts under `srcs/`.

## What this repository does

At a high level, the workflow:

1. prepares groundwater and rainfall data,
2. identifies likely upstream and rainfall drivers for each monitoring well,
3. classifies wells as inland or coastal,
4. builds a station-level modeling input table,
5. calibrates gray-box models for selected stations, and
6. exports fit summaries, comparisons, figures, and diagnostics.

## Research context

This repository is related to published work on groundwater process analysis, reference-level estimation, and signal-processing-based hydrogeological interpretation.

## Related publications

1. Hsu, S.M., Ouédraogo, A.R. & Chen, YW. **A data-driven approach to establishing groundwater reference levels through hydrogeological process analysis in central Taiwan.** *Hydrogeology Journal* **34**, 103–124 (2026). DOI: https://doi.org/10.1007/s10040-025-02992-2
2. Ouédraogo, A. R., Hsu, S. M., & Wang, Y. **Estimating the Average Magnitude of Pumping Surrounding Monitoring Wells Using Signal Processing.** *Journal of Hydrologic Engineering* **28**(4): 05023002 (2023). ASCE. DOI: https://doi.org/10.1061/JHYEFF.HEENG-5760

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

## Quick start

### 1. Clone the repository

```bash
git clone https://github.com/Rekin226/groundwater-prediction-graybox-model.git
cd groundwater-prediction-graybox-model
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

This repository does not currently include a `requirements.txt` or `pyproject.toml`, so dependencies must be installed manually.

```bash
pip install pandas numpy scipy scikit-learn matplotlib geopandas shapely pyproj pyshp tqdm numba
```

Depending on your environment, you may also need system libraries required by GeoPandas, Fiona, or GDAL.

### 4. Generate gray-box input relationships

```bash
python srcs/corr_up_rf.py
```

This step is intended to generate:

- groundwater metadata filtered to the study area,
- station pairings,
- coastal/inland classification,
- `data/gray_box_input.csv`,
- visualization outputs in `workspace/`.

### 5. Activate selected stations

Edit `data/gray_box_input.csv` and set:

- `active = 1` for stations you want to calibrate,
- `active = 0` for stations you want to skip.

### 6. Run gray-box calibration

```bash
python srcs/gray_box_driver.py
```

### 7. Run diagnostics

```bash
python srcs/diagnostics_report.py
```

Optional:

```bash
python srcs/diagnostics_pairing_search.py
```

## Main workflow

The repository supports the following high-level workflow:

1. Prepare and clean groundwater and rainfall metadata/time series.
2. Pair each groundwater station with:
   - an upstream groundwater station,
   - and a rainfall station.
3. Classify stations into **inland** or **coastal** groups.
4. Build `data/gray_box_input.csv`.
5. Run gray-box calibration for active stations.
6. Save figures and model-fit summaries to `workspace/`.
7. Run diagnostics on stations with low `R²`.

## Core scripts

### `srcs/corr_up_rf.py`
Builds the main station-level modeling input relationships.

Main tasks:

- filter groundwater stations inside the study boundary,
- prepare groundwater metadata and time series,
- load rainfall metadata and daily rainfall data,
- identify upstream groundwater station links,
- identify the best rainfall station by lagged correlation,
- classify stations as coastal or inland,
- generate `data/gray_box_input.csv`,
- create maps and other visual outputs in `workspace/`.

### `srcs/gray_box_driver.py`
Runs the batch calibration workflow.

It:

- loads `data/gray_box_input.csv`,
- filters rows where `active == 1`,
- groups stations by `group` (`inland` / `coastal`),
- launches `gw_shell.py` in parallel for each selected station.

### `srcs/gw_shell.py`
Per-station model fitting and result export.

Responsibilities include:

- loading groundwater and rainfall data,
- computing daily groundwater series,
- extracting tidal indicators (`amp`, `amt`) from hourly groundwater data using STFT,
- estimating rainfall and upstream lags,
- fitting two model variants:
  - `base`,
  - `filtered`,
- comparing model performance using RMSE and `R²`,
- exporting:
  - plot images to `workspace/muli_model/`,
  - best-fit summaries to `workspace/gw_fit_results.csv`,
  - per-model comparisons to `workspace/gw_fit_model_compare.csv`.

### `srcs/gw_subroutine.py`
Contains the gray-box groundwater model equations and wrappers used during calibration.

Implemented models include:

- inland model,
- coastal model,
- inland model with filtered upstream driver,
- coastal model with filtered upstream driver.

### `srcs/classif_inld_coast.py`
Standalone coastal/inland classification workflow based on:

- distance to coastline,
- spectral evidence of the M2 tidal component.

### `srcs/input_prepar.py`
Earlier-stage data preparation utility for:

- groundwater metadata extraction from shapefiles,
- groundwater time series filtering,
- rainfall metadata preparation,
- writing intermediate CSV inputs.

### `srcs/diagnostics_report.py`
Generates a diagnostics CSV for stations with low model performance.

It checks for issues such as:

- parameters near optimization bounds,
- tiny coefficients,
- weak rainfall, upstream, or tidal contributions,
- suspicious lag values.

### `srcs/diagnostics_pairing_search.py`
Searches for improved rainfall/upstream pairings for stations with poor `R²` values by re-evaluating candidate combinations.

### `srcs/test.py`
Simple helper script for filtering model-fit results below an `R²` threshold.

## Run a single-station fit manually

Example command pattern:

```bash
python srcs/gw_shell.py st_id=st1 gw_st=station_id gw_x=0 gw_y=0 ups_id=none rf_id=RF001 lag_days=5 group_name=inland
```

Arguments are passed in `key=value` format.

## Modeling concept

This project uses a **gray-box** approach, blending:

- **physics-inspired process structure** (groundwater balance / response equations), and
- **data-driven parameter estimation** using nonlinear optimization.

The implemented models account for combinations of:

- groundwater recession toward an equilibrium level,
- rainfall recharge effects,
- upstream groundwater influence,
- tidal amplitude forcing,
- and, for coastal sites:
  - tidal modulation (`AMT`),
  - sea-level interaction / submarine groundwater discharge terms.

Two model families are tested for each station:

1. **Base model**
2. **Filtered model** with a low-pass filtered upstream signal

The best model is selected using fit error metrics such as RMSE and `R²`.

## Expected input data

The repository currently does **not** include the full modeling datasets. The scripts expect files under `data/` such as:

- `data/gray_box_input.csv`
- `data/gw_data2.csv`
- `data/rf_data.csv`
- `data/gw_meta2.csv`

Several scripts also reference GIS and intermediate files such as:

- `data/input_gw_st.csv`
- `data/input_rf_st.csv`
- `data/gw_data.csv`
- `data/rf_meta.csv`
- `data/GIS/...`
- shapefiles for:
  - groundwater stations,
  - rainfall stations,
  - coastline / sea polygons,
  - Zhuoshui Alluvial Fan boundary.

Because these datasets are not committed, you must supply them locally before running the full workflow.

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

## Contributing

Contributions are welcome, especially improvements to:

- documentation,
- reproducibility,
- dependency management,
- code structure,
- testing,
- and example datasets or tutorials.

If you plan to contribute:

1. fork the repository,
2. create a feature branch,
3. make your changes,
4. open a pull request with a clear description.

Because this is a research-oriented repository, please describe:

- what changed,
- why it changed,
- and whether the change affects reproducibility, results, or workflows.

## Limitations

- This repository currently contains **code only**; core datasets are not included.
- Some scripts use relative paths and assume execution from specific working directories.
- GIS-dependent scripts require shapefiles and a working geospatial Python stack.
- There is no packaged CLI or pinned dependency file yet.
- File naming and script organization reflect an active research workflow and may include legacy conventions.

## Roadmap ideas

Useful additions to this repository would include:

- a `requirements.txt` or `environment.yml`,
- a small sample dataset or synthetic example,
- a documented end-to-end reproducible example,
- clearer separation between raw data, processed data, and outputs,
- unit tests for core preprocessing and model functions,
- consistent argument parsing for all scripts,
- a dedicated `How to cite` section.

## License

No license file is currently present in the repository. If you want others to use or contribute to this project, consider adding a license such as MIT, BSD-3-Clause, or GPL.

## Author

Repository owner: [Rekin226](https://github.com/Rekin226)
