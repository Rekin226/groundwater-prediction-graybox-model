import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr, pearsonr
from scipy.signal import butter, filtfilt
from numpy.fft import fft, fftfreq
from tqdm import tqdm
import os
import sys
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
from pyproj import CRS, Transformer
from matplotlib.animation import FuncAnimation, PillowWriter

sys.path.insert(0, '/Users/rekin226/Desktop/Postdoc/code_space')
from rklib import StationMapFig, NorthArrow, ScaleBar, setup_font, savefig as rklib_savefig

TWD97_CRS = CRS.from_string("+proj=tmerc +lat_0=0 +lon_0=121 +k=0.9999 +x_0=250000 +y_0=0 +ellps=GRS80 +units=m +no_defs")
BOUNDARY_SHP = "../data/Zhuoshui Alluvial Fan/Zhuoshui Alluvial Fan.shp"
GW_META_IN = "../data/raw/gw_stations_prefilter.csv"
GW_DATA_IN = "../data/raw/gw_timeseries_raw.csv"
GW_META_OUT = "../data/gw_stations.csv"
GW_DATA_OUT = "../data/gw_timeseries.csv"
RF_META_IN = "../data/rf_stations.csv"
RF_DATA_IN = "../data/rf_timeseries.csv"

# Coastal / inland classification parameters
CUTOFF_CPD = 0.5
FS_CPD = 24.0
FILTER_ORDER = 5
M2_TARGET_CPD = 1.9323
M2_TOL_CPD = 0.05
M2_MIN_REL_AMP = 0.1
COASTAL_MAX_DIST_M = 5000.0


def filter_stations_within_boundary(df_gw_station, shp_path=BOUNDARY_SHP):
    print("Filtering stations within boundary...")
    if not os.path.exists(shp_path):
        print(f"Shapefile not found at {shp_path}. Skipping filtering.")
        return df_gw_station

    try:
        boundary_gdf = gpd.read_file(shp_path)
        if boundary_gdf.crs is None:
            boundary_gdf.set_crs(TWD97_CRS, inplace=True)
        else:
            boundary_gdf = boundary_gdf.to_crs(TWD97_CRS)

        geometry = [Point(xy) for xy in zip(df_gw_station['TM_X97'], df_gw_station['TM_Y97'])]
        gdf_stations = gpd.GeoDataFrame(df_gw_station, crs=TWD97_CRS, geometry=geometry)
        stations_within = gpd.sjoin(gdf_stations, boundary_gdf, how="inner", predicate="intersects")

        print(f"Stations before filtering: {len(df_gw_station)}")
        filtered = df_gw_station.loc[stations_within.index].reset_index(drop=True)
        print(f"Stations after filtering: {len(filtered)}")
        return filtered
    except Exception as exc:
        print(f"Error filtering stations with shapefile: {exc}")
        return df_gw_station


def find_matching_column(station_id, data_columns):
    if station_id in data_columns:
        return station_id
    if len(station_id) == 7:
        padded = station_id.zfill(8)
        if padded in data_columns:
            return padded
    if station_id.startswith('0') and station_id[1:] in data_columns:
        return station_id[1:]
    return None


def prepare_groundwater_data(
    gw_meta_path=GW_META_IN,
    gw_data_path=GW_DATA_IN,
    shp_path=BOUNDARY_SHP,
    gw_meta_out=GW_META_OUT,
    gw_data_out=GW_DATA_OUT
):
    print("\nStep 1: Data preparation")
    try:
        df_gw_station = pd.read_csv(gw_meta_path)
    except FileNotFoundError as exc:
        print(f"Error loading groundwater metadata: {exc}")
        return None, None

    df_gw_station = filter_stations_within_boundary(df_gw_station, shp_path)
    df_gw_station['Station'] = df_gw_station['Station'].astype(str)

    try:
        df_gw_data = pd.read_csv(gw_data_path, index_col=0, parse_dates=True)
    except FileNotFoundError as exc:
        print(f"Error loading groundwater data: {exc}")
        return None, None

    df_gw_data.columns = df_gw_data.columns.astype(str)

    data_columns = pd.Index(df_gw_data.columns)
    column_mapping = {}
    missing_stations = []
    for station in df_gw_station['Station']:
        match = find_matching_column(station, data_columns)
        if match:
            column_mapping[station] = match
        else:
            missing_stations.append(station)

    if missing_stations:
        print("Warning: No matching data columns for stations:", missing_stations)
        df_gw_station = df_gw_station[~df_gw_station['Station'].isin(missing_stations)].reset_index(drop=True)

    if df_gw_station.empty:
        print("No groundwater stations remain after filtering. Aborting.")
        return None, None

    df_gw_station['st_id'] = [f"st{i+1}" for i in range(len(df_gw_station))]

    ordered_station_ids = df_gw_station['Station'].tolist()
    selected_columns = [column_mapping[station] for station in ordered_station_ids]
    df_gw_data_filtered = df_gw_data[selected_columns].copy()
    rename_map = dict(zip(selected_columns, df_gw_station['st_id']))
    df_gw_data_filtered.rename(columns=rename_map, inplace=True)

    df_gw_station.to_csv(gw_meta_out, index=False)
    df_gw_data_filtered.to_csv(gw_data_out, index=True)
    print(f"Filtered groundwater metadata saved to {gw_meta_out}")
    print(f"Filtered groundwater data saved to {gw_data_out}")
    #print(df_gw_data_filtered)

    return df_gw_station, df_gw_data_filtered


def load_rainfall_data(rf_meta_path=RF_META_IN, rf_data_path=RF_DATA_IN):
    print("Loading rainfall metadata and data...")
    try:
        df_rain_station = pd.read_csv(rf_meta_path)
        df_rf_data = pd.read_csv(rf_data_path, index_col=0, parse_dates=True)
    except FileNotFoundError as exc:
        print(f"Error loading rainfall files: {exc}")
        return None, None

    df_rf_data.columns = df_rf_data.columns.astype(str)
    if not isinstance(df_rf_data.index, pd.DatetimeIndex):
        raise ValueError("Index of rf_data must be datetime for resampling")

    df_rf_data_daily = df_rf_data.resample('D').sum()
    return df_rain_station, df_rf_data_daily


def verify_rainfall_stations(df_rain_station, df_rf_data):
    if df_rain_station is None or df_rf_data is None:
        return

    rain_station_ids = df_rain_station['rf_id'].astype(str).str.strip()
    rain_data_cols = pd.Index(df_rf_data.columns).astype(str).str.strip()
    missing_in_data = sorted(set(rain_station_ids) - set(rain_data_cols))
    missing_in_metadata = sorted(set(rain_data_cols) - set(rain_station_ids))

    if missing_in_data or missing_in_metadata:
        if missing_in_data:
            print("Warning: Rain stations missing in rf_data:", missing_in_data)
        if missing_in_metadata:
            print("Warning: rf_data columns missing in metadata:", missing_in_metadata)
    else:
        print("Rainfall station IDs verified against rf_data columns.")

    matching_stations = sorted(set(rain_station_ids).intersection(set(rain_data_cols)))
    print(f"{len(matching_stations)} rainfall stations are present in both metadata and rf_data.")


def high_pass_filter(data: np.ndarray, cutoff: float, fs: float, order: int = 5) -> np.ndarray:
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype="high", analog=False)
    return filtfilt(b, a, data)


def identify_top_dominant_frequencies(signal: np.ndarray, top_n: int = 5) -> pd.DataFrame:
    n = len(signal)
    if n < 2:
        return pd.DataFrame(columns=["Frequency", "Amplitude"])

    fft_values = fft(signal)
    fft_values = 2.0 / n * np.abs(fft_values[: n // 2])
    freqs_cph = fftfreq(n, 1.0)[: n // 2]
    freqs_cpd = freqs_cph * 24.0

    power_spectrum = np.abs(fft_values) ** 2
    if power_spectrum.size == 0:
        return pd.DataFrame(columns=["Frequency", "Amplitude"])

    top_indices = np.argsort(power_spectrum)[-top_n:][::-1]
    top_freqs = freqs_cpd[top_indices]
    top_amplitudes = fft_values[top_indices]
    return pd.DataFrame({"Frequency": top_freqs, "Amplitude": top_amplitudes})


def find_m2_peak(df_top_freqs: pd.DataFrame):
    if df_top_freqs.empty:
        return None, None

    mask = (
        (df_top_freqs["Frequency"] >= M2_TARGET_CPD - M2_TOL_CPD)
        & (df_top_freqs["Frequency"] <= M2_TARGET_CPD + M2_TOL_CPD)
    )
    subset = df_top_freqs[mask]
    if subset.empty:
        return None, None

    idx = subset["Amplitude"].idxmax()
    row = subset.loc[idx]
    return float(row["Frequency"]), float(row["Amplitude"])


def build_coastline_geometry(data_dir: str):
    sea_path = os.path.join(data_dir, "water", "sea_TWD97.shp")
    fan_path = os.path.join(data_dir, "Zhuoshui Alluvial Fan", "Zhuoshui Alluvial Fan.shp")

    if not os.path.exists(sea_path) or not os.path.exists(fan_path):
        raise FileNotFoundError("Sea or alluvial fan shapefile missing for coastline computation")

    sea = gpd.read_file(sea_path)
    fan = gpd.read_file(fan_path)

    if sea.crs is None and fan.crs is not None:
        sea = sea.set_crs(fan.crs)
    elif fan.crs is None and sea.crs is not None:
        fan = fan.set_crs(sea.crs)

    if sea.crs != fan.crs:
        fan = fan.to_crs(sea.crs)

    coastline = gpd.overlay(sea, fan, how="intersection")
    if coastline.empty:
        raise RuntimeError("Coastline intersection is empty")

    return gpd.GeoSeries([coastline.unary_union], crs=sea.crs)


def compute_station_distances_to_coast(df_st: pd.DataFrame, coastline: gpd.GeoSeries) -> pd.Series:
    if df_st.empty:
        return pd.Series(dtype=float)

    gdf_st = gpd.GeoDataFrame(
        df_st.copy(),
        geometry=[Point(xy) for xy in zip(df_st["TM_X97"], df_st["TM_Y97"])],
        crs=coastline.crs,
    )
    return gdf_st.geometry.distance(coastline.iloc[0])


def _scan_lagged_correlation(dgw: pd.Series, rf: pd.Series, max_lag_days: int = 45, min_overlap_days: int = 100):
    """Scan lags to find best Pearson correlation, mirroring gw_rf_pairing logic."""
    best_corr = None
    best_lag = None
    best_p = None

    for lag in range(0, max_lag_days + 1):
        rf_shifted = rf.shift(lag) if lag > 0 else rf
        tmp = pd.concat([dgw, rf_shifted], axis=1).dropna()
        if len(tmp) < min_overlap_days:
            continue
        r, p = pearsonr(tmp.iloc[:, 0], tmp.iloc[:, 1])
        if best_corr is None or abs(r) > abs(best_corr):
            best_corr = r
            best_lag = lag
            best_p = p

    return best_corr, best_lag, best_p


def _quality_label(corr_value: float) -> str:
    if corr_value is None or np.isnan(corr_value):
        return 'None'
    mag = abs(corr_value)
    if mag >= 0.6:
        return 'Strong'
    if mag >= 0.4:
        return 'Medium'
    if mag >= 0.2:
        return 'Weak'
    return 'None'


def groundwater_rainfall_pairing(df_gw_station, df_gw_data_daily, df_rain_station, df_rf_data_daily):
    """Step 3: pair each GW station with best RF station, then attach to df_input later."""
    if df_gw_station is None or df_gw_data_daily is None or df_rain_station is None or df_rf_data_daily is None:
        return pd.DataFrame()

    # Ensure IDs are strings
    df_gw_station = df_gw_station.copy()
    df_rain_station = df_rain_station.copy()
    df_gw_station['Station'] = df_gw_station['Station'].astype(str)
    df_rain_station['rf_id'] = df_rain_station['rf_id'].astype(str)

    gw_coords = df_gw_station.set_index('Station')[['TM_X97', 'TM_Y97']]
    rf_coords = df_rain_station.set_index('rf_id')[['TM_X97', 'TM_Y97']]

    results = []

    for _, gw_row in df_gw_station.iterrows():
        gw_id = str(gw_row['Station'])
        st_id = gw_row.get('st_id', np.nan)

        if st_id not in df_gw_data_daily.columns:
            continue

        gw_series = df_gw_data_daily[st_id]
        dgw = gw_series.diff().dropna()
        if dgw.empty:
            continue

        try:
            gw_x, gw_y = gw_coords.loc[gw_id]
        except KeyError:
            continue

        best_entry = None

        for _, rf_row in df_rain_station.iterrows():
            rf_id = str(rf_row['rf_id'])
            if rf_id not in df_rf_data_daily.columns:
                continue

            rf_series = df_rf_data_daily[rf_id]
            try:
                rf_x, rf_y = rf_coords.loc[rf_id]
            except KeyError:
                continue

            dist_m = float(np.hypot(rf_x - gw_x, rf_y - gw_y))
            if dist_m > 25_000.0:
                continue

            aligned = pd.concat([dgw, rf_series], axis=1, join='inner').dropna()
            if aligned.empty:
                continue

            corr, lag, p_val = _scan_lagged_correlation(aligned.iloc[:, 0], aligned.iloc[:, 1])
            if corr is None:
                continue

            entry = {
                'GW_station': gw_id,
                'GW_st_id': st_id,
                'GW_TM_X97': gw_x,
                'GW_TM_Y97': gw_y,
                'Best_RF_id': rf_id,
                'Distance_m': dist_m,
                'Max_corr': corr,
                'Best_lag_days': lag,
                'p_value': p_val,
            }

            if best_entry is None or abs(corr) > abs(best_entry['Max_corr']):
                best_entry = entry

        if best_entry is None:
            results.append({
                'GW_station': gw_id,
                'GW_st_id': st_id,
                'Best_RF_id': np.nan,
                'Distance_m': np.nan,
                'Max_corr': np.nan,
                'Best_lag_days': np.nan,
                'p_value': np.nan,
                'Quality': 'None',
            })
            continue

        quality = _quality_label(best_entry['Max_corr'])
        if best_entry['p_value'] is not None and best_entry['p_value'] > 0.05:
            quality = 'None'

        best_entry['Quality'] = quality
        results.append(best_entry)

    return pd.DataFrame(results)


def classify_coastal_inland(df_gw_station, df_gw_data):
    print("\nStep 4: Coastal and inland classification")
    if df_gw_station is None or df_gw_data is None:
        print("Missing groundwater inputs for classification. Skipping Step 4.")
        return pd.DataFrame(), None

    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    try:
        coastline = build_coastline_geometry(data_dir)
    except Exception as exc:
        print(f"Unable to build coastline geometry: {exc}")
        return pd.DataFrame(), None

    df_meta = df_gw_station.copy()
    df_meta['Station'] = df_meta['Station'].astype(str)
    df_meta['dist_to_coast_m'] = compute_station_distances_to_coast(df_meta, coastline)
    df_meta['is_near_coast'] = df_meta['dist_to_coast_m'] <= COASTAL_MAX_DIST_M

    freq_records = []
    for _, row in df_meta.iterrows():
        st_id = row['st_id']
        station = row['Station']
        if st_id not in df_gw_data.columns:
            freq_records.append({
                'Station': station,
                'st_id': st_id,
                'dom_freq_cpd': np.nan,
                'dom_amp': np.nan,
                'm2_freq_cpd': np.nan,
                'm2_amp': np.nan,
                'is_m2_like': False,
            })
            continue

        series = df_gw_data[st_id].dropna()
        if len(series) < 10:
            freq_records.append({
                'Station': station,
                'st_id': st_id,
                'dom_freq_cpd': np.nan,
                'dom_amp': np.nan,
                'm2_freq_cpd': np.nan,
                'm2_amp': np.nan,
                'is_m2_like': False,
            })
            continue

        try:
            filtered = high_pass_filter(series.to_numpy(), CUTOFF_CPD, FS_CPD, FILTER_ORDER)
        except Exception as exc:
            print(f"High-pass filter failed for station {station}: {exc}")
            freq_records.append({
                'Station': station,
                'st_id': st_id,
                'dom_freq_cpd': np.nan,
                'dom_amp': np.nan,
                'm2_freq_cpd': np.nan,
                'm2_amp': np.nan,
                'is_m2_like': False,
            })
            continue

        df_top = identify_top_dominant_frequencies(filtered, top_n=10)
        if df_top.empty:
            freq_records.append({
                'Station': station,
                'st_id': st_id,
                'dom_freq_cpd': np.nan,
                'dom_amp': np.nan,
                'm2_freq_cpd': np.nan,
                'm2_amp': np.nan,
                'is_m2_like': False,
            })
            continue

        idx_max = df_top['Amplitude'].idxmax()
        dom_freq = float(df_top.loc[idx_max, 'Frequency'])
        dom_amp = float(df_top.loc[idx_max, 'Amplitude'])

        m2_freq, m2_amp = find_m2_peak(df_top)
        if m2_amp is None:
            is_m2_like = False
        else:
            max_amp = float(df_top['Amplitude'].max())
            rel_amp = m2_amp / max_amp if max_amp > 0 else 0.0
            is_m2_like = rel_amp >= M2_MIN_REL_AMP

        freq_records.append({
            'Station': station,
            'st_id': st_id,
            'dom_freq_cpd': dom_freq,
            'dom_amp': dom_amp,
            'm2_freq_cpd': m2_freq,
            'm2_amp': m2_amp,
            'is_m2_like': bool(is_m2_like),
        })

    df_freq = pd.DataFrame(freq_records)
    df_result = df_meta.merge(df_freq, on=['Station', 'st_id'], how='left')

    df_result['group'] = 'inland'
    mask_coastal = df_result['is_near_coast'] & df_result['is_m2_like']
    df_result.loc[mask_coastal, 'group'] = 'coastal'

    out_path = os.path.join(data_dir, 'intermediate', 'gw_coastal_inland_class.csv')
    df_result.to_csv(out_path, index=False)
    print(f"Coastal/inland classification saved to {out_path}")

    print("Group counts:")
    print(df_result['group'].value_counts())

    return df_result, coastline


def plot_coastal_stations(df_class, coastline):
    if df_class is None or df_class.empty or coastline is None:
        print("Skipping coastal station map (missing data).")
        return

    try:
        gdf_stations = gpd.GeoDataFrame(
            df_class.copy(),
            geometry=[Point(xy) for xy in zip(df_class['TM_X97'], df_class['TM_Y97'])],
            crs=coastline.crs,
        )

        gdf_coastal = gdf_stations[gdf_stations['group'] == 'coastal']

        fan_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data")),
                                "Zhuoshui Alluvial Fan", "Zhuoshui Alluvial Fan.shp")
        gdf_fan = gpd.read_file(fan_path)
        if gdf_fan.crs is None and coastline.crs is not None:
            gdf_fan = gdf_fan.set_crs(coastline.crs)
        elif gdf_fan.crs != coastline.crs:
            gdf_fan = gdf_fan.to_crs(coastline.crs)

        gdf_coastal_wgs84 = gdf_coastal.to_crs("EPSG:4326") if not gdf_coastal.empty else gdf_coastal
        x = gdf_coastal_wgs84.geometry.x.tolist()
        y = gdf_coastal_wgs84.geometry.y.tolist()

        setup_font()
        fig_obj = StationMapFig(
            x, y, fan_path,
            station_color="#c0392b",
            north_arrow=True,
            scale_bar=True,
            scale_km=5,
            figsize=(8, 8),
            dpi=300,
        )
        fig, ax = fig_obj.plot()
        ax.set_title("Coastal groundwater stations (WGS84)")

        out_map_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "workspace", "maps"))
        os.makedirs(out_map_dir, exist_ok=True)
        out_map_path = os.path.join(out_map_dir, "coastal_stations_map_wgs84.tiff")
        rklib_savefig(fig, out_map_path)
        plt.close(fig)
        print(f"Coastal stations map saved to {out_map_path}")
    except Exception as exc:
        print(f"Failed to plot coastal stations: {exc}")


def groundwater_upstream_pairing(df_gw_station, df_gw_data, df_rain_station, df_rf_data):
    print("\nStep 2: Groundwater upstream pairing")
    if df_gw_station is None or df_gw_data is None:
        print("Groundwater data missing. Skipping Step 2.")
        return None

    if not isinstance(df_gw_data.index, pd.DatetimeIndex):
        raise ValueError("Index of groundwater data must be datetime for resampling")

    df_gw_data_daily = df_gw_data.resample('D').mean()

    if df_rf_data is not None:
        if not isinstance(df_rf_data.index, pd.DatetimeIndex):
            raise ValueError("Index of rainfall data must be datetime for resampling")
        df_rf_data_daily = df_rf_data
        common_index = df_gw_data_daily.index.intersection(df_rf_data_daily.index)
        df_gw_data_daily = df_gw_data_daily.loc[common_index]
        df_rf_data_daily = df_rf_data_daily.loc[common_index]
        verify_rainfall_stations(df_rain_station, df_rf_data_daily)
    else:
        df_rf_data_daily = None

    min_x = 161250
    max_x = 220000
    step = 6600

    groups = []
    for x_start in range(min_x, max_x, step):
        mask = (df_gw_station['TM_X97'] >= x_start) & (df_gw_station['TM_X97'] < x_start + step)
        group = df_gw_station[mask].sort_values('TM_X97')
        if len(group) > 0:
            groups.append(group)

    print("Finding upstream stations...")
    groundwater_and_upstream = []
    for idx, group in enumerate(groups):
        if idx < len(groups) - 1:
            next_group = groups[idx + 1]
            coords1 = group[['TM_X97', 'TM_Y97']].values
            coords2 = next_group[['TM_X97', 'TM_Y97']].values
            dist_matrix = cdist(coords1, coords2)
            min_dist_idx = np.argmin(dist_matrix, axis=1)

            for j in range(len(group)):
                gw_station = group.iloc[j]
                up_station = next_group.iloc[min_dist_idx[j]]
                groundwater_and_upstream.append({
                    'gw_st': str(gw_station['Station']),
                    'gw_id': gw_station['st_id'],
                    'gw_TM_X97': gw_station['TM_X97'],
                    'gw_TM_Y97': gw_station['TM_Y97'],
                    'ups_station': str(up_station['Station']),
                    'ups_id': up_station['st_id'],
                    'ups_TM_X97': up_station['TM_X97'],
                    'ups_TM_Y97': up_station['TM_Y97']
                })
        else:
            for _, gw_station in group.iterrows():
                groundwater_and_upstream.append({
                    'gw_st': str(gw_station['Station']),
                    'gw_id': gw_station['st_id'],
                    'gw_TM_X97': gw_station['TM_X97'],
                    'gw_TM_Y97': gw_station['TM_Y97'],
                    'ups_station': 'none',
                    'ups_id': 'none',
                    'ups_TM_X97': np.nan,
                    'ups_TM_Y97': np.nan
                })

    df_input = pd.DataFrame(groundwater_and_upstream)
    if df_input.empty:
        print("No upstream relationships computed.")
        return df_input

    print("Calculating correlations...")
    df_input['spearmanr'] = np.nan
    df_input['correlation'] = ''

    for i in tqdm(range(len(df_input)), desc="Calculating correlation"):
        gw_id = df_input.loc[i, 'gw_id']
        up_id = df_input.loc[i, 'ups_id']

        if up_id == 'none':
            df_input.loc[i, 'correlation'] = 'No Upstream'
            continue

        if gw_id not in df_gw_data_daily.columns or up_id not in df_gw_data_daily.columns:
            df_input.loc[i, 'correlation'] = 'Missing Data'
            continue

        df_pair = df_gw_data_daily[[gw_id, up_id]].dropna()
        if len(df_pair) > 10:
            r, _ = spearmanr(df_pair.iloc[:, 0], df_pair.iloc[:, 1])
            r = round(r, 3)
            df_input.loc[i, 'spearmanr'] = r

            if abs(r) >= 0.7:
                df_input.loc[i, 'correlation'] = 'Strong'
            elif abs(r) >= 0.3:
                df_input.loc[i, 'correlation'] = 'Medium'
            else:
                df_input.loc[i, 'correlation'] = 'Low'
        else:
            df_input.loc[i, 'correlation'] = 'Insufficient Data'

    df_input['st_id'] = df_input['gw_id']

    # Keep upstream ID for plotting, but drop unused details
    df_input = df_input.drop(
        columns=['ups_station', 'ups_TM_X97', 'ups_TM_Y97'],
        errors='ignore',
    )

    desired_order = [
        'gw_st', 'st_id', 'gw_TM_X97', 'gw_TM_Y97',
        'ups_id', 'spearmanr', 'correlation'
    ]
    existing_cols = [col for col in desired_order if col in df_input.columns]
    df_input = df_input[existing_cols]

    return df_input, df_gw_data_daily, df_rf_data_daily


def generate_visualization(df_input, df_gw_station, df_rain_station, shp_path=BOUNDARY_SHP):
    print("Generating visualization...")
    if df_input is None or df_input.empty:
        print("No data available for visualization.")
        return

    try:
        setup_font()
        fig, ax = plt.subplots(figsize=(8, 10))

        transformer = Transformer.from_crs(TWD97_CRS, CRS.from_epsg(4326), always_xy=True)

        if os.path.exists(shp_path):
            boundary = gpd.read_file(shp_path).to_crs('EPSG:4326')
        else:
            print(f"Shapefile not found at {shp_path}. Skipping map background.")
            boundary = None

        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

        if boundary is not None:
            boundary.plot(ax=ax, edgecolor='black', facecolor='none')

        ax.grid(True, linestyle='--', alpha=0.4)

        for _, station in df_gw_station.iterrows():
            lon, lat = transformer.transform(station['TM_X97'], station['TM_Y97'])
            ax.scatter(lon, lat, color='blue', s=10)
            ax.annotate(station['st_id'], (lon, lat), fontsize=12,
                        xytext=(4, 4), textcoords='offset points', ha='left')

        # Draw upstream links using station coordinates from df_gw_station
        gw_coord_lookup = {
            str(s['st_id']): (s['TM_X97'], s['TM_Y97'])
            for _, s in df_gw_station.iterrows()
        }

        for _, row in df_input.iterrows():
            if row['ups_id'] == 'none':
                continue

            gw_id = str(row['st_id'])
            up_id = str(row['ups_id'])

            if gw_id not in gw_coord_lookup or up_id not in gw_coord_lookup:
                continue

            x1, y1 = gw_coord_lookup[gw_id]
            x2, y2 = gw_coord_lookup[up_id]
            lon1, lat1 = transformer.transform(x1, y1)
            lon2, lat2 = transformer.transform(x2, y2)
            ax.plot([lon1, lon2], [lat1, lat2], color='red', linewidth=0.8)

        blue_dot = plt.Line2D([], [], marker='o', color='blue', markersize=5, label='Observation Wells', linestyle='none')
        red_line = plt.Line2D([], [], color='red', linewidth=1, label='Upstream Links')
        handles = [blue_dot, red_line]
        if boundary is not None:
            black_line = plt.Line2D([], [], color='black', linewidth=1, label='Study Area Boundary')
            handles.append(black_line)
        ax.legend(handles=handles, loc='upper left')
        ax.set_title("Mapping Upstream Relations")

        NorthArrow(ax, location='lower right').draw()
        ScaleBar(ax, location='lower center', length_km=10).draw()

        map_out = '../workspace/maps/final_map_V3.tiff'
        rklib_savefig(fig, map_out)
        print(f"Final map saved to {map_out}")
    except Exception as exc:
        print(f"Error during visualization: {exc}")
        import traceback
        traceback.print_exc()


def plot_station_map(df_gw_station, df_rain_station, shp_path=BOUNDARY_SHP):
    """Plot a clean map showing groundwater observation wells and rainfall stations."""
    print("Generating station map (GW + rainfall)...")
    try:
        setup_font()
        fig, ax = plt.subplots(figsize=(8, 10))

        transformer = Transformer.from_crs(TWD97_CRS, CRS.from_epsg(4326), always_xy=True)

        if os.path.exists(shp_path):
            boundary = gpd.read_file(shp_path).to_crs('EPSG:4326')
        else:
            print(f"Shapefile not found at {shp_path}. Skipping map background.")
            boundary = None

        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

        if boundary is not None:
            boundary.plot(ax=ax, edgecolor='black', facecolor='none')

        ax.grid(True, linestyle='--', alpha=0.4)

        texts = []
        point_xs, point_ys = [], []

        for _, station in df_gw_station.iterrows():
            lon, lat = transformer.transform(station['TM_X97'], station['TM_Y97'])
            ax.scatter(lon, lat, color='blue', s=18, zorder=5)
            texts.append(ax.text(lon, lat, station['st_id'], fontsize=8, color='blue'))
            point_xs.append(lon)
            point_ys.append(lat)

        if df_rain_station is not None:
            for _, rf_station in df_rain_station.iterrows():
                rf_lon, rf_lat = transformer.transform(rf_station['TM_X97'], rf_station['TM_Y97'])
                rf_id = str(rf_station['rf_id']) if 'rf_id' in rf_station else ''
                ax.scatter(rf_lon, rf_lat, color='green', marker='^', s=22, zorder=5)
                texts.append(ax.text(rf_lon, rf_lat, rf_id, fontsize=8, color='green'))
                point_xs.append(rf_lon)
                point_ys.append(rf_lat)

        try:
            from adjustText import adjust_text
            adjust_text(texts, x=point_xs, y=point_ys, ax=ax,
                        arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
        except ImportError:
            print("adjustText not installed — labels may overlap. Run: pip install adjustText")

        blue_dot = plt.Line2D([], [], marker='o', color='blue', markersize=5, label='Observation Wells', linestyle='none')
        green_tri = plt.Line2D([], [], marker='^', color='green', markersize=5, label='Rainfall Stations', linestyle='none')
        handles = [blue_dot, green_tri]
        if boundary is not None:
            handles.append(plt.Line2D([], [], color='black', linewidth=1, label='Study Area Boundary'))
        ax.legend(handles=handles, loc='upper left')
        ax.set_title("Groundwater & Rainfall Stations")

        NorthArrow(ax, location='lower right').draw()
        ScaleBar(ax, location='lower center', length_km=10).draw()

        map_out = '../workspace/maps/station_map.tiff'
        rklib_savefig(fig, map_out)
        print(f"Station map saved to {map_out}")
    except Exception as exc:
        print(f"Error during station map: {exc}")
        import traceback
        traceback.print_exc()


def plot_station_map_subplots(df_gw_station, df_rain_station, shp_path=BOUNDARY_SHP):
    """Two-panel subplot map: (a) groundwater stations, (b) rainfall stations."""
    print("Generating subplot station map...")
    try:
        from adjustText import adjust_text
        has_adjust_text = True
    except ImportError:
        has_adjust_text = False
        print("adjustText not installed — labels may overlap. Run: pip install adjustText")

    try:
        setup_font()
        fig, axes = plt.subplots(1, 2, figsize=(16, 10))

        transformer = Transformer.from_crs(TWD97_CRS, CRS.from_epsg(4326), always_xy=True)

        if os.path.exists(shp_path):
            boundary = gpd.read_file(shp_path).to_crs('EPSG:4326')
        else:
            print(f"Shapefile not found at {shp_path}. Skipping map background.")
            boundary = None

        # --- (a) Groundwater stations ---
        ax_gw = axes[0]
        if boundary is not None:
            boundary.plot(ax=ax_gw, edgecolor='black', facecolor='none')
        ax_gw.grid(True, linestyle='--', alpha=0.4)
        ax_gw.set_xlabel('Longitude', fontsize=13, fontweight='bold')
        ax_gw.set_ylabel('Latitude', fontsize=13, fontweight='bold')
        ax_gw.set_title('Groundwater Stations', fontsize=14, fontweight='bold')
        ax_gw.text(0.02, 0.98, '(a)', transform=ax_gw.transAxes,
                   fontsize=13, fontweight='bold', va='top', ha='left')

        gw_texts, gw_xs, gw_ys = [], [], []
        for _, station in df_gw_station.iterrows():
            lon, lat = transformer.transform(station['TM_X97'], station['TM_Y97'])
            ax_gw.scatter(lon, lat, color='blue', s=18, zorder=5)
            gw_texts.append(ax_gw.text(lon, lat, station['st_id'], fontsize=12, color='blue'))
            gw_xs.append(lon)
            gw_ys.append(lat)

        if has_adjust_text:
            adjust_text(gw_texts, x=gw_xs, y=gw_ys, ax=ax_gw,
                        arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

        ax_gw.legend(handles=[plt.Line2D([], [], marker='o', color='blue', markersize=5,
                                          label='Observation Wells', linestyle='none')],
                     loc='upper right')
        NorthArrow(ax_gw, location='lower right').draw()
        ScaleBar(ax_gw, location='lower center', length_km=10).draw()

        # --- (b) Rainfall stations ---
        ax_rf = axes[1]
        if boundary is not None:
            boundary.plot(ax=ax_rf, edgecolor='black', facecolor='none')
        ax_rf.grid(True, linestyle='--', alpha=0.4)
        ax_rf.set_xlabel('Longitude', fontsize=13, fontweight='bold')
        ax_rf.set_ylabel('Latitude', fontsize=13, fontweight='bold')
        ax_rf.set_title('Rainfall Stations', fontsize=14, fontweight='bold')
        ax_rf.text(0.02, 0.98, '(b)', transform=ax_rf.transAxes,
                   fontsize=13, fontweight='bold', va='top', ha='left')

        if df_rain_station is not None:
            rf_texts, rf_xs, rf_ys = [], [], []
            for _, rf_station in df_rain_station.iterrows():
                rf_lon, rf_lat = transformer.transform(rf_station['TM_X97'], rf_station['TM_Y97'])
                rf_id = str(rf_station['rf_id']) if 'rf_id' in rf_station else ''
                ax_rf.scatter(rf_lon, rf_lat, color='green', marker='^', s=22, zorder=5)
                rf_texts.append(ax_rf.text(rf_lon, rf_lat, rf_id, fontsize=12, color='green'))
                rf_xs.append(rf_lon)
                rf_ys.append(rf_lat)

            if has_adjust_text:
                adjust_text(rf_texts, x=rf_xs, y=rf_ys, ax=ax_rf,
                            arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

        ax_rf.legend(handles=[plt.Line2D([], [], marker='^', color='green', markersize=5,
                                          label='Rainfall Stations', linestyle='none')],
                     loc='upper right')
        NorthArrow(ax_rf, location='lower right').draw()
        ScaleBar(ax_rf, location='lower center', length_km=10).draw()

        fig.tight_layout()
        map_out = '../workspace/maps/station_map_subplots.tiff'
        rklib_savefig(fig, map_out)
        print(f"Subplot station map saved to {map_out}")
    except Exception as exc:
        print(f"Error during subplot station map: {exc}")
        import traceback
        traceback.print_exc()


def animate_upstream_links(
    df_input,
    df_gw_station,
    df_rain_station,
    shp_path=BOUNDARY_SHP,
    out_path="../workspace/animation_V3.gif",
    interval_ms=300,
):
    print("Generating animated visualization...")
    if df_input is None or df_input.empty or df_gw_station is None or df_gw_station.empty:
        print("Missing inputs for animation.")
        return

    link_rows = df_input[
        (df_input['ups_id'] != 'none')
        & (df_input['correlation'].isin(['Strong', 'Medium']))
    ].copy()

    if link_rows.empty:
        print("No Strong/Medium upstream links available for animation.")
        return

    try:
        setup_font()
        fig, ax = plt.subplots(figsize=(8, 10))
        transformer = Transformer.from_crs(TWD97_CRS, CRS.from_epsg(4326), always_xy=True)

        if os.path.exists(shp_path):
            boundary = gpd.read_file(shp_path).to_crs('EPSG:4326')
        else:
            print(f"Shapefile not found at {shp_path}. Skipping map background.")
            boundary = None

        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.grid(True)
        ax.set_aspect('equal', adjustable='datalim')

        if boundary is not None:
            boundary.plot(ax=ax, edgecolor='black', facecolor='none')

        for _, station in df_gw_station.iterrows():
            lon, lat = transformer.transform(station['TM_X97'], station['TM_Y97'])
            ax.scatter(lon, lat, color='blue', s=10)
            ax.annotate(str(station['st_id']), (lon, lat), fontsize=6, xytext=(2, 2), textcoords='offset points')

        if df_rain_station is not None:
            for _, rf_station in df_rain_station.iterrows():
                rf_lon, rf_lat = transformer.transform(rf_station['TM_X97'], rf_station['TM_Y97'])
                rf_id = str(rf_station.get('rf_id', ''))
                ax.scatter(rf_lon, rf_lat, color='green', s=10)
                ax.annotate(rf_id, (rf_lon, rf_lat), fontsize=6, xytext=(2, 2), textcoords='offset points')

        gw_coord_lookup = {
            str(s['st_id']): (s['TM_X97'], s['TM_Y97'])
            for _, s in df_gw_station.iterrows()
            if pd.notna(s['st_id'])
        }

        line_artists = []
        for _, row in link_rows.iterrows():
            gw_id = str(row['st_id'])
            up_id = str(row['ups_id'])
            if gw_id not in gw_coord_lookup or up_id not in gw_coord_lookup:
                continue

            x1, y1 = gw_coord_lookup[gw_id]
            x2, y2 = gw_coord_lookup[up_id]
            lon1, lat1 = transformer.transform(x1, y1)
            lon2, lat2 = transformer.transform(x2, y2)
            (line,) = ax.plot([lon1, lon2], [lat1, lat2], color='red', linewidth=0.8, alpha=0.0)
            line_artists.append(line)

        if not line_artists:
            print("No valid upstream links to animate after filtering.")
            plt.close(fig)
            return

        blue_dot = plt.Line2D([], [], marker='o', color='blue', markersize=5, label='Observation Wells', linestyle='none')
        green_dot = plt.Line2D([], [], marker='o', color='green', markersize=5, label='Rainfall Stations', linestyle='none')
        red_line = plt.Line2D([], [], color='red', linewidth=1, label='Upstream Links')
        handles = [blue_dot, green_dot, red_line]
        if boundary is not None:
            black_line = plt.Line2D([], [], color='black', linewidth=1, label='Study Area Boundary')
            handles.append(black_line)
        ax.legend(handles=handles, loc='upper left')
        ax.set_title("Mapping Upstream Relations (Animated)")

        def update(frame):
            for idx, line in enumerate(line_artists):
                line.set_alpha(1.0 if idx <= frame else 0.0)
            return line_artists

        anim = FuncAnimation(
            fig,
            update,
            frames=len(line_artists),
            interval=interval_ms,
            blit=True,
            repeat=False,
        )

        out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "workspace"))
        os.makedirs(out_dir, exist_ok=True)
        gif_path = os.path.join(out_dir, os.path.basename(out_path))
        writer = PillowWriter(fps=max(1, int(1000 / interval_ms)))
        anim.save(gif_path, writer=writer)
        plt.close(fig)
        print(f"Animation saved to {gif_path}")
    except Exception as exc:
        print(f"Error during animation: {exc}")
        import traceback
        traceback.print_exc()


def main():
    df_gw_station, df_gw_data = prepare_groundwater_data()
    df_rain_station, df_rf_data = load_rainfall_data()
    df_input, df_gw_daily, df_rf_daily = groundwater_upstream_pairing(df_gw_station, df_gw_data, df_rain_station, df_rf_data)

    # Step 3: groundwater and rainfall pairing
    print("\nStep 3: Groundwater and rainfall pairing")
    df_gw_rf = groundwater_rainfall_pairing(df_gw_station, df_gw_daily, df_rain_station, df_rf_daily)

    if df_gw_rf is not None and not df_gw_rf.empty:
        df_gw_rf = df_gw_rf.rename(columns={
            'Best_RF_id': 'rf_id',
            'Distance_m': 'distance_m',
            'Max_corr': 'max_corr',
            'Best_lag_days': 'lag_days',
            'Quality': 'quality',
        })
        df_input = df_input.merge(
            df_gw_rf[['GW_st_id', 'rf_id', 'distance_m', 'max_corr', 'lag_days', 'quality']],
            left_on='st_id',
            right_on='GW_st_id',
            how='left',
        ).drop(columns=['GW_st_id'], errors='ignore')

    # Step 4: coastal vs inland classification
    df_class, coastline = classify_coastal_inland(df_gw_station, df_gw_data)
    if df_class is not None and not df_class.empty:
        df_input = df_input.merge(df_class[['st_id', 'group']], on='st_id', how='left')
    plot_coastal_stations(df_class, coastline)

    if df_input is None or df_input.empty:
        print("No results to save.")
        return

    # Sort rows by correlation class (Strong -> Low -> others) and Spearman magnitude
    correlation_order = ['Strong', 'Medium', 'Low', 'Insufficient Data', 'Missing Data', 'No Upstream']
    cat_type = pd.CategoricalDtype(correlation_order, ordered=True)
    if 'correlation' in df_input.columns:
        df_input['correlation'] = df_input['correlation'].astype(cat_type)
    if 'spearmanr' in df_input.columns:
        df_input = df_input.sort_values(by=['correlation', 'spearmanr'], ascending=[True, False])
    else:
        df_input = df_input.sort_values(by=['correlation'], ascending=True)

    # add a column active == 0 for all rows in df_input as last column
    df_input['active'] = 0

    output_path = '../data/gray_box_input.csv'
    df_input.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Results saved to {output_path}")

    print("\nSummary of Correlations:")
    print(df_input['correlation'].value_counts())

    print("\nFirst few rows of result:")
    print(df_input)

    generate_visualization(df_input, df_gw_station, df_rain_station)
    plot_station_map(df_gw_station, df_rain_station)
    plot_station_map_subplots(df_gw_station, df_rain_station)
    animate_upstream_links(df_input, df_gw_station, df_rain_station)


if __name__ == "__main__":
    main()
