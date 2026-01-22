import logging
import os

import numpy as np
import pandas as pd
from numpy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


# ==========================
# Parameters (tunable)
# ==========================

# Sampling and filter parameters
CUTOFF_CPD = 0.5        # high-pass cutoff in cycles per day
FS_CPD = 24.0           # sampling frequency (hourly data -> 24 samples/day)
FILTER_ORDER = 5

# M2 target frequency and search window
M2_TARGET_CPD = 1.9323
M2_TOL_CPD = 0.05       # +/- window in cpd around M2

# Minimum relative amplitude for treating M2 peak as significant
M2_MIN_REL_AMP = 0.1    # relative to max amplitude in station spectrum

# Distance threshold to define "near coast" (in meters, TWD97 is metric)
COASTAL_MAX_DIST_M = 5000.0


# ==========================
# Signal processing helpers
# ==========================


def high_pass_filter(data: np.ndarray, cutoff: float, fs: float, order: int = 5) -> np.ndarray:
	"""High-pass Butterworth filter.

	Parameters
	----------
	data : np.ndarray
		1D signal array.
	cutoff : float
		Cutoff frequency in cycles per day.
	fs : float
		Sampling frequency in cycles per day.
	order : int
		Filter order.
	"""

	nyquist = 0.5 * fs
	normal_cutoff = cutoff / nyquist
	b, a = butter(order, normal_cutoff, btype="high", analog=False)
	return filtfilt(b, a, data)


def identify_top_dominant_frequencies(signal: np.ndarray, top_n: int = 5) -> pd.DataFrame:
	"""Return top N dominant frequencies of a signal (in cpd).

	Parameters
	----------
	signal : np.ndarray
		1D signal after filtering (NaNs removed).
	top_n : int
		Number of dominant frequencies to return.
	"""

	n = len(signal)
	if n < 2:
		return pd.DataFrame(columns=["Frequency", "Amplitude"])

	# Sampling interval in hours (1 hour), then convert to cycles per day
	T_hours = 1.0
	fft_values = fft(signal)
	fft_values = 2.0 / n * np.abs(fft_values[: n // 2])
	freqs_cph = fftfreq(n, T_hours)[: n // 2]  # cycles per hour
	freqs_cpd = freqs_cph * 24.0

	power_spectrum = np.abs(fft_values) ** 2
	if power_spectrum.size == 0:
		return pd.DataFrame(columns=["Frequency", "Amplitude"])

	top_indices = np.argsort(power_spectrum)[-top_n:][::-1]
	top_freqs = freqs_cpd[top_indices]
	top_amplitudes = fft_values[top_indices]
	return pd.DataFrame({"Frequency": top_freqs, "Amplitude": top_amplitudes})


def find_m2_peak(df_top_freqs: pd.DataFrame) -> tuple[float | None, float | None]:
	"""Find the strongest peak near M2 frequency.

	Returns
	-------
	(m2_freq, m2_amp) or (None, None) if nothing in window.
	"""

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


# ==========================
# Coastline / geometry
# ==========================


def build_coastline_geometry(data_dir: str) -> gpd.GeoSeries:
	"""Build coastline geometry as intersection between sea and alluvial fan.

	Parameters
	----------
	data_dir : str
		Path to the data directory (../data).
	"""

	sea_path = os.path.join(data_dir, "water", "sea_TWD97.shp")
	fan_path = os.path.join(data_dir, "Zhuoshui Alluvial Fan", "Zhuoshui Alluvial Fan.shp")

	if not os.path.exists(sea_path):
		raise FileNotFoundError(f"Sea shapefile not found: {sea_path}")
	if not os.path.exists(fan_path):
		raise FileNotFoundError(f"Alluvial fan shapefile not found: {fan_path}")

	sea = gpd.read_file(sea_path)
	fan = gpd.read_file(fan_path)

	# Ensure both are in the same CRS; if one CRS is missing, copy from the other.
	if sea.crs is None and fan.crs is not None:
		sea = sea.set_crs(fan.crs)
	elif fan.crs is None and sea.crs is not None:
		fan = fan.set_crs(sea.crs)

	if sea.crs != fan.crs:
		fan = fan.to_crs(sea.crs)

	logging.info("Computing intersection between sea and alluvial fan to obtain coastline...")
	coastline = gpd.overlay(sea, fan, how="intersection")
	if coastline.empty:
		raise RuntimeError("Intersection of sea and alluvial fan is empty; cannot derive coastline.")

	# Dissolve all geometries into a single (multi-)geometry for distance computation
	coastline_union = coastline.unary_union
	return gpd.GeoSeries([coastline_union], crs=sea.crs)


def compute_station_distances_to_coast(
	df_st: pd.DataFrame, coastline: gpd.GeoSeries
) -> pd.Series:
	"""Compute distance from each station to coastline geometry.

	Parameters
	----------
	df_st : DataFrame
		Must contain TM_X97, TM_Y97 columns.
	coastline : GeoSeries
		Single-element GeoSeries with coastline geometry in same CRS.
	"""

	if df_st.empty:
		return pd.Series(dtype=float)

	gdf_st = gpd.GeoDataFrame(
		df_st.copy(),
		geometry=[Point(xy) for xy in zip(df_st["TM_X97"], df_st["TM_Y97"])],
		crs=coastline.crs,
	)

	coast_geom = coastline.iloc[0]
	distances = gdf_st.geometry.distance(coast_geom)
	return distances


# ==========================
# Main workflow
# ==========================


def main() -> None:
	base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	data_dir = os.path.join(base_dir, "data")

	# 1. Load station metadata
	gw_meta_path = os.path.join(data_dir, "gw_meta2.csv")
	gw_ts_path = os.path.join(data_dir, "gw_data2.csv")

	logging.info("Reading groundwater station metadata from %s", gw_meta_path)
	df_st = pd.read_csv(gw_meta_path)

	# Expect at least: Station, st_id, TM_X97, TM_Y97
	required_cols = {"Station", "TM_X97", "TM_Y97"}
	missing_cols = required_cols - set(df_st.columns)
	if missing_cols:
		raise KeyError(f"Missing required columns in input_gw_st.csv: {missing_cols}")

	# Ensure Station is string and strip leading zeros for consistency
	df_st["Station"] = df_st["Station"].astype(str).str.lstrip("0")

	# 2. Load groundwater time series
	logging.info("Reading groundwater time series from %s", gw_ts_path)
	df_gw = pd.read_csv(gw_ts_path, index_col=0, parse_dates=True)

	# Align column naming with Station IDs (strip leading zeros)
	df_gw.columns = df_gw.columns.astype(str).str.lstrip("0")

	# Keep only stations that have both metadata and time series
	common_stations = sorted(set(df_st["Station"]) & set(df_gw.columns))
	if not common_stations:
		raise RuntimeError("No overlapping stations between input_gw_st.csv and gw_data.csv")

	df_st = df_st[df_st["Station"].isin(common_stations)].reset_index(drop=True)
	df_gw = df_gw[common_stations]

	logging.info("Number of stations with both metadata and time series: %d", len(common_stations))

	# 3. Build coastline geometry and compute distances
	coastline = build_coastline_geometry(data_dir)
	dist_to_coast = compute_station_distances_to_coast(df_st, coastline)
	df_st["dist_to_coast_m"] = dist_to_coast
	df_st["is_near_coast"] = df_st["dist_to_coast_m"] <= COASTAL_MAX_DIST_M

	logging.info(
		"Distance to coast: min=%.1f m, median=%.1f m, max=%.1f m",
		df_st["dist_to_coast_m"].min(),
		df_st["dist_to_coast_m"].median(),
		df_st["dist_to_coast_m"].max(),
	)

	# 4. Frequency analysis per station
	records = []
	for station in common_stations:
		series = df_gw[station].dropna()
		if len(series) < 10:
			logging.warning("Station %s: insufficient data for FFT (n=%d)", station, len(series))
			records.append(
				{
					"Station": station,
					"dom_freq_cpd": np.nan,
					"dom_amp": np.nan,
					"m2_freq_cpd": np.nan,
					"m2_amp": np.nan,
					"is_m2_like": False,
				}
			)
			continue

		data = series.to_numpy()

		try:
			filtered = high_pass_filter(data, CUTOFF_CPD, FS_CPD, FILTER_ORDER)
		except Exception as e:
			logging.warning("Station %s: high-pass filter failed (%s)", station, e)
			records.append(
				{
					"Station": station,
					"dom_freq_cpd": np.nan,
					"dom_amp": np.nan,
					"m2_freq_cpd": np.nan,
					"m2_amp": np.nan,
					"is_m2_like": False,
				}
			)
			continue

		df_top = identify_top_dominant_frequencies(filtered, top_n=10)
		if df_top.empty:
			logging.warning("Station %s: FFT returned no usable peaks", station)
			records.append(
				{
					"Station": station,
					"dom_freq_cpd": np.nan,
					"dom_amp": np.nan,
					"m2_freq_cpd": np.nan,
					"m2_amp": np.nan,
					"is_m2_like": False,
				}
			)
			continue

		# Dominant overall peak
		idx_max = df_top["Amplitude"].idxmax()
		dom_freq = float(df_top.loc[idx_max, "Frequency"])
		dom_amp = float(df_top.loc[idx_max, "Amplitude"])

		# M2 candidate peak
		m2_freq, m2_amp = find_m2_peak(df_top)

		if m2_amp is None:
			is_m2_like = False
		else:
			max_amp = float(df_top["Amplitude"].max())
			rel_amp = m2_amp / max_amp if max_amp > 0 else 0.0
			is_m2_like = rel_amp >= M2_MIN_REL_AMP

		records.append(
			{
				"Station": station,
				"dom_freq_cpd": dom_freq,
				"dom_amp": dom_amp,
				"m2_freq_cpd": m2_freq,
				"m2_amp": m2_amp,
				"is_m2_like": bool(is_m2_like),
			}
		)

	df_freq = pd.DataFrame.from_records(records)

	# 5. Merge and classify stations into coastal / inland
	df_result = df_st.merge(df_freq, on="Station", how="left")

	df_result["group"] = "inland"
	mask_coastal = df_result["is_near_coast"] & df_result["is_m2_like"]
	df_result.loc[mask_coastal, "group"] = "coastal"

	n_coastal = int((df_result["group"] == "coastal").sum())
	n_inland = int((df_result["group"] == "inland").sum())
	logging.info("Number of coastal stations: %d", n_coastal)
	logging.info("Number of inland stations: %d", n_inland)

	# 6. Save classification table
	out_path = os.path.join(data_dir, "gw_coastal_inland_class.csv")
	cols_order = [
		col
		for col in [
			"Station",
			"st_id" if "st_id" in df_result.columns else None,
			"NAME_C" if "NAME_C" in df_result.columns else None,
			"TM_X97",
			"TM_Y97",
			"dist_to_coast_m",
			"is_near_coast",
			"dom_freq_cpd",
			"dom_amp",
			"m2_freq_cpd",
			"m2_amp",
			"is_m2_like",
			"group",
		]
		if col is not None
	]

	df_result[cols_order].to_csv(out_path, index=False)
	logging.info("Coastal/inland classification written to %s", out_path)

	# 6b. Add coastal/inland group to gw_upstream_rain_pairs
	try:
		pairs_path = os.path.join(data_dir, "gw_upstream_rain_pairs.csv")
		if os.path.exists(pairs_path):
			df_pairs = pd.read_csv(pairs_path)
			group_map = dict(zip(df_result["Station"], df_result["group"]))
			# Map by groundwater_id (string) to group; if missing, leave NaN
			df_pairs["group"] = df_pairs["groundwater_id"].astype(str).str.lstrip("0").map(group_map)
			df_pairs.to_csv(pairs_path, index=False)
			logging.info("Added coastal/inland group column to gw_upstream_rain_pairs.csv")
		else:
			logging.warning("gw_upstream_rain_pairs.csv not found at %s; skipping group merge", pairs_path)
	except Exception as e:
		logging.warning("Failed to update gw_upstream_rain_pairs.csv with group column: %s", e)

	# 7. Plot coastal stations, study area, and coastline
	try:
		fan_path = os.path.join(data_dir, "Zhuoshui Alluvial Fan", "Zhuoshui Alluvial Fan.shp")
		gdf_fan = gpd.read_file(fan_path)
		# Ensure both fan and stations are in a projected CRS before converting to WGS84
		if gdf_fan.crs is None and coastline.crs is not None:
			gdf_fan = gdf_fan.set_crs(coastline.crs)
		elif gdf_fan.crs != coastline.crs:
			gdf_fan = gdf_fan.to_crs(coastline.crs)

		gdf_stations = gpd.GeoDataFrame(
			df_result.copy(),
			geometry=[Point(xy) for xy in zip(df_result["TM_X97"], df_result["TM_Y97"])],
			crs=coastline.crs,
		)

		gdf_coastal = gdf_stations[gdf_stations["group"] == "coastal"]

		# Reproject to WGS84 for plotting in lon/lat
		gdf_fan_wgs84 = gdf_fan.to_crs("EPSG:4326")
		gdf_coastal_wgs84 = gdf_coastal.to_crs("EPSG:4326") if not gdf_coastal.empty else gdf_coastal

		fig, ax = plt.subplots(figsize=(8, 8))

		gdf_fan_wgs84.plot(ax=ax, color="lightgrey", edgecolor="black", linewidth=0.5, label="Study area")

		if not gdf_coastal_wgs84.empty:
			gdf_coastal_wgs84.plot(ax=ax, color="red", markersize=20, marker="o", label="Coastal stations")

		ax.set_xlabel("Longitude")
		ax.set_ylabel("Latitude")
		ax.set_title("Coastal groundwater stations (WGS84)")

		handles, labels = ax.get_legend_handles_labels()
		if handles:
			ax.legend(loc="best")

		ax.set_aspect("equal")

		out_map_dir = os.path.join(base_dir, "workspace")
		os.makedirs(out_map_dir, exist_ok=True)
		out_map_path = os.path.join(out_map_dir, "coastal_stations_map_wgs84.png")
		plt.tight_layout()
		plt.savefig(out_map_path, dpi=300)
		plt.close(fig)
		logging.info("Coastal stations map saved to %s", out_map_path)
	except Exception as e:
		logging.warning("Failed to create coastal stations map: %s", e)


if __name__ == "__main__":
	main()

