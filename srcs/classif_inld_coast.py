import logging
import os
import sys

import numpy as np
import pandas as pd
from numpy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from adjustText import adjust_text

sys.path.insert(0, '/Users/rekin226/Desktop/Postdoc/code_space')
from rklib import StationMapFig, setup_font, savefig as rklib_savefig


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
	required_cols = {"Station", "st_id", "TM_X97", "TM_Y97"}
	missing_cols = required_cols - set(df_st.columns)
	if missing_cols:
		raise KeyError(f"Missing required columns in input_gw_st.csv: {missing_cols}")

	# Ensure Station is string; st_id is used to match time-series columns
	df_st["Station"] = df_st["Station"].astype(str).str.lstrip("0")
	df_st["st_id"] = df_st["st_id"].astype(str)

	# 2. Load groundwater time series
	logging.info("Reading groundwater time series from %s", gw_ts_path)
	df_gw = pd.read_csv(gw_ts_path, index_col=0, parse_dates=True)
	df_gw.columns = df_gw.columns.astype(str)

	# Keep only stations that have both metadata and time series (match on st_id)
	common_stations = sorted(set(df_st["st_id"]) & set(df_gw.columns))
	if not common_stations:
		raise RuntimeError("No overlapping stations between input_gw_st.csv and gw_data.csv")

	df_st = df_st[df_st["st_id"].isin(common_stations)].reset_index(drop=True)
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
					"st_id": station,
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
					"st_id": station,
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
					"st_id": station,
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
				"st_id": station,
				"dom_freq_cpd": dom_freq,
				"dom_amp": dom_amp,
				"m2_freq_cpd": m2_freq,
				"m2_amp": m2_amp,
				"is_m2_like": bool(is_m2_like),
			}
		)

	df_freq = pd.DataFrame.from_records(records)

	# 5. Merge and classify stations into coastal / inland
	df_result = df_st.merge(df_freq, on="st_id", how="left")

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

	# 7. Plot coastal and inland stations on the same map
	try:
		fan_path = os.path.join(data_dir, "Zhuoshui Alluvial Fan", "Zhuoshui Alluvial Fan.shp")
		gdf_stations = gpd.GeoDataFrame(
			df_result.copy(),
			geometry=[Point(xy) for xy in zip(df_result["TM_X97"], df_result["TM_Y97"])],
			crs=coastline.crs,
		)
		gdf_wgs84 = gdf_stations.to_crs("EPSG:4326")
		gdf_coastal = gdf_wgs84[gdf_wgs84["group"] == "coastal"]
		gdf_inland = gdf_wgs84[gdf_wgs84["group"] == "inland"]

		x_coastal = gdf_coastal.geometry.x.tolist()
		y_coastal = gdf_coastal.geometry.y.tolist()
		labels_coastal = gdf_coastal["st_id"].tolist()
		x_inland = gdf_inland.geometry.x.tolist()
		y_inland = gdf_inland.geometry.y.tolist()
		labels_inland = gdf_inland["st_id"].tolist()

		setup_font()
		# Use StationMapFig with coastal stations as the base map
		fig_obj = StationMapFig(
			x_coastal, y_coastal, fan_path,
			station_color="#c0392b",
			north_arrow=True,
			scale_bar=True,
			scale_km=10,
			figsize=(8, 8),
			dpi=300,
		)
		fig, ax = fig_obj.plot()

		# Overlay inland stations with a distinct marker and color
		sc_inland = ax.scatter(
			x_inland, y_inland,
			c="#2980b9", marker="^", edgecolors="black", s=35, zorder=5,
		)

		# Retrieve the coastal scatter collection added by StationMapFig
		sc_coastal = next(
			c for c in ax.collections if hasattr(c, 'get_offsets')
			and len(c.get_offsets()) == len(x_coastal)
		)

		# Build all labels for both groups, then adjust to avoid overlap
		all_x = x_coastal + x_inland
		all_y = y_coastal + y_inland
		all_labels = labels_coastal + labels_inland
		texts = [
			ax.text(xi, yi, str(lbl), fontsize=9, zorder=7)
			for xi, yi, lbl in zip(all_x, all_y, all_labels)
		]
		adjust_text(
			texts,
			x=all_x, y=all_y,
			ax=ax,
			add_objects=[sc_coastal, sc_inland],
			arrowprops=dict(arrowstyle='-', color='gray', lw=0.5),
			expand=(2.0, 2.5),
			force_text=(0.8, 1.0),
			force_points=(1.5, 2.0),
			force_objects=(1.5, 2.0),
			min_arrow_len=4,
		)

		# Legend
		legend_handles = [
			Line2D([0], [0], marker='o', color='w', markerfacecolor='#c0392b',
				   markeredgecolor='black', markersize=8, label='Coastal'),
			Line2D([0], [0], marker='^', color='w', markerfacecolor='#2980b9',
				   markeredgecolor='black', markersize=8, label='Inland'),
		]
		ax.legend(handles=legend_handles, loc='upper left', fontsize=10)

		# Title and axis labels: bold, fontsize 14
		ax.set_title("Classified Coastal and Inland Stations", fontsize=14, fontweight='bold')
		ax.set_xlabel("Longitude", fontsize=14, fontweight='bold')
		ax.set_ylabel("Latitude", fontsize=14, fontweight='bold')

		out_map_dir = os.path.join(base_dir, "workspace", "maps")
		os.makedirs(out_map_dir, exist_ok=True)
		out_map_path = os.path.join(out_map_dir, "coastal_inland_stations_map_wgs84.tiff")
		rklib_savefig(fig, out_map_path)
		plt.close(fig)
		logging.info("Coastal/inland classification map saved to %s", out_map_path)
	except Exception as e:
		logging.warning("Failed to create coastal/inland stations map: %s", e)


if __name__ == "__main__":
	main()

