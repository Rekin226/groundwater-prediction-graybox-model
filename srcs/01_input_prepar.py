"""
This script prepares input data for groundwater and rainfall analysis in the Zhuoshui Alluvial Fan region.
It processes groundwater station metadata from shapefiles, filters and cleans groundwater time series data,
filters stations within a specified boundary, and prepares rainfall station metadata and time series data.
The processed metadata and time series are saved as CSV files for further modeling or analysis.
"""

import os
import pandas as pd
import glob
import matplotlib.pyplot as plt
#from numpy.fft import fft, fftfreq
#from scipy.signal import butter, filtfilt
import shapefile as shp
import logging
import sys
import os
#import jfft

def process_shapefile(sf, zone):
    df_meta = pd.DataFrame(sf.records(), columns=[field[0] for field in sf.fields[1:]])
    df_meta = df_meta[df_meta['GW_ZONE'] == zone]
    df_meta = df_meta[['ST_NO', 'NAME_C', 'TM_X97', 'TM_Y97']]
    df_meta['ST_NO'] = df_meta['ST_NO'].str.lstrip('0')
    df_meta = df_meta[~df_meta['ST_NO'].str.startswith(('8', '1'))]
    return df_meta


csv_file_path = os.path.abspath("../data/gw_clean/gw_all_2012-2022.csv")
df_gw_st = pd.read_csv(csv_file_path)
# Convert the 'date time' column to datetime and set it as the index
df_gw_st['date time'] = pd.to_datetime(df_gw_st['date time'])
df_gw_st.set_index('date time', inplace=True)

# Start reading the data from '2012-01-01 00:00'
df_gw_st = df_gw_st.loc['2012-01-01 00:00':'2020-12-31 23:00']
# Remove columns with NaN values
df_gw_st = df_gw_st.dropna(axis=1)

# Drop ST_NO column with column head starting from 08 and 10
df_gw_st = df_gw_st.loc[:, ~df_gw_st.columns.str.startswith('08')]
df_gw_st = df_gw_st.loc[:, ~df_gw_st.columns.str.startswith('10')]

# remove the first 0 for all columns in df_gw_st
df_gw_st.columns = df_gw_st.columns.str.lstrip('0')
#print(df_gw_st.columns)

zone = '濁水溪沖積扇'
try:
    sf1 = shp.Reader('../data/GIS/gwobwell_e/gwobwell_e.shp')
except Exception as e:
    logging.error("Error reading shapefile gwobwell_e: %s", e)
    sys.exit(1)
try:
    sf2 = shp.Reader('../data/GIS/gwobwell_a/gwobwell_a.shp')
except Exception as e:
    logging.error("Error reading shapefile gwobwell_a: %s", e)
    sys.exit(1)

df_meta1 = process_shapefile(sf1, zone)
df_meta2 = process_shapefile(sf2, zone)
df_meta = pd.concat([df_meta1, df_meta2], ignore_index=True)
df_meta = df_meta[~df_meta['NAME_C'].str.contains(r'\(2\)|\(3\)|\(4\)|\(5\)', regex=True, na=False)]

df_input = pd.DataFrame({'ST_NO': df_meta['ST_NO'].unique()})
df_input['ST_NO'] = df_input['ST_NO'].astype(str).str.lstrip('0')
df_meta['ST_NO'] = df_meta['ST_NO'].astype(str).str.lstrip('0')
df_input = df_input.merge(df_meta, on='ST_NO', how='left')
missing = df_input['NAME_C'].isna()
if missing.any():
    logging.warning("Unmatched station metadata for stations: %s", df_input.loc[missing, 'ST_NO'].tolist())
df_input.rename(columns={'ST_NO': 'Station'}, inplace=True)

stations = df_input['Station'].tolist()
df_gw_st = df_gw_st[df_gw_st.columns.intersection(stations)]
df_input = df_input[df_input['Station'].isin(df_gw_st.columns)]

# add a columns st_id with st1, st2, ...
df_input = df_input.reset_index(drop=True)
df_input['st_id'] = ['st' + str(i+1) for i in range(len(df_input))]

# Filter the stations within the shapefile boundary
shp_path = "data/Zhuoshui Alluvial Fan/Zhuoshui Alluvial Fan.shp"
if os.path.exists(shp_path):
    try:
        sf_boundary = shp.Reader(shp_path)
        boundary_shape = sf_boundary.shapes()[0]
        boundary_points = boundary_shape.points
        boundary_polygon = plt.Polygon(boundary_points)

        def is_within_boundary(x, y, polygon):
            return polygon.get_path().contains_point((x, y))

        df_input['within_boundary'] = df_input.apply(lambda row: is_within_boundary(row['TM_X97'], row['TM_Y97'], boundary_polygon), axis=1)
        df_input = df_input[df_input['within_boundary']]
        df_input.drop(columns=['within_boundary'], inplace=True)
    except Exception as e:
        logging.error("Error processing boundary shapefile: %s", e)
else:
    logging.warning("Boundary shapefile not found: %s", shp_path)


print(df_input)

# save df_input to csv
df_input.to_csv("data/raw/gw_stations_prefilter.csv", index=False)


# print the list csv file in ../data/rf_clean folder

def list_rainfall_files(directory='../data/rf_clean'):
    try:
        return sorted([entry.path for entry in os.scandir(directory) if entry.is_file() and entry.name.endswith('.csv')])
    except FileNotFoundError:
        logging.error("Rainfall directory not found: %s", directory)
        return []

rf_files = list_rainfall_files()
print(f'Found {len(rf_files)} rainfall csv files in ../data/rf_clean folder.')

# a dataframe to store the rf data for all stations
df_rf_st = pd.DataFrame()
for file in sorted(rf_files):
    df_rf = pd.read_csv(file)
    # Convert the 'date time' column to datetime and set it as the index
    df_rf['Date Time'] = pd.to_datetime(df_rf['Date Time'])
    df_rf.set_index('Date Time', inplace=True)
    # Start reading the data from '2012-01-01 00:00'
    df_rf = df_rf.loc['2012-01-01 00:00':'2020-12-31 23:00']
    # Remove columns with NaN values
    df_rf = df_rf.dropna(axis=1)
    # remove the first 0 for all columns in df_rf
    df_rf.columns = df_rf.columns.str.lstrip('0')

    # Make columns unique per file using station code from filename
    station_code = os.path.basename(file).split('-')[0]  # e.g. C0K390
    df_rf.columns = [f"{station_code}_{c}" for c in df_rf.columns]

    # merge to df_rf_st
    if df_rf_st.empty:
        df_rf_st = df_rf
    else:
        # either join...
        df_rf_st = df_rf_st.join(df_rf, how='outer')
        # ...or concat:
        # df_rf_st = pd.concat([df_rf_st, df_rf], axis=1)

# Remove rf from each column name which in this format C0K390_rf
df_rf_st.columns = [col.replace('_rf', '') for col in df_rf_st.columns]
 # Rename the index to 'date time'
df_rf_st.index.name = 'date time'
# save df_rf_st to csv as rf_timeseries.csv (in raw/ as it requires manual curation before use)
#df_rf_st.to_csv("data/raw/rf_timeseries_raw.csv")

# read the rainfall-station-up2date.csv
df_rf_meta = pd.read_csv('../data/GIS/rainfall-station-up2date.csv')
# rename x, y to TM_X97, TM_Y97
df_rf_meta.rename(columns={'x': 'TM_X97', 'y': 'TM_Y97'}, inplace=True)
# replace the column name rf_st with rf_id
df_rf_meta.rename(columns={'rf_st': 'rf_id'}, inplace=True)

print(df_rf_meta)

# save df_rf_meta to csv
df_rf_meta.to_csv("data/raw/rf_stations_prefilter.csv", index=False)




