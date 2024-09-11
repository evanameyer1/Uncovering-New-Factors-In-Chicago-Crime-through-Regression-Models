import pandas as pd
from scipy.spatial import cKDTree
import geopandas as gpd
import numpy as np
import os
from datetime import datetime

#### Reading in Files
def read_in(directory: str, target: list[str]) -> dict[str, pd.DataFrame]:
    """
    Reads CSV files from the specified directory and loads them into a dictionary.

    Args:
        directory (str): Path to the directory containing the data.
        target (list[str]): List of file names to be read.

    Returns:
        dict: A dictionary of DataFrames with file names (without extension) as keys.
    """
    data = {}
    for filename in os.listdir(directory):
        if filename in target:
            file_path = os.path.join(directory, filename)
            data[filename[:-4]] = pd.read_csv(file_path)
            print(f'{filename[:-4]} successfully read in')
    return data

# Define directory and target files to load
directory = '../../data/processed'
target = ['clean_vacant_buildings.csv', 'clean_bike_stations.csv', 'clean_bus_stops.csv', 'clean_police_stations.csv', 'clean_train_stations.csv']
data = read_in(directory, target)

# Load the clean_crime dataset and add a 'point' column for geometry analysis
clean_crime = pd.read_csv('../../data/processed/clean_crime.csv')
clean_crime['point'] = clean_crime.apply(lambda row: (row['lat'], row['long']), axis=1)

#### Proximity Analysis Function using cKDTree for Efficiency
def proximity_scan(df: pd.DataFrame, col_name: str, distances: list[float]) -> pd.DataFrame:
    """
    Conducts proximity analysis for a given dataset against crime data, finding the count of points within
    specified distances using cKDTree for efficiency.

    Args:
        df (pd.DataFrame): The dataset with lat/long columns.
        col_name (str): A label for the dataset (used in column names for proximity results).
        distances (list[float]): List of distances (in miles) to scan for proximity.

    Returns:
        pd.DataFrame: A DataFrame with crime data and proximity counts for each distance.
    """
    clean_crime_gdf = gpd.GeoDataFrame(clean_crime, geometry=gpd.points_from_xy(clean_crime.long, clean_crime.lat))
    df_gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.long, df.lat))

    df_coords = np.array(list(zip(df_gdf.geometry.x, df_gdf.geometry.y)))
    tree = cKDTree(df_coords)

    output = []
    distances_in_degrees = [d / 69 for d in distances]  # Convert distances from miles to degrees

    for crime_idx, crime_row in clean_crime_gdf.iterrows():
        crime_cnt = [0 for _ in range(len(distances))]
        crime_point = crime_row.geometry

        for idx, distance in enumerate(distances_in_degrees):
            indices = tree.query_ball_point([crime_point.x, crime_point.y], distance)
            crime_cnt[idx] = len(indices)

        output.append(crime_cnt)

    temp_df = pd.DataFrame(output, columns=[f'{col_name}_distance_{str(distance)}' for distance in distances])
    return temp_df

# Perform proximity analysis for multiple datasets
crime_with_police = proximity_scan(data['clean_police_stations'], 'police_stations', [0.1, 0.3, 0.5, 1, 3, 5])
crime_with_bikes = proximity_scan(data['clean_bike_stations'], 'bike_stations', [0.1, 0.3, 0.5, 1, 3, 5])
crime_with_buses = proximity_scan(data['clean_bus_stops'], 'bus_stops', [0.1, 0.3, 0.5, 1, 3, 5])
crime_with_trains = proximity_scan(data['clean_train_stations'], 'train_stations', [0.1, 0.3, 0.5, 1, 3, 5])
crime_with_vacant_buildings = proximity_scan(data['clean_vacant_buildings'], 'vacant_buildings', [0.1, 0.3, 0.5, 1, 3, 5])

#### Time-Related Proximity Scan for 311 and Bike Rides
def patch_datetypes(data: list[pd.DataFrame]) -> None:
    """
    Patches date types in the given datasets, converting 'start_date' and 'end_date' columns to datetime format.

    Args:
        data (list[pd.DataFrame]): List of DataFrames to be patched.
    """
    for df in data:
        df['start_date'] = pd.to_datetime(df['start_date'])
        df['end_date'] = pd.to_datetime(df['end_date'])

# Load and patch date columns for 311-related datasets
clean_alleylights = pd.read_csv('../../data/processed/clean_alleylights.csv')
clean_streetlights_allout = pd.read_csv('../../data/processed/clean_streetlights_allout.csv')
clean_streetlights_oneout = pd.read_csv('../../data/processed/clean_streetlights_oneout.csv')
clean_crime['date'] = pd.to_datetime(clean_crime['date'])

patch_datetypes([clean_alleylights, clean_streetlights_allout, clean_streetlights_oneout])

#### Proximity Scan for Time-Sensitive Datasets (311 Datasets)
def proximity_scan_311(df: pd.DataFrame, col_name: str, distances: list[float]) -> pd.DataFrame:
    """
    Conducts proximity analysis for 311-related datasets, considering both time and proximity constraints.

    Args:
        df (pd.DataFrame): The 311 dataset with lat/long and time columns.
        col_name (str): A label for the dataset (used in column names for proximity results).
        distances (list[float]): List of distances (in miles) to scan for proximity.

    Returns:
        pd.DataFrame: A DataFrame with crime data and proximity counts for each distance.
    """
    clean_crime_gdf = gpd.GeoDataFrame(clean_crime, geometry=gpd.points_from_xy(clean_crime.long, clean_crime.lat))
    output = []
    distances_in_degrees = [d / 69 for d in distances]  # Convert distances from miles to degrees

    for crime_idx, crime_row in clean_crime_gdf.iterrows():
        crime_cnt = [0 for _ in range(len(distances))]
        crime_point = crime_row.geometry

        df_filtered = df[(df.start_date <= crime_row['date']) & ((df.end_date.isna()) | (df.end_date >= crime_row['date']))]
        df_gdf = gpd.GeoDataFrame(df_filtered, geometry=gpd.points_from_xy(df_filtered.long, df_filtered.lat))
        df_coords = np.array(list(zip(df_gdf.geometry.x, df_gdf.geometry.y)))
        tree = cKDTree(df_coords)

        for idx, distance in enumerate(distances_in_degrees):
            indices = tree.query_ball_point([crime_point.x, crime_point.y], distance)
            crime_cnt[idx] = len(indices)

        output.append(crime_cnt)

    temp_df = pd.DataFrame(output, columns=[f'{col_name}_distance_{str(distance)}' for distance in distances])
    return temp_df

# Proximity analysis for 311-related datasets
crime_with_alleylights = proximity_scan_311(clean_alleylights, 'alleylights', [0.1, 0.3, 0.5, 1, 3, 5])
crime_with_streetlights_allout = proximity_scan_311(clean_streetlights_allout, 'streetlights_allout', [0.1, 0.3, 0.5, 1, 3, 5])
crime_with_streetlights_oneout = proximity_scan_311(clean_streetlights_oneout, 'streetlights_oneout', [0.1, 0.3, 0.5, 1, 3, 5])

#### Bike Rides Proximity Scan with Binary Search and Haversine Distance Calculation
def print_timestamped_message(message: str) -> None:
    """Prints a message with a timestamp."""
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}")

def convert_to_numpy(df: pd.DataFrame) -> np.ndarray:
    """
    Converts a DataFrame with date, long, lat columns to a NumPy array with radians.

    Args:
        df (pd.DataFrame): DataFrame to convert.

    Returns:
        np.ndarray: Numpy array with cleaned date and converted coordinates.
    """
    base_time = datetime(2016, 1, 1, 0, 0, 0, 0)
    df['cleaned_date'] = (df['date'] - base_time).dt.total_seconds()
    df = df.sort_values(by='cleaned_date').reset_index(drop=True)
    df_np = df[['cleaned_date', 'long', 'lat']].to_numpy()
    df_np[:, 1] = np.radians(df_np[:, 1])
    df_np[:, 2] = np.radians(df_np[:, 2])
    return df_np

def binary_search_crime(arr: np.ndarray, crime_time: float, last_row_idx: list[int]) -> list[tuple[int, int]]:
    """
    Performs a binary search to find the time range of nearby bike rides within the 5, 10, and 15-minute windows.

    Args:
        arr (np.ndarray): Numpy array of bike ride data.
        crime_time (float): Time of the crime (in seconds).
        last_row_idx (list[int]): List of indices from the last row search.

    Returns:
        list[tuple[int, int]]: List of start and end indices for each time window.
    """
    times = [300, 600, 900]  # 5, 10, 15 minutes in seconds
    target_times = [(crime_time - t, crime_time + t) for t in times]
    final_idx = []

    def find_start_index(arr, start_time, last_row_idx):
        left, right = last_row_idx, len(arr) - 1
        while left <= right:
            mid = (left + right) // 2
            curr_time = arr[mid][0]
            if curr_time < start_time:
                left = mid + 1
            else:
                right = mid - 1
        return left

    def find_end_index(arr, end_time, last_row_idx):
        left, right = last_row_idx, len(arr) - 1
        while left <= right:
            mid = (left + right) // 2
            curr_time = arr[mid][0]
            if curr_time > end_time:
                right = mid - 1
            else:
                left = mid + 1
        return right

    for i, (start_time, end_time) in enumerate(target_times):
        start_idx = find_start_index(arr, start_time, last_row_idx[i])
        end_idx = find_end_index(arr, end_time, last_row_idx[i])
        
        if start_idx <= end_idx and start_idx < len(arr) and end_idx >= 0:
            final_idx.append((start_idx, end_idx))
        else:
            final_idx.append((-1, -1))  # if no valid index is found within the time range

    return final_idx

def compute_haversine(arr: np.ndarray, distances: list[float], target_pnt: tuple[float, float]) -> np.ndarray:
    """
    Computes the Haversine distance between crime points and bike ride points.

    Args:
        arr (np.ndarray): Array of bike ride data points.
        distances (list[float]): List of distance thresholds.
        target_pnt (tuple[float, float]): Target point (crime location).

    Returns:
        np.ndarray: Array with counts of bike rides within each distance threshold.
    """
    row_lat = arr[:, 2]
    row_lon = arr[:, 1]
    crime_lat, crime_lon = target_pnt[1], target_pnt[0]

    dlon = row_lon - crime_lon
    dlat = row_lat - crime_lat

    a = np.sin(dlat / 2)**2 + np.cos(crime_lat) * np.cos(row_lat) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 3956  # Radius of Earth in miles
    distances_in_miles = c * r

    cnt = np.zeros(len(distances))
    for i, distance in enumerate(distances):
        cnt[i] = np.sum(distances_in_miles <= distance)

    return cnt

def proximity_scan_bike_rides(df: pd.DataFrame, distances: list[float]) -> list[list[np.ndarray]]:
    """
    Conducts proximity analysis for bike rides relative to crime events within time windows and distance thresholds.

    Args:
        df (pd.DataFrame): DataFrame with bike ride data.
        distances (list[float]): List of distances to scan for proximity.

    Returns:
        list[list[np.ndarray]]: List of counts of nearby bike rides within each distance for each time window.
    """
    print_timestamped_message("Starting proximity_scan_bike_rides")
    crime_np = convert_to_numpy(clean_crime)
    df_np = convert_to_numpy(df)

    delta_cnts = [[] for _ in range(3)]

    for i, crime_row in enumerate(crime_np):
        if i % 1000 == 0: print_timestamped_message(f"Processing crime row {i} with time: {crime_row[0]}")
        last_row_idx = [0] * 3  # 5 min, 10 min, 15 min

        target_indicies = binary_search_crime(df_np, crime_row[0], last_row_idx)
        last_row_idx = [start_idx for (start_idx, end_idx) in target_indicies]

        for idx, target_tuple in enumerate(target_indicies):
            if target_tuple[0] != -1 and target_tuple[1] != -1:
                target_arr = df_np[target_tuple[0]:target_tuple[1]]
                crime_pnt = (crime_row[1], crime_row[2])
                counts = compute_haversine(target_arr, distances, crime_pnt)
            else:
                counts = [0, 0, 0]
            delta_cnts[idx].append(counts)

    print_timestamped_message("Completed proximity_scan_bike_rides")
    return delta_cnts

# Perform proximity scan for bike rides
clean_bike_trips = pd.read_csv('../../data/processed/clean_bike_trips.csv')
clean_bike_trips['date'] = pd.to_datetime(clean_bike_trips['date'])
crime_with_bike_trips_cnts = proximity_scan_bike_rides(clean_bike_trips, [0.1, 0.3, 0.5])

# Create a new DataFrame to store results for different time intervals and distances
crime_with_bike_trips = clean_crime[['id']]

for i, time in enumerate(['5_min', '10_min', '15_min']):
    for j, distance in enumerate([0.1, 0.3, 0.5]):
        bike_rides_counts = [counts[j] for counts in crime_with_bike_trips_cnts[i]]
        crime_with_bike_trips[f'bike_rides_within_{distance}_and_{time}'] = bike_rides_counts

#### Save Finalized Dataset
crime_with_proximity = pd.concat(
    [clean_crime, crime_with_police, crime_with_bikes, crime_with_buses, crime_with_trains,
     crime_with_alleylights, crime_with_streetlights_allout, crime_with_streetlights_oneout, crime_with_bike_trips],
    axis=1).drop('point', axis=1)

crime_with_proximity.to_csv('../../data/pre_training/crime_with_proximity.csv', index=False)