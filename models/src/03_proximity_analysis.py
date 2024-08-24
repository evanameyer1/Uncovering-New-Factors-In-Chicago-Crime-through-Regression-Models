import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
import numpy as np
import os
from datetime import datetime
from typing import List, Tuple, Dict

# Read in target CSV files from a directory and store them in a dictionary
def read_in(directory: str, target: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Read in target CSV files from a specified directory and store them in a dictionary.

    Args:
        directory (str): Directory containing the target CSV files.
        target (List[str]): List of filenames to read.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary with filenames (without extension) as keys and DataFrames as values.
    """
    data = {}
    for filename in os.listdir(directory):
        if filename in target:
            file_path = os.path.join(directory, filename)
            data[filename[:-4]] = pd.read_csv(file_path)
            print(f'{filename[:-4]} successfully read in')
    return data

# Conduct proximity analysis for a given DataFrame based on distances
def proximity_scan(df: pd.DataFrame, col_name: str, distances: List[float], clean_crime: pd.DataFrame) -> pd.DataFrame:
    """
    Perform proximity analysis for a given dataset based on specified distances.

    Args:
        df (pd.DataFrame): DataFrame containing locations (latitude, longitude).
        col_name (str): Column name prefix for the output columns.
        distances (List[float]): List of distances (in miles) to check proximity.
        clean_crime (pd.DataFrame): DataFrame containing crime data.

    Returns:
        pd.DataFrame: DataFrame with proximity counts for each distance.
    """
    clean_crime_gdf = gpd.GeoDataFrame(clean_crime, geometry=gpd.points_from_xy(clean_crime.long, clean_crime.lat))
    df_gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.long, df.lat))
    
    df_coords = np.array(list(zip(df_gdf.geometry.x, df_gdf.geometry.y)))
    tree = cKDTree(df_coords)
    
    output = []
    distances_in_degrees = [d / 69 for d in distances]  # Convert distances from miles to degrees

    for crime_row in clean_crime_gdf.itertuples():
        crime_cnt = [0 for _ in range(len(distances))]
        crime_point = crime_row.geometry
        
        for idx, distance in enumerate(distances_in_degrees):
            indices = tree.query_ball_point([crime_point.x, crime_point.y], distance)
            crime_cnt[idx] = len(indices)
        
        output.append(crime_cnt)
    
    return pd.DataFrame(output, columns=[f'{col_name}_distance_{str(distance)}' for distance in distances])

# Conduct proximity analysis for 311-related datasets considering time frames
def proximity_scan_311(df: pd.DataFrame, col_name: str, distances: List[float], clean_crime: pd.DataFrame) -> pd.DataFrame:
    """
    Perform proximity analysis for 311-related datasets with time frame considerations.

    Args:
        df (pd.DataFrame): DataFrame containing 311 service requests (latitude, longitude, start_date, end_date).
        col_name (str): Column name prefix for the output columns.
        distances (List[float]): List of distances (in miles) to check proximity.
        clean_crime (pd.DataFrame): DataFrame containing crime data.

    Returns:
        pd.DataFrame: DataFrame with proximity counts for each distance and time filter.
    """
    clean_crime_gdf = gpd.GeoDataFrame(clean_crime, geometry=gpd.points_from_xy(clean_crime.long, clean_crime.lat))
    
    output = []
    distances_in_degrees = [d / 69 for d in distances]  # Convert distances from miles to degrees

    for crime_row in clean_crime_gdf.itertuples():
        crime_cnt = [0 for _ in range(len(distances))]
        crime_point = crime_row.geometry
        
        df_filtered = df[(df.start_date <= crime_row.date) & ((df.end_date.isna()) | (df.end_date >= crime_row.date))]
        df_gdf = gpd.GeoDataFrame(df_filtered, geometry=gpd.points_from_xy(df_filtered.long, df_filtered.lat))
        df_coords = np.array(list(zip(df_gdf.geometry.x, df_gdf.geometry.y)))
        tree = cKDTree(df_coords)

        for idx, distance in enumerate(distances_in_degrees):
            indices = tree.query_ball_point([crime_point.x, crime_point.y], distance)
            crime_cnt[idx] = len(indices)
        
        output.append(crime_cnt)
    
    return pd.DataFrame(output, columns=[f'{col_name}_distance_{str(distance)}' for distance in distances])

# Convert 'start_date' and 'end_date' columns to datetime for all DataFrames in the list
def patch_datetypes(data: List[pd.DataFrame]) -> None:
    """
    Convert 'start_date' and 'end_date' columns to datetime for all DataFrames in the provided list.

    Args:
        data (List[pd.DataFrame]): List of DataFrames with 'start_date' and 'end_date' columns to convert.
    """
    for df in data:
        df['start_date'] = pd.to_datetime(df['start_date'])
        df['end_date'] = pd.to_datetime(df['end_date'])

# Print a message with a timestamp
def print_timestamped_message(message: str) -> None:
    """
    Print a message with the current timestamp.

    Args:
        message (str): The message to be printed.
    """
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}")

# Convert a DataFrame with date, lat, and long columns into a NumPy array
def convert_to_numpy(df: pd.DataFrame) -> np.ndarray:
    """
    Convert a DataFrame with date, latitude, and longitude columns into a NumPy array.

    Args:
        df (pd.DataFrame): DataFrame with 'date', 'lat', and 'long' columns.

    Returns:
        np.ndarray: Converted NumPy array with date in seconds since the base time, latitude, and longitude in radians.
    """
    base_time = datetime(2016, 1, 1, 0, 0, 0, 0)
    df['cleaned_date'] = (df['date'] - base_time).dt.total_seconds()
    df = df.sort_values(by='cleaned_date').reset_index(drop=True)
    df = df[['cleaned_date', 'long', 'lat']]
    
    df_np = df.to_numpy()
    df_np[:, 1] = np.radians(df_np[:, 1])
    df_np[:, 2] = np.radians(df_np[:, 2])

    return df_np

# Perform a binary search to find the start and end indices within a specified time range
def binary_search_crime(arr: np.ndarray, crime_time: float, last_row_idx: List[int]) -> List[Tuple[int, int]]:
    """
    Perform binary search to find the start and end indices of events within specified time ranges relative to a crime time.

    Args:
        arr (np.ndarray): NumPy array with event data (time, lat, long).
        crime_time (float): Crime time in seconds since the base time.
        last_row_idx (List[int]): List of last row indices used for starting the search.

    Returns:
        List[Tuple[int, int]]: List of tuples representing start and end indices for each time range.
    """
    times = [300, 600, 900]  # 5, 10, 15 minutes in seconds
    target_times = [(crime_time - t, crime_time + t) for t in times]
    final_idx = []

    def find_start_index(arr: np.ndarray, start_time: float, last_row_idx: int) -> int:
        left, right = last_row_idx, len(arr) - 1
        while left <= right:
            mid = (left + right) // 2
            curr_time = arr[mid][0]
            if curr_time < start_time:
                left = mid + 1
            else:
                right = mid - 1
        return left

    def find_end_index(arr: np.ndarray, end_time: float, last_row_idx: int) -> int:
        left, right = last_row_idx, len(arr) - 1
        while left <= right:
            mid = (left + right) // 2
            curr_time = arr[mid][0]
            if curr_time > end_time:
                right = mid - 1
            else:
                left = mid + 1
        return right

    for start_time, end_time in target_times:
        start_idx = find_start_index(arr, start_time, last_row_idx[0])
        end_idx = find_end_index(arr, end_time, last_row_idx[0])
        
        if start_idx <= end_idx and start_idx < len(arr) and end_idx >= 0:
            final_idx.append((start_idx, end_idx))
        else:
            final_idx.append((-1, -1))  # if no valid index is found within the time range

    return final_idx

# Compute the haversine distance between an array of points and a target point
def compute_haversine(arr: np.ndarray, distances: List[float], target_pnt: Tuple[float, float]) -> np.ndarray:
    """
    Compute the haversine distance between an array of points and a target point.

    Args:
        arr (np.ndarray): NumPy array with latitude and longitude columns (in radians).
        distances (List[float]): List of distance thresholds to compute proximity counts.
        target_pnt (Tuple[float, float]): Target point (longitude, latitude) to calculate distance from.

    Returns:
        np.ndarray: Array containing the number of points within each specified distance.
    """
    row_lat = arr[:, 2]
    row_lon = arr[:, 1]
    crime_lat, crime_lon = target_pnt[1], target_pnt[0]
    
    dlon = row_lon - crime_lon
    dlat = row_lat - crime_lat

    a = np.sin(dlat / 2)**2 + np.cos(crime_lat) * np.cos(row_lat) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 3956  # Radius of earth in miles
    distances_in_miles = c * r

    cnt = np.zeros(len(distances))
    for i, distance in enumerate(distances):
        cnt[i] = np.sum(distances_in_miles <= distance)

    return cnt

# Conduct proximity analysis for bike rides based on time intervals and distances
def proximity_scan_bike_rides(df: pd.DataFrame, distances: List[float], clean_crime: pd.DataFrame) -> List[np.ndarray]:
    """
    Perform proximity analysis for bike rides based on time intervals and distances.

    Args:
        df (pd.DataFrame): DataFrame containing bike ride data.
        distances (List[float]): List of distances (in miles) to check proximity.
        clean_crime (pd.DataFrame): DataFrame containing crime data.

    Returns:
        List[np.ndarray]: List of arrays containing proximity counts for each time interval and distance.
    """
    print_timestamped_message("Starting proximity_scan_bike_rides")
    crime_np = convert_to_numpy(clean_crime)
    df_np = convert_to_numpy(df)

    delta_cnts = [[] for _ in range(3)]

    for i, crime_row in enumerate(crime_np):
        if i % 1000 == 0: 
            print_timestamped_message(f"Processing crime row {i} with time: {crime_row[0]}")
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
        if i % 1000 == 0: 
            print_timestamped_message(f"Found the following bike ride counts for distance {distances[idx]} {counts}")
    
    print_timestamped_message("Completed proximity_scan_bike_rides")
    return delta_cnts

# Function to handle proximity scanning for static datasets
def run_static_proximity_scans(data: Dict[str, pd.DataFrame], clean_crime: pd.DataFrame) -> List[pd.DataFrame]:
    """
    Handle proximity scanning for static datasets (e.g., police stations, bike stations, bus stops).

    Args:
        data (Dict[str, pd.DataFrame]): Dictionary containing static datasets (e.g., police stations, bike stations).
        clean_crime (pd.DataFrame): DataFrame containing crime data.

    Returns:
        List[pd.DataFrame]: List of DataFrames with proximity counts for each static dataset.
    """
    crime_with_police = proximity_scan(data['clean_police_stations'], 'police_stations', [0.1, 0.3, 0.5, 1, 3, 5], clean_crime)
    crime_with_bikes = proximity_scan(data['clean_bike_stations'], 'bike_stations', [0.1, 0.3, 0.5, 1, 3, 5], clean_crime)
    crime_with_buses = proximity_scan(data['clean_bus_stops'], 'bus_stops', [0.1, 0.3, 0.5, 1, 3, 5], clean_crime)
    crime_with_trains = proximity_scan(data['clean_train_stations'], 'train_stations', [0.1, 0.3, 0.5, 1, 3, 5], clean_crime)
    crime_with_vacant_buildings = proximity_scan(data['clean_vacant_buildings'], 'vacant_buildings', [0.1, 0.3, 0.5, 1, 3, 5], clean_crime)

    return [
        crime_with_police, crime_with_bikes, crime_with_buses,
        crime_with_trains, crime_with_vacant_buildings
    ]

# Function to handle proximity scanning for time-related datasets
def run_time_related_scans(clean_crime: pd.DataFrame) -> List[pd.DataFrame]:
    """
    Handle proximity scanning for time-related datasets (e.g., alleylights, streetlights).

    Args:
        clean_crime (pd.DataFrame): DataFrame containing crime data.

    Returns:
        List[pd.DataFrame]: List of DataFrames with proximity counts for each time-related dataset.
    """
    clean_alleylights = pd.read_csv('../../data/processed/clean_alleylights.csv')
    clean_streetlights_allout = pd.read_csv('../../data/processed/clean_streetlights_allout.csv')
    clean_streetlights_oneout = pd.read_csv('../../data/processed/clean_streetlights_oneout.csv')
    
    patch_datetypes([clean_alleylights, clean_streetlights_allout, clean_streetlights_oneout])
    
    crime_with_alleylights = proximity_scan_311(clean_alleylights, 'alleylights', [0.1, 0.3, 0.5, 1, 3, 5], clean_crime)
    crime_with_streetlights_allout = proximity_scan_311(clean_streetlights_allout, 'streetlights_allout', [0.1, 0.3, 0.5, 1, 3, 5], clean_crime)
    crime_with_streetlights_oneout = proximity_scan_311(clean_streetlights_oneout, 'streetlights_oneout', [0.1, 0.3, 0.5, 1, 3, 5], clean_crime)

    return [
        crime_with_alleylights, crime_with_streetlights_allout, crime_with_streetlights_oneout
    ]

# Function to handle proximity scanning for bike rides
def run_bike_rides_scan(clean_crime: pd.DataFrame) -> pd.DataFrame:
    """
    Handle proximity scanning for bike rides based on time intervals and distances.

    Args:
        clean_crime (pd.DataFrame): DataFrame containing crime data.

    Returns:
        pd.DataFrame: DataFrame containing proximity counts for bike rides.
    """
    clean_bike_trips = pd.read_csv('../../data/processed/clean_bike_trips.csv')
    clean_bike_trips['date'] = pd.to_datetime(clean_bike_trips['date'])
    
    crime_with_bike_trips_cnts = proximity_scan_bike_rides(clean_bike_trips, [0.1, 0.3, 0.5], clean_crime)
    crime_with_bike_trips = clean_crime[['id']]
    
    for i, time in enumerate(['5_min', '10_min', '15_min']):
        for j, distance in enumerate([0.1, 0.3, 0.5]):
            bike_rides_counts = [counts[j] for counts in crime_with_bike_trips_cnts[i]]
            crime_with_bike_trips[f'bike_rides_within_{distance}_and_{time}'] = bike_rides_counts
    
    return crime_with_bike_trips

# Main function to run the complete process
def main() -> None:
    """
    Main function to orchestrate the proximity scanning process for static datasets, time-related datasets,
    and bike rides, then save the final results.
    """
    directory = '../../data/processed'
    target = ['clean_vacant_buildings.csv', 'clean_bike_stations.csv', 'clean_bus_stops.csv', 'clean_police_stations.csv', 'clean_train_stations.csv']
    
    data = read_in(directory, target)
    clean_crime = pd.read_csv('../../data/processed/clean_crime.csv')
    clean_crime['point'] = clean_crime.apply(lambda row: (row['lat'], row['long']), axis=1)

    # Run proximity scans for static datasets
    static_scans = run_static_proximity_scans(data, clean_crime)
    
    # Run proximity scans for time-related datasets
    time_related_scans = run_time_related_scans(clean_crime)
    
    # Run proximity scans for bike rides
    crime_with_bike_trips = run_bike_rides_scan(clean_crime)
    
    # Concatenate all results and save the final dataset
    crime_with_proximity = pd.concat([
        clean_crime, *static_scans, *time_related_scans, crime_with_bike_trips
    ], axis=1).drop('point', axis=1)

    crime_with_proximity.to_csv('../../data/pre_training/crime_with_proximity.csv', index=False)

if __name__ == "__main__":
    main()