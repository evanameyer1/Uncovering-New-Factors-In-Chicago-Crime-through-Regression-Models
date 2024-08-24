import pandas as pd
from geopy.distance import geodesic  # deprecated
import geopandas as gpd
from scipy.spatial import cKDTree
import numpy as np
import os

directory = '../../data/processed'
target = [
    'clean_vacant_buildings.csv', 
    'clean_bike_stations.csv', 
    'clean_bus_stops.csv', 
    'clean_police_stations.csv', 
    'clean_train_stations.csv'
]

# Creating a dict to store datasets with long,lat columns
def read_in(directory, target):
    data = {}
    for filename in os.listdir(directory):
        if filename in target:
            file_path = os.path.join(directory, filename)
            data[filename[:-4]] = pd.read_csv(file_path)
            print(f'{filename[:-4]} successfully read in')
    return data

data = read_in(directory, target)

clean_crime = pd.read_csv('../../data/processed/clean_crime.csv')
# Create a column of tuples consisting of each crime's geopoint
clean_crime['point'] = clean_crime.apply(lambda row: (row['lat'], row['long']), axis=1)

# Determine a count of objects within pre-defined distances for each crime
def proximity_scan(df, col_name, distances):
    clean_crime_gdf = gpd.GeoDataFrame(clean_crime, geometry=gpd.points_from_xy(clean_crime.long, clean_crime.lat))
    df_gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.long, df.lat))
    
    # Using a KD-Tree for computational efficiency
    df_coords = np.array(list(zip(df_gdf.geometry.x, df_gdf.geometry.y)))
    tree = cKDTree(df_coords) 
    
    output = []
    distances_in_degrees = [d / 69 for d in distances]  # Convert distances from miles to degrees

    for crime_idx, crime_row in clean_crime_gdf.iterrows():            
        crime_cnt = [0 for _ in range(len(distances))]
        crime_point = crime_row.geometry
        
        # For each crime, query all points within each pre-defined distance
        for idx, distance in enumerate(distances_in_degrees):
            indices = tree.query_ball_point([crime_point.x, crime_point.y], distance)
            crime_cnt[idx] = len(indices)
        
        output.append(crime_cnt)
    
    temp_df = pd.DataFrame(output, columns=[f'{col_name}_distance_{str(distance)}' for distance in distances])
    return temp_df

# Creating temporary dataframes with the count of the following (distances are in terms of miles)
crime_with_police = proximity_scan(data['clean_police_stations'], 'police_stations', [0.1, 0.3, 0.5, 1, 3, 5])
crime_with_bikes = proximity_scan(data['clean_bike_stations'], 'bike_stations', [0.1, 0.3, 0.5, 1, 3, 5])
crime_with_buses = proximity_scan(data['clean_bus_stops'], 'bus_stops', [0.1, 0.3, 0.5, 1, 3, 5])
crime_with_trains = proximity_scan(data['clean_train_stations'], 'train_stations', [0.1, 0.3, 0.5, 1, 3, 5])
crime_with_vacant_buildings = proximity_scan(data['clean_vacant_buildings'], 'vacant_buildings', [0.1, 0.3, 0.5, 1, 3, 5])

# Read in 311 datasets, as time is a factor to determine nearby objects at the time of each crime
clean_alleylights = pd.read_csv('../../data/processed/clean_alleylights.csv')
clean_streetlights_allout = pd.read_csv('../../data/processed/clean_streetlights_allout.csv')
clean_streetlights_oneout = pd.read_csv('../../data/processed/clean_streetlights_oneout.csv')

def patch_datetypes(data):
    for df in data:
        df['start_date'] = pd.to_datetime(df['start_date'])
        df['end_date'] = pd.to_datetime(df['end_date'])

def proximity_scan_311(df, col_name, distances):
    clean_crime_gdf = gpd.GeoDataFrame(clean_crime, geometry=gpd.points_from_xy(clean_crime.long, clean_crime.lat))
    
    output = []
    distances_in_degrees = [d / 69 for d in distances]  # Convert distances from miles to degrees

    for crime_idx, crime_row in clean_crime_gdf.iterrows():    
        crime_cnt = [0 for _ in range(len(distances))]
        crime_point = crime_row.geometry
        
        # Create a new KD-Tree for each crime, composed of all 311 entries that are within the time frame of the crime
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

patch_datetypes([clean_alleylights, clean_streetlights_allout, clean_streetlights_oneout])

crime_with_alleylights = proximity_scan_311(clean_alleylights, 'alleylights', [0.1, 0.3, 0.5, 1, 3, 5])
crime_with_streetlights_allout = proximity_scan_311(clean_streetlights_allout, 'streetlights_allout', [0.1, 0.3, 0.5, 1, 3, 5])
crime_with_streetlights_oneout = proximity_scan_311(clean_streetlights_oneout, 'streetlights_oneout', [0.1, 0.3, 0.5, 1, 3, 5])

# Combine into final proximity dataset
crime_with_proximity = pd.concat([
    clean_crime, 
    crime_with_police, 
    crime_with_bikes, 
    crime_with_buses, 
    crime_with_trains, 
    crime_with_alleylights, 
    crime_with_streetlights_allout, 
    crime_with_streetlights_oneout
], axis=1).drop('point', axis=1)

crime_with_proximity.to_csv('../../data/pre_training/crime_with_proximity.csv', index=False)
