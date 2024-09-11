import pandas as pd
import os
from shapely.geometry import Point, Polygon
from shapely import geometry
import ast

directory = '../../data/processed'
void = ['clean_areas.csv', 'clean_disadvantaged_areas.csv', 'clean_police_districts.csv', 
        'clean_public_healthindicator.csv', 'clean_train_ridership.csv', 'clean_bike_trips.csv']

# Function to read in all CSV files in the directory, except for those in the void list
def read_in(directory: str, void: list[str]) -> dict[str, pd.DataFrame]:
    """
    Reads in CSV files from the specified directory, except for the files in the void list.

    Args:
        directory (str): The path to the directory containing the CSV files.
        void (list): A list of file names to exclude from reading.

    Returns:
        dict: A dictionary containing DataFrames with keys as file names (without .csv extension).
    """
    data = {}
    for filename in os.listdir(directory):
        if filename not in void and filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            data[filename[:-4]] = pd.read_csv(file_path)
            print(f'{filename[:-4]} successfully read in')
    return data

data = read_in(directory, void)
clean_police_districts = pd.read_csv('../../data/processed/clean_police_districts.csv')
clean_areas = pd.read_csv('../../data/processed/clean_areas.csv')
clean_disadvantaged_areas = pd.read_csv('../../data/processed/clean_disadvantaged_areas.csv')

#### Data Cleaning of Geom Types
# Function to parse polygon strings from the 'geom' column of CSV files
def parse_polygon1(polygon_string: str) -> Polygon:
    """
    Parses a polygon string in the format 'POLYGON (...)' into a Shapely Polygon object.

    Args:
        polygon_string (str): A string representation of a polygon.

    Returns:
        Polygon: A Shapely Polygon object.
    """
    points = polygon_string.strip('POLYGON ((').strip('))').split(', ')
    points = [tuple(map(float, point.split())) for point in points]
    return Polygon(points)

# Function to parse polygon strings that are lists of tuples
def parse_polygon2(polygon_string: str) -> Polygon:
    """
    Parses a polygon string that is a list of tuples (from JSON format) into a Shapely Polygon object.

    Args:
        polygon_string (str): A string representation of a polygon (list of tuples).

    Returns:
        Polygon: A Shapely Polygon object.
    """
    points = ast.literal_eval(polygon_string)
    return Polygon(points)

# Function to swap coordinates of a polygon
def swap_coordinates(polygon: Polygon) -> Polygon:
    """
    Swaps the coordinates of a polygon (lat, long -> long, lat).

    Args:
        polygon (Polygon): A Shapely Polygon object.

    Returns:
        Polygon: A Shapely Polygon object with swapped coordinates.
    """
    if polygon.is_empty:
        return polygon
    swapped_coords = [(y, x) for x, y in polygon.exterior.coords]
    return Polygon(swapped_coords)

# Apply functions to clean up and swap coordinates for police districts and areas
clean_police_districts['geom'] = clean_police_districts['geom'].apply(parse_polygon1)
clean_areas['poly'] = clean_areas['poly'].apply(parse_polygon2)
clean_disadvantaged_areas['poly'] = clean_disadvantaged_areas['poly'].apply(parse_polygon1)
clean_disadvantaged_areas['poly'] = clean_disadvantaged_areas['poly'].apply(swap_coordinates)

# Adding centroids to the areas and disadvantaged areas DataFrame
clean_areas['centroid'] = clean_areas['poly'].apply(lambda poly: poly.centroid)
clean_disadvantaged_areas['centroid'] = clean_disadvantaged_areas['poly'].apply(lambda poly: poly.centroid)

#### Assigning Districts to Each Dataset
# Function to determine which police district each point belongs to
def determine_within(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns a police district to each row in the DataFrame based on its centroid or coordinates.

    Args:
        df (pd.DataFrame): A DataFrame containing points with centroids or lat/long.

    Returns:
        pd.DataFrame: The DataFrame with assigned districts, filtering out points that don't belong to any district.
    """
    statuses = []
    districts = []
    cent = 'centroid' in df.columns  # Check if the DataFrame has centroid column

    for i in range(len(df)):
        point = df.loc[i, 'centroid'] if cent else geometry.Point(df.loc[i, 'long'], df.loc[i, 'lat'])
        status = 0

        # Check if point lies within a police district polygon
        for index, row in clean_police_districts.iterrows():
            district = row['district']
            geom = row['geom']
            if geom.contains(point): 
                status = 1
                break
        statuses.append(status)
        districts.append(district)

    df['status'] = statuses
    df['district'] = districts
    df = df[df['status'] == 1].drop('status', axis=1)

    if cent: 
        df.drop('centroid', axis=1, inplace=True)

    return df

# Function to assign districts to multiple DataFrames
def determine_districts(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Applies the determine_within function to all DataFrames in the provided dictionary.

    Args:
        data (dict): A dictionary containing DataFrames.

    Returns:
        dict: Updated dictionary with districts assigned to each DataFrame.
    """
    for df_name, df in data.items():
        data[df_name] = determine_within(df)
        print(f'{df_name} successfully completed')
    return data

clean_disadvantaged_areas = determine_within(clean_disadvantaged_areas)
clean_areas = determine_within(clean_areas)

#### Assigning Areas Stats to Each Crime
# Function to assign an area to each crime based on centroid or coordinates
def determine_area_for_crimes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns an area to each crime entry based on its centroid or coordinates.

    Args:
        df (pd.DataFrame): A DataFrame containing crime data with centroids or lat/long.

    Returns:
        pd.DataFrame: The DataFrame with assigned areas, filtering out points that don't belong to any area.
    """
    statuses = []
    areas = []
    cent = 'centroid' in df.columns  # Check if the DataFrame has centroid column

    for i in range(len(df)):
        point = df.loc[i, 'centroid'] if cent else geometry.Point(df.loc[i, 'long'], df.loc[i, 'lat'])
        status = 0

        # Check if point lies within a given area polygon
        for index, row in clean_areas.iterrows():
            area = row['id']
            geom = row['poly']
            if geom.contains(point): 
                status = 1
                break
        statuses.append(status)
        areas.append(area)

    df['status'] = statuses
    df['areas'] = areas
    df = df[df['status'] == 1].drop('status', axis=1)

    if cent: 
        df.drop('centroid', axis=1, inplace=True)

    return df

# Assign areas for train and bike stations
clean_bike_trips = pd.read_csv('../../data/processed/clean_bike_trips.csv')
clean_train_ridership = pd.read_csv('../../data/processed/clean_train_ridership.csv')

clean_disadvantaged_areas['centroid'] = clean_disadvantaged_areas['poly'].apply(lambda poly: poly.centroid)
clean_disadvantaged_areas['poly'].drop_duplicates(inplace=True)
clean_disadvantaged_areas.reset_index(drop=True, inplace=True)

# Determine areas for train stations, bike stations, and disadvantaged areas
trains_with_areas = determine_area_for_crimes(data['clean_train_stations'])
bikes_with_areas = determine_area_for_crimes(data['clean_bike_stations'])
disadvantaged_areas_within_areas = determine_area_for_crimes(clean_disadvantaged_areas)

#### Data Cleaning to Finalize Datasets
clean_bike_trips = clean_bike_trips[['station_id', 'date', 'lat', 'long']].merge(
    right=bikes_with_areas[['id', 'district', 'areas']], 
    how='left', left_on='station_id', right_on='id').dropna(subset=['district'])

clean_train_ridership = clean_train_ridership[['date', 'line', 'station_name', 'lat', 'long', 'rides']].merge(
    right=trains_with_areas[['station_name', 'district', 'areas']], 
    how='left', on='station_name').dropna(subset=['district'])

clean_police_districts = clean_police_districts.merge(
    right=clean_disadvantaged_areas.groupby('district').agg('count'), 
    how='left', on='district').fillna(0).rename(columns={'poly': 'disadvantaged_score'})

clean_public_healthindicator = pd.read_csv('../../data/processed/clean_public_healthindicator.csv')

agg_public_healthindicator = pd.merge(
    left=clean_public_healthindicator, 
    right=clean_areas[['id', 'district']], 
    on='id', how='left').drop('id', axis=1)

agg_public_healthindicator = agg_public_healthindicator.groupby('district').agg('mean').reset_index()

#### Save Datasets with District and Areas Data
def save_data(data: dict[str, pd.DataFrame]) -> None:
    """
    Saves the cleaned DataFrames to CSV files in the processed directory.

    Args:
        data (dict): A dictionary containing DataFrames to be saved.
    """
    for df_name, df in data.items():
        df.to_csv(f'../../data/processed/{df_name}.csv', index=False)

save_data(data)