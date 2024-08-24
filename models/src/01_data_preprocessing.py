import pandas as pd
from shapely.geometry import Polygon
import warnings
from typing import List

warnings.filterwarnings('ignore')

# Crime Dataset
def clean_crime_data(file_path: str, output_path: str) -> None:
    """
    Cleans and processes raw crime data by filtering relevant columns, handling missing values,
    formatting date columns, and filtering the date range.

    Args:
        file_path (str): Path to the raw crime data CSV file.
        output_path (str): Path to save the cleaned crime data CSV file.
    """
    raw_crime = pd.read_csv(file_path)
    raw_crime = raw_crime[['Case Number', 'Date', 'Primary Type', 'Latitude', 'Longitude']]
    raw_crime = raw_crime.dropna(subset=['Latitude']).reset_index(drop=True)
    raw_crime = raw_crime.rename(columns={
        'Case Number': 'id', 
        'Date': 'date', 
        'Primary Type': 'type', 
        'Latitude': 'lat', 
        'Longitude': 'long'
    })
    raw_crime['date'] = pd.to_datetime(raw_crime['date'])
    raw_crime = raw_crime[(raw_crime['date'] >= '2016-01-01') & (raw_crime['date'] <= '2020-12-31')]
    raw_crime.to_csv(output_path, index=False)

# 311-Related Datasets (Alleylights and Streetlights Reported Outages)
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans 311-related datasets by filtering relevant columns, handling missing values,
    formatting date columns, and filtering the date range.

    Args:
        df (pd.DataFrame): DataFrame containing the raw 311-related data.

    Returns:
        pd.DataFrame: Cleaned DataFrame with relevant columns and formatted data.
    """
    df = df[['Service Request Number', 'Creation Date', 'Completion Date', 'Type of Service Request', 'Latitude', 'Longitude']]
    df = df.rename(columns={
        'Creation Date': 'start_date', 
        'Completion Date': 'end_date', 
        'Service Request Number': 'id', 
        'Type of Service Request': 'type', 
        'Latitude': 'lat', 
        'Longitude': 'long'
    })
    df = df.dropna(subset=['lat', 'long']).drop_duplicates(subset=['id'])
    df['start_date'] = pd.to_datetime(df['start_date'])
    df['end_date'] = pd.to_datetime(df['end_date'])
    df = df[(df['start_date'] >= '2016-01-01') & (df['start_date'] <= '2020-12-31')]
    df = df.sort_values(by='start_date').reset_index(drop=True)
    return df

def clean_311_datasets() -> None:
    """
    Cleans and processes multiple 311-related datasets such as alleylights, streetlights (all out), 
    and streetlights (one out) by loading, cleaning, and saving the cleaned datasets.
    """
    raw_alleylights = pd.read_csv('../../data/raw/raw_alleylights.csv')
    clean_alleylights = clean_data(raw_alleylights)
    clean_alleylights.to_csv('../../data/processed/clean_alleylights.csv', index=False)

    raw_streetlights_allout = pd.read_csv('../../data/raw/raw_streetlights_allout.csv')
    clean_streetlights_allout = clean_data(raw_streetlights_allout)
    clean_streetlights_allout.to_csv('../../data/processed/clean_streetlights_allout.csv', index=False)

    raw_streetlights_oneout = pd.read_csv('../../data/raw/raw_streetlights_oneout.csv')
    clean_streetlights_oneout = clean_data(raw_streetlights_oneout)
    clean_streetlights_oneout.to_csv('../../data/processed/clean_streetlights_oneout.csv', index=False)

# Vacant Buildings Dataset
def clean_vacant_buildings(file_path: str, output_path: str) -> None:
    """
    Cleans and processes the vacant buildings dataset by filtering relevant columns, handling missing values,
    formatting date columns, and filtering the date range.

    Args:
        file_path (str): Path to the raw vacant buildings data CSV file.
        output_path (str): Path to save the cleaned vacant buildings data CSV file.
    """
    raw_vacant_buildings = pd.read_csv(file_path)
    raw_vacant_buildings = raw_vacant_buildings[['DATE SERVICE REQUEST WAS RECEIVED', 'SERVICE REQUEST NUMBER', 'LATITUDE', 'LONGITUDE']]
    raw_vacant_buildings = raw_vacant_buildings.rename(columns={
        'DATE SERVICE REQUEST WAS RECEIVED': 'date', 
        'SERVICE REQUEST NUMBER': 'id', 
        'LATITUDE': 'lat', 
        'LONGITUDE': 'long'
    })
    raw_vacant_buildings = raw_vacant_buildings.dropna(subset=['lat', 'long'])
    raw_vacant_buildings['date'] = pd.to_datetime(raw_vacant_buildings['date'])
    raw_vacant_buildings = raw_vacant_buildings[(raw_vacant_buildings['date'] >= '2016-01-01') & (raw_vacant_buildings['date'] <= '2020-12-31')]
    raw_vacant_buildings = raw_vacant_buildings.sort_values(by='date').reset_index(drop=True)
    raw_vacant_buildings.to_csv(output_path, index=False)

# Bike Trips Dataset
def clean_bike_trips(file_path: str, output_path: str) -> pd.DataFrame:
    """
    Cleans and processes the bike trips dataset, including handling bike stations data,
    and returns a cleaned bike stations DataFrame.

    Args:
        file_path (str): Path to the raw bike trips data CSV file.
        output_path (str): Path to save the cleaned bike trips data CSV file.

    Returns:
        pd.DataFrame: Cleaned bike stations DataFrame.
    """
    raw_bike_trips = pd.read_csv(file_path)
    raw_bike_trips = raw_bike_trips[['TRIP ID', 'START TIME', 'STOP TIME', 'FROM STATION ID', 'FROM STATION NAME', 'TO STATION ID', 'TO STATION NAME']]
    
    # Create DataFrames for bike stations from both start and end stations
    bike_stations_a = raw_bike_trips[['FROM STATION ID', 'FROM STATION NAME']].rename(columns={'FROM STATION ID': 'id', 'FROM STATION NAME': 'station'})
    bike_stations_b = raw_bike_trips[['TO STATION ID', 'TO STATION NAME']].rename(columns={'TO STATION ID': 'id', 'TO STATION NAME': 'station'})
    bike_stations_in_use = pd.concat([bike_stations_a, bike_stations_b]).drop_duplicates().reset_index(drop=True)

    # Load and clean the bike stations dataset
    raw_bike_stations = pd.read_csv('../../data/raw/raw_bike_stations.csv')
    raw_bike_stations = raw_bike_stations[raw_bike_stations.Status == 'In Service']
    raw_bike_stations = raw_bike_stations[['ID', 'Station Name', 'Latitude', 'Longitude']].rename(columns={'ID': 'id', 'Station Name': 'station', 'Latitude': 'lat', 'Longitude': 'long'})
    raw_bike_stations = raw_bike_stations.merge(bike_stations_a, on='station', how='inner').drop_duplicates()
    raw_bike_stations = raw_bike_stations[~raw_bike_stations.station.duplicated()].reset_index(drop=True)
    raw_bike_stations.to_csv('../../data/processed/clean_bike_stations.csv', index=False)

    # Simplify and clean bike trips data
    raw_bike_trips_a = raw_bike_trips[['TRIP ID', 'START TIME', 'FROM STATION ID', 'FROM STATION NAME']].rename(columns={'TRIP ID': 'id', 'START TIME': 'date', 'FROM STATION ID': 'station_id', 'FROM STATION NAME': 'station_name'})
    raw_bike_trips_b = raw_bike_trips[['TRIP ID', 'START TIME', 'TO STATION ID', 'TO STATION NAME']].rename(columns={'TRIP ID': 'id', 'START TIME': 'date', 'TO STATION ID': 'station_id', 'TO STATION NAME': 'station_name'})
    concat_bike_trips = pd.concat([raw_bike_trips_a, raw_bike_trips_b]).reset_index(drop=True)
    concat_bike_trips['date'] = pd.to_datetime(concat_bike_trips['date'], format='%m/%d/%Y %I:%M:%S %p')
    concat_bike_trips = pd.merge(concat_bike_trips, raw_bike_stations, left_on='station_id', right_on='id', how='left')
    concat_bike_trips = concat_bike_trips[['date', 'station_id', 'station_name', 'lat', 'long']].drop_duplicates().reset_index(drop=True)
    concat_bike_trips['id'] = concat_bike_trips.index  # Create a bike trips ID column based on the index
    concat_bike_trips = concat_bike_trips[(concat_bike_trips['date'] >= '2016-01-01') & (concat_bike_trips['date'] <= '2020-12-31')]
    concat_bike_trips.to_csv(output_path, index=False)

    return raw_bike_stations

# Disadvantaged Areas Dataset
def convert_to_polygon_1(df: pd.DataFrame) -> List[Polygon]:
    """
    Converts textual geometry data into shapely Polygon objects for disadvantaged areas dataset.

    Args:
        df (pd.DataFrame): DataFrame containing geometry data as text.

    Returns:
        List[Polygon]: List of shapely Polygon objects representing geometries.
    """
    updated_polygons = []
    for row in df['the_geom']:
        target = row.replace('MULTIPOLYGON (((', '').replace(')))', '')
        points = target.split(', ')
        final = [(float(p.split()[0]), float(p.split()[1])) for p in points]
        polygons = Polygon(final)
        updated_polygons.append(polygons)
    return updated_polygons

def clean_disadvantaged_areas(file_path: str, output_path: str) -> None:
    """
    Cleans and processes the disadvantaged areas dataset by converting geometry data 
    into polygons and saving the cleaned data.

    Args:
        file_path (str): Path to the raw disadvantaged areas data CSV file.
        output_path (str): Path to save the cleaned disadvantaged areas data CSV file.
    """
    raw_disadvantaged_areas = pd.read_csv(file_path)
    clean_disadvantaged_areas = pd.DataFrame()
    clean_disadvantaged_areas['poly'] = convert_to_polygon_1(raw_disadvantaged_areas)
    clean_disadvantaged_areas.to_csv(output_path, index=False)

# Areas Dataset
def convert_to_polygon_2(df: pd.DataFrame) -> List[Polygon]:
    """
    Converts textual geometry data into shapely Polygon objects for the areas dataset.

    Args:
        df (pd.DataFrame): DataFrame containing geometry data as text.

    Returns:
        List[Polygon]: List of shapely Polygon objects representing geometries.
    """
    updated_polygons = []
    for row in df['the_geom']:
        target = row.replace('MULTIPOLYGON (((', '').replace(')))', '')
        points = target.split(', ')
        final = [(float(p.split()[1].replace('(', '').replace(')', '')), float(p.split()[0].replace('(', '').replace(')', ''))) for p in points]
        polygons = Polygon(final)
        updated_polygons.append(polygons)
    return updated_polygons

def clean_areas(file_path: str, output_path: str) -> None:
    """
    Cleans and processes the areas dataset by converting geometry data into polygons 
    and saving the cleaned data.

    Args:
        file_path (str): Path to the raw areas data CSV file.
        output_path (str): Path to save the cleaned areas data CSV file.
    """
    raw_areas = pd.read_csv(file_path)
    raw_areas['poly'] = convert_to_polygon_2(raw_areas)
    raw_areas = raw_areas[['AREA_NUMBE', 'poly']].rename(columns={'AREA_NUMBE': 'id'})
    raw_areas.to_csv(output_path, index=False)

# Police Stations Dataset
def clean_police_stations(file_path: str, output_path: str) -> None:
    """
    Cleans and processes the police stations dataset by renaming columns, handling missing values, 
    and saving the cleaned data.

    Args:
        file_path (str): Path to the raw police stations data CSV file.
        output_path (str): Path to save the cleaned police stations data CSV file.
    """
    raw_police_stations = pd.read_csv(file_path)
    raw_police_stations = raw_police_stations[['DISTRICT', 'LATITUDE', 'LONGITUDE']].rename(columns={'DISTRICT': 'id', 'LATITUDE': 'lat', 'LONGITUDE': 'long'})
    raw_police_stations.loc[0, 'id'] = 0  # Assign the station labeled "headquarters" an ID of 0
    raw_police_stations.to_csv(output_path, index=False)

# Public Health Indicators Dataset
def clean_public_health_indicator(file_path: str, output_path: str) -> None:
    """
    Cleans and processes the public health indicators dataset by filtering relevant columns, 
    renaming columns, normalizing percentage columns, and saving the cleaned data.

    Args:
        file_path (str): Path to the raw public health indicators data CSV file.
        output_path (str): Path to save the cleaned public health indicators data CSV file.
    """
    raw_public_healthindicator = pd.read_csv(file_path)
    raw_public_healthindicator = raw_public_healthindicator[['Community Area', 'Unemployment', 'Per Capita Income', 'No High School Diploma', 'Dependency', 'Crowded Housing', 'Below Poverty Level']]
    raw_public_healthindicator = raw_public_healthindicator.rename(columns={
        'Community Area': 'id', 'Unemployment': 'unemployment', 'Per Capita Income': 'per_capita_income',
        'No High School Diploma': 'no_hs_dip', 'Dependency': 'gov_depend', 'Crowded Housing': 'crowded_housing',
        'Below Poverty Level': 'below_pov'
    })
    raw_public_healthindicator[['unemployment', 'no_hs_dip', 'gov_depend', 'crowded_housing', 'below_pov']] /= 100  # Normalize percent columns
    raw_public_healthindicator.to_csv(output_path, index=False)

# Police Districts Dataset
def clean_police_districts(file_path: str, output_path: str) -> None:
    """
    Cleans and processes the police districts dataset by converting geometry data into polygons 
    and saving the cleaned data.

    Args:
        file_path (str): Path to the raw police districts data CSV file.
        output_path (str): Path to save the cleaned police districts data CSV file.
    """
    raw_police_districts = pd.read_csv(file_path)
    raw_police_districts = raw_police_districts[['DIST_NUM', 'the_geom']].rename(columns={'DIST_NUM': 'district'})
    raw_police_districts['geom'] = convert_to_polygon_2(raw_police_districts)
    raw_police_districts = raw_police_districts[['district', 'geom']]
    raw_police_districts.to_csv(output_path, index=False)

# Bus Stops Dataset
def clean_bus_stops(file_path: str, output_path: str) -> None:
    """
    Cleans and processes the bus stops dataset by filtering relevant columns, 
    handling route data, and saving the cleaned data.

    Args:
        file_path (str): Path to the raw bus stops data CSV file.
        output_path (str): Path to save the cleaned bus stops data CSV file.
    """
    raw_bus_stops = pd.read_csv(file_path)
    raw_bus_stops = raw_bus_stops[raw_bus_stops['STATUS'] == 'In Service']
    raw_bus_stops['routes_split'] = raw_bus_stops['ROUTESSTPG'].str.split(',')
    raw_bus_stops = raw_bus_stops.explode('routes_split').rename(columns={'y': 'lat', 'x': 'long', 'routes_split': 'route', 'systemstop': 'stop_id'}).reset_index(drop=True)
    raw_bus_stops.to_csv(output_path, index=False)

# Train Stations Dataset
def clean_train_stations(file_path: str, output_path: str) -> pd.DataFrame:
    """
    Cleans and processes the train stations dataset and returns the cleaned DataFrame.

    Args:
        file_path (str): Path to the raw train stations data CSV file.
        output_path (str): Path to save the cleaned train stations data CSV file.

    Returns:
        pd.DataFrame: Cleaned train stations DataFrame.
    """
    raw_train_stations = pd.read_csv(file_path)
    raw_train_stations = raw_train_stations.rename(columns={'x': 'long', 'y': 'lat', 'name': 'station_name', 'rail line': 'line', 'station id': 'id'})
    raw_train_stations = raw_train_stations[['id', 'line', 'station_name', 'lat', 'long']]
    raw_train_stations.to_csv(output_path, index=False)
    return raw_train_stations

# Train Ridership Dataset
def clean_train_ridership(file_path: str, output_path: str, train_stations: pd.DataFrame) -> None:
    """
    Cleans and processes the train ridership dataset by merging it with the train stations data,
    handling missing values, and saving the cleaned data.

    Args:
        file_path (str): Path to the raw train ridership data CSV file.
        output_path (str): Path to save the cleaned train ridership data CSV file.
        train_stations (pd.DataFrame): DataFrame containing the cleaned train stations data.
    """
    raw_train_ridership = pd.read_csv(file_path)
    raw_train_ridership = raw_train_ridership.rename(columns={'stationname': 'station_name'})
    raw_train_ridership = pd.merge(train_stations[['line', 'station_name', 'lat', 'long']], raw_train_ridership[['station_name', 'date', 'rides']], on='station_name', how='right')
    raw_train_ridership = raw_train_ridership.dropna(subset=['lat', 'long']).reset_index(drop=True)
    raw_train_ridership['date'] = pd.to_datetime(raw_train_ridership['date'])
    raw_train_ridership = raw_train_ridership[(raw_train_ridership['date'] >= '2016-01-01') & (raw_train_ridership['date'] <= '2020-12-31')]
    raw_train_ridership.to_csv(output_path, index=False)

# Main function to clean all datasets
def main() -> None:
    """Main function to clean all datasets by calling specific cleaning functions for each dataset."""
    clean_crime_data('../../data/raw/raw_crime.csv', '../../data/processed/clean_crime.csv')
    clean_311_datasets()
    clean_vacant_buildings('../../data/raw/raw_vacant_buildings.csv', '../../data/processed/clean_vacant_buildings.csv')
    
    raw_bike_stations = clean_bike_trips('../../data/raw/raw_bike_trips.csv', '../../data/processed/clean_bike_trips.csv')
    
    clean_disadvantaged_areas('../../data/raw/raw_disadvantaged_areas.csv', '../../data/processed/clean_disadvantaged_areas.csv')
    clean_areas('../../data/raw/raw_areas.csv', '../../data/processed/clean_areas.csv')
    clean_police_stations('../../data/raw/raw_police_stations.csv', '../../data/processed/clean_police_stations.csv')
    clean_public_health_indicator('../../data/raw/raw_publichealth_indicator.csv', '../../data/processed/clean_public_healthindicator.csv')
    clean_police_districts('../../data/raw/raw_police_districts.csv', '../../data/processed/clean_police_districts.csv')
    clean_bus_stops('../../data/raw/raw_bus_stops.csv', '../../data/processed/clean_bus_stops.csv')

    raw_train_stations = clean_train_stations('../../data/raw/raw_train_stations.csv', '../../data/processed/clean_train_stations.csv')
    clean_train_ridership('../../data/raw/raw_train_ridership.csv', '../../data/processed/clean_train_ridership.csv', raw_train_stations)

if __name__ == "__main__":
    main()