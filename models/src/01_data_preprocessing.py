import pandas as pd
from shapely.geometry import Polygon
import warnings

# Suppressing warnings
warnings.filterwarnings('ignore')

#### Crime Dataset
def process_crime_data(file_path: str) -> pd.DataFrame:
    """
    Reads, cleans, and processes the crime dataset. Filters for non-null lat/long and specified date range.

    Args:
        file_path (str): The path to the raw crime data CSV.

    Returns:
        pd.DataFrame: Cleaned and processed crime data.
    """
    raw_crime = pd.read_csv(file_path)
    
    # Select relevant columns
    raw_crime = raw_crime[['Case Number', 'Date', 'Primary Type', 'Latitude', 'Longitude']]
    raw_crime = raw_crime[raw_crime.Latitude.notnull()]
    
    # Sort by Date, reset index, and filter by date range
    raw_crime = raw_crime.rename(columns={
        'Case Number': 'id', 
        'Date': 'date', 
        'Primary Type': 'type', 
        'Latitude': 'lat', 
        'Longitude': 'long'
    }).sort_values(by='date').reset_index(drop=True)
    
    raw_crime['date'] = pd.to_datetime(raw_crime['date'])
    raw_crime = raw_crime[(raw_crime['date'] >= '2016-01-01') & (raw_crime['date'] <= '2020-12-31')].reset_index(drop=True)
    
    # Save the processed dataset
    raw_crime.to_csv('../../data/processed/clean_crime.csv', index=False)
    return raw_crime

# Process and clean crime data
process_crime_data('../../data/raw/raw_crime.csv')

#### Cleaning 311-Related Datasets
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans a dataset by selecting, renaming columns, handling missing values, and filtering date range.

    Args:
        df (pd.DataFrame): DataFrame to clean.

    Returns:
        pd.DataFrame: Cleaned and filtered DataFrame.
    """
    # Selecting and renaming columns
    df = df[['Service Request Number', 'Creation Date', 'Completion Date', 'Type of Service Request', 'Latitude', 'Longitude']]
    df.rename(columns={
        'Creation Date': 'start_date', 
        'Completion Date': 'end_date', 
        'Service Request Number': 'id', 
        'Type of Service Request': 'type', 
        'Latitude': 'lat', 
        'Longitude': 'long'
    }, inplace=True)

    # Remove rows with missing lat/long and drop duplicates
    df.dropna(subset=['lat', 'long'], inplace=True)
    df.drop_duplicates(subset=['id'], inplace=True)

    # Convert date columns and filter by date range
    df['start_date'] = pd.to_datetime(df['start_date'])
    df['end_date'] = pd.to_datetime(df['end_date'])
    df = df[(df['start_date'] >= '2016-01-01') & (df['start_date'] <= '2020-12-31')].sort_values(by='start_date').reset_index(drop=True)
    
    return df

# Cleaning 311-related datasets (Alleylights and Streetlights)
alleylights = clean_data(pd.read_csv('../../data/raw/raw_alleylights.csv'))
alleylights.to_csv('../../data/processed/clean_alleylights.csv', index=False)

streetlights_allout = clean_data(pd.read_csv('../../data/raw/raw_streetlights_allout.csv'))
streetlights_allout.to_csv('../../data/processed/clean_streetlights_allout.csv', index=False)

streetlights_oneout = clean_data(pd.read_csv('../../data/raw/raw_streetlights_oneout.csv'))
streetlights_oneout.to_csv('../../data/processed/clean_streetlights_oneout.csv', index=False)

#### Vacant Buildings Dataset
def process_vacant_buildings(file_path: str) -> pd.DataFrame:
    """
    Cleans and processes the vacant buildings dataset.

    Args:
        file_path (str): Path to the vacant buildings dataset.

    Returns:
        pd.DataFrame: Cleaned and processed DataFrame.
    """
    df = pd.read_csv(file_path)
    
    # Selecting and renaming columns
    df = df[['DATE SERVICE REQUEST WAS RECEIVED', 'SERVICE REQUEST NUMBER', 'LATITUDE', 'LONGITUDE']]
    df.rename(columns={
        'DATE SERVICE REQUEST WAS RECEIVED': 'date', 
        'SERVICE REQUEST NUMBER': 'id', 
        'LATITUDE': 'lat', 
        'LONGITUDE': 'long'
    }, inplace=True)
    
    # Remove null lat/long values
    df.dropna(subset=['lat', 'long'], inplace=True)
    
    # Convert date column and filter by date range
    df['date'] = pd.to_datetime(df['date'])
    df = df[(df['date'] >= '2016-01-01') & (df['date'] <= '2020-12-31')].sort_values(by='date').reset_index(drop=True)
    
    df.to_csv('../../data/processed/clean_vacant_buildings.csv', index=False)
    return df

# Process vacant buildings data
process_vacant_buildings('../../data/raw/raw_vacant_buildings.csv')

#### Bike Trips and Stations Datasets
def process_bike_stations_and_trips(trip_file: str, station_file: str) -> pd.DataFrame:
    """
    Cleans and merges bike trip and station data.

    Args:
        trip_file (str): Path to the bike trips dataset.
        station_file (str): Path to the bike stations dataset.

    Returns:
        pd.DataFrame: Merged and cleaned bike trips data.
    """
    # Reading raw bike trip data
    raw_bike_trips = pd.read_csv(trip_file)
    raw_bike_stations = pd.read_csv(station_file)

    # Processing bike stations
    raw_bike_stations = raw_bike_stations[raw_bike_stations.Status == 'In Service'][['ID', 'Station Name', 'Latitude', 'Longitude']]
    raw_bike_stations.rename(columns={'ID': 'id', 'Station Name': 'station', 'Latitude': 'lat', 'Longitude': 'long'}, inplace=True)
    raw_bike_stations.drop_duplicates(inplace=True)

    # Processing and merging bike trips
    raw_bike_trips = raw_bike_trips[['TRIP ID', 'START TIME', 'FROM STATION ID', 'FROM STATION NAME', 'TO STATION ID', 'TO STATION NAME']]
    raw_bike_trips_a = raw_bike_trips[['TRIP ID', 'START TIME', 'FROM STATION ID', 'FROM STATION NAME']].rename(
        columns={'TRIP ID': 'id', 'START TIME': 'date', 'FROM STATION ID': 'station_id', 'FROM STATION NAME': 'station_name'})
    raw_bike_trips_b = raw_bike_trips[['TRIP ID', 'START TIME', 'TO STATION ID', 'TO STATION NAME']].rename(
        columns={'TRIP ID': 'id', 'START TIME': 'date', 'TO STATION ID': 'station_id', 'TO STATION NAME': 'station_name'})

    concat_bike_trips = pd.concat([raw_bike_trips_a, raw_bike_trips_b]).reset_index(drop=True)
    concat_bike_trips['date'] = pd.to_datetime(concat_bike_trips['date'], format='%m/%d/%Y %I:%M:%S %p')
    concat_bike_trips = concat_bike_trips.sort_values(by='date').reset_index(drop=True)
    
    # Merging trips with stations and filtering by date range
    concat_bike_trips = pd.merge(concat_bike_trips, raw_bike_stations, left_on='station_id', right_on='id', how='left')
    concat_bike_trips = concat_bike_trips[['date', 'station_id', 'station_name', 'lat', 'long']].drop_duplicates().reset_index(drop=True)
    concat_bike_trips['id'] = concat_bike_trips.index
    concat_bike_trips = concat_bike_trips[(concat_bike_trips['date'] >= '2016-01-01') & (concat_bike_trips['date'] <= '2020-12-31')]

    concat_bike_trips.to_csv('../../data/processed/clean_bike_trips.csv', index=False)
    return concat_bike_trips

# Process bike trips and stations
process_bike_stations_and_trips('../../data/raw/raw_bike_trips.csv', '../../data/raw/raw_bike_stations.csv')

#### Function to Convert Geometry for Datasets
def convert_to_polygon(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Converts string representation of polygons into Shapely Polygon objects.

    Args:
        df (pd.DataFrame): DataFrame with a column of polygon strings.
        column_name (str): The name of the column containing polygon strings.

    Returns:
        pd.DataFrame: DataFrame with converted Polygon objects.
    """
    updated_polygons = []
    for row in df[column_name]:
        target = row.replace('MULTIPOLYGON (((', '').replace(')))', '')
        points = target.split(', ')
        final = [(float(p.split()[1]), float(p.split()[0])) for p in points]
        polygons = Polygon(final)
        updated_polygons.append(polygons)
    
    df['poly'] = updated_polygons
    return df

#### Process Disadvantaged Areas and Police Districts
def process_disadvantaged_areas_and_police_districts(disadvantaged_file: str, police_file: str) -> pd.DataFrame:
    """
    Processes and saves disadvantaged areas and police districts datasets by converting geometries to polygons.

    Args:
        disadvantaged_file (str): Path to disadvantaged areas dataset.
        police_file (str): Path to police districts dataset.
    """
    # Disadvantaged areas
    disadvantaged_areas = pd.read_csv(disadvantaged_file)
    disadvantaged_areas = convert_to_polygon(disadvantaged_areas, 'the_geom')
    disadvantaged_areas.to_csv('../../data/processed/clean_disadvantaged_areas.csv', index=False)

    # Police districts
    police_districts = pd.read_csv(police_file)
    police_districts.rename(columns={'DIST_NUM': 'district'}, inplace=True)
    police_districts = convert_to_polygon(police_districts, 'the_geom')
    police_districts = police_districts[['district', 'poly']]
    police_districts.to_csv('../../data/processed/clean_police_districts.csv', index=False)

# Process disadvantaged areas and police districts
process_disadvantaged_areas_and_police_districts('../../data/raw/raw_disadvantaged_areas.csv', '../../data/raw/raw_police_districts.csv')

#### Function to Clean and Process Vacant Buildings Dataset
def process_vacant_buildings(file_path: str) -> pd.DataFrame:
    """
    Cleans and processes the vacant buildings dataset by selecting relevant columns, handling missing data, 
    and filtering by date range.

    Args:
        file_path (str): Path to the vacant buildings CSV.

    Returns:
        pd.DataFrame: Cleaned and processed vacant buildings dataset.
    """
    df = pd.read_csv(file_path)
    df = df[['DATE SERVICE REQUEST WAS RECEIVED', 'SERVICE REQUEST NUMBER', 'LATITUDE', 'LONGITUDE']]
    df.rename(columns={'DATE SERVICE REQUEST WAS RECEIVED': 'date', 'SERVICE REQUEST NUMBER': 'id', 
                       'LATITUDE': 'lat', 'LONGITUDE': 'long'}, inplace=True)
    df.dropna(subset=['lat', 'long'], inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    df = df[(df['date'] >= '2016-01-01') & (df['date'] <= '2020-12-31')].sort_values(by='date').reset_index(drop=True)
    df.to_csv('../../data/processed/clean_vacant_buildings.csv', index=False)
    return df

#### Function to Clean Bike Stations and Trips Dataset
def process_bike_trips(trip_file: str, station_file: str) -> pd.DataFrame:
    """
    Processes and merges bike trips and stations datasets, filtering by date range and station status.

    Args:
        trip_file (str): Path to the bike trips dataset CSV.
        station_file (str): Path to the bike stations dataset CSV.

    Returns:
        pd.DataFrame: Cleaned and merged bike trips data with corresponding station details.
    """
    raw_bike_trips = pd.read_csv(trip_file)
    raw_bike_stations = pd.read_csv(station_file)

    raw_bike_stations = raw_bike_stations[raw_bike_stations.Status == 'In Service'][['ID', 'Station Name', 'Latitude', 'Longitude']]
    raw_bike_stations.rename(columns={'ID': 'id', 'Station Name': 'station', 'Latitude': 'lat', 'Longitude': 'long'}, inplace=True)
    raw_bike_stations.drop_duplicates(inplace=True)

    raw_bike_trips_a = raw_bike_trips[['TRIP ID', 'START TIME', 'FROM STATION ID', 'FROM STATION NAME']].rename(
        columns={'TRIP ID': 'id', 'START TIME': 'date', 'FROM STATION ID': 'station_id', 'FROM STATION NAME': 'station_name'})
    raw_bike_trips_b = raw_bike_trips[['TRIP ID', 'START TIME', 'TO STATION ID', 'TO STATION NAME']].rename(
        columns={'TRIP ID': 'id', 'START TIME': 'date', 'TO STATION ID': 'station_id', 'TO STATION NAME': 'station_name'})

    concat_bike_trips = pd.concat([raw_bike_trips_a, raw_bike_trips_b]).reset_index(drop=True)
    concat_bike_trips['date'] = pd.to_datetime(concat_bike_trips['date'], format='%m/%d/%Y %I:%M:%S %p')
    concat_bike_trips = concat_bike_trips.sort_values(by='date').reset_index(drop=True)

    concat_bike_trips = pd.merge(concat_bike_trips, raw_bike_stations, left_on='station_id', right_on='id', how='left')
    concat_bike_trips = concat_bike_trips[['date', 'station_id', 'station_name', 'lat', 'long']].drop_duplicates().reset_index(drop=True)
    concat_bike_trips['id'] = concat_bike_trips.index
    concat_bike_trips = concat_bike_trips[(concat_bike_trips['date'] >= '2016-01-01') & (concat_bike_trips['date'] <= '2020-12-31')]

    concat_bike_trips.to_csv('../../data/processed/clean_bike_trips.csv', index=False)
    return concat_bike_trips

#### Function to Convert String Geometry to Shapely Polygon
def convert_to_polygon(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Converts a column with string representation of polygons into Shapely Polygon objects.

    Args:
        df (pd.DataFrame): DataFrame with polygon data in string format.
        column_name (str): The column name containing the string polygon data.

    Returns:
        pd.DataFrame: DataFrame with added polygon column in Shapely Polygon format.
    """
    updated_polygons = []
    for row in df[column_name]:
        target = row.replace('MULTIPOLYGON (((', '').replace(')))', '')
        points = target.split(', ')
        final = [(float(p.split()[1]), float(p.split()[0])) for p in points]
        polygons = Polygon(final)
        updated_polygons.append(polygons)
    
    df['poly'] = updated_polygons
    return df

#### Function to Process Disadvantaged Areas Dataset
def process_disadvantaged_areas(file_path: str) -> pd.DataFrame:
    """
    Processes and converts the disadvantaged areas dataset by converting geometries to Shapely polygons.

    Args:
        file_path (str): Path to the disadvantaged areas dataset.

    Returns:
        pd.DataFrame: Cleaned and processed disadvantaged areas dataset.
    """
    df = pd.read_csv(file_path)
    df = convert_to_polygon(df, 'the_geom')
    df.to_csv('../../data/processed/clean_disadvantaged_areas.csv', index=False)
    return df

#### Function to Process Areas Dataset
def process_areas(file_path: str) -> pd.DataFrame:
    """
    Processes and converts the areas dataset by converting geometries to Shapely polygons.

    Args:
        file_path (str): Path to the areas dataset.

    Returns:
        pd.DataFrame: Cleaned and processed areas dataset with polygons.
    """
    df = pd.read_csv(file_path)
    df = df[['the_geom', 'AREA_NUMBE']]
    df = convert_to_polygon(df, 'the_geom')
    df.rename(columns={'AREA_NUMBE': 'id'}, inplace=True)
    df.to_csv('../../data/processed/clean_areas.csv', index=False)
    return df

#### Function to Process Police Stations Dataset
def process_police_stations(file_path: str) -> pd.DataFrame:
    """
    Processes and cleans the police stations dataset by renaming columns and converting data types.

    Args:
        file_path (str): Path to the police stations dataset.

    Returns:
        pd.DataFrame: Cleaned police stations dataset.
    """
    df = pd.read_csv(file_path)
    df = df[['DISTRICT', 'LATITUDE', 'LONGITUDE']]
    df.rename(columns={'DISTRICT': 'id', 'LATITUDE': 'lat', 'LONGITUDE': 'long'}, inplace=True)
    df.to_csv('../../data/processed/clean_police_stations.csv', index=False)
    return df

#### Function to Process Public Health Indicators Dataset
def process_public_health_indicators(file_path: str) -> pd.DataFrame:
    """
    Processes the public health indicators dataset by renaming columns, converting values, and normalizing percentages.

    Args:
        file_path (str): Path to the public health indicators dataset.

    Returns:
        pd.DataFrame: Cleaned and normalized public health indicators dataset.
    """
    df = pd.read_csv(file_path)
    df.rename(columns={
        'Community Area': 'id',
        'Unemployment': 'unemployment',
        'Per Capita Income': 'per_capita_income',
        'No High School Diploma': 'no_hs_dip',
        'Dependency': 'gov_depend',
        'Crowded Housing': 'crowded_housing',
        'Below Poverty Level': 'below_pov'
    }, inplace=True)
    
    # Normalize percentage columns
    df['unemployment'] = df['unemployment'] / 100
    df['no_hs_dip'] = df['no_hs_dip'] / 100
    df['gov_depend'] = df['gov_depend'] / 100
    df['crowded_housing'] = df['crowded_housing'] / 100
    df['below_pov'] = df['below_pov'] / 100

    df.to_csv('../../data/processed/clean_public_healthindicator.csv', index=False)
    return df

#### Function to Process Police Districts Dataset
def process_police_districts(file_path: str) -> pd.DataFrame:
    """
    Processes the police districts dataset by converting geometry strings into Shapely polygons.

    Args:
        file_path (str): Path to the police districts dataset.

    Returns:
        pd.DataFrame: Cleaned police districts dataset with polygon geometries.
    """
    df = pd.read_csv(file_path)
    df.rename(columns={'DIST_NUM': 'district'}, inplace=True)
    df = convert_to_polygon(df, 'the_geom')
    df.to_csv('../../data/processed/clean_police_districts.csv', index=False)
    return df

#### Function to Process Bus Stops Dataset
def process_bus_stops(file_path: str) -> pd.DataFrame:
    """
    Processes the bus stops dataset by filtering and cleaning relevant columns.

    Args:
        file_path (str): Path to the bus stops dataset.

    Returns:
        pd.DataFrame: Cleaned bus stops dataset.
    """
    df = pd.read_csv(file_path)
    df = df[df['STATUS'] == 'In Service']
    df['routes_split'] = df['ROUTESSTPG'].str.split(',')
    df = df.explode('routes_split')
    df.columns = df.columns.str.lower()
    df.rename(columns={'y': 'lat', 'x': 'long', 'routes_split': 'route', 'systemstop': 'stop_id'}, inplace=True)
    df = df[['stop_id', 'route', 'name', 'lat', 'long']].reset_index(drop=True)
    df.to_csv('../../data/processed/clean_bus_stops.csv', index=False)
    return df

#### Function to Process Train Stations Dataset
def process_train_stations(file_path: str) -> pd.DataFrame:
    """
    Processes the train stations dataset by renaming columns and selecting relevant data.

    Args:
        file_path (str): Path to the train stations dataset.

    Returns:
        pd.DataFrame: Cleaned train stations dataset.
    """
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.lower()
    df.rename(columns={'x': 'long', 'y': 'lat', 'name': 'station_name', 'rail line': 'line', 'station id': 'id'}, inplace=True)
    df = df[['id', 'line', 'station_name', 'lat', 'long']].reset_index(drop=True)
    df.to_csv('../../data/processed/clean_train_stations.csv', index=False)
    return df

#### Function to Process Train Ridership Dataset
def process_train_ridership(file_path: str, train_stations_path: str) -> pd.DataFrame:
    """
    Processes the train ridership dataset by merging it with the train stations dataset and filtering by date range.

    Args:
        file_path (str): Path to the train ridership dataset.
        train_stations_path (str): Path to the train stations dataset.

    Returns:
        pd.DataFrame: Cleaned and processed train ridership dataset.
    """
    df = pd.read_csv(file_path)
    df.rename(columns={'stationname': 'station_name'}, inplace=True)
    
    train_stations = pd.read_csv(train_stations_path)
    df = pd.merge(train_stations[['line', 'station_name', 'lat', 'long']], df[['station_name', 'date', 'rides']], on='station_name', how='right')
    
    df.dropna(subset=['lat', 'long'], inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    df = df[(df['date'] >= '2016-01-01') & (df['date'] <= '2020-12-31')].reset_index(drop=True)
    
    df.to_csv('../../data/processed/clean_train_ridership.csv', index=False)
    return df