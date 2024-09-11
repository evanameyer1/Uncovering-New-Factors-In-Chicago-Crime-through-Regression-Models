import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from shapely.geometry import Point, Polygon
from shapely import geometry


#### Reading in datasets
def load_datasets() -> dict[str, pd.DataFrame]:
    """
    Loads the necessary datasets into a dictionary of DataFrames.

    Returns:
        dict: Dictionary containing the loaded DataFrames.
    """
    crime_with_proximity = pd.read_csv('../../data/pre_training/crime_with_proximity.csv')
    clean_bike_trips = pd.read_csv('../../data/processed/clean_bike_trips.csv')
    agg_public_healthindicator = pd.read_csv('../../data/processed/agg_public_healthindicator.csv')
    clean_police_districts = pd.read_csv('../../data/processed/clean_police_districts.csv')
    clean_train_ridership = pd.read_csv('../../data/processed/clean_train_ridership.csv')
    clean_areas = pd.read_csv('../../data/processed/clean_areas.csv')
    clean_crime = pd.read_csv('../../data/processed/clean_crime.csv')
    clean_dis_areas = pd.read_csv('../../data/processed/disadvantaged_areas_within_areas.csv')

    return {
        'crime_with_proximity': crime_with_proximity,
        'clean_bike_trips': clean_bike_trips,
        'agg_public_healthindicator': agg_public_healthindicator,
        'clean_police_districts': clean_police_districts,
        'clean_train_ridership': clean_train_ridership,
        'clean_areas': clean_areas,
        'clean_crime': clean_crime,
        'clean_dis_areas': clean_dis_areas
    }

data = load_datasets()
crime_with_proximity = data['crime_with_proximity']
clean_crime = data['clean_crime']
crime_with_proximity = crime_with_proximity.merge(clean_crime[['id', 'areas']], on='id', how='inner')
crime_with_proximity.rename(columns={'areas': 'area_id'}, inplace=True)

crime_with_proximity['date'] = pd.to_datetime(crime_with_proximity['date'])
crime_with_proximity['hour'] = crime_with_proximity['date'].dt.hour
crime_with_proximity['day'] = crime_with_proximity['date'].dt.date
crime_with_proximity['date_hour'] = crime_with_proximity['date'].dt.floor('h')
crime_with_proximity.sort_values('date', inplace=True)

#### Calculate District Crimes Over Time
def calculate_crimes_over_time(crime_df: pd.DataFrame, group_by_col: str, time_windows: list[int], label_prefix: str) -> pd.DataFrame:
    """
    Calculates the number of crimes over specified time windows for each district/area.

    Args:
        crime_df (pd.DataFrame): DataFrame containing the crime data.
        group_by_col (str): Column name by which to group (e.g., 'district', 'area_id').
        time_windows (list[int]): List of time windows in hours.
        label_prefix (str): Prefix for the resulting columns.

    Returns:
        pd.DataFrame: DataFrame containing the crime counts for each time window.
    """
    crimes_over_hours = crime_df.groupby([group_by_col, pd.Grouper(key='date', freq='h')])['id'].count().reset_index().rename(columns={'id': f'{label_prefix}_crimes_this_hour'})
    
    for window in time_windows:
        crimes_over_hours[f'{label_prefix}_crimes_{window}_hours_prev'] = crimes_over_hours.groupby(group_by_col)[f'{label_prefix}_crimes_this_hour'].rolling(window=window, min_periods=1).sum().shift(1).reset_index(level=0, drop=True)
    
    return crimes_over_hours

# Calculate district crimes
time_windows = [1, 3, 6, 12, 24]
district_crimes_over_hours = calculate_crimes_over_time(crime_with_proximity, 'district', time_windows, 'district')
area_crimes_over_hours = calculate_crimes_over_time(crime_with_proximity, 'area_id', time_windows, 'area')

# Merge with crime data
district_crimes_over_hours = pd.merge(left=crime_with_proximity[['id', 'district', 'date_hour', 'hour', 'day']], right=district_crimes_over_hours, left_on=['district', 'date_hour'], right_on=['district', 'date'], how='left')
area_crimes_over_hours = pd.merge(left=crime_with_proximity[['id', 'area_id', 'date_hour', 'hour', 'day']], right=area_crimes_over_hours, left_on=['area_id', 'date_hour'], right_on=['area_id', 'date'], how='left')

#### Adding Disadvantaged Areas
def parse_polygon(polygon_string: str) -> Polygon:
    """
    Parses a polygon string into a Shapely Polygon object.

    Args:
        polygon_string (str): String representation of a polygon.

    Returns:
        Polygon: Shapely Polygon object.
    """
    points = polygon_string.strip('POLYGON ((').strip('))').split(', ')
    points = [tuple(map(float, point.split())) for point in points]
    return Polygon(points)

def swap_coordinates(polygon: Polygon) -> Polygon:
    """
    Swaps the coordinates of a polygon (lat, long to long, lat).

    Args:
        polygon (Polygon): Shapely Polygon object.

    Returns:
        Polygon: Polygon with swapped coordinates.
    """
    if polygon.is_empty:
        return polygon
    swapped_coords = [(y, x) for x, y in polygon.exterior.coords]
    return Polygon(swapped_coords)

# Clean disadvantaged areas
clean_dis_areas = data['clean_dis_areas']
clean_dis_areas['poly'] = clean_dis_areas['poly'].apply(parse_polygon)
clean_dis_areas['poly'] = clean_dis_areas['poly'].apply(swap_coordinates)
clean_dis_areas['id'] = clean_dis_areas.index

# Create dictionary mapping areas to disadvantaged areas
def map_dis_areas(clean_dis_areas: pd.DataFrame) -> dict[int, list[tuple[int, Polygon]]]:
    """
    Maps each area to its disadvantaged areas using geometries.

    Args:
        clean_dis_areas (pd.DataFrame): DataFrame containing disadvantaged areas and geometries.

    Returns:
        dict: Dictionary mapping area ids to disadvantaged areas and their geometries.
    """
    dis_areas_to_areas = {}
    for idx, row in clean_dis_areas.iterrows():
        if row['areas'] in dis_areas_to_areas:
            dis_areas_to_areas[row['areas']].append((row['id'], row['poly']))
        else:
            dis_areas_to_areas[row['areas']] = [(row['id'], row['poly'])]
    return dis_areas_to_areas

dis_areas_to_areas = map_dis_areas(clean_dis_areas)

# Determine disadvantaged areas for crimes
def determine_dis_area_for_crimes(df: pd.DataFrame, perc: int, dis_areas_to_areas: dict[int, list[tuple[int, Polygon]]]) -> pd.DataFrame:
    """
    Determines which disadvantaged area each crime belongs to based on location.

    Args:
        df (pd.DataFrame): Crime DataFrame.
        perc (int): Percentage increment to show progress.
        dis_areas_to_areas (dict): Dictionary mapping areas to disadvantaged areas and geometries.

    Returns:
        pd.DataFrame: DataFrame with disadvantaged area IDs added.
    """
    dis_areas = []
    perc_cnt = perc

    for i in range(len(df)):
        point = geometry.Point(df.loc[i, 'long'], df.loc[i, 'lat'])
        curr_dis_area = None

        if df.loc[i, 'area_id'] in dis_areas_to_areas:
            for (area, geom) in dis_areas_to_areas[df.loc[i, 'area_id']]:
                if geom.contains(point):
                    curr_dis_area = area
                    break
        
        dis_areas.append(curr_dis_area)

        if i > 0 and i % (round(len(df) * (perc_cnt / 100))) == 0:
            print(f"{perc_cnt}% - Row {i}/{len(df)} completed")
            perc_cnt += perc

    df['dis_area_id'] = dis_areas
    return df

crime_with_proximity = determine_dis_area_for_crimes(crime_with_proximity, 2, dis_areas_to_areas)

#### Adding Crime Counts by Disadvantaged Areas
dis_area_crimes_over_hours = calculate_crimes_over_time(crime_with_proximity, 'dis_area_id', time_windows, 'dis_area')
crime_with_proximity_dis_area = crime_with_proximity.dropna(subset=['dis_area_id'], axis=0)

# Merge with disadvantaged area crime data
dis_area_crimes_over_hours = pd.merge(
    left=crime_with_proximity_dis_area[['id', 'dis_area_id', 'date_hour', 'hour', 'day']],
    right=dis_area_crimes_over_hours, 
    left_on=['dis_area_id', 'date_hour'], 
    right_on=['dis_area_id', 'date'], 
    how='left'
)

#### Group and Merge Other Data
clean_bike_trips = data['clean_bike_trips']
clean_bike_trips['date'] = pd.to_datetime(clean_bike_trips['date'])
clean_bike_trips['hour'] = clean_bike_trips['date'].dt.hour
clean_bike_trips['date'] = clean_bike_trips['date'].dt.date

# Group bike rides by date, hour, and district
grouped_bike_trips = clean_bike_trips.groupby(['date', 'hour', 'district'])['station_id'].agg('count').reset_index().rename(columns={'station_id': 'hourly_bike_rides'})
grouped_bike_trips['date'] = pd.to_datetime(grouped_bike_trips['date'])

# Merge bike trips, crime data, and other features
final_df = pd.merge(left=crime_with_proximity, right=grouped_bike_trips, on=['date', 'hour', 'district'], how='left').drop(['hour', 'day'], axis=1).fillna(0)

agg_public_healthindicator = data['agg_public_healthindicator']
agg_public_healthindicator.columns = ['district_' + col if col != 'district' else 'district' for col in agg_public_healthindicator.columns]

final_df = pd.merge(left=final_df, right=agg_public_healthindicator, on='district', how='left')
final_df = pd.merge(left=final_df, right=data['clean_police_districts'][['district', 'disadvantaged_score']], on='district', how='left')

clean_train_ridership = data['clean_train_ridership']
clean_train_ridership['date'] = pd.to_datetime(clean_train_ridership['date'])
grouped_train_ridership = clean_train_ridership.groupby(['date', 'district'])['rides'].agg('sum').reset_index()

final_df = pd.merge(left=final_df, right=grouped_train_ridership, on=['date', 'district'], how='left').fillna(0)
final_df = final_df.drop_duplicates(subset=['id'])

# Final merging of district, area, and disadvantaged area crime stats
final_df = pd.merge(left=final_df, right=district_crimes_over_hours, on='id', how='inner')
final_df = pd.merge(left=final_df, right=area_crimes_over_hours, on='id', how='inner')
final_df = pd.merge(left=final_df, right=dis_area_crimes_over_hours, on='id', how='inner')

#### Base DataFrames for Areas, Districts, and Disadvantaged Areas
def create_area_base_df() -> pd.DataFrame:
    """
    Creates a base DataFrame with all possible area and date combinations.

    Returns:
        pd.DataFrame: Area base DataFrame.
    """
    date_range = pd.date_range(start='2016-01-01 00:00:00', end='2020-12-31 23:00:00', freq='h')
    areas = np.arange(1, 78)
    area_base_df = pd.DataFrame([(area, date) for area in areas for date in date_range], columns=['area_id', 'date_hour'])
    area_base_df['day'] = area_base_df['date_hour'].dt.day
    area_base_df['hour'] = area_base_df['date_hour'].dt.hour
    area_base_df['year'] = area_base_df['date_hour'].dt.year
    area_base_df['month'] = area_base_df['date_hour'].dt.month
    area_base_df['day_of_week'] = area_base_df['date_hour'].dt.dayofweek
    return area_base_df

# Example usage to create base area DataFrame
area_base_df = create_area_base_df()

# Continue with similar logic for disadvantaged areas and districts...

#### Normalization and Saving Data
def normalize_and_save(df: pd.DataFrame, columns_to_normalize: list[str], output_path: str) -> None:
    """
    Normalizes specified columns in the DataFrame using MinMaxScaler and saves the result to a CSV file.

    Args:
        df (pd.DataFrame): DataFrame to normalize.
        columns_to_normalize (list[str]): List of columns to normalize.
        output_path (str): File path to save the normalized DataFrame.
    """
    scaler = MinMaxScaler()
    df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
    df.fillna(0, inplace=True)
    df.to_csv(output_path, index=False)

# Columns to normalize
area_columns_to_normalize = [
    'area_unemployment', 'area_per_capita_income', 'area_no_hs_dip', 'area_gov_depend', 'area_crowded_housing', 'area_below_pov',
    'police_stations_distance_0.1', 'police_stations_distance_0.3', 'police_stations_distance_0.5', 'police_stations_distance_1',
    'police_stations_distance_3', 'police_stations_distance_5', 'bike_stations_distance_0.1', 'bike_stations_distance_0.3',
    'bike_stations_distance_0.5', 'bike_stations_distance_1', 'bike_stations_distance_3', 'bike_stations_distance_5',
    #... (continue listing columns to normalize)
]

# Normalize and save the final DataFrames
normalize_and_save(final_df, area_columns_to_normalize, '../../data/pre_training/area_pre_feature_selection.csv')