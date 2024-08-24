import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import List, Dict

# Load CSV files
def load_data() -> Dict[str, pd.DataFrame]:
    """
    Load all relevant datasets.

    Returns:
        dict: A dictionary containing DataFrames for all relevant datasets.
    """
    return {
        "crime_with_proximity": pd.read_csv('../../data/pre_training/crime_with_proximity.csv'),
        "clean_bike_trips": pd.read_csv('../../data/processed/clean_bike_trips.csv'),
        "agg_public_healthindicator": pd.read_csv('../../data/processed/agg_public_healthindicator.csv'),
        "clean_police_districts": pd.read_csv('../../data/processed/clean_police_districts.csv'),
        "clean_train_ridership": pd.read_csv('../../data/processed/clean_train_ridership.csv'),
        "clean_areas": pd.read_csv('../../data/processed/clean_areas.csv'),
        "clean_crime": pd.read_csv('../../data/processed/clean_crime.csv')
    }

# Preprocess the crime data
def preprocess_crime_data(crime_with_proximity: pd.DataFrame, clean_crime: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess crime data by merging area information and adding date/time features.

    Args:
        crime_with_proximity (pd.DataFrame): The dataset with crime and proximity features.
        clean_crime (pd.DataFrame): The dataset with crime data.

    Returns:
        pd.DataFrame: The preprocessed crime data with added date and time features.
    """
    crime_with_proximity = crime_with_proximity.merge(clean_crime[['id', 'areas']], on='id', how='inner')
    crime_with_proximity = crime_with_proximity.rename(columns={'areas': 'area_id'})
    crime_with_proximity['date'] = pd.to_datetime(crime_with_proximity['date'])
    crime_with_proximity['hour'] = crime_with_proximity['date'].dt.hour
    crime_with_proximity['day'] = crime_with_proximity['date'].dt.date
    crime_with_proximity['date_hour'] = crime_with_proximity['date'].dt.floor('h')
    crime_with_proximity.sort_values('date', inplace=True)
    return crime_with_proximity

# Compute rolling crime counts for districts and areas
def compute_crime_over_time(crime_with_proximity: pd.DataFrame, group_col: str, prefix: str) -> pd.DataFrame:
    """
    Compute rolling crime counts over various time windows.

    Args:
        crime_with_proximity (pd.DataFrame): The dataset with crime and proximity features.
        group_col (str): The column to group by (e.g., 'district' or 'area_id').
        prefix (str): The prefix for the new crime count columns (e.g., 'district' or 'area').

    Returns:
        pd.DataFrame: A DataFrame containing rolling crime counts.
    """
    crime_over_time = crime_with_proximity.groupby([group_col, pd.Grouper(key='date', freq='h')])['id'].count().reset_index().rename(columns={'id': f'{prefix}_crimes_this_hour'})
    time_windows = [1, 3, 6, 12, 24]

    for window in time_windows:
        crime_over_time[f'{prefix}_crimes_{window}_hours_prev'] = crime_over_time.groupby(group_col)[f'{prefix}_crimes_this_hour'].rolling(window=window, min_periods=1).sum().shift(1).reset_index(level=0, drop=True)

    return crime_over_time

# Merge rolling crime counts with the main dataset
def merge_crime_counts(crime_with_proximity: pd.DataFrame, district_crimes: pd.DataFrame, area_crimes: pd.DataFrame) -> pd.DataFrame:
    """
    Merge rolling crime counts back into the main crime dataset.

    Args:
        crime_with_proximity (pd.DataFrame): The dataset with crime and proximity features.
        district_crimes (pd.DataFrame): The dataset containing rolling district crime counts.
        area_crimes (pd.DataFrame): The dataset containing rolling area crime counts.

    Returns:
        pd.DataFrame: The crime dataset merged with district and area crime counts.
    """
    district_crimes = district_crimes[['district', 'date_hour', 'district_crimes_this_hour'] + [f'district_crimes_{w}_hours_prev' for w in [1, 3, 6, 12, 24]]]
    area_crimes = area_crimes[['area_id', 'date_hour', 'area_crimes_this_hour'] + [f'area_crimes_{w}_hours_prev' for w in [1, 3, 6, 12, 24]]]

    crime_with_proximity = pd.merge(crime_with_proximity, district_crimes, on=['district', 'date_hour'], how='left')
    crime_with_proximity = pd.merge(crime_with_proximity, area_crimes, on=['area_id', 'date_hour'], how='left')
    
    return crime_with_proximity

# Process and merge bike trips and train ridership data
def process_bike_train_data(crime_with_proximity: pd.DataFrame, clean_bike_trips: pd.DataFrame, clean_train_ridership: pd.DataFrame) -> pd.DataFrame:
    """
    Process and merge bike trips and train ridership data with the main crime dataset.

    Args:
        crime_with_proximity (pd.DataFrame): The dataset with crime and proximity features.
        clean_bike_trips (pd.DataFrame): The dataset containing bike trip data.
        clean_train_ridership (pd.DataFrame): The dataset containing train ridership data.

    Returns:
        pd.DataFrame: The dataset merged with bike trips and train ridership data.
    """
    clean_bike_trips['date'] = pd.to_datetime(clean_bike_trips['date'])
    clean_bike_trips['hour'] = clean_bike_trips['date'].dt.hour
    clean_bike_trips['date'] = clean_bike_trips['date'].dt.date

    grouped_bike_trips = clean_bike_trips.groupby(['date', 'hour', 'district'])['station_id'].agg('count').reset_index().rename(columns={'station_id': 'hourly_bike_rides'})
    grouped_bike_trips['date'] = pd.to_datetime(grouped_bike_trips['date'])
    
    clean_train_ridership['date'] = pd.to_datetime(clean_train_ridership['date'])
    grouped_train_ridership = clean_train_ridership.groupby(['date', 'district'])['rides'].agg('sum').reset_index()

    final_df = pd.merge(crime_with_proximity, grouped_bike_trips, on=['date', 'hour', 'district'], how='left').fillna(0)
    final_df = pd.merge(final_df, grouped_train_ridership, on=['date', 'district'], how='left').fillna(0)
    
    return final_df

# Merge additional datasets such as public health indicators and disadvantaged scores
def merge_additional_data(final_df: pd.DataFrame, agg_public_healthindicator: pd.DataFrame, clean_police_districts: pd.DataFrame) -> pd.DataFrame:
    """
    Merge public health indicators and disadvantaged scores with the main dataset.

    Args:
        final_df (pd.DataFrame): The main dataset.
        agg_public_healthindicator (pd.DataFrame): The dataset containing aggregated public health indicators.
        clean_police_districts (pd.DataFrame): The dataset containing police district data.

    Returns:
        pd.DataFrame: The dataset merged with public health indicators and disadvantaged scores.
    """
    agg_public_healthindicator.columns = ['district_' + col if col != 'district' else 'district' for col in agg_public_healthindicator.columns]
    final_df = pd.merge(final_df, agg_public_healthindicator, on='district', how='left')
    final_df = pd.merge(final_df, clean_police_districts[['district', 'disadvantaged_score']], on='district', how='left')
    return final_df

# Normalize specified columns in the dataset
def normalize_columns(df: pd.DataFrame, columns_to_normalize: List[str]) -> pd.DataFrame:
    """
    Normalize the specified columns in the DataFrame using MinMaxScaler.

    Args:
        df (pd.DataFrame): The dataset containing columns to be normalized.
        columns_to_normalize (List[str]): A list of column names to be normalized.

    Returns:
        pd.DataFrame: The dataset with normalized columns.
    """
    scaler = MinMaxScaler()
    df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
    return df

# Save the final preprocessed datasets
def save_final_datasets(area_final_df: pd.DataFrame, district_final_df: pd.DataFrame) -> None:
    """
    Save the final area and district pre-feature selection datasets.

    Args:
        area_final_df (pd.DataFrame): The final pre-feature selection dataset for areas.
        district_final_df (pd.DataFrame): The final pre-feature selection dataset for districts.
    """
    area_final_df.fillna(0, inplace=True)
    district_final_df.fillna(0, inplace=True)
    area_final_df.to_csv('../../data/pre_training/area_pre_feature_selection.csv', index=False)
    district_final_df.to_csv('../../data/pre_training/district_pre_feature_selection.csv', index=False)

# Main function to orchestrate the process
def main() -> None:
    """
    Main function to orchestrate the data processing workflow, including loading data, preprocessing,
    computing rolling crime counts, merging additional data, normalizing columns, and saving final datasets.
    """
    # Load data
    data = load_data()
    
    # Preprocess crime data
    crime_with_proximity = preprocess_crime_data(data["crime_with_proximity"], data["clean_crime"])

    # Compute rolling crime counts
    district_crimes = compute_crime_over_time(crime_with_proximity, 'district', 'district')
    area_crimes = compute_crime_over_time(crime_with_proximity, 'area_id', 'area')

    # Merge rolling crime counts with the main dataset
    crime_with_proximity = merge_crime_counts(crime_with_proximity, district_crimes, area_crimes)

    # Process and merge bike trips and train ridership data
    final_df = process_bike_train_data(crime_with_proximity, data["clean_bike_trips"], data["clean_train_ridership"])

    # Merge additional datasets
    final_df = merge_additional_data(final_df, data["agg_public_healthindicator"], data["clean_police_districts"])

    # Prepare area and district datasets for normalization
    area_columns_to_normalize = [...]  # Define your area columns to normalize
    district_columns_to_normalize = [...]  # Define your district columns to normalize
    area_final_df = normalize_columns(final_df, area_columns_to_normalize)
    district_final_df = normalize_columns(final_df, district_columns_to_normalize)

    # Save the final datasets
    save_final_datasets(area_final_df, district_final_df)

if __name__ == "__main__":
    main()