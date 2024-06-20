import pandas as pd
import datetime

# Load the raw crime data
raw_crime = pd.read_csv('../../data/raw/raw_crime.csv')
# Identify duplicated case numbers in the dataset
duplicated_case_numbers = raw_crime[raw_crime['Case Number'].duplicated() == True]
# Identify completely duplicated rows in the dataset
duplicated_rows = raw_crime[raw_crime.duplicated() == True]
# Select relevant columns for further processing
raw_crime = raw_crime[['Case Number', 'Date', 'Primary Type', 'Latitude', 'Longitude']]
# Remove rows with null Latitude values
raw_crime = raw_crime[raw_crime.Latitude.isnull() == False]
# Reset the index of the dataframe
raw_crime.reset_index(drop=True, inplace=True)
# Sort the dataframe by the Date column
raw_crime = raw_crime.sort_values(by='Date').reset_index(drop=True)
# Rename the columns to more appropriate names
raw_crime = raw_crime.rename(columns={'Case Number' : 'id', 'Date': 'date', 'Primary Type' : 'type', 'Latitude' : 'lat', 'Longitude' : 'long'})
# Sort the dataframe by the newly renamed date column
raw_crime = raw_crime.sort_values(by='date').reset_index(drop=True)
# Save the cleaned crime data to a CSV file
raw_crime.to_csv('../../data/processed/clean_crime.csv', index = False)

# Load the raw alley lights data
raw_alleylights = pd.read_csv('../../data/raw/raw_alleylights.csv')
# Select relevant columns for further processing
raw_alleylights = raw_alleylights[['Creation Date', 'Service Request Number', 'Type of Service Request', 'Latitude', 'Longitude']]
# Rename the columns to more appropriate names
raw_alleylights.rename(columns={'Creation Date' : 'date', 'Service Request Number' : 'id', 'Type of Service Request' : 'type', 'Latitude': 'lat', 'Longitude' : 'long'}, inplace=True)
# Remove rows with null Latitude and Longitude values
raw_alleylights.dropna(subset=['lat', 'long'], inplace=True)
# Sort the dataframe by the date column
raw_alleylights.sort_values(by='date', inplace=True)
# Reset the index of the dataframe
raw_alleylights.reset_index(drop=True, inplace=True)
# Remove duplicated rows from the dataframe
raw_alleylights = raw_alleylights[raw_alleylights.duplicated() == False]
# Update the type column to 'alley'
raw_alleylights['type'] = 'alley'

# Function to clean various dataframes
def clean_data(df, type):
    df = df[['Creation Date', 'Service Request Number', 'Type of Service Request', 'Latitude', 'Longitude']]
    df.rename(columns={'Creation Date' : 'date', 'Service Request Number' : 'id', 'Type of Service Request' : 'type', 'Latitude': 'lat', 'Longitude' : 'long'}, inplace=True)
    perc_null = sum(df.lat.isnull()) / len(df)
    if perc_null > 0.005:
        print(f'perc_null of {perc_null} is too large to clean')
    else:
        df.dropna(subset=['lat', 'long'], inplace=True)
        print(f'successfully removed {sum(df.lat.isnull())} nulls or {perc_null}%')
    df.sort_values(by='date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['type'] = type
    return df

# Load and clean streetlights all-out data
raw_streetlights_allout = pd.read_csv('../../data/raw/raw_streetlights_allout.csv')
raw_streetlights_allout = clean_data(raw_streetlights_allout, 'sl_all')

# Load and clean streetlights one-out data
raw_streetlights_oneout = pd.read_csv('../../data/raw/raw_streetlights_oneout.csv')
raw_streetlights_oneout = clean_data(raw_streetlights_oneout, 'sl_one')

# Load and clean vacant buildings data
raw_vacant_buildings = pd.read_csv('../../data/raw/raw_vacant_buildings.csv')
raw_vacant_buildings = raw_vacant_buildings[['DATE SERVICE REQUEST WAS RECEIVED', 'SERVICE REQUEST NUMBER', 'SERVICE REQUEST TYPE', 'LATITUDE', 'LONGITUDE']]
raw_vacant_buildings.rename(columns={'DATE SERVICE REQUEST WAS RECEIVED' : 'date', 'SERVICE REQUEST NUMBER' : 'id', 'SERVICE REQUEST TYPE' : 'type', 'LATITUDE': 'lat', 'LONGITUDE' : 'long'}, inplace=True)
raw_vacant_buildings.dropna(subset=['lat', 'long'], inplace=True)
raw_vacant_buildings.sort_values(by='date', inplace=True)
raw_vacant_buildings.reset_index(drop=True, inplace=True)
raw_vacant_buildings['type'] = 'vacant_building'

# Combine all the cleaned dataframes
dfs = [raw_alleylights, raw_streetlights_allout, raw_streetlights_oneout, raw_vacant_buildings]
raw_311 = pd.concat(dfs)
# Convert the date column to datetime format
raw_311['date'] = [datetime.datetime.strptime(raw_311.date.iloc[i], '%m/%d/%Y') for i in range(len(raw_311))]
# Sort the dataframe by the date column
raw_311.sort_values(by='date', inplace=True)
# Reset the index of the dataframe
raw_311.reset_index(drop=True, inplace=True)
# Remove the first four rows of the dataframe
clean_311 = raw_311.tail(-4)
# Save the cleaned 311 data to a CSV file
clean_311.to_csv('../../data/processed/clean_311.csv', index = False)

# Load the raw bike trips data
raw_bike_trips = pd.read_csv('../../data/raw/raw_bike_trips.csv')
# Select relevant columns for further processing
raw_bike_trips = raw_bike_trips[['TRIP ID', 'START TIME', 'STOP TIME', 'FROM STATION ID', 'FROM STATION NAME', 'TO STATION ID', 'TO STATION NAME']]

# Create dataframes for the bike stations
bike_stations_a = raw_bike_trips[['FROM STATION ID', 'FROM STATION NAME']]
bike_stations_b = raw_bike_trips[['TO STATION ID', 'TO STATION NAME']]
# Rename the columns for consistency
bike_stations_a.rename(columns={'FROM STATION ID':'id','FROM STATION NAME':'station'}, inplace=True)
bike_stations_b.rename(columns={'TO STATION ID':'id','TO STATION NAME':'station'}, inplace=True)

# Concatenate the bike stations dataframes and remove duplicates
bike_stations_in_use = pd.concat([bike_stations_a, bike_stations_b])
bike_stations_in_use.drop_duplicates(inplace=True)
bike_stations_in_use.reset_index(drop=True, inplace=True)

# Load the raw bike stations data
raw_bike_stations = pd.read_csv('../../data/raw/raw_bike_stations.csv')
# Filter the dataframe to include only in-service stations
raw_bike_stations = raw_bike_stations[raw_bike_stations.Status == 'In Service']
# Select relevant columns for further processing
raw_bike_stations = raw_bike_stations[['ID', 'Station Name', 'Latitude', 'Longitude']]
# Rename the columns for consistency
raw_bike_stations.rename(columns={'ID' : 'id', 'Station Name' : 'station', 'Latitude' : 'lat', 'Longitude' : 'long'}, inplace=True)
# Reorder the columns
raw_bike_stations = raw_bike_stations[['station', 'lat', 'long', 'id']]
# Merge the bike stations data with the bike stations in use data
raw_bike_stations = raw_bike_stations.merge(bike_stations_a, on='station', how='inner')
# Remove duplicates
raw_bike_stations.drop_duplicates(inplace=True)
raw_bike_stations = raw_bike_stations[raw_bike_stations.station.duplicated() == False]
# Reset the index of the dataframe
raw_bike_stations.reset_index(drop=True, inplace=True)
# Rename the id column to avoid duplication
raw_bike_stations.rename(columns={'id_x' : 'id'}, inplace=True)
# Reorder the columns
raw_bike_stations = raw_bike_stations[['station', 'lat', 'long', 'id']]
# Save the cleaned bike stations data to a CSV file
raw_bike_stations.to_csv('../../data/processed/clean_bike_stations.csv', index = False)

# Create dataframes for the bike trips
raw_bike_trips_a = raw_bike_trips[['TRIP ID', 'START TIME', 'FROM STATION ID', 'FROM STATION NAME']]
raw_bike_trips_b = raw_bike_trips[['TRIP ID', 'START TIME', 'TO STATION ID', 'TO STATION NAME']]
# Rename the columns for consistency
raw_bike_trips_a.rename(columns={'TRIP ID' : 'id', 'START TIME' : 'date', 'FROM STATION ID' : 'station_id', 'FROM STATION NAME' : 'station_name'}, inplace=True)
raw_bike_trips_b.rename(columns={'TRIP ID' : 'id', 'START TIME' : 'date', 'TO STATION ID' : 'station_id', 'TO STATION NAME' : 'station_name'}, inplace=True)
# Concatenate the bike trips dataframes
concat_bike_trips = pd.concat([raw_bike_trips_a, raw_bike_trips_b])
# Reset the index of the dataframe
concat_bike_trips.reset_index(drop=True, inplace=True)
# Convert the date column to datetime format
concat_bike_trips['date'] = pd.to_datetime(concat_bike_trips['date'], format='%m/%d/%Y %I:%M:%S %p')
# Sort the dataframe by the date column
concat_bike_trips.sort_values(by='date', inplace=True)
# Reset the index of the dataframe
concat_bike_trips.reset_index(drop=True, inplace=True)
# Save the cleaned bike trips data to a CSV file
concat_bike_trips.to_csv('../../data/processed/clean_bike_trips.csv', index = False)

# Function to convert MULTIPOLYGON data to a list of tuples
def convert_to_polygon_1(df):
    updated_polygons = []
    for row in df['the_geom']:
        target = row.replace('MULTIPOLYGON (((', '').replace(')))', '')
        points = target.split(', ')
        final = []
        for point in points:
            temp = point.split(' ')
            tup = float(temp[1].replace('(', '').replace(')', '')), float(temp[0].replace('(', '').replace(')', '')) 
            final.append(tup)
        updated_polygons.append(final)
    return updated_polygons

# Load the raw disadvantaged areas data
raw_disadvantaged_areas = pd.read_csv('../../data/raw/raw_disadvantaged_areas.csv')
# Convert the geometry data to a list of tuples
clean_disadvantaged_areas = pd.DataFrame()
clean_disadvantaged_areas['poly'] = convert_to_polygon_1(raw_disadvantaged_areas)
# Save the cleaned disadvantaged areas data to a CSV file
clean_disadvantaged_areas.to_csv('../../data/processed/clean_disadvantaged_areas.csv', index = False)

# Load the raw bus stations data
raw_bus_stations = pd.read_csv('../../data/raw/raw_bus_stations.csv')
# Select relevant columns for further processing
raw_bus_stations = raw_bus_stations[['stop_id','cta_stop_name','lat','long']]
# Save the cleaned bus stations data to a CSV file
raw_bus_stations.to_csv('../../data/processed/clean_bus_stations.csv', index=False)

# Load the raw CTA stations data
raw_cta_stations = pd.read_csv('../../data/raw/raw_cta_stations.csv')
# Select relevant columns for further processing
raw_cta_stations = raw_cta_stations[['station_number','station_name','lat','long']]
# Save the cleaned CTA stations data to a CSV file
raw_cta_stations.to_csv('../../data/processed/clean_cta_stations.csv', index=False)

# Load the raw ridership data
raw_ridership = pd.read_csv('../../data/raw/raw_ridership.csv')
# Save the cleaned ridership data to a CSV file
raw_ridership.to_csv('../../data/processed/clean_ridership.csv', index=False)

# Load the raw areas data
raw_areas = pd.read_csv('../../data/raw/raw_areas.csv')
# Select relevant columns for further processing
raw_areas = raw_areas[['the_geom', 'AREA_NUMBE']]

# Function to convert MULTIPOLYGON data to a list of tuples
def convert_to_polygon_2(df):
    updated_polygons = []
    for row in df['the_geom']:
        target = row.replace('MULTIPOLYGON (((', '').replace(')))', '')
        points = target.split(', ')
        final = []
        for point in points:
            temp = point.split(' ')
            tup = float(temp[1].replace('(', '').replace(')', '')), float(temp[0].replace('(', '').replace(')', '')) 
            final.append(tup)
        updated_polygons.append(final)
    return updated_polygons

# Convert the geometry data to a list of tuples
raw_areas['poly'] = convert_to_polygon_2(raw_areas)
# Select relevant columns for further processing and rename them
raw_areas = raw_areas[['AREA_NUMBE', 'poly']]
raw_areas.rename(columns={'AREA_NUMBE' : 'id'}, inplace=True)
# Save the cleaned areas data to a CSV file
raw_areas.to_csv('../../data/processed/clean_areas.csv', index=False)

# Load the raw police stations data
raw_police_stations = pd.read_csv('../../data/raw/raw_police_stations.csv')
# Select relevant columns for further processing
raw_police_stations = raw_police_stations[['DISTRICT', 'LATITUDE', 'LONGITUDE']]
# Rename the columns for consistency
raw_police_stations.rename(columns={'DISTRICT' : 'id', 'LATITUDE' : 'lat', 'LONGITUDE' : 'long'}, inplace=True)
# Fix the id value for the first row
raw_police_stations.id.iloc[0] = 0
# Save the cleaned police stations data to a CSV file
raw_police_stations.to_csv('../../data/processed/clean_police_stations.csv', index=False)

# Load the raw public health indicator data
raw_public_healthindicator = pd.read_csv('../../data/raw/raw_publichealth_indicator.csv')
# Select relevant columns for further processing and rename them
raw_public_healthindicator = raw_public_healthindicator[['Community Area', 'Unemployment', 'Per Capita Income', 'No High School Diploma', 'Dependency', 'Crowded Housing', 'Below Poverty Level']]
raw_public_healthindicator.rename(columns={'Community Area' : 'id', 'Community Area Name' : 'name', 'Unemployment' : 'unemployment', 'Per Capita Income' : 'per_capita_income', 'No High School Diploma' : 'no_hs_dip', 'Dependency' : 'gov_depend', 'Crowded Housing' : 'crowded_housing', 'Below Poverty Level' : 'below_pov'}, inplace=True)
# Convert percentages to proportions
raw_public_healthindicator.unemployment = raw_public_healthindicator.unemployment / 100
raw_public_healthindicator.no_hs_dip = raw_public_healthindicator.no_hs_dip / 100
raw_public_healthindicator.gov_depend = raw_public_healthindicator.gov_depend / 100
raw_public_healthindicator.crowded_housing = raw_public_healthindicator.crowded_housing / 100
raw_public_healthindicator.below_pov = raw_public_healthindicator.below_pov / 100
# Save the cleaned public health indicator data to a CSV file
raw_public_healthindicator.to_csv('../../data/processed/clean_public_healthindicator.csv', index=False)

# Load the raw police sentiment data
raw_police_sentiment = pd.read_csv('../../data/raw/raw_police_sentiment.csv')

# Select relevant columns for further processing
raw_police_sentiment = raw_police_sentiment[[
    'DISTRICT', 'SECTOR', 'START_DATE', 'END_DATE',
    'SAFETY', 'TRUST', 'T_LISTEN', 'T_RESPECT',
    'S_RACE_AFRICAN_AMERICAN', 'S_RACE_ASIAN_AMERICAN', 'S_RACE_HISPANIC', 'S_RACE_WHITE',
    'S_AGE_LOW', 'S_AGE_MEDIUM', 'S_AGE_HIGH',
    'S_SEX_FEMALE', 'S_SEX_MALE',
    'S_EDUCATION_LOW', 'S_EDUCATION_MEDIUM', 'S_EDUCATION_HIGH',
    'S_INCOME_LOW', 'S_INCOME_MEDIUM', 'S_INCOME_HIGH'
]]

# Convert column names to lowercase
raw_police_sentiment.columns = [col.lower() for col in raw_police_sentiment.columns]

# Convert date columns to datetime format
raw_police_sentiment['start_date'] = pd.to_datetime(raw_police_sentiment['start_date'])
raw_police_sentiment['end_date'] = pd.to_datetime(raw_police_sentiment['end_date'])

# Drop rows with null values in the 'district' and 'sector' columns
raw_police_sentiment.dropna(subset=['district','sector'], inplace=True)

# Reset the index of the dataframe
raw_police_sentiment.reset_index(drop=True, inplace=True)

# List of columns to normalize
columns_to_normalize = [
    'safety', 'trust', 't_listen', 't_respect', 's_race_african_american',
    's_race_asian_american', 's_race_hispanic', 's_race_white', 's_age_low',
    's_age_medium', 's_age_high', 's_sex_female', 's_sex_male',
    's_education_low', 's_education_medium', 's_education_high',
    's_income_low', 's_income_medium', 's_income_high'
]

# Normalize the selected columns to a 0-1 range
for column in columns_to_normalize:
    min_value = raw_police_sentiment[column].min()
    max_value = raw_police_sentiment[column].max()
    raw_police_sentiment[column] = (raw_police_sentiment[column] - min_value) / (max_value - min_value)

# Save the cleaned police sentiment data to a CSV file
raw_police_sentiment.to_csv('../../data/processed/clean_police_sentiment.csv', index=False)