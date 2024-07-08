import pandas as pd
import datetime
from shapely.geometry import Point, Polygon
from shapely import geometry
import warnings
warnings.filterwarnings('ignore')

# Crime Dataset
raw_crime = pd.read_csv('../../data/raw/raw_crime.csv')
raw_crime = raw_crime[['Case Number', 'Date', 'Primary Type', 'Latitude', 'Longitude']]
raw_crime = raw_crime[raw_crime.Latitude.isnull() == False]
raw_crime.reset_index(drop=True, inplace=True)
raw_crime = raw_crime.sort_values(by='Date').reset_index(drop=True)
raw_crime = raw_crime.rename(columns={'Case Number' : 'id', 'Date': 'date', 'Primary Type' : 'type', 'Latitude' : 'lat', 'Longitude' : 'long'})
raw_crime['date'] = pd.to_datetime(raw_crime['date'])
raw_crime = raw_crime[(raw_crime['date'] >= '2016-01-01') & (raw_crime['date'] <= '2020-12-31')]
raw_crime.reset_index(drop=True, inplace=True)
raw_crime.to_csv('../../data/processed/clean_crime.csv', index=False)

# Handling the 3 311-Related Datasets
def clean_data(df):
    df = df[['Service Request Number', 'Creation Date', 'Completion Date', 'Type of Service Request', 'Latitude', 'Longitude']]
    df.rename(columns={'Creation Date':'start_date', 'Completion Date':'end_date', 'Service Request Number':'id', 'Type of Service Request':'type', 'Latitude':'lat', 'Longitude':'long'}, inplace=True)
    df.dropna(subset=['lat', 'long'], inplace=True)
    df.drop_duplicates(subset=['id'], inplace=True)
    
    df['start_date'] = pd.to_datetime(df['start_date'])
    df['end_date'] = pd.to_datetime(df['end_date'])
    df = df[(df['start_date'] >= '2016-01-01') & (df['start_date'] <= '2020-12-31')]

    df.sort_values(by='start_date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# Alley Light Outages Dataset
raw_alleylights = pd.read_csv('../../data/raw/raw_alleylights.csv')
raw_alleylights = clean_data(raw_alleylights)
raw_alleylights.to_csv('../../data/processed/clean_alleylights.csv', index=False)

# Streetlights Outage Datasets
raw_streetlights_allout = pd.read_csv('../../data/raw/raw_streetlights_allout.csv')
raw_streetlights_allout = clean_data(raw_streetlights_allout)
raw_streetlights_allout.to_csv('../../data/processed/clean_streetlights_allout.csv', index=False)

raw_streetlights_oneout = pd.read_csv('../../data/raw/raw_streetlights_oneout.csv')
raw_streetlights_oneout = clean_data(raw_streetlights_oneout)
raw_streetlights_oneout.to_csv('../../data/processed/clean_streetlights_oneout.csv', index=False)


# Vacant Buildings Dataset
raw_vacant_buildings = pd.read_csv('../../data/raw/raw_vacant_buildings.csv')
raw_vacant_buildings = raw_vacant_buildings[['DATE SERVICE REQUEST WAS RECEIVED', 'SERVICE REQUEST NUMBER', 'SERVICE REQUEST TYPE', 'LATITUDE', 'LONGITUDE']]
raw_vacant_buildings.rename(columns={'DATE SERVICE REQUEST WAS RECEIVED' : 'date', 'SERVICE REQUEST NUMBER' : 'id', 'SERVICE REQUEST TYPE' : 'type', 'LATITUDE': 'lat', 'LONGITUDE' : 'long'}, inplace=True)
raw_vacant_buildings.dropna(subset=['lat', 'long'], inplace=True)
raw_vacant_buildings['date'] = pd.to_datetime(raw_vacant_buildings['date'])
raw_vacant_buildings = raw_vacant_buildings[(raw_vacant_buildings['date'] >= '2016-01-01') & (raw_vacant_buildings['date'] <= '2020-12-31')]
raw_vacant_buildings.sort_values(by='date', inplace=True)
raw_vacant_buildings.reset_index(drop=True, inplace=True)
raw_vacant_buildings.to_csv('../../data/processed/clean_vacant_buildings.csv', index=False)

# Bike Trips Dataset
raw_bike_trips = pd.read_csv('../../data/raw/raw_bike_trips.csv')
raw_bike_trips = raw_bike_trips[['TRIP ID', 'START TIME', 'STOP TIME', 'FROM STATION ID', 'FROM STATION NAME', 'TO STATION ID', 'TO STATION NAME']]
bike_stations_a = raw_bike_trips[['FROM STATION ID', 'FROM STATION NAME']]
bike_stations_b = raw_bike_trips[['TO STATION ID', 'TO STATION NAME']]
bike_stations_a.rename(columns={'FROM STATION ID':'id','FROM STATION NAME':'station'}, inplace=True)
bike_stations_b.rename(columns={'TO STATION ID':'id','TO STATION NAME':'station'}, inplace=True)
bike_stations_in_use = pd.concat([bike_stations_a, bike_stations_b])
bike_stations_in_use.drop_duplicates(inplace=True)
bike_stations_in_use.reset_index(drop=True, inplace=True)

# Bike Stations Dataset
raw_bike_stations = pd.read_csv('../../data/raw/raw_bike_stations.csv')
raw_bike_stations = raw_bike_stations[raw_bike_stations.Status == 'In Service']
raw_bike_stations = raw_bike_stations[['ID', 'Station Name', 'Latitude', 'Longitude']]
raw_bike_stations.rename(columns={'ID' : 'id', 'Station Name' : 'station', 'Latitude' : 'lat', 'Longitude' : 'long'}, inplace=True)
raw_bike_stations = raw_bike_stations[['station', 'lat', 'long', 'id']]
raw_bike_stations = raw_bike_stations.merge(bike_stations_a, on='station', how='inner')
raw_bike_stations.drop_duplicates(inplace=True)
raw_bike_stations = raw_bike_stations[raw_bike_stations.station.duplicated() == False]
raw_bike_stations.reset_index(drop=True, inplace=True)
raw_bike_stations.rename(columns={'id_x' : 'id'}, inplace=True)
raw_bike_stations = raw_bike_stations[['station', 'lat', 'long', 'id']]
raw_bike_stations.to_csv('../../data/processed/clean_bike_stations.csv', index=False)

# Cleaned Bike Trips Dataset
raw_bike_trips_a = raw_bike_trips[['TRIP ID', 'START TIME', 'FROM STATION ID', 'FROM STATION NAME']]
raw_bike_trips_b = raw_bike_trips[['TRIP ID', 'START TIME', 'TO STATION ID', 'TO STATION NAME']]
raw_bike_trips_a.rename(columns={'TRIP ID' : 'id', 'START TIME' : 'date', 'FROM STATION ID' : 'station_id', 'FROM STATION NAME' : 'station_name'}, inplace=True)
raw_bike_trips_b.rename(columns={'TRIP ID' : 'id', 'START TIME' : 'date', 'TO STATION ID' : 'station_id', 'TO STATION NAME' : 'station_name'}, inplace=True)
concat_bike_trips = pd.concat([raw_bike_trips_a, raw_bike_trips_b])
concat_bike_trips.reset_index(drop=True, inplace=True)
concat_bike_trips['date'] = pd.to_datetime(concat_bike_trips['date'], format='%m/%d/%Y %I:%M:%S %p')
concat_bike_trips.sort_values(by='date', inplace=True)
concat_bike_trips.reset_index(drop=True, inplace=True)
concat_bike_trips = pd.merge(concat_bike_trips, raw_bike_stations, left_on='station_id', right_on='id', how='left')
concat_bike_trips = concat_bike_trips[['date', 'station_id', 'station_name', 'lat', 'long']].drop_duplicates()
concat_bike_trips.reset_index(inplace=True, drop=True)
concat_bike_trips['id'] = concat_bike_trips.index
concat_bike_trips = concat_bike_trips[(concat_bike_trips['date'] >= '2016-01-01') & (concat_bike_trips['date'] <= '2020-12-31')]
concat_bike_trips.to_csv('../../data/processed/clean_bike_trips.csv', index=False)

# Disadvantaged Areas Dataset
raw_disadvantaged_areas = pd.read_csv('../../data/raw/raw_disadvantaged_areas.csv')
def convert_to_polygon_1(df):
    updated_polygons = []
    for row in df['the_geom']:
        target = row.replace('MULTIPOLYGON (((', '').replace(')))', '')
        points = target.split(', ')
        final = []
        for point in points:
            temp = point.split(' ')
            tup = float(temp[0]), float(temp[1]) 
            final.append(tup)
        updated_polygons.append(final)
    return updated_polygons

clean_disadvantaged_areas = pd.DataFrame()
clean_disadvantaged_areas['poly'] = convert_to_polygon_1(raw_disadvantaged_areas)
clean_disadvantaged_areas.to_csv('../../data/processed/clean_disadvantaged_areas.csv', index=False)

# Areas Dataset
raw_areas = pd.read_csv('../../data/raw/raw_areas.csv')
raw_areas = raw_areas[['the_geom', 'AREA_NUMBE']]
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

raw_areas['poly'] = convert_to_polygon_2(raw_areas)
raw_areas = raw_areas[['AREA_NUMBE', 'poly']]
raw_areas.rename(columns={'AREA_NUMBE' : 'id'}, inplace=True)
raw_areas.to_csv('../../data/processed/clean_areas.csv', index=False)

# Police Stations Dataset
raw_police_stations = pd.read_csv('../../data/raw/raw_police_stations.csv')
raw_police_stations = raw_police_stations[['DISTRICT', 'LATITUDE', 'LONGITUDE']]
raw_police_stations.rename(columns={'DISTRICT' : 'id', 'LATITUDE' : 'lat', 'LONGITUDE' : 'long'}, inplace=True)
raw_police_stations.id.iloc[0] = 0
raw_police_stations.to_csv('../../data/processed/clean_police_stations.csv', index=False)

# Public Health Indicators Dataset
raw_public_healthindicator = pd.read_csv('../../data/raw/raw_publichealth_indicator.csv')
raw_public_healthindicator = raw_public_healthindicator[['Community Area', 'Unemployment', 'Per Capita Income', 'No High School Diploma', 'Dependency', 'Crowded Housing', 'Below Poverty Level']]
raw_public_healthindicator.rename(columns={'Community Area' : 'id', 'Unemployment' : 'unemployment', 'Per Capita Income' : 'per_capita_income', 'No High School Diploma' : 'no_hs_dip', 'Dependency' : 'gov_depend', 'Crowded Housing' : 'crowded_housing', 'Below Poverty Level' : 'below_pov'}, inplace=True)
raw_public_healthindicator.unemployment = raw_public_healthindicator.unemployment / 100
raw_public_healthindicator.no_hs_dip = raw_public_healthindicator.no_hs_dip / 100
raw_public_healthindicator.gov_depend = raw_public_healthindicator.gov_depend / 100
raw_public_healthindicator.crowded_housing = raw_public_healthindicator.crowded_housing / 100
raw_public_healthindicator.below_pov = raw_public_healthindicator.below_pov / 100
raw_public_healthindicator.to_csv('../../data/processed/clean_public_healthindicator.csv', index=False)

# Police Districts Geometry Dataset
raw_police_districts = pd.read_csv('../../data/raw/raw_police_districts.csv')
raw_police_districts.rename(columns={'DIST_NUM':'district'}, inplace=True)
raw_police_districts = raw_police_districts[['district','the_geom']]
def convert_to_polygon(df):
    updated_polygons = []
    for row in df['the_geom']:
        target = row.replace('MULTIPOLYGON (((', '').replace(')))', '')
        points = target.split(', ')
        final = []
        for point in points:
            temp = point.split(' ')
            tup = float(temp[1].replace(')', '').replace('(', '')), float(temp[0].replace(')', '').replace('(', '')) 
            final.append(tup)
        polygons = Polygon(final)
        updated_polygons.append(polygons)
    return updated_polygons

raw_police_districts['geom'] = convert_to_polygon(raw_police_districts)
raw_police_districts = raw_police_districts[['district', 'geom']]
raw_police_districts.to_csv('../../data/processed/clean_police_districts.csv', index=False)

# Bus Stops Dataset
raw_bus_stops = pd.read_csv('../../data/raw/raw_bus_stops.csv')
raw_bus_stops = raw_bus_stops[raw_bus_stops['STATUS'] == 'In Service']
raw_bus_stops['routes_split'] = raw_bus_stops['ROUTESSTPG'].str.split(',')
raw_bus_stops = raw_bus_stops.explode('routes_split')
raw_bus_stops.columns = raw_bus_stops.columns.str.lower()
raw_bus_stops.rename(columns={'y':'lat','x':'long','routes_split':'route','systemstop':'stop_id'}, inplace=True)
raw_bus_stops = raw_bus_stops[['stop_id','route','name','lat','long']]
raw_bus_stops.reset_index(drop=True, inplace=True)
raw_bus_stops.to_csv('../../data/processed/clean_bus_stops.csv', index=False)

# Train Stations Dataset
raw_train_stations = pd.read_csv('../../data/raw/raw_train_stations.csv')
raw_train_stations.columns = raw_train_stations.columns.str.lower()
raw_train_stations.rename(columns={'x':'long','y':'lat','name':'station_name','rail line':'line','station id':'id'}, inplace=True)
raw_train_stations = raw_train_stations[['id','line','station_name','lat','long']]
raw_train_stations.to_csv('../../data/processed/clean_train_stations.csv', index=False)

# Train Ridership Dataset
raw_train_ridership = pd.read_csv('../../data/raw/raw_train_ridership.csv')
raw_train_ridership.rename(columns={'stationname':'station_name'}, inplace=True)
raw_train_ridership = pd.merge(left=raw_train_stations[['line','station_name','lat','long']], right=raw_train_ridership[['station_name','date','rides']], on='station_name', how='right')
raw_train_ridership.dropna(subset=['lat','long'], inplace=True)
raw_train_ridership.reset_index(drop=True, inplace=True)
raw_train_ridership['date'] = pd.to_datetime(raw_train_ridership['date'])
raw_train_ridership = raw_train_ridership[(raw_train_ridership['date'] >= '2016-01-01') & (raw_train_ridership['date'] <= '2020-12-31')]
raw_train_ridership.to_csv('../../data/processed/clean_train_ridership.csv', index=False)