import pandas as pd
import os
from shapely.geometry import Point, Polygon
from shapely import geometry
import ast

directory = '../../data/processed'
void = ['clean_areas.csv', 'clean_disadvantaged_areas.csv', 'clean_police_districts.csv', 'clean_public_healthindicator.csv', 'clean_train_ridership.csv', 'clean_bike_trips.csv']

# Creating a Dict to Store Datasets with long,lat columns
def read_in(directory, void):
    data = {}
    for filename in os.listdir(directory):
        if filename not in void and filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            data[filename[:-4]] = pd.read_csv(file_path)
            print(f'{filename[:-4]} successfully read in')
    return data

data = read_in(directory, void)

# Manually reading in datasets with Geom datatypes for manual cleaning
clean_police_districts = pd.read_csv('../../data/processed/clean_police_districts.csv')
clean_public_healthindicator = pd.read_csv('../../data/processed/clean_public_healthindicator.csv')
clean_areas = pd.read_csv('../../data/processed/clean_areas.csv')
clean_disadvantaged_areas = pd.read_csv('../../data/processed/clean_disadvantaged_areas.csv')

# Converting a polygon string to a polygon datatype
def parse_polygon1(polygon_string):
    points = polygon_string.strip('POLYGON ((').strip('))').split(', ')
    points = [tuple(map(float, point.split())) for point in points]
    return Polygon(points)

# Converting a list of tuples to a polygon datatype
def parse_polygon2(polygon_string):
    points = ast.literal_eval(polygon_string)
    return Polygon(points)

# Swapping coordinates of polygon object storing long,lats as flipped
def swap_coordinates(polygon):
    if polygon.is_empty:
        return polygon
    swapped_coords = [(y, x) for x, y in polygon.exterior.coords]
    return Polygon(swapped_coords)

clean_police_districts['geom'] = clean_police_districts['geom'].apply(parse_polygon1)
clean_areas['poly'] = clean_areas['poly'].apply(parse_polygon2)
clean_disadvantaged_areas['poly'] = clean_disadvantaged_areas['poly'].apply(parse_polygon2)
clean_disadvantaged_areas['poly'] = clean_disadvantaged_areas['poly'].apply(swap_coordinates)

# Determining centroid of each polygon object, to pull long,lat data from it for later functions
clean_areas['centroid'] = clean_areas['poly'].apply(lambda poly : poly.centroid)
clean_public_healthindicator = pd.merge(left=clean_public_healthindicator, right=clean_areas[['id','centroid']], on='id', how='left')
clean_public_healthindicator.drop('id', axis=1, inplace=True)
clean_disadvantaged_areas['centroid'] = clean_disadvantaged_areas['poly'].apply(lambda poly : poly.centroid)

# Assigning districts to each dataset
def determine_within(df):
    statuses = []
    districts = []
    cent = True if 'centroid' in df.columns else False # Adding case for datasets without long,lat columns

    for i in range(len(df)):
        point = df.centroid.loc[i] if cent else geometry.Point(df.lat.loc[i], df.long.loc[i]) # Using a point object
        status = 0

        for index, row in clean_police_districts.iterrows(): # Iterating through each district until finding correct one
            district = row['district']
            geom = row['geom']
            if geom.contains(point): 
                status = 1
                break
        statuses.append(status)
        districts.append(district)

    df['status'] = statuses
    df['district'] = districts
    df = df[df['status'] == 1].drop('status', axis=1) # Removing any rows that are not in an districts (out of the city boundaries)

    if cent: df.drop('centroid', axis=1, inplace=True) # Centroid column no longer needed

    return df

# Assigning districts to all datasets with long,lat columns
def determine_districts(data):
    for df_name, df in data.items():
        data[df_name] = determine_within(df)
        print(f'{df_name} successfully completed')
    return data

clean_disadvantaged_areas = determine_within(clean_disadvantaged_areas)
clean_public_healthindicator = determine_within(clean_public_healthindicator)
data = determine_districts(data)

# Using the found district of each station to quickly assign district data to bike and train ridership (since they are both station dependent)
clean_bike_trips = pd.read_csv('../../data/processed/clean_bike_trips.csv')
clean_train_ridership = pd.read_csv('../../data/processed/clean_train_ridership.csv')
clean_bike_trips = clean_bike_trips.merge(right=data['clean_bike_stations'][['id', 'district']], how='left', left_on='station_id', right_on='id').dropna(subset=['district'])
clean_train_ridership = clean_train_ridership.merge(right=data['clean_train_stations'][['station_name', 'district']], how='left', on='station_name').dropna(subset=['district'])

# Determining how many official disadvantaged areas are within each police district
clean_police_districts = clean_police_districts.merge(right=clean_disadvantaged_areas.groupby('district').agg('count'), how='left', on='district').fillna(0).rename(columns={'poly':'disadvantaged_score'})

clean_bike_trips.to_csv('../../data/processed/clean_bike_trips.csv', index=False)
clean_train_ridership.to_csv('../../data/processed/clean_train_ridership.csv', index=False)
clean_police_districts.to_csv('../../data/processed/clean_police_districts.csv', index=False)
clean_public_healthindicator.to_csv('../../data/processed/clean_public_healthindicator.csv', index=False)

def save_data():
    for df_name, df in data.items():
        df.to_csv(f'../../data/processed/{df_name}.csv', index=False)

save_data()