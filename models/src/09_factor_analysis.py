import pandas as pd
import numpy as np
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
from shapely import geometry
import geopandas as gpd
import geodatasets
from sklearn.preprocessing import MinMaxScaler


# Load data
crime_data = pd.read_csv('../../data/pre_training/pre_aggregation_crime_data.csv')
top_features = [
    'bus_stops_distance_0.1', 'alleylights_distance_0.1', 'police_stations_distance_1',
    'streetlights_oneout_distance_0.1', 'streetlights_allout_distance_0.5', 'streetlights_oneout_distance_0.3',
    'streetlights_allout_distance_0.3', 'bike_rides_within_0.5_and_15_min', 'area_crimes_3_hours_prev',
    'area_crimes_1_hours_prev', 'alleylights_distance_0.3', 'bike_rides_within_0.1_and_10_min',
    'bike_rides_within_0.1_and_5_min'
]
crime_data_selected = crime_data[top_features].fillna(0)

# Normalize to prepare for factor analysis
crime_data_selected_scaler = MinMaxScaler()
crime_data_selected_normalized = pd.DataFrame(crime_data_selected_scaler.fit_transform(crime_data_selected), columns=top_features)

# Adequacy Test
chi_square_val, p_val = calculate_bartlett_sphericity(crime_data_selected_normalized)
print(f"Bartlett's Test: chi-square value: {chi_square_val}, p-value: {p_val}")
kmo_all, kmo_model = calculate_kmo(crime_data_selected_normalized)
print(f"KMO Test: All: {kmo_all}, Model: {kmo_model}")

# Factor analysis
fa = FactorAnalyzer()
fa.fit(crime_data_selected)
eigenvalues, _ = fa.get_eigenvalues()
print("Eigenvalues:", eigenvalues)

# Scree Plot for factor selection
num_factors_kaiser = sum(eigenvalues > 1)
plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, marker='o')
plt.xlabel('Number of Factors')
plt.ylabel('Eigenvalue')
plt.title('Scree Plot')
plt.axhline(y=1, color='r', linestyle='--')  # Kaiser criterion threshold
plt.show()

# Factor analysis with Varimax rotation
fa = FactorAnalyzer(n_factors=num_factors_kaiser, rotation='varimax')
fa.fit(crime_data_selected)
fa_loadings = pd.DataFrame(data=fa.loadings_, columns=[f'Factor {i+1}' for i in range(num_factors_kaiser)], index=crime_data_selected.columns)

# Factor analysis with five factors for comparison
fa_five = FactorAnalyzer(n_factors=5, rotation='varimax')
fa_five.fit(crime_data_selected)
fa_five_loadings = pd.DataFrame(data=fa_five.loadings_, columns=[f'Factor {i+1}' for i in range(5)], index=crime_data_selected.columns)

# Function to evaluate factors
def evaluate_factors(factor_df: pd.DataFrame, threshold: float) -> None:
    """
    Evaluates and prints the features with factor loadings above a specified threshold.

    Args:
    - factor_df: DataFrame with factor loadings.
    - threshold: Minimum loading value to consider a feature.
    """
    s = 0
    for factor in factor_df:
        features = list(factor_df[factor_df[factor] > threshold].index)
        features_with_vals = [(feature, float(round(factor_df[factor][feature], 3))) for feature in features]
        print(factor, features_with_vals)
        s += sum([val for _, val in features_with_vals])
    print(f'Cumulative Factor Loadings: {s}')


# Evaluate the factor loadings
evaluate_factors(fa_loadings, 0.4)
evaluate_factors(fa_five_loadings, 0.4)

# Factor scores over districts
transformed_data = pd.DataFrame(fa_five.transform(crime_data_selected), 
                                columns=['Bike Activity', 'Alleylight Availability', 'Streetlight Outages', 
                                         'Partial Streetlight Outages', 'Recent Crime Activity'])
crime_data_with_factors = pd.concat([crime_data, transformed_data], axis=1)

factors_by_district = crime_data_with_factors.groupby('district')[
    ['Bike Activity', 'Alleylight Availability', 'Streetlight Outages', 'Partial Streetlight Outages', 'Recent Crime Activity']
].agg('mean').reset_index()

# Load additional geospatial data
clean_police_districts = pd.read_csv('../../data/processed/clean_police_districts.csv')
clean_areas = pd.read_csv('../../data/processed/clean_areas.csv')
disadvantaged_areas_within_areas = pd.read_csv('../../data/processed/disadvantaged_areas_within_areas.csv')


# Function to parse polygons from WKT format
def parse_polygon(polygon_string: str) -> Polygon:
    points = polygon_string.strip('POLYGON ((').strip('))').split(', ')
    points = [tuple(map(float, point.split())) for point in points]
    return Polygon(points)


# Function to swap the coordinates of a polygon
def swap_coordinates(polygon: Polygon) -> Polygon:
    if polygon.is_empty:
        return polygon
    swapped_coords = [(y, x) for x, y in polygon.exterior.coords]
    return Polygon(swapped_coords)


# Preprocess the geospatial data by parsing polygons
clean_police_districts['geom'] = clean_police_districts['geom'].apply(parse_polygon).apply(swap_coordinates)
clean_areas['poly'] = clean_areas['poly'].apply(parse_polygon).apply(swap_coordinates)
disadvantaged_areas_within_areas['poly'] = disadvantaged_areas_within_areas['poly'].apply(parse_polygon).apply(swap_coordinates)

# Group areas by districts and disadvantaged areas by area
district_to_areas = {}
for idx, row in clean_areas.iterrows():
    district_to_areas.setdefault(row['district'], []).append((row['id'], row['poly']))

disadvantaged_areas_to_areas = {}
for idx, row in disadvantaged_areas_within_areas.iterrows():
    disadvantaged_areas_to_areas.setdefault(row['areas'], []).append((row['id'], row['poly']))


# Function to determine areas for crimes
def determine_area_for_crimes(df: pd.DataFrame, perc: int) -> pd.DataFrame:
    """
    Determines the areas and disadvantaged areas for each crime and assigns them to the DataFrame.

    Args:
    - df: DataFrame containing crime data with lat/long coordinates.
    - perc: Percentage increment for logging progress.

    Returns:
    - DataFrame with assigned areas and disadvantaged areas.
    """
    statuses = []
    areas, dis_areas = [], []
    perc_cnt = perc

    for i in range(len(df)):
        point = geometry.Point(df.loc[i, 'long'], df.loc[i, 'lat'])
        status = 0

        curr_area = None
        for area, geom in district_to_areas[df.loc[i, 'district']]:
            curr_area = area
            if geom.contains(point):
                status = 1
                break
        statuses.append(status)
        areas.append(curr_area)

        curr_dis_area = None
        if curr_area in disadvantaged_areas_to_areas:
            for dis_area, geom in disadvantaged_areas_to_areas[curr_area]:
                if geom.contains(point):
                    curr_dis_area = dis_area
                    break
        dis_areas.append(curr_dis_area)

        if i > 0 and i % (round(len(df) * (perc_cnt/100))) == 0:
            print(f"{perc_cnt}%- Row {i}/{len(df)} completed")
            perc_cnt += perc

    df['status'] = statuses
    df['areas'] = areas
    df['dis_areas'] = dis_areas
    df = df[df['status'] == 1].drop(['status'], axis=1)

    return df


# Assign areas and disadvantaged areas to the crime data
crime_data_with_factors_and_areas = determine_area_for_crimes(crime_data_with_factors, 5)

# Group factors by areas and disadvantaged areas
factors_by_area = crime_data_with_factors_and_areas.groupby('areas')[
    ['Bike Activity', 'Alleylight Availability', 'Streetlight Outages', 'Partial Streetlight Outages', 'Recent Crime Activity']
].agg('mean').reset_index()

factors_by_dis_area = crime_data_with_factors_and_areas.groupby('dis_areas')[
    ['Bike Activity', 'Alleylight Availability', 'Streetlight Outages', 'Partial Streetlight Outages', 'Recent Crime Activity']
].agg('mean').reset_index()

# Rename columns for merging later
factors_by_area.rename(columns={'areas': 'area'}, inplace=True)
clean_areas.rename(columns={'id': 'area', 'poly': 'geom'}, inplace=True)
factors_by_dis_area.rename(columns={'dis_areas': 'area'}, inplace=True)
disadvantaged_areas_within_areas.rename(columns={'id': 'area', 'poly': 'geom'}, inplace=True)

# Create GeoDataFrames for geospatial plotting
geo_districts = gpd.GeoDataFrame(clean_police_districts[['district', 'geom']], geometry='geom')
geo_areas = gpd.GeoDataFrame(clean_areas[['area', 'geom']], geometry='geom')
geo_dis_areas = gpd.GeoDataFrame(disadvantaged_areas_within_areas[['area', 'geom']], geometry='geom')
chicago = gpd.read_file(geodatasets.get_path("geoda.chicago_commpop"))


# Function to normalize data and merge with GeoDataFrame
def normalize_and_merge(geo_df: gpd.GeoDataFrame, factors_df: pd.DataFrame, exclude_columns: list, merge_column: str) -> gpd.GeoDataFrame:
    """
    Normalizes the factors and merges with a GeoDataFrame for plotting.

    Args:
    - geo_df: GeoDataFrame for geographical boundaries.
    - factors_df: DataFrame containing factors to be normalized.
    - exclude_columns: List of columns to exclude from normalization.
    - merge_column: Column to merge the data on.

    Returns:
    - Merged GeoDataFrame with normalized factors.
    """
    scaler = MinMaxScaler()
    factor_columns = factors_df.drop(columns=exclude_columns)
    normalized_factors = pd.DataFrame(scaler.fit_transform(factor_columns), 
                                      columns=factor_columns.columns, 
                                      index=factors_df[exclude_columns[0]])
    normalized_df = pd.concat([factors_df[exclude_columns], normalized_factors], axis=1)
    merged_gdf = geo_df.merge(normalized_df, on=merge_column)
    return merged_gdf


# Function to plot normalized heatmaps
def plot_normalized_heatmap(
    geo_df: gpd.GeoDataFrame, factors_df: pd.DataFrame, factor_column: str, t: str, 
    title: str, threshold: float = 0.1, label: bool = True, background: bool = True, 
    savefig: str = None
) -> None:
    """
    Plots a heatmap of normalized factors on a geographical map.

    Args:
    - geo_df: GeoDataFrame for geographical boundaries.
    - factors_df: DataFrame containing normalized factors.
    - factor_column: Column to plot.
    - t: Geographical column for labeling.
    - title: Plot title.
    - threshold: Threshold for labeling high/low areas.
    - label: Whether to label areas on the plot.
    - background: Whether to include background map (e.g., Chicago boundary).
    - savefig: File path to save the plot.
    """
    merged_gdf = normalize_and_merge(geo_df, factors_df, [t], t)
    fig, ax = plt.subplots(figsize=(8, 8), facecolor="#f9f9f6")
    ax.set_facecolor("#e0e0e0")

    if background:
        chicago.plot(color='lightgrey', alpha=1, ax=ax)

    merged_gdf.plot(column=factor_column, cmap='OrRd', ax=ax)

    sm = plt.cm.ScalarMappable(cmap='OrRd', norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.036, pad=0.1)
    cbar.set_ticks([0, 1])
    cbar.ax.set_xticklabels(['Low', 'High'])
    cbar.outline.set_visible(False)

    if label:
        for _, row in merged_gdf.iterrows():
            label_text = row[t]
            if row[factor_column] > 1 - threshold:
                plt.text(row.geom.centroid.x, row.geom.centroid.y, s=label_text, 
                         horizontalalignment='center', verticalalignment='center', fontsize=10, color='white', fontweight='bold')
            elif row[factor_column] < threshold:
                plt.text(row.geom.centroid.x, row.geom.centroid.y, s=label_text, 
                         horizontalalignment='center', verticalalignment='center', fontsize=10, color='black', fontweight='bold')

    plt.title(f"The Impact of {factor_column} on Each {title}'s Total Crime", fontsize=14, pad=14)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    if savefig is not None:
        plt.savefig(savefig)

    plt.show()


# Plot heatmaps for factors by district, area, and disadvantaged area
for factor_col in factors_by_district.columns[2:]:
    plot_normalized_heatmap(geo_districts, factors_by_district, factor_col, 'district', 'Police Districts', label=False, savefig=f'../results/analysis/district_{factor_col.replace(" ", "_").lower()}_heatmap.png')

for factor_col in factors_by_area.columns[1:]:
    plot_normalized_heatmap(geo_areas, factors_by_area, factor_col, 'area', 'Neighborhoods', label=False, savefig=f'../results/analysis/area_{factor_col.replace(" ", "_").lower()}_heatmap.png')

for factor_col in factors_by_dis_area.columns[1:]:
    plot_normalized_heatmap(geo_dis_areas, factors_by_dis_area, factor_col, 'area', 'Socioeconomically Disadvantaged Neighborhoods', label=False, savefig=f'../results/analysis/dis_area_{factor_col.replace(" ", "_").lower()}_heatmap.png')


# Sum heatmap analysis for normal factor analysis visualization
factors_by_district_sum = crime_data_with_factors.groupby('district')[
    ['Bike Activity', 'Alleylight Availability', 'Streetlight Outages', 'Partial Streetlight Outages', 'Recent Crime Activity']
].agg('sum').reset_index()

factors_by_area_sum = crime_data_with_factors_and_areas.groupby('areas')[
    ['Bike Activity', 'Alleylight Availability', 'Streetlight Outages', 'Partial Streetlight Outages', 'Recent Crime Activity']
].agg('sum').reset_index()

factors_by_dis_area_sum = crime_data_with_factors_and_areas.groupby('dis_areas')[
    ['Bike Activity', 'Alleylight Availability', 'Streetlight Outages', 'Partial Streetlight Outages', 'Recent Crime Activity']
].agg('sum').reset_index()

# Rename columns for merging
factors_by_area_sum.rename(columns={'areas': 'area'}, inplace=True)
factors_by_dis_area_sum.rename(columns={'dis_areas': 'area'}, inplace=True)

# Plot sum heatmaps for factors by district, area, and disadvantaged area
for factor_col in factors_by_district_sum.columns[2:]:
    plot_normalized_heatmap(geo_districts, factors_by_district_sum, factor_col, 'district', 'Police Districts', label=False, savefig=f'../results/analysis/district_{factor_col.replace(" ", "_").lower()}_sum_heatmap.png')

for factor_col in factors_by_area_sum.columns[1:]:
    plot_normalized_heatmap(geo_areas, factors_by_area_sum, factor_col, 'area', 'Neighborhoods', label=False, savefig=f'../results/analysis/area_{factor_col.replace(" ", "_").lower()}_sum_heatmap.png')

for factor_col in factors_by_dis_area_sum.columns[1:]:
    plot_normalized_heatmap(geo_dis_areas, factors_by_dis_area_sum, factor_col, 'area', 'Socioeconomically Disadvantaged Neighborhoods', label=False, savefig=f'../results/analysis/dis_area_{factor_col.replace(" ", "_").lower()}_sum_heatmap.png')