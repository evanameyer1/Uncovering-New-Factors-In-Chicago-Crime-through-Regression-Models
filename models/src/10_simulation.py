import pandas as pd
import numpy as np
import xgboost as xgb
import geopandas as gpd
import geodatasets
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
import warnings
import json

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

# Load XGBoost model
final_district_xgb_model = xgb.Booster()
final_district_xgb_model.load_model('models/final_district_xgb_model.json')

# Top features used in simulation
top_features = [
    'bus_stops_distance_0.1', 'alleylights_distance_0.1', 'police_stations_distance_1',
    'streetlights_oneout_distance_0.1', 'streetlights_allout_distance_0.5',
    'streetlights_oneout_distance_0.3', 'streetlights_allout_distance_0.3',
    'bike_rides_within_0.5_and_15_min', 'area_crimes_3_hours_prev', 'area_crimes_1_hours_prev',
    'alleylights_distance_0.3', 'bike_rides_within_0.1_and_10_min', 'bike_rides_within_0.1_and_5_min'
]

# Load data
area_df = pd.read_csv('../../data/pre_training/area_pre_feature_selection_nonnormalized.csv')
dis_area_df = pd.read_csv('../../data/pre_training/dis_area_pre_feature_selection_nonnormalized.csv')
district_df = pd.read_csv('../../data/pre_training/district_pre_feature_selection_nonnormalized.csv')

# Select the relevant features
area_df_selected = area_df[top_features]
dis_area_df_selected = dis_area_df[top_features]
district_df_selected = district_df[top_features]

# Define factor mappings for simulations
factor_mapping = {
    'Bike Activity': [
        'bike_rides_within_0.5_and_15_min', 'bike_rides_within_0.1_and_10_min', 'bike_rides_within_0.1_and_5_min'
    ],
    'Alleylight Outages': [
        'alleylights_distance_0.1', 'alleylights_distance_0.3'
    ],
    'Streetlight Outages': [
        'streetlights_allout_distance_0.5', 'streetlights_allout_distance_0.3'
    ],
    'Partial Streetlight Outages': [
        'streetlights_oneout_distance_0.1', 'streetlights_oneout_distance_0.3'
    ],
    'Recent Crime Activity': [
        'area_crimes_3_hours_prev', 'area_crimes_1_hours_prev'
    ]
}

# Simulation function
def simulation(df: pd.DataFrame, ref_df: pd.DataFrame, factors: dict, thresholds: list) -> dict:
    """
    Simulates changes in the factors for the dataset, with random variations based on thresholds.

    Args:
    - df: DataFrame of the original data.
    - ref_df: DataFrame containing reference columns like district/area IDs.
    - factors: Dictionary mapping factor names to their respective columns.
    - thresholds: List of thresholds for percentage changes.

    Returns:
    - Dictionary of simulated dataframes.
    """
    simmed_dfs = {}
    for factor in factors.keys():
        target_cols = factors[factor]
        for threshold in thresholds:
            threshold = round(threshold, 2)
            temp_df = df.copy()

            if threshold < 0:
                random_factors = np.random.uniform(1 - abs(threshold), 1, size=temp_df[target_cols].shape)
                temp_df.loc[:, target_cols] = temp_df[target_cols] * random_factors
            else:
                random_factors = np.random.uniform(1, 1 + threshold, size=temp_df[target_cols].shape)
                random_additions = np.random.choice([0, 1], size=temp_df[target_cols].shape, p=[1 - threshold, threshold])
                temp_df.loc[:, target_cols] = (temp_df[target_cols] + random_additions) * random_factors

            temp_df = pd.concat([temp_df, ref_df], axis=1)
            simmed_dfs[f'{factor.replace(" ", "_").lower()}_{threshold}'] = temp_df

    return simmed_dfs

# Generate simulations for different regions and factors
district_sims = simulation(district_df[top_features], district_df[['district', 'date_hour']], factor_mapping, [-0.1, -0.05, 0.05, 0.1])
area_sims = simulation(area_df[top_features], area_df[['area_id', 'date_hour']], factor_mapping, [-0.1, -0.05, 0.05, 0.1])
dis_area_sims = simulation(dis_area_df[top_features], dis_area_df[['dis_area_id', 'date_hour']], factor_mapping, [-0.1, -0.05, 0.05, 0.1])

# Single factor mappings for smaller simulations
single_factor_mappings = {'Bus Stops': 'bus_stops_distance_0.1', 'Police Stations': 'police_stations_distance_1'}
single_factor_district_sims = simulation(district_df[top_features], district_df[['district', 'date_hour']], single_factor_mappings, [-0.1, -0.05, 0.05, 0.1])
single_factor_area_sims = simulation(area_df[top_features], area_df[['area_id', 'date_hour']], single_factor_mappings, [-0.1, -0.05, 0.05, 0.1])
single_factor_dis_area_sims = simulation(dis_area_df[top_features], dis_area_df[['dis_area_id', 'date_hour']], single_factor_mappings, [-0.1, -0.05, 0.05, 0.1])

# Function to predict the simulations
def predict_simulations(model: xgb.Booster, drops: list, sim_dfs: dict, filepath: str = None) -> dict:
    """
    Runs the XGBoost model on simulated datasets and predicts crime levels.

    Args:
    - model: Pre-trained XGBoost Booster model.
    - drops: List of columns to drop before prediction.
    - sim_dfs: Dictionary of simulated dataframes.
    - filepath: Optional path to save the predictions.

    Returns:
    - Dictionary of predictions for each simulation.
    """
    predictions = {}
    for threshold, sim_df in sim_dfs.items():
        dmatrix = xgb.DMatrix(sim_df.drop(drops, axis=1))
        prediction = model.predict(dmatrix)
        predictions[threshold] = prediction
        print(f'Predictions for {threshold} completed')

    # Save predictions if a filepath is provided
    if filepath:
        with open(filepath, 'w') as f:
            json.dump(predictions, f)

    return predictions

# Predicting for various regions and factors
district_predictions = predict_simulations(final_district_xgb_model, ['district', 'date_hour'], district_sims, '../../data/simulations/district_predictions.json')
area_predictions = predict_simulations(final_district_xgb_model, ['area_id', 'date_hour'], area_sims, '../../data/simulations/area_predictions.json')
dis_area_predictions = predict_simulations(final_district_xgb_model, ['dis_area_id', 'date_hour'], dis_area_sims, '../../data/simulations/dis_area_predictions.json')

# Control predictions (no simulation) for comparison
district_dmatrix = xgb.DMatrix(district_df_selected)
control_district_predictions = final_district_xgb_model.predict(district_dmatrix)

area_dmatrix = xgb.DMatrix(area_df_selected)
control_area_predictions = final_district_xgb_model.predict(area_dmatrix)

dis_area_dmatrix = xgb.DMatrix(dis_area_df_selected)
control_dis_area_predictions = final_district_xgb_model.predict(dis_area_dmatrix)

# Function to calculate residuals across geospatial factors
def calculate_residuals(predictions: dict, test_labels: np.ndarray, geodata: pd.Series, filepath: str = None) -> dict:
    """
    Calculates residuals (differences between predicted and actual values) by geographical area.

    Args:
    - predictions: Dictionary of predicted values.
    - test_labels: Array of actual crime values.
    - geodata: Series of geographical identifiers (e.g., district IDs).
    - filepath: Optional path to save residuals.

    Returns:
    - Dictionary of residuals for each threshold.
    """
    residuals = {}
    for threshold, prediction in predictions.items():
        temp_df = pd.DataFrame(data={'geo': geodata, 'actual_crime': test_labels, 'new_crime': prediction})
        avg_grouped_residuals = temp_df.groupby('geo')[['actual_crime', 'new_crime']].agg('mean').reset_index()
        avg_grouped_residuals.rename(columns={'actual_crime': 'avg_actual_crime', 'new_crime': 'avg_new_crime'}, inplace=True)
        avg_grouped_residuals['avg_diff'] = avg_grouped_residuals['avg_actual_crime'] - avg_grouped_residuals['avg_new_crime']
        avg_grouped_residuals['avg_perc_diff'] = (avg_grouped_residuals['avg_new_crime'] - avg_grouped_residuals['avg_actual_crime']) / avg_grouped_residuals['avg_actual_crime'] * 100

        tot_grouped_residuals = temp_df.groupby('geo')[['actual_crime', 'new_crime']].agg('sum').reset_index()
        tot_grouped_residuals.rename(columns={'actual_crime': 'tot_actual_crime', 'new_crime': 'tot_new_crime'}, inplace=True)
        tot_grouped_residuals['tot_diff'] = tot_grouped_residuals['tot_actual_crime'] - tot_grouped_residuals['tot_new_crime']
        tot_grouped_residuals['tot_perc_diff'] = (tot_grouped_residuals['tot_new_crime'] - tot_grouped_residuals['tot_actual_crime']) / tot_grouped_residuals['tot_actual_crime'] * 100

        residuals[threshold] = pd.merge(avg_grouped_residuals, tot_grouped_residuals, on='geo', how='inner')

    if filepath:
        residuals_serializable = {k: v.values.tolist() for k, v in residuals.items()}
        with open(filepath, 'w') as f:
            json.dump(residuals_serializable, f)

    return residuals

# Calculating residuals for each region
district_residuals = calculate_residuals(district_predictions, control_district_predictions, district_df['district'], '../../data/simulations/district_residuals.json')
area_residuals = calculate_residuals(area_predictions, control_area_predictions, area_df['area_id'], '../../data/simulations/area_residuals.json')
dis_area_residuals = calculate_residuals(dis_area_predictions, control_dis_area_predictions, dis_area_df['dis_area_id'], '../../data/simulations/dis_area_residuals.json')

# Load the geographical data from the processed files
clean_police_districts = pd.read_csv('../../data/processed/clean_police_districts.csv')
clean_areas = pd.read_csv('../../data/processed/clean_areas.csv')
disadvantaged_areas_within_areas = pd.read_csv('../../data/processed/disadvantaged_areas_within_areas.csv')

# Function to parse polygons from string format
def parse_polygon(polygon_string: str) -> Polygon:
    """
    Parses a polygon string into a Shapely Polygon object.

    Args:
    - polygon_string: The string representing a polygon.

    Returns:
    - A Shapely Polygon object.
    """
    points = polygon_string.strip('POLYGON ((').strip('))').split(', ')
    points = [tuple(map(float, point.split())) for point in points]
    return Polygon(points)

# Function to swap x and y coordinates in a polygon
def swap_coordinates(polygon: Polygon) -> Polygon:
    """
    Swaps x and y coordinates of a polygon's exterior points.

    Args:
    - polygon: A Shapely Polygon object.

    Returns:
    - Polygon with swapped coordinates.
    """
    if polygon.is_empty:
        return polygon
    swapped_coords = [(y, x) for x, y in polygon.exterior.coords]
    return Polygon(swapped_coords)

# Apply the parsing and swapping functions to the police districts
clean_police_districts['geom'] = clean_police_districts['geom'].apply(parse_polygon)
clean_police_districts['geom'] = clean_police_districts['geom'].apply(swap_coordinates)

# Apply the same to the areas and disadvantaged areas
clean_areas['poly'] = clean_areas['poly'].apply(parse_polygon)
clean_areas['poly'] = clean_areas['poly'].apply(swap_coordinates)

disadvantaged_areas_within_areas['poly'] = disadvantaged_areas_within_areas['poly'].apply(parse_polygon)
disadvantaged_areas_within_areas['poly'] = disadvantaged_areas_within_areas['poly'].apply(swap_coordinates)
disadvantaged_areas_within_areas['id'] = disadvantaged_areas_within_areas.index

# Rename columns to be consistent
clean_areas.rename(columns={'id': 'area', 'poly': 'geom'}, inplace=True)
disadvantaged_areas_within_areas.rename(columns={'id': 'area', 'poly': 'geom'}, inplace=True)

# Create GeoDataFrames for police districts, areas, and disadvantaged areas
geo_districts = gpd.GeoDataFrame(clean_police_districts[['district', 'geom']], geometry='geom')
geo_areas = gpd.GeoDataFrame(clean_areas[['area', 'geom']], geometry='geom')
geo_dis_areas = gpd.GeoDataFrame(disadvantaged_areas_within_areas[['area', 'geom']], geometry='geom')

# Load the city-wide Chicago map data
chicago = gpd.read_file(geodatasets.get_path("geoda.chicago_commpop"))

# Visualizing differences by geography (mapping, etc.)
def plot_normalized_heatmap(geo_df: gpd.GeoDataFrame, target_column: pd.DataFrame, t: str, title: str, background: bool = True, savefig: str = None, remove_outliers: bool = True):
    """
    Plots heatmaps based on geographical residual differences.

    Args:
    - geo_df: GeoDataFrame containing geographical shapes.
    - target_column: DataFrame containing target data (e.g., residuals).
    - t: String for the key to join data.
    - title: Title of the plot.
    - background: Whether to include background map (e.g., Chicago boundaries).
    - savefig: Filepath to save the figure.
    - remove_outliers: Whether to remove statistical outliers.
    """
    merged_gdf = geo_df.merge(target_column, left_on=t, right_on='geo')

    if remove_outliers:
        Q1 = merged_gdf[target_column.columns[1]].quantile(0.25)
        Q3 = merged_gdf[target_column.columns[1]].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        merged_gdf = merged_gdf[(merged_gdf[target_column.columns[1]] >= lower_bound) & 
                                (merged_gdf[target_column.columns[1]] <= upper_bound)]

    vmin = merged_gdf['avg_perc_diff'].min()
    vmax = merged_gdf['avg_perc_diff'].max()

    rounded_vmin = round(vmin, 2)
    rounded_vmax = round(vmax, 2)

    if rounded_vmax <= -0.1:
        cmap = 'Greens'
    elif rounded_vmin >= 0.1:
        cmap = 'YlOrRd'
    else:
        cmap = 'RdYlGn_r'

    fig, ax = plt.subplots(figsize=(8, 8), facecolor="#f9f9f6")
    ax.set_facecolor("#e0e0e0")

    if background:
        chicago.plot(color='lightgrey', alpha=1, ax=ax)

    merged_gdf.plot(column=target_column.columns[1], cmap=cmap, ax=ax)

    norm = plt.Normalize(vmin=rounded_vmin, vmax=rounded_vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.036, pad=0.1)
    cbar.set_ticks([rounded_vmin, rounded_vmax])
    cbar.set_ticklabels(['Decreased Crime', 'Increased Crime'])

    plt.title(title, fontsize=16, pad=14)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    if savefig:
        plt.savefig(savefig)

    plt.show()

# Plot residuals heatmaps for various areas
plot_normalized_heatmap(geo_districts, district_residuals, 'district', 'District Residual Differences')
plot_normalized_heatmap(geo_areas, area_residuals, 'area', 'Area Residual Differences')
plot_normalized_heatmap(geo_dis_areas, dis_area_residuals, 'area', 'Disadvantaged Areas Residual Differences')
