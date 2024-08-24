# Import necessary libraries
import pandas as pd
import numpy as np
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px

# Load the crime data
crime_data = pd.read_csv('../../data/pre_training/pre_aggregation_crime_data.csv')

# Select top features for factor analysis
top_features = [
    'bus_stops_distance_0.1',
    'alleylights_distance_0.1',
    'police_stations_distance_1',
    'streetlights_oneout_distance_0.1',
    'streetlights_allout_distance_0.5',
    'streetlights_oneout_distance_0.3',
    'streetlights_allout_distance_0.3',
    'bike_rides_within_0.5_and_15_min',
    'area_crimes_3_hours_prev',
    'area_crimes_1_hours_prev',
    'alleylights_distance_0.3',
    'bike_rides_within_0.1_and_10_min',
    'bike_rides_within_0.1_and_5_min'
]

# Select the relevant columns and fill missing values with 0
crime_data_selected = crime_data[top_features].fillna(0)

# Normalize the data to prepare for factor analysis
crime_data_selected_scaler = MinMaxScaler()
crime_data_selected_scaler.set_output(transform='pandas')
crime_data_selected_normalized = crime_data_selected_scaler.fit_transform(crime_data_selected)

# Adequacy Test
chi_square_val, p_val = calculate_bartlett_sphericity(crime_data_selected_normalized)
print(f"Bartlett's test chi-square value: {chi_square_val}, p-value: {p_val}")

kmo_all, kmo_model = calculate_kmo(crime_data_selected_normalized)
print(f"KMO Test - Overall: {kmo_all}, Model: {kmo_model}")

# Perform factor analysis and obtain eigenvalues
fa = FactorAnalyzer()
fa.fit(crime_data_selected)
eigenvalues, _ = fa.get_eigenvalues()
print("Eigenvalues:", eigenvalues)

# Plot the scree plot to visualize eigenvalues and retain factors where eigenvalue > 1
num_factors_kaiser = sum(eigenvalues > 1)
plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, marker='o')
plt.xlabel('Number of Factors')
plt.ylabel('Eigenvalue')
plt.title('Scree Plot')
plt.axhline(y=1, color='r', linestyle='--')  # Kaiser criterion threshold
plt.show()

# Perform factor analysis with varimax rotation
fa = FactorAnalyzer(n_factors=num_factors_kaiser, rotation='varimax')
fa.fit(crime_data_selected)
fa_loadings = pd.DataFrame(data=fa.loadings_, columns=[f'Factor {i+1}' for i in range(num_factors_kaiser)], index=crime_data_selected.columns)

# Perform factor analysis with five factors
fa_five = FactorAnalyzer(n_factors=5, rotation='varimax')
fa_five.fit(crime_data_selected)
fa_five_loadings = pd.DataFrame(data=fa_five.loadings_, columns=['Factor 1', 'Factor 2', 'Factor 3', 'Factor 4', 'Factor 5'], index=crime_data_selected.columns)

def evaluate_factors(factor_df: pd.DataFrame, threshold: float) -> None:
    """
    Evaluate factors by selecting features with factor loadings greater than the threshold.
    
    Args:
        factor_df (pd.DataFrame): DataFrame containing factor loadings.
        threshold (float): Threshold for selecting significant factor loadings.
    """
    cumulative_loadings = 0
    for factor in factor_df:
        features = list(factor_df[factor_df[factor] > threshold].index)
        features_with_vals = [(feature, float(round(factor_df[factor][feature], 3))) for feature in features]
        print(f"{factor}: {features_with_vals}")
        cumulative_loadings += sum([val for _, val in features_with_vals])
    print(f'Cumulative Factor Loadings: {cumulative_loadings}')

# Evaluate factor loadings with a threshold of 0.4
evaluate_factors(fa_loadings, 0.4)
evaluate_factors(fa_five_loadings, 0.4)

# Calculate factor scores for each district
transformed_data = pd.DataFrame(fa_five.transform(crime_data_selected), columns=['Bike Activity', 'Alleylight Availability', 'Streetlight Outages', 'Partial Streetlight Outages', 'Recent Crime Activity'])
crime_data_with_factors = pd.concat([crime_data, transformed_data], axis=1)
factors_by_district = crime_data_with_factors.groupby('district')[['Bike Activity', 'Alleylight Availability', 'Streetlight Outages', 'Partial Streetlight Outages', 'Recent Crime Activity']].agg('mean').reset_index()

# Plot crime data with factors using Plotly
fig = px.scatter_mapbox(
    crime_data_with_factors,
    lat="lat",
    lon="long",
    color="Bike Activity",
    size="district_crimes_this_hour",
    hover_name="district",
    hover_data=["Bike Activity", "district_crimes_this_hour"],
    color_continuous_scale=px.colors.cyclical.IceFire,
    size_max=15,
    zoom=10,
    title="Bike Activity and Crime Distribution in Chicago"
)

fig.update_layout(mapbox_style="carto-positron")
fig.show()