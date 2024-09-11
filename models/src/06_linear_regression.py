import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

# Load datasets
area_feature_training_sample = pd.read_csv('../../data/pre_training/area_feature_training_sample.csv')
area_target_training_sample = pd.read_csv('../../data/pre_training/area_target_training_sample.csv')
district_feature_training_sample = pd.read_csv('../../data/pre_training/district_feature_training_sample.csv')
district_target_training_sample = pd.read_csv('../../data/pre_training/district_target_training_sample.csv')

# Drop 'crime_status' from the feature sets
area_feature_training_sample.drop('crime_status', axis=1, inplace=True)
district_feature_training_sample.drop('crime_status', axis=1, inplace=True)

area_feature_testing_data = pd.read_csv('../../data/pre_training/area_feature_testing_data.csv')
area_target_testing_data = pd.read_csv('../../data/pre_training/area_target_testing_data.csv')
district_feature_testing_data = pd.read_csv('../../data/pre_training/district_feature_testing_data.csv')
district_target_testing_data = pd.read_csv('../../data/pre_training/district_target_testing_data.csv')

def generate_correlation_heatmap(df: pd.DataFrame, figsize: tuple, title: str, save_name: str) -> None:
    """
    Generates a correlation heatmap for a given DataFrame.

    Args:
    df (pd.DataFrame): The dataframe containing the data to be correlated.
    figsize (tuple): The size of the heatmap.
    title (str): The title of the heatmap.
    save_name (str): The filename to save the heatmap image.

    Returns:
    None
    """
    mask = np.triu(np.ones_like(df.corr(), dtype=bool))  # Generate mask for the upper triangle
    plt.figure(figsize=figsize)
    sns.heatmap(df.corr(), annot=True, mask=mask, vmin=-1, vmax=1)
    plt.title(title)
    plt.savefig(f'../results/linear_regression/{save_name}.png')
    plt.show()

def compute_vif(feature_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes Variance Inflation Factor (VIF) for each feature to check for multicollinearity.

    Args:
    feature_df (pd.DataFrame): DataFrame containing features for which VIF needs to be calculated.

    Returns:
    pd.DataFrame: A DataFrame containing features and their corresponding VIF values.
    """
    print(f"{datetime.now()} - Starting VIF computation")
    X = feature_df.copy()
    X['intercept'] = 1  # Add a constant term for VIF calculation
    vif = pd.DataFrame()
    vif["feature"] = X.columns
    vif["vif"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif = vif[vif['feature'] != 'intercept']
    print(f"{datetime.now()} - Completed VIF computation")
    return vif

def optimize_vif(feature_df: pd.DataFrame, vif_threshold: float) -> pd.DataFrame:
    """
    Iteratively drops features with the highest VIF value until all features are below the given threshold.

    Args:
    feature_df (pd.DataFrame): The input DataFrame with features.
    vif_threshold (float): The VIF threshold above which features are dropped.

    Returns:
    pd.DataFrame: A DataFrame containing the remaining features after optimization.
    """
    print(f"{datetime.now()} - Starting VIF optimization")
    df = feature_df.copy()
    vif_df = compute_vif(df)

    while (vif_df['vif'] >= vif_threshold).any():
        largest_vif_feature = vif_df.loc[vif_df['vif'].idxmax(), 'feature']
        print(f"{datetime.now()} - Dropping feature: {largest_vif_feature} with VIF score of: {vif_df['vif'].max()}")
        df = df.drop(columns=[largest_vif_feature])
        vif_df = compute_vif(df)

    print(f"{datetime.now()} - Completed VIF optimization")
    return vif_df

# Optimize VIF for area and district datasets
area_selected_features_ten = optimize_vif(area_feature_training_sample, 10)
district_selected_features_ten = optimize_vif(district_feature_training_sample, 10)

# Filter selected features for training and testing datasets
area_feature_training_sample = area_feature_training_sample[area_selected_features_ten['feature'].values]
area_feature_testing_data = area_feature_testing_data[area_selected_features_ten['feature'].values]
district_feature_training_sample = district_feature_training_sample[district_selected_features_ten['feature'].values]
district_feature_testing_data = district_feature_testing_data[district_selected_features_ten['feature'].values]

# Further optimize VIF with a threshold of 5
area_selected_features_five = optimize_vif(area_feature_training_sample, 5)
district_selected_features_five = optimize_vif(district_feature_training_sample, 5)

# Filter selected features based on VIF threshold of 5
area_feature_training_sample = area_feature_training_sample[area_selected_features_five['feature'].values]
area_feature_testing_data = area_feature_testing_data[area_selected_features_five['feature'].values]
district_feature_training_sample = district_feature_training_sample[district_selected_features_five['feature'].values]
district_feature_testing_data = district_feature_testing_data[district_selected_features_five['feature'].values]

def feature_selection_sfs(X_train: pd.DataFrame, y_train: pd.DataFrame) -> list:
    """
    Performs Sequential Forward Selection (SFS) to select the best features based on negative mean squared error.

    Args:
    X_train (pd.DataFrame): Training features.
    y_train (pd.DataFrame): Target variable for training.

    Returns:
    list: List of selected feature names.
    """
    model = LinearRegression()
    sfs = SFS(model, k_features='best', forward=True, floating=False, scoring='neg_mean_squared_error', n_jobs=15, cv=5, verbose=2)
    sfs.fit(X_train, y_train)

    best_avg_score = -1 * float('inf')
    best_subset = None

    # Iterate through the subsets to find the best feature set
    for subset, values in sfs.subsets_.items():
        avg_score = values['avg_score']
        if avg_score > best_avg_score:
            best_avg_score = avg_score
            best_subset = values['feature_idx']

    return [col for idx, col in enumerate(X_train.columns) if idx in best_subset]

# Select best features using SFS
area_best_features = feature_selection_sfs(area_feature_training_sample, area_target_training_sample)
district_best_features = feature_selection_sfs(district_feature_training_sample, district_target_training_sample)

def train_model(X_train: pd.DataFrame, y_train: pd.DataFrame) -> LinearRegression:
    """
    Trains a Linear Regression model on the given training data.

    Args:
    X_train (pd.DataFrame): Training features.
    y_train (pd.DataFrame): Target variable for training.

    Returns:
    LinearRegression: Trained Linear Regression model.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Train final models for area and district datasets
area_final_model = train_model(area_feature_training_sample[area_best_features], area_target_training_sample)
district_final_model = train_model(district_feature_training_sample[district_best_features], district_target_training_sample)

def evaluate_model(model: LinearRegression, X_test: pd.DataFrame, y_test: pd.DataFrame) -> None:
    """
    Evaluates the trained model on the test data and prints performance metrics.

    Args:
    model (LinearRegression): Trained Linear Regression model.
    X_test (pd.DataFrame): Test features.
    y_test (pd.DataFrame): True test targets.

    Returns:
    None
    """
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    # Print metrics
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"R^2 Score: {r2}")

# Evaluate models for different scenarios
evaluate_model(area_final_model, area_feature_testing_data, area_target_testing_data)
evaluate_model(district_final_model, district_feature_testing_data, district_target_testing_data)

def analyze_feature_importances(model: LinearRegression, feature_columns: pd.Index) -> pd.DataFrame:
    """
    Analyzes the feature importances by examining the coefficients of a trained linear model.

    Args:
    model (LinearRegression): Trained linear model.
    feature_columns (pd.Index): List of feature names.

    Returns:
    pd.DataFrame: DataFrame containing features and their coefficient values.
    """
    coef_df = pd.DataFrame({
        'Feature': feature_columns,
        'Coefficient': model.coef_
    })

    # Sort by absolute coefficient values
    coef_df['Abs_Coefficient'] = coef_df['Coefficient'].abs()
    return coef_df.sort_values(by='Abs_Coefficient', ascending=False)

# Analyze feature importances for both area and district models
area_coef_df = analyze_feature_importances(area_final_model, area_feature_training_sample.columns)
district_coef_df = analyze_feature_importances(district_final_model, district_feature_training_sample.columns)

def plot_feature_importances(coef_df: pd.DataFrame, title: str, save_name: str) -> None:
    """
    Plots feature importances from the linear model's coefficients.

    Args:
    coef_df (pd.DataFrame): DataFrame containing features and their coefficients.
    title (str): Plot title.
    save_name (str): Name of the file to save the plot.

    Returns:
    None
    """
    plt.figure(figsize=(8, 10))
    plt.barh(coef_df['Feature'], coef_df['Coefficient'], color='blue')
    plt.xlabel('Coefficient Value')
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f'../results/linear_regression/{save_name}.png')
    plt.show()

# Plot feature importances for area and district models
plot_feature_importances(area_coef_df, 'Feature Importances from Area Model', 'area_feature_coefficients')
plot_feature_importances(district_coef_df, 'Feature Importances from District Model', 'district_feature_coefficients')

# Correlation heatmap generation
generate_correlation_heatmap(area_feature_training_sample, (35, 15), "Correlation Heatmap of Area Model's Final Features", 'area_final_features_corr')
generate_correlation_heatmap(district_feature_training_sample, (30,10), "Correlation Heatmap of District Model's Final Features", 'district_final_features_corr')