# Import necessary libraries
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import statsmodels.api as sm

# Load datasets
area_feature_training_sample = pd.read_csv('../../data/pre_training/area_feature_training_sample.csv')
area_target_training_sample = pd.read_csv('../../data/pre_training/area_target_training_sample.csv')
district_feature_training_sample = pd.read_csv('../../data/pre_training/district_feature_training_sample.csv')
district_target_training_sample = pd.read_csv('../../data/pre_training/district_target_training_sample.csv')

# Remove 'crime_status' column
area_feature_training_sample.drop('crime_status', axis=1, inplace=True)
district_feature_training_sample.drop('crime_status', axis=1, inplace=True)

# Load testing datasets
area_feature_testing_data = pd.read_csv('../../data/pre_training/area_feature_testing_data.csv')
area_target_testing_data = pd.read_csv('../../data/pre_training/area_target_testing_data.csv')
district_feature_testing_data = pd.read_csv('../../data/pre_training/district_feature_testing_data.csv')
district_target_testing_data = pd.read_csv('../../data/pre_training/district_target_testing_data.csv')

def generate_correlation_heatmap(df: pd.DataFrame, figsize: tuple, title: str, save_name: str) -> None:
    """
    Generate a heatmap to visualize correlations between features.
    
    Args:
        df (pd.DataFrame): The input dataframe for which the heatmap is generated.
        figsize (tuple): The size of the heatmap figure.
        title (str): The title of the heatmap.
        save_name (str): The filename for saving the heatmap image.
    """
    # Generate a mask to only show the bottom triangle
    mask = np.triu(np.ones_like(df.corr(), dtype=bool))

    # Generate heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(df.corr(), annot=True, mask=mask, vmin=-1, vmax=1)
    plt.title(title)
    plt.savefig(f'../results/linear_regression/{save_name}.png')
    plt.show()

def compute_vif(feature_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Variance Inflation Factor (VIF) for all features in a dataframe to detect multicollinearity.

    Args:
        feature_df (pd.DataFrame): The input feature dataframe.

    Returns:
        pd.DataFrame: A dataframe containing features and their corresponding VIF values.
    """
    print(f"{datetime.now()} - Starting VIF computation")
    X = feature_df.copy()
    X['intercept'] = 1  # Add constant for VIF computation
    
    # Calculate VIF values
    vif = pd.DataFrame()
    vif["feature"] = X.columns
    vif["vif"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif = vif[vif['feature'] != 'intercept']  # Remove intercept
    
    print(f"{datetime.now()} - Completed VIF computation")
    return vif

def optimize_vif(feature_df: pd.DataFrame, vif_threshold: float) -> pd.DataFrame:
    """
    Optimize feature set by iteratively dropping features with high VIF values.

    Args:
        feature_df (pd.DataFrame): The input feature dataframe.
        vif_threshold (float): The VIF threshold for dropping features.

    Returns:
        pd.DataFrame: A dataframe with optimized features and their corresponding VIF values.
    """
    print(f"{datetime.now()} - Starting VIF optimization")
    df = feature_df.copy()
    vif_df = compute_vif(feature_df)
    
    while (vif_df['vif'] >= vif_threshold).any():
        print(f"{datetime.now()} - Current VIF values:\n{vif_df}")
        largest_vif_feature = vif_df.loc[vif_df['vif'].idxmax(), 'feature']
        print(f"{datetime.now()} - Dropping feature: {largest_vif_feature} with VIF score of: {vif_df['vif'].max()}")
        df = df.drop(columns=[largest_vif_feature])
        vif_df = compute_vif(df)
    
    print(f"{datetime.now()} - Completed VIF optimization")
    return vif_df

# Optimize features by VIF for both area and district datasets
area_selected_features_ten = optimize_vif(area_feature_training_sample, 10)
district_selected_features_ten = optimize_vif(district_feature_training_sample, 10)

# Update training and testing datasets with selected features
area_feature_training_sample = area_feature_training_sample[list(area_selected_features_ten['feature'].values)]
area_feature_testing_data = area_feature_testing_data[list(area_selected_features_ten['feature'].values)]
district_feature_training_sample = district_feature_training_sample[list(district_selected_features_ten['feature'].values)]
district_feature_testing_data = district_feature_testing_data[list(district_selected_features_ten['feature'].values)]

# Further optimize features by VIF using a lower threshold
area_selected_features_five = optimize_vif(area_feature_training_sample, 5)
district_selected_features_five = optimize_vif(district_feature_training_sample, 5)

# Update training and testing datasets with further optimized features
area_feature_training_sample = area_feature_training_sample[list(area_selected_features_five['feature'].values)]
area_feature_testing_data = area_feature_testing_data[list(area_selected_features_five['feature'].values)]
district_feature_training_sample = district_feature_training_sample[list(district_selected_features_five['feature'].values)]
district_feature_testing_data = district_feature_testing_data[list(district_selected_features_five['feature'].values)]

def select_best_features(model, feature_data: pd.DataFrame, target_data: pd.DataFrame, cv: int = 5) -> list:
    """
    Select the best subset of features using Sequential Feature Selector (SFS).

    Args:
        model: The machine learning model to be used for feature selection.
        feature_data (pd.DataFrame): The input feature dataframe.
        target_data (pd.DataFrame): The target dataframe.
        cv (int): The number of cross-validation folds.

    Returns:
        list: The list of the best feature indices.
    """
    sfs = SFS(model, k_features='best', forward=True, floating=False, scoring='neg_mean_squared_error', cv=cv, n_jobs=15, verbose=2)
    sfs.fit(feature_data, target_data)
    
    best_avg_score = -1 * float('inf')
    best_subset = None
    
    # Find the best feature subset
    for subset, values in sfs.subsets_.items():
        avg_score = values['avg_score']
        if avg_score > best_avg_score:
            best_avg_score = avg_score
            best_subset = values['feature_idx']
    
    return [col for idx, col in enumerate(feature_data.columns) if idx in best_subset]

# Select the best features for both area and district models
area_best_features = select_best_features(LinearRegression(), area_feature_training_sample, area_target_training_sample)
district_best_features = select_best_features(LinearRegression(), district_feature_training_sample, district_target_training_sample)

# Update training and testing datasets with the best selected features
area_feature_training_sample = area_feature_training_sample[area_best_features]
area_feature_testing_data = area_feature_testing_data[area_best_features]
district_feature_training_sample = district_feature_training_sample[district_best_features]
district_feature_testing_data = district_feature_testing_data[district_best_features]

def train_model(feature_data: pd.DataFrame, target_data: pd.DataFrame) -> LinearRegression:
    """
    Train a linear regression model on the provided data.

    Args:
        feature_data (pd.DataFrame): The input feature dataframe for training.
        target_data (pd.DataFrame): The target dataframe for training.

    Returns:
        LinearRegression: The trained linear regression model.
    """
    model = LinearRegression()
    model.fit(feature_data, target_data)
    return model

# Train the final area and district models
area_final_model = train_model(area_feature_training_sample, area_target_training_sample)
district_final_model = train_model(district_feature_training_sample, district_target_training_sample)

def evaluate_model(model, feature_data: pd.DataFrame, target_data: pd.DataFrame) -> dict:
    """
    Evaluate a trained model using various metrics.

    Args:
        model: The trained machine learning model.
        feature_data (pd.DataFrame): The input feature dataframe for evaluation.
        target_data (pd.DataFrame): The target dataframe for evaluation.

    Returns:
        dict: A dictionary containing evaluation metrics.
    """
    predictions = model.predict(feature_data)
    mse = mean_squared_error(target_data, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(target_data, predictions)
    r2 = r2_score(target_data, predictions)
    
    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R^2": r2
    }

# Evaluate area model performance
area_false_metrics = evaluate_model(area_final_model, area_feature_testing_data.loc[area_target_testing_data[area_target_testing_data['area_crimes_this_hour'] == 0].index].reset_index(drop=True), area_target_testing_data[area_target_testing_data['area_crimes_this_hour'] == 0].reset_index(drop=True))
area_true_metrics = evaluate_model(area_final_model, area_feature_testing_data.loc[area_target_testing_data[area_target_testing_data['area_crimes_this_hour'] > 0].index].reset_index(drop=True), area_target_testing_data[area_target_testing_data['area_crimes_this_hour'] > 0].reset_index(drop=True))

# Print area model performance metrics
print("Area Model Performance Metrics for False Crime Status:")
print(area_false_metrics)
print("Area Model Performance Metrics for True Crime Status:")
print(area_true_metrics)

# Evaluate district model performance
district_false_metrics = evaluate_model(district_final_model, district_feature_testing_data.loc[district_target_testing_data[district_target_testing_data['district_crimes_this_hour'] == 0].index].reset_index(drop=True), district_target_testing_data[district_target_testing_data['district_crimes_this_hour'] == 0].reset_index(drop=True))
district_true_metrics = evaluate_model(district_final_model, district_feature_testing_data.loc[district_target_testing_data[district_target_testing_data['district_crimes_this_hour'] > 0].index].reset_index(drop=True), district_target_testing_data[district_target_testing_data['district_crimes_this_hour'] > 0].reset_index(drop=True))

# Print district model performance metrics
print("District Model Performance Metrics for False Crime Status:")
print(district_false_metrics)
print("District Model Performance Metrics for True Crime Status:")
print(district_true_metrics)

def analyze_feature_importances(model, feature_data: pd.DataFrame, plot_title: str, save_name: str) -> None:
    """
    Analyze and plot feature importances for a linear regression model.

    Args:
        model: The trained machine learning model.
        feature_data (pd.DataFrame): The input feature dataframe.
        plot_title (str): The title for the plot.
        save_name (str): The filename for saving the plot.
    """
    coef_df = pd.DataFrame({
        'Feature': feature_data.columns,
        'Coefficient': model.coef_
    })
    
    coef_df['Abs_Coefficient'] = coef_df['Coefficient'].abs()
    coef_df = coef_df.sort_values(by='Abs_Coefficient', ascending=False)
    
    plt.figure(figsize=(8, 10))
    plt.barh(coef_df['Feature'], coef_df['Coefficient'], color='blue')
    plt.xlabel('Coefficient Value')
    plt.title(plot_title)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f'../results/linear_regression/{save_name}.png')
    plt.show()

# Analyze feature importances for both models
analyze_feature_importances(area_final_model, area_feature_training_sample, 'Feature Importances from Area Model', 'area_feature_coefficients')
analyze_feature_importances(district_final_model, district_feature_training_sample, 'Feature Importances from District Model', 'district_feature_coefficients')

def run_ols_analysis(feature_data: pd.DataFrame, target_data: pd.DataFrame) -> None:
    """
    Run OLS regression analysis and print the summary.

    Args:
        feature_data (pd.DataFrame): The input feature dataframe.
        target_data (pd.DataFrame): The target dataframe.
    """
    feature_data_const = sm.add_constant(feature_data)
    sm_model = sm.OLS(target_data, feature_data_const).fit()
    print(sm_model.summary())

# Run OLS analysis for both models
run_ols_analysis(area_feature_training_sample, area_target_training_sample)
run_ols_analysis(district_feature_training_sample, district_target_training_sample)

# Generate correlation heatmaps for both models
generate_correlation_heatmap(area_feature_training_sample, (35, 15), "Correlation Heatmap of Area Model's Final Features", 'area_final_features_corr')
generate_correlation_heatmap(district_feature_training_sample, (30,10), "Correlation Heatmap of District Model's Final Features", 'district_final_features_corr')
