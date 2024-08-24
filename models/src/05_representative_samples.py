import pandas as pd
import numpy as np
from typing import List
from sklearn.preprocessing import MinMaxScaler

# Load pre-feature selection data
def load_pre_feature_selection_data() -> dict:
    """
    Load pre-feature selection datasets for both area and district data.

    Returns:
        dict: A dictionary containing DataFrames for area and district pre-feature selection data.
    """
    return {
        "area_pre_feature_selection": pd.read_csv('../../data/pre_training/area_pre_feature_selection.csv'),
        "district_pre_feature_selection": pd.read_csv('../../data/pre_training/district_pre_feature_selection.csv')
    }

# Split dataset into training and testing sets
def split_train_test(features: pd.DataFrame, target: pd.DataFrame, year: int) -> dict:
    """
    Split the features and target data into training and testing sets based on the specified year.

    Args:
        features (pd.DataFrame): DataFrame containing the features.
        target (pd.DataFrame): DataFrame containing the target variable.
        year (int): Year used to split the data into training (before the year) and testing (equal to the year).

    Returns:
        dict: A dictionary containing training and testing sets for features and target variables.
    """
    return {
        "train_features": features[features['year'] < year].reset_index(drop=True),
        "test_features": features[features['year'] == year].reset_index(drop=True),
        "train_target": target[target['year'] < year].reset_index(drop=True).drop('year', axis=1),
        "test_target": target[target['year'] == year].reset_index(drop=True).drop('year', axis=1)
    }

# Target and frequency encoding for categorical columns
def target_and_frequency_encoding(training_data: pd.DataFrame, testing_data: pd.DataFrame, group_col: str, target_col: str) -> tuple:
    """
    Perform target encoding and frequency encoding on a specified categorical column.

    Args:
        training_data (pd.DataFrame): Training data DataFrame.
        testing_data (pd.DataFrame): Testing data DataFrame.
        group_col (str): Categorical column to be encoded.
        target_col (str): Target column used for target encoding.

    Returns:
        tuple: Tuple containing the modified training and testing DataFrames with encoded columns.
    """
    means = training_data.groupby(group_col)[target_col].mean()
    freq = training_data[group_col].value_counts() / len(training_data)
    
    training_data[f'{group_col}_target_encoded'] = training_data[group_col].map(means)
    testing_data[f'{group_col}_target_encoded'] = testing_data[group_col].map(means)
    
    training_data[f'{group_col}_freq_encoded'] = training_data[group_col].map(freq)
    testing_data[f'{group_col}_freq_encoded'] = testing_data[group_col].map(freq)
    
    return training_data, testing_data

# Patch datatypes to optimize memory usage
def patch_datatypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize memory usage by converting column data types.

    Args:
        df (pd.DataFrame): DataFrame to optimize.

    Returns:
        pd.DataFrame: Optimized DataFrame with reduced memory usage.
    """
    float_cols = df.select_dtypes(include=['float64']).columns
    df[float_cols] = df[float_cols].astype(np.float32)

    int_cols = df.select_dtypes(include=['int64']).columns
    df[int_cols] = df[int_cols].astype(np.int32)
    
    return df

# Combine features and targets
def combine_features_targets(features: pd.DataFrame, target: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Combine features and target columns into a single DataFrame.

    Args:
        features (pd.DataFrame): DataFrame containing features.
        target (pd.DataFrame): DataFrame containing the target variable.
        target_col (str): Name of the target column to be used in the combined DataFrame.

    Returns:
        pd.DataFrame: Combined DataFrame with a new column 'crime_status' indicating whether the target variable is greater than 0.
    """
    combined_df = pd.concat([features, target], axis=1)
    combined_df['crime_status'] = combined_df[target_col] > 0
    return combined_df

# Apply Sturges' formula to determine the number of bins
def sturges_formula(n: int) -> int:
    """
    Determine the number of bins using Sturges' formula.

    Args:
        n (int): Number of observations.

    Returns:
        int: Number of bins calculated using Sturges' formula.
    """
    return int(np.ceil(np.log2(n) + 1))

# Bin the DataFrame based on the number of bins
def bin_dataframe(df: pd.DataFrame, exempt: List[str], bins: int) -> pd.DataFrame:
    """
    Bin numerical columns of a DataFrame while leaving exempt columns unchanged.

    Args:
        df (pd.DataFrame): DataFrame to bin.
        exempt (List[str]): List of columns to be exempted from binning.
        bins (int): Number of bins for binning numerical columns.

    Returns:
        pd.DataFrame: DataFrame with binned numerical columns.
    """
    binned_df = pd.DataFrame()
    for col in df.columns:
        if col not in exempt:
            binned_df[col] = pd.cut(df[col], bins=bins, labels=False)
        else:
            binned_df[col] = df[col]
    return binned_df

# Perform stratified sampling based on binned data
def stratified_sampling(df_false: pd.DataFrame, df_true: pd.DataFrame, exempt: List[str]) -> pd.DataFrame:
    """
    Perform stratified sampling to balance the dataset by matching the number of false samples to the number of true samples.

    Args:
        df_false (pd.DataFrame): DataFrame containing false samples.
        df_true (pd.DataFrame): DataFrame containing true samples.
        exempt (List[str]): List of columns to be exempted from binning.

    Returns:
        pd.DataFrame: Balanced DataFrame after stratified sampling.
    """
    false_bins = sturges_formula(len(df_false))
    true_bins = sturges_formula(len(df_true))

    df_false_binned = bin_dataframe(df_false, exempt, false_bins)
    df_false_binned['combined'] = df_false_binned.apply(lambda row: tuple(row), axis=1)
    combined_weight = df_false_binned['combined'].value_counts(normalize=True)
    df_false_binned['combined_weight'] = df_false_binned['combined'].apply(lambda x: combined_weight[x])
    
    df_false_sample = df_false_binned.sample(n=len(df_true), weights=df_false_binned['combined_weight']).drop(['combined', 'combined_weight'], axis=1).reset_index()
    sampled_df_false = df_false.loc[df_false_sample.index]
    
    return pd.concat([sampled_df_false, df_true]).sample(frac=1).reset_index(drop=True)

# Save datasets to CSV
def save_datasets(prefix: str, feature_train: pd.DataFrame, target_train: pd.DataFrame, feature_test: pd.DataFrame, target_test: pd.DataFrame) -> None:
    """
    Save the training and testing datasets to CSV files.

    Args:
        prefix (str): Prefix for the output filenames.
        feature_train (pd.DataFrame): DataFrame containing the training features.
        target_train (pd.DataFrame): DataFrame containing the training target.
        feature_test (pd.DataFrame): DataFrame containing the testing features.
        target_test (pd.DataFrame): DataFrame containing the testing target.
    """
    feature_train.to_csv(f'../../data/pre_training/{prefix}_feature_training_sample.csv', index=False)
    target_train.to_csv(f'../../data/pre_training/{prefix}_target_training_sample.csv', index=False)
    feature_test.to_csv(f'../../data/pre_training/{prefix}_feature_testing_data.csv', index=False)
    target_test.to_csv(f'../../data/pre_training/{prefix}_target_testing_data.csv', index=False)

# Main function to orchestrate the process
def main() -> None:
    """
    Main function to orchestrate the data loading, splitting, encoding, datatype optimization, 
    stratified sampling, and saving of datasets.
    """
    # Load data
    data = load_pre_feature_selection_data()

    # Split the area and district datasets into training and testing datasets
    area_split = split_train_test(data["area_pre_feature_selection"].drop('area_crimes_this_hour', axis=1),
                                  data["area_pre_feature_selection"][['year', 'area_crimes_this_hour']], 2020)
    district_split = split_train_test(data["district_pre_feature_selection"].drop('district_crimes_this_hour', axis=1),
                                      data["district_pre_feature_selection"][['year', 'district_crimes_this_hour']], 2020)

    # Perform target and frequency encoding
    area_split["train_features"], area_split["test_features"] = target_and_frequency_encoding(
        area_split["train_features"], area_split["test_features"], 'area_id', 'area_crimes_this_hour'
    )
    district_split["train_features"], district_split["test_features"] = target_and_frequency_encoding(
        district_split["train_features"], district_split["test_features"], 'district', 'district_crimes_this_hour'
    )

    # Patch datatypes
    area_split["train_features"] = patch_datatypes(area_split["train_features"])
    area_split["test_features"] = patch_datatypes(area_split["test_features"])
    district_split["train_features"] = patch_datatypes(district_split["train_features"])
    district_split["test_features"] = patch_datatypes(district_split["test_features"])

    # Combine features and targets
    area_combined = combine_features_targets(area_split["train_features"], area_split["train_target"], 'area_crimes_this_hour')
    district_combined = combine_features_targets(district_split["train_features"], district_split["train_target"], 'district_crimes_this_hour')

    # Separate true and false samples for stratified sampling
    area_combined_false = area_combined[area_combined['crime_status'] == False].reset_index(drop=True)
    area_combined_true = area_combined[area_combined['crime_status'] == True].reset_index(drop=True)
    district_combined_false = district_combined[district_combined['crime_status'] == False].reset_index(drop=True)
    district_combined_true = district_combined[district_combined['crime_status'] == True].reset_index(drop=True)

    # Perform stratified sampling
    area_training_sample = stratified_sampling(area_combined_false, area_combined_true,
                                               ['day', 'hour', 'year', 'month', 'day_of_week', 'crime_status', 'area_id_target_encoded', 'area_id_freq_encoded'])
    district_training_sample = stratified_sampling(district_combined_false, district_combined_true,
                                                   ['day', 'hour', 'year', 'month', 'day_of_week', 'crime_status', 'district_target_encoded', 'district_freq_encoded'])

    # Split into feature and target datasets
    area_feature_training_sample = area_training_sample.drop('area_crimes_this_hour', axis=1)
    area_target_training_sample = area_training_sample[['area_crimes_this_hour']]
    district_feature_training_sample = district_training_sample.drop('district_crimes_this_hour', axis=1)
    district_target_training_sample = district_training_sample[['district_crimes_this_hour']]

    # Save datasets
    save_datasets('area', area_feature_training_sample, area_target_training_sample, area_split["test_features"], area_split["test_target"])
    save_datasets('district', district_feature_training_sample, district_target_training_sample, district_split["test_features"], district_split["test_target"])

if __name__ == "__main__":
    main()
