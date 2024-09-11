import pandas as pd
import numpy as np

#### Loading Datasets
def load_data() -> dict[str, pd.DataFrame]:
    """
    Loads the area and district datasets for feature selection.

    Returns:
        dict: Dictionary containing loaded datasets for area and district.
    """
    area_pre_feature_selection = pd.read_csv('../../data/pre_training/area_pre_feature_selection.csv')
    district_pre_feature_selection = pd.read_csv('../../data/pre_training/district_pre_feature_selection.csv')
    
    return {
        'area_pre_feature_selection': area_pre_feature_selection,
        'district_pre_feature_selection': district_pre_feature_selection
    }

data = load_data()
area_pre_feature_selection = data['area_pre_feature_selection']
district_pre_feature_selection = data['district_pre_feature_selection']

#### Splitting Data into Features and Targets
area_features = area_pre_feature_selection.drop('area_crimes_this_hour', axis=1)
district_features = district_pre_feature_selection.drop('district_crimes_this_hour', axis=1)

area_target = area_pre_feature_selection[['year', 'area_crimes_this_hour']]
district_target = district_pre_feature_selection[['year', 'district_crimes_this_hour']]

#### Splitting Data into Training and Testing
def split_data(features: pd.DataFrame, target: pd.DataFrame, year_col: str, split_year: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the dataset into training and testing based on the specified year.

    Args:
        features (pd.DataFrame): Feature dataset.
        target (pd.DataFrame): Target dataset.
        year_col (str): Column name representing the year.
        split_year (int): Year to split the data.

    Returns:
        tuple: Training and testing features, and training and testing targets.
    """
    feature_training_data = features[features[year_col] < split_year].reset_index(drop=True)
    feature_testing_data = features[features[year_col] == split_year].reset_index(drop=True)

    target_training_data = target[target[year_col] < split_year].reset_index(drop=True)
    target_testing_data = target[target[year_col] == split_year].reset_index(drop=True)
    
    return feature_training_data, feature_testing_data, target_training_data, target_testing_data

# Splitting area and district datasets into training and testing sets
area_feature_training_data, area_feature_testing_data, area_target_training_data, area_target_testing_data = split_data(area_features, area_target, 'year', 2020)
district_feature_training_data, district_feature_testing_data, district_target_training_data, district_target_testing_data = split_data(district_features, district_target, 'year', 2020)

#### Dropping Columns and Preparing Features
def prepare_features(feature_data: pd.DataFrame, target_data: pd.DataFrame, drop_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepares features by dropping specified columns and removing year from the target data.

    Args:
        feature_data (pd.DataFrame): Feature dataset.
        target_data (pd.DataFrame): Target dataset.
        drop_cols (list[str]): Columns to drop from the feature data.

    Returns:
        tuple: Prepared feature and target datasets.
    """
    feature_data = feature_data.drop(drop_cols, axis=1)
    target_data = target_data.drop('year', axis=1)
    return feature_data, target_data

# Dropping unnecessary columns
area_feature_training_data, area_target_training_data = prepare_features(area_feature_training_data, area_target_training_data, ['date_hour'])
area_feature_testing_data, area_target_testing_data = prepare_features(area_feature_testing_data, area_target_testing_data, ['date_hour'])

district_feature_training_data, district_target_training_data = prepare_features(district_feature_training_data, district_target_training_data, ['date_hour'])
district_feature_testing_data, district_target_testing_data = prepare_features(district_feature_testing_data, district_target_testing_data, ['date_hour'])

#### Target and Frequency Encoding of District/Area Columns
def encode_columns(training_data: pd.DataFrame, testing_data: pd.DataFrame, col: str, means: pd.Series, freq: pd.Series, target_encoded_col: str, freq_encoded_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Encodes a specified column using target and frequency encoding.

    Args:
        training_data (pd.DataFrame): Training dataset.
        testing_data (pd.DataFrame): Testing dataset.
        col (str): Column to encode.
        means (pd.Series): Mean values for target encoding.
        freq (pd.Series): Frequency values for frequency encoding.
        target_encoded_col (str): Name of the new target-encoded column.
        freq_encoded_col (str): Name of the new frequency-encoded column.

    Returns:
        tuple: Updated training and testing datasets.
    """
    training_data[target_encoded_col] = training_data[col].map(means)
    testing_data[target_encoded_col] = testing_data[col].map(means)

    training_data[freq_encoded_col] = training_data[col].map(freq)
    testing_data[freq_encoded_col] = testing_data[col].map(freq)
    
    return training_data, testing_data

# Target and frequency encoding for area and district columns
area_means = area_pre_feature_selection.groupby('area_id')['area_crimes_this_hour'].mean()
district_means = district_pre_feature_selection.groupby('district')['district_crimes_this_hour'].mean()

area_freq = area_pre_feature_selection['area_id'].value_counts() / len(area_pre_feature_selection)
district_freq = district_pre_feature_selection['district'].value_counts() / len(district_pre_feature_selection)

area_feature_training_data, area_feature_testing_data = encode_columns(area_feature_training_data, area_feature_testing_data, 'area_id', area_means, area_freq, 'area_id_target_encoded', 'area_id_freq_encoded')
district_feature_training_data, district_feature_testing_data = encode_columns(district_feature_training_data, district_feature_testing_data, 'district', district_means, district_freq, 'district_target_encoded', 'district_freq_encoded')

# Dropping the original area_id and district columns
area_feature_training_data.drop('area_id', axis=1, inplace=True)
area_feature_testing_data.drop('area_id', axis=1, inplace=True)

district_feature_training_data.drop('district', axis=1, inplace=True)
district_feature_testing_data.drop('district', axis=1, inplace=True)

#### Patching Data Types
def patch_datatypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts float64 columns to float32 and int64 columns to int32 to reduce memory usage.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with patched data types.
    """
    float_cols = df.select_dtypes(include=['float64']).columns
    df[float_cols] = df[float_cols].astype(np.float32)

    int_cols = df.select_dtypes(include=['int64']).columns
    df[int_cols] = df[int_cols].astype(np.int32)    
      
    return df

# Patch data types
area_feature_training_data = patch_datatypes(area_feature_training_data)
area_feature_testing_data = patch_datatypes(area_feature_testing_data)
district_feature_training_data = patch_datatypes(district_feature_training_data)
district_feature_testing_data = patch_datatypes(district_feature_testing_data)

#### Selecting Representative Sample Using Sturges' Formula
def sturges_formula(n: int) -> int:
    """
    Sturges' formula to determine the number of bins.

    Args:
        n (int): Number of samples.

    Returns:
        int: Number of bins.
    """
    return int(np.ceil(np.log2(n) + 1))

def bin_dataframe(df: pd.DataFrame, exempt: list[str], bins: int) -> pd.DataFrame:
    """
    Bins the columns of a DataFrame, excluding specified columns.

    Args:
        df (pd.DataFrame): DataFrame to bin.
        exempt (list[str]): List of columns to exempt from binning.
        bins (int): Number of bins to use.

    Returns:
        pd.DataFrame: Binned DataFrame.
    """
    binned_df = pd.DataFrame()
    for col in df.columns:
        if col not in exempt:
            binned_df[col] = pd.cut(df[col], bins=bins, labels=False)
        else:
            binned_df[col] = df[col]
    
    return binned_df

# Sturges' formula and binning for area and district training sets
area_false_bins = sturges_formula(len(area_feature_training_data))
district_false_bins = sturges_formula(len(district_feature_training_data))

area_training_combined = pd.concat([area_feature_training_data, area_target_training_data], axis=1)
area_training_combined['crime_status'] = area_training_combined['area_crimes_this_hour'] > 0

area_training_combined_false = area_training_combined[area_training_combined['crime_status'] == False].reset_index(drop=True)
area_training_combined_true = area_training_combined[area_training_combined['crime_status'] == True].reset_index(drop=True)

# Binning and sampling
area_training_combined_false_binned = bin_dataframe(area_training_combined_false, ['day', 'hour', 'year', 'month', 'day_of_week', 'crime_status', 'area_id_target_encoded', 'area_id_freq_encoded'], area_false_bins)
area_training_combined_false_binned['combined'] = area_training_combined_false_binned.apply(lambda row: tuple(row), axis=1)
area_combined_false_weight = area_training_combined_false_binned['combined'].value_counts(normalize=True)
area_training_combined_false_binned['combined_weight'] = area_training_combined_false_binned['combined'].apply(lambda x: area_combined_false_weight[x])
area_training_combined_false_binned_sample = area_training_combined_false_binned.sample(n=len(area_training_combined_true), weights=area_training_combined_false_binned['combined_weight']).drop(['combined', 'combined_weight'], axis=1).reset_index()

# Creating balanced samples for area
area_training_false_sample = area_training_combined_false.loc[area_training_combined_false_binned_sample.index]
area_training_sample = pd.concat([area_training_false_sample, area_training_combined_true]).sample(frac=1).reset_index(drop=True)

# Prepare final features and targets for area and district
area_feature_training_sample = area_training_sample.drop('area_crimes_this_hour', axis=1)
area_target_training_sample = area_training_sample[['area_crimes_this_hour']]

#### Saving Pre-processed Data
def save_data(df: pd.DataFrame, path: str) -> None:
    """
    Saves the DataFrame to a CSV file.

    Args:
        df (pd.DataFrame): DataFrame to save.
        path (str): Path to save the CSV file.
    """
    df.to_csv(path, index=False)

# Save area and district samples, features, and targets
save_data(area_feature_training_sample, '../../data/pre_training/area_feature_training_sample.csv')
save_data(area_target_training_sample, '../../data/pre_training/area_target_training_sample.csv')
save_data(area_feature_testing_data, '../../data/pre_training/area_feature_testing_data.csv')
save_data(area_target_testing_data, '../../data/pre_training/area_target_testing_data.csv')

save_data(district_feature_training_data, '../../data/pre_training/district_feature_training_sample.csv')
save_data(district_target_training_data, '../../data/pre_training/district_target_training_sample.csv')
save_data(district_feature_testing_data, '../../data/pre_training/district_feature_testing_data.csv')
save_data(district_target_testing_data, '../../data/pre_training/district_target_testing_data.csv')