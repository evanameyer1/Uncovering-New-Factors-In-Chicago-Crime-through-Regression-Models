import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
import pickle
from sklearn.inspection import permutation_importance

# Load training and testing datasets
area_feature_training_sample = pd.read_csv('../../data/pre_training/area_feature_training_sample.csv')
area_target_training_sample = pd.read_csv('../../data/pre_training/area_target_training_sample.csv')
district_feature_training_sample = pd.read_csv('../../data/pre_training/district_feature_training_sample.csv')
district_target_training_sample = pd.read_csv('../../data/pre_training/district_target_training_sample.csv')

area_feature_testing_data = pd.read_csv('../../data/pre_training/area_feature_testing_data.csv')
area_target_testing_data = pd.read_csv('../../data/pre_training/area_target_testing_data.csv')
district_feature_testing_data = pd.read_csv('../../data/pre_training/district_feature_testing_data.csv')
district_target_testing_data = pd.read_csv('../../data/pre_training/district_target_testing_data.csv')

# Drop 'crime_status' column for both datasets
area_feature_training_sample.drop('crime_status', axis=1, inplace=True)
district_feature_training_sample.drop('crime_status', axis=1, inplace=True)


class Evaluator:
    """
    Evaluator class for evaluating RandomForestRegressor model performance.
    """
    def __init__(self, model: RandomForestRegressor, test_features: pd.DataFrame, test_labels: pd.Series):
        self.model = model
        self.test_features = test_features
        self.test_labels = test_labels
        self.sample_size = len(self.test_labels)
        self.independent_vars = len(test_features.columns)
        self.predictions = None
        self.residuals = None

    def set_residuals(self, residual: pd.Series):
        """
        Set the residuals between true and predicted values.
        """
        self.residuals = residual

    def gather_predictions(self):
        """
        Generate predictions using the trained model.
        """
        self.model.set_params(verbose=0)
        self.predictions = self.model.predict(self.test_features)
        self.residuals = self.test_labels - self.predictions
    
    def mae(self) -> float:
        """
        Calculate the Mean Absolute Error (MAE).
        """
        if self.residuals is None:
            raise ValueError("Predictions need to be gathered first")
        return np.mean(np.abs(self.residuals))
    
    def mse(self) -> float:
        """
        Calculate the Mean Squared Error (MSE).
        """
        if self.residuals is None:
            raise ValueError("Predictions need to be gathered first")
        return np.mean(self.residuals ** 2)
    
    def rmse(self) -> float:
        """
        Calculate the Root Mean Squared Error (RMSE).
        """
        return np.sqrt(self.mse())
    
    def relative_rmse(self) -> float:
        """
        Calculate the Relative RMSE by normalizing RMSE by the maximum test label value.
        """
        return self.rmse() / max(self.test_labels)

    def r_squared(self) -> float:
        """
        Calculate the R-squared value, a measure of the model's goodness-of-fit.
        """
        ss_res = np.sum(self.residuals ** 2)
        ss_tot = np.sum((self.test_labels - np.mean(self.test_labels)) ** 2)
        return 1 - (ss_res / ss_tot)

    def adjusted_r_squared(self) -> float:
        """
        Calculate the adjusted R-squared, which adjusts the R-squared for the number of predictors.
        """
        r2 = self.r_squared()
        return 1 - (1 - r2) * ((self.sample_size - 1) / (self.sample_size - self.independent_vars - 1))
    
    def median_absolute_error(self) -> float:
        """
        Calculate the Median Absolute Error.
        """
        return np.median(np.abs(self.residuals))

    def plot_residuals(self, title: str, save_name: str) -> None:
        """
        Plot the residuals as a percentage difference between test labels and predictions.
        """
        percent_diff = (self.residuals / max(self.test_labels)) * 100
        
        plt.figure(figsize=(10, 5))
        plt.plot(percent_diff, label='Percent Difference', marker='o')
        
        avg_percent_diff = np.mean(percent_diff)
        plt.axhline(y=avg_percent_diff, color='r', linestyle='dashed', label='Average Percent Difference')
        plt.text(0, avg_percent_diff + 1, f'Avg: {avg_percent_diff:.5f}%', color='r', fontsize=12)

        plt.title(f'Percent Difference between Test Labels and Predicted Values - {title} Model')
        plt.xlabel('Sample Index')
        plt.ylabel('Percent Difference (%)')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'../results/random_forest/{save_name}.png')
        plt.show()


def evaluate(model: RandomForestRegressor, test_features: pd.DataFrame, test_labels: pd.Series, title: str, save_name: str) -> None:
    """
    Evaluate a RandomForestRegressor model using various evaluation metrics and plot the residuals.

    Args:
    - model: Trained RandomForestRegressor model.
    - test_features: Test dataset features.
    - test_labels: Actual test dataset labels.
    - title: Model title (for plot).
    - save_name: File name to save the residual plot.
    """
    ev = Evaluator(model, test_features, test_labels)
    ev.gather_predictions()
    
    print(f"Mean Absolute Error (MAE): {ev.mae():.6f}")
    print(f"Mean Squared Error (MSE): {ev.mse():.6f}")
    print(f"Root Mean Squared Error (RMSE): {ev.rmse():.6f}")
    print(f"Relative Root Mean Squared Error (Relative RMSE): {ev.relative_rmse():.6f}")
    print(f"R-squared (R²): {ev.r_squared():.6f}")
    print(f"Adjusted R-squared: {ev.adjusted_r_squared():.6f}")
    print(f"Median Absolute Error: {ev.median_absolute_error():.6f}")
    
    ev.plot_residuals(title, save_name)


# Train initial RandomForest models for Area and District datasets
area_target_training_sample = area_target_training_sample.values.ravel()
area_target_testing_data = area_target_testing_data.values.ravel()
area_model = RandomForestRegressor(n_jobs=15, verbose=1)
area_model.fit(area_feature_training_sample, area_target_training_sample)
evaluate(area_model, area_feature_testing_data, area_target_testing_data, 'Area', 'initial_area_residuals')

district_target_training_sample = district_target_training_sample.values.ravel()
district_target_testing_data = district_target_testing_data.values.ravel()
district_model = RandomForestRegressor(n_jobs=15, verbose=1)
district_model.fit(district_feature_training_sample, district_target_training_sample)
evaluate(district_model, district_feature_testing_data, district_target_testing_data, 'District', 'initial_district_residuals')


def feature_selection_by_cumsum(df: pd.DataFrame, cum_sum: float) -> pd.Series:
    """
    Selects features based on cumulative sum of importance.

    Args:
    - df: DataFrame containing features and their importance values.
    - cum_sum: Cumulative sum threshold for feature selection.

    Returns:
    - Series containing selected features.
    """
    df_cum = np.cumsum(df['importance'])
    return df['feature'][df_cum <= cum_sum]


def iterate_through_cumsum(df: pd.DataFrame, training_sample: tuple, testing_data: tuple, cum_sum_range: np.ndarray) -> tuple:
    """
    Iterates through different cumulative sum values to select features and train RandomForestRegressor models.

    Args:
    - df: DataFrame containing feature importance.
    - training_sample: Tuple containing training features and targets.
    - testing_data: Tuple containing testing features and targets.
    - cum_sum_range: Range of cumulative sum values to iterate through.

    Returns:
    - Best features and DataFrame with results.
    """
    best_score = 0
    best_features = None
    results = []

    for cum_sum in cum_sum_range:
        temp_selected_features = feature_selection_by_cumsum(df, cum_sum)
        temp_model = RandomForestRegressor(n_jobs=20, verbose=1)
        temp_model.fit(training_sample[0][temp_selected_features], training_sample[1])
        acc_after_feature_selection = temp_model.score(testing_data[0][temp_selected_features], testing_data[1])
        print(f'Accuracy after feature selection at cum sum {cum_sum}: {acc_after_feature_selection:.4f}')

        results.append({'cum_sum': cum_sum, 'accuracy': acc_after_feature_selection})

        if acc_after_feature_selection > best_score:
            best_score = acc_after_feature_selection
            best_features = temp_selected_features
    
    results_df = pd.DataFrame(results)
    return best_features, results_df


def forward_selection(training_sample: tuple, testing_data: tuple, features: list) -> list:
    """
    Performs forward selection of features based on accuracy improvement.

    Args:
    - training_sample: Tuple containing training features and targets.
    - testing_data: Tuple containing testing features and targets.
    - features: List of feature names to iterate through.

    Returns:
    - List of best features selected.
    """
    best_score = 0
    best_features = None

    for idx in range(len(features)):
        temp_selected_features = features[:idx+1]
        temp_model = RandomForestRegressor(n_jobs=20, verbose=1)
        temp_model.fit(training_sample[0][temp_selected_features], training_sample[1])
        acc_after_feature_selection = temp_model.score(testing_data[0][temp_selected_features], testing_data[1])
        print(f'Accuracy with features {temp_selected_features}: {acc_after_feature_selection:.4f}')

        if acc_after_feature_selection > best_score:
            best_score = acc_after_feature_selection
            best_features = temp_selected_features
    
    return best_features

def print_importances(df: pd.DataFrame) -> None:
    """
    Prints the feature importance from the dataframe.

    Args:
    - df: DataFrame containing 'feature' and 'importance' columns.
    
    Returns:
    - None
    """
    for idx, row in df.iterrows():
        print(f"{row['feature']} - importance: {row['importance']}")


def save_fig(params: np.ndarray, accs: np.ndarray, version: str, title: str) -> pd.DataFrame:
    """
    Save a plot of the parameter tuning results and return a DataFrame of the results.

    Args:
    - params: Array of parameter values.
    - accs: Array of accuracy values.
    - version: Version name (for saving the file).
    - title: Plot title and file name.
    
    Returns:
    - pd.DataFrame: DataFrame containing parameter values and accuracies.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(params, accs, marker='o', linestyle='-', color='b')
    plt.title(f'Effect of {title} on Model Accuracy')
    plt.xlabel(title)
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.savefig(f'../results/random_forest/feature_tuning/{version}/{title}.png')
    plt.show()

    return pd.DataFrame(data={'params': params, 'accs': accs})


def graph_parameter(
        training_samplesets: tuple, 
        testing_datasets: tuple, 
        parameter: str, 
        parameter_start: float, 
        parameter_end: float, 
        parameter_step: float, 
        min_iteration: int, 
        tolerance: float = 0.01, 
        max_tolerable_changes: int = 3) -> tuple:
    """
    Generate accuracy vs parameter graph and find the best parameter value for a RandomForest model.

    Args:
    - training_samplesets: Tuple containing training features and targets.
    - testing_datasets: Tuple containing testing features and targets.
    - parameter: The parameter to tune (e.g., 'n_estimators', 'max_depth').
    - parameter_start: Starting value of the parameter.
    - parameter_end: End value of the parameter.
    - parameter_step: Increment for the parameter values.
    - min_iteration: Minimum iterations for grid search.
    - tolerance: Tolerance to measure improvement in accuracy.
    - max_tolerable_changes: Max number of consecutive iterations without improvement.

    Returns:
    - tuple: Arrays of parameter values and corresponding accuracies.
    """
    accuracies = []
    no_improvement_count = 0
    drop_count = 0

    if parameter_end == -1:
        p = parameter_start
        while True:
            model = RandomForestRegressor(**{parameter: p}, n_jobs=18)
            model.fit(training_samplesets[0], training_samplesets[1])
            acc = model.score(testing_datasets[0], testing_datasets[1])

            accuracies.append((p, acc))
            print(f'Accuracy of {acc:.4%} found for parameter {parameter} at {p}')

            if len(accuracies) > 1 and min_iteration < 0:
                if acc < accuracies[-2][1] * (1 - tolerance):
                    drop_count += 1
                elif acc < accuracies[-2][1] * (1 + tolerance):
                    no_improvement_count += 1
                else:
                    drop_count = 0
                    no_improvement_count = 0
                
                if drop_count > max_tolerable_changes or no_improvement_count > max_tolerable_changes:
                    break

            p += parameter_step
            min_iteration -= 1

    else:
        parameters = [x for x in np.arange(parameter_start, parameter_end, parameter_step)]
        for p in parameters:
            model = RandomForestRegressor(**{parameter: p}, n_jobs=18)
            model.fit(training_samplesets[0], training_samplesets[1])
            acc = model.score(testing_datasets[0], testing_datasets[1])
            accuracies.append((p, acc))
            print(f'Accuracy of {acc:.4%} found for parameter {parameter} at {p}')

    param_values, accuracy_values = zip(*accuracies)

    plt.figure(figsize=(15, 10))
    plt.plot(param_values, accuracy_values, marker='o', linestyle='-', color='b')
    plt.title(f'Effect of {parameter} on Model Accuracy')
    plt.xlabel(parameter)
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.show()

    return param_values, accuracy_values


# Running Grid Search for optimal hyperparameters
def run_grid_search(
    training_features: pd.DataFrame, 
    training_labels: pd.Series, 
    param_grid: dict, 
    cv: int = 5, 
    n_jobs: int = 10, 
    verbose: int = 3
) -> GridSearchCV:
    """
    Performs GridSearchCV to find the best parameters for a RandomForestRegressor.

    Args:
    - training_features: Training feature set.
    - training_labels: Training labels.
    - param_grid: Dictionary of parameters to search.
    - cv: Number of cross-validation folds.
    - n_jobs: Number of jobs to run in parallel.
    - verbose: Verbosity level for logging.
    
    Returns:
    - GridSearchCV object with the best parameters and score.
    """
    model = RandomForestRegressor()
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, n_jobs=n_jobs, verbose=verbose, scoring='r2')
    grid_search.fit(training_features, training_labels)
    
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    print(f'Best Parameters: {best_params}')
    print(f'Best R2 Score: {best_score:.4f}')

    return grid_search


# Feature Importance plotting for Random Forest models
def plot_feature_importances(
        feature_importances: np.ndarray, 
        feature_names: list, 
        title: str, 
        save_name: str) -> None:
    """
    Plots and saves feature importance from the RandomForestRegressor model.

    Args:
    - feature_importances: Array of feature importance values.
    - feature_names: List of feature names.
    - title: Plot title.
    - save_name: Filename to save the plot.
    
    Returns:
    - None
    """
    feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importances})
    feature_importance_df.sort_values(by='importance', ascending=False, inplace=True)

    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance_df['feature'], feature_importance_df['importance'], color='b')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f'../results/random_forest/{save_name}.png')
    plt.show()


def plot_accuracy_vs_threshold(
        df_combined: pd.DataFrame, 
        predicted_column: str = 'predicted', 
        target_column: str = 'target', 
        threshold_range: tuple = (0.1, 3), 
        threshold_step: float = 0.1, 
        save_path: str = None) -> None:
    """
    Plots accuracy vs threshold percentage for model predictions.

    Args:
    - df_combined: DataFrame containing predicted and target values.
    - predicted_column: Column name for predicted values.
    - target_column: Column name for actual target values.
    - threshold_range: Range of threshold percentages to evaluate.
    - threshold_step: Step size for the threshold percentage.
    - save_path: File path to save the plot.
    
    Returns:
    - None
    """
    accuracies = []
    max_predicted_value = np.max(np.abs(df_combined[predicted_column]))

    for threshold_percentage in np.arange(threshold_range[0], threshold_range[1], threshold_step):
        threshold_value = (threshold_percentage / 100) * max_predicted_value
        correct_predictions = np.sum(np.abs(df_combined[predicted_column] - df_combined[target_column]) <= threshold_value)
        total_predictions = len(df_combined)
        accuracy = correct_predictions / total_predictions
        accuracies.append(accuracy)

    plt.figure(figsize=(8, 6))
    plt.plot(np.arange(threshold_range[0], threshold_range[1], threshold_step), [accuracy * 100 for accuracy in accuracies], marker='o')
    plt.title('Accuracy vs Threshold Percentage')
    plt.xlabel('Threshold Percentage (%)')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)

    plt.show()


# Feature ablation analysis
def feature_ablation(
        model: RandomForestRegressor, 
        training_sample: tuple, 
        test_data: tuple) -> list:
    """
    Perform feature ablation by dropping one feature at a time and assessing model accuracy.

    Args:
    - model: Trained RandomForestRegressor model.
    - training_sample: Tuple containing training features and targets.
    - test_data: Tuple containing testing features and targets.

    Returns:
    - list: A list of tuples with feature names and their corresponding performance metrics after ablation.
    """
    accs = []
    features = training_sample[0].columns

    for feature in features:
        temp_training_sample = training_sample[0].drop(feature, axis=1)
        temp_testing_data = test_data[0].drop(feature, axis=1)
        model.set_params(n_jobs=10, verbose=1)
        model.fit(temp_training_sample, training_sample[1])

        evaluator = Evaluator(model, temp_testing_data, test_data[1])
        evaluator.gather_predictions()
        scores = {
            'MAE': evaluator.mae(),
            'R^2': evaluator.r_squared(),
            'RMSE': evaluator.rmse(),
            'Relative RMSE': evaluator.relative_rmse()
        }
        accs.append((feature, scores))
        print(f'{feature} ablation has resulting scores of: {scores}')
    
    return accs


def calculate_differences(
        original_accs: dict, 
        ablation_accs: list) -> dict:
    """
    Calculate differences in accuracy metrics between the original model and the feature ablation results.

    Args:
    - original_accs: Dictionary of original model accuracy metrics.
    - ablation_accs: List of accuracy metrics after feature ablation.

    Returns:
    - dict: Dictionary containing the differences in accuracy metrics for each feature.
    """
    out = {}
    scores = ['MAE', 'R^2', 'RMSE', 'Relative RMSE']

    for feature, accs in ablation_accs:
        out[feature] = [(score, accs[score] - original_accs[score]) for score in scores]
    
    return out


# Partial Dependency and Permutation Importance
def plot_partial_dependency(model: RandomForestRegressor, training_features: pd.DataFrame, save_path: str) -> None:
    """
    Plot Partial Dependency for features in a RandomForestRegressor model.

    Args:
    - model: Trained RandomForestRegressor model.
    - training_features: Training features used for the model.
    - save_path: Path to save the resulting plot.
    
    Returns:
    - None
    """
    fig, ax = plt.subplots(figsize=(20, 15))
    PartialDependenceDisplay.from_estimator(
        model, 
        training_features, 
        features=training_features.columns, 
        ax=ax, 
        n_jobs=12, 
        grid_resolution=50, 
        percentiles=(0, 1)
    )

    fig.suptitle('Partial Dependency Plots', fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(save_path)
    plt.show()


def permutation_importance_analysis(
        model: RandomForestRegressor, 
        training_features: pd.DataFrame, 
        training_labels: pd.Series, 
        save_path: str) -> None:
    """
    Calculate and plot permutation importance for the given RandomForestRegressor model.

    Args:
    - model: Trained RandomForestRegressor model.
    - training_features: Training feature set.
    - training_labels: Training labels.
    - save_path: Path to save the permutation importance plot.
    
    Returns:
    - None
    """
    perm_result = permutation_importance(model, training_features, training_labels, n_repeats=10, random_state=42)

    perm_importance_df = pd.DataFrame({
        'Feature': training_features.columns,
        'Importance': perm_result.importances_mean
    })

    perm_importance_df = perm_importance_df.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(8, 4))
    plt.barh(perm_importance_df['Feature'], perm_importance_df['Importance'], color='skyblue')
    plt.xlabel('Permutation Importance')
    plt.ylabel('Feature')
    plt.title('Permutation Importance of Features')
    plt.tight_layout()
    plt.gca().invert_yaxis()
    plt.savefig(save_path)
    plt.show()


# Save the final district model
def save_model(model: RandomForestRegressor, file_path: str) -> None:
    """
    Save the trained RandomForestRegressor model using pickle.

    Args:
    - model: Trained RandomForestRegressor model.
    - file_path: Path to save the model.
    
    Returns:
    - None
    """
    with open(file_path, 'wb') as file:
        pickle.dump(model, file)
