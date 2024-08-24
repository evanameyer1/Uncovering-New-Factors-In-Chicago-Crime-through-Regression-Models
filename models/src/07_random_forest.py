# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
import seaborn as sns
import matplotlib.gridspec as gridspec

# Load datasets
area_feature_training_sample = pd.read_csv('../../data/pre_training/area_feature_training_sample.csv')
area_target_training_sample = pd.read_csv('../../data/pre_training/area_target_training_sample.csv')
district_feature_training_sample = pd.read_csv('../../data/pre_training/district_feature_training_sample.csv')
district_target_training_sample = pd.read_csv('../../data/pre_training/district_target_training_sample.csv')
area_feature_testing_data = pd.read_csv('../../data/pre_training/area_feature_testing_data.csv')
area_target_testing_data = pd.read_csv('../../data/pre_training/area_target_testing_data.csv')
district_feature_testing_data = pd.read_csv('../../data/pre_training/district_feature_testing_data.csv')
district_target_testing_data = pd.read_csv('../../data/pre_training/district_target_testing_data.csv')

# Remove 'crime_status' column
area_feature_training_sample.drop('crime_status', axis=1, inplace=True)
district_feature_training_sample.drop('crime_status', axis=1, inplace=True)

class Evaluator:
    """
    A class to evaluate model performance on test data with various metrics.
    
    Attributes:
        model (object): The machine learning model to evaluate.
        test_features (pd.DataFrame): The feature data for testing.
        test_labels (pd.Series): The true labels for testing.
    """
    def __init__(self, model, test_features: pd.DataFrame, test_labels: pd.Series):
        self.model = model
        self.test_features = test_features
        self.test_labels = test_labels
        self.sample_size = len(self.test_labels)
        self.independent_vars = len(test_features.columns)
        self.predictions = None
        self.residuals = None

    def set_residuals(self, residual: pd.Series) -> None:
        """Set the residuals for evaluation."""
        self.residuals = residual

    def gather_predictions(self) -> None:
        """Generate predictions using the model and calculate residuals."""
        self.model.set_params(verbose=0)
        self.predictions = self.model.predict(self.test_features)
        self.residuals = self.test_labels - self.predictions
    
    def mae(self) -> float:
        """Calculate Mean Absolute Error (MAE)."""
        if self.residuals is None:
            raise ValueError("Predictions need to be gathered first")
        return np.mean(np.abs(self.residuals))
    
    def mse(self) -> float:
        """Calculate Mean Squared Error (MSE)."""
        if self.residuals is None:
            raise ValueError("Predictions need to be gathered first")
        return np.mean(self.residuals ** 2)
    
    def rmse(self) -> float:
        """Calculate Root Mean Squared Error (RMSE)."""
        return np.sqrt(self.mse())
    
    def relative_rmse(self) -> float:
        """Calculate Relative RMSE by normalizing RMSE with the maximum label value."""
        return self.rmse() / max(self.test_labels)

    def r_squared(self) -> float:
        """Calculate the R-squared value."""
        ss_res = np.sum(self.residuals ** 2)
        ss_tot = np.sum((self.test_labels - np.mean(self.test_labels)) ** 2)
        return 1 - (ss_res / ss_tot)

    def adjusted_r_squared(self) -> float:
        """Calculate the Adjusted R-squared value."""
        r2 = self.r_squared()
        return 1 - (1 - r2) * ((self.sample_size - 1) / (self.sample_size - self.independent_vars - 1))
    
    def median_absolute_error(self) -> float:
        """Calculate the Median Absolute Error."""
        return np.median(np.abs(self.residuals))

    def plot_residuals(self, title: str, save_name: str) -> None:
        """
        Plot the residuals of the model predictions and save the plot.
        
        Args:
            title (str): The title of the plot.
            save_name (str): The filename for saving the plot.
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

def evaluate(model, test_features: pd.DataFrame, test_labels: pd.Series, title: str, save_name: str) -> None:
    """
    Evaluate model performance using the Evaluator class and print results.
    
    Args:
        model (object): The machine learning model to evaluate.
        test_features (pd.DataFrame): The feature data for testing.
        test_labels (pd.Series): The true labels for testing.
        title (str): The title of the plot.
        save_name (str): The filename for saving the plot.
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

# Train the initial Area Random Forest model
area_target_training_sample = area_target_training_sample.values.ravel()
area_target_testing_data = area_target_testing_data.values.ravel()
area_model = RandomForestRegressor(n_jobs=15, verbose=1)
area_model.fit(area_feature_training_sample, area_target_training_sample)
evaluate(area_model, area_feature_testing_data, area_target_testing_data, 'Area', 'initial_area_residuals')

# Train the initial District Random Forest model
district_target_training_sample = district_target_training_sample.values.ravel()
district_target_testing_data = district_target_testing_data.values.ravel()
district_model = RandomForestRegressor(n_jobs=15, verbose=1)
district_model.fit(district_feature_training_sample, district_target_training_sample)
evaluate(district_model, district_feature_testing_data, district_target_testing_data, 'District', 'initial_district_residuals')

def print_importances(df: pd.DataFrame) -> None:
    """Print the feature importances."""
    for idx, row in df.iterrows():
        print(row['feature'], '- importance:', row['importance'])

def feature_selection_by_cumsum(df: pd.DataFrame, cum_sum: float) -> pd.Series:
    """
    Select features based on cumulative sum of importances.

    Args:
        df (pd.DataFrame): DataFrame containing feature importances.
        cum_sum (float): The cumulative sum threshold for selecting features.

    Returns:
        pd.Series: The selected features.
    """
    df_cum = np.cumsum(df['importance'])
    return df['feature'][df_cum <= cum_sum]

def iterate_through_cumsum(df: pd.DataFrame, training_sample: tuple, testing_data: tuple, cum_sum_range: np.ndarray) -> tuple:
    """
    Iterate through different cumulative sum thresholds to select the best features.

    Args:
        df (pd.DataFrame): DataFrame containing feature importances.
        training_sample (tuple): Tuple of training features and target.
        testing_data (tuple): Tuple of testing features and target.
        cum_sum_range (np.ndarray): Range of cumulative sum thresholds to test.

    Returns:
        tuple: The best features and a DataFrame of results.
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

# Calculate feature importances for the Area and District models
area_importances = area_model.feature_importances_
area_feature_names = area_feature_training_sample.columns
area_feature_importance_df = pd.DataFrame({'feature': area_feature_names, 'importance': area_importances})
area_feature_importance_df.sort_values(by='importance', ascending=False, inplace=True)
print_importances(area_feature_importance_df)

district_importances = district_model.feature_importances_
district_feature_names = district_feature_training_sample.columns
district_feature_importance_df = pd.DataFrame({'feature': district_feature_names, 'importance': district_importances})
district_feature_importance_df.sort_values(by='importance', ascending=False, inplace=True)
print_importances(district_feature_importance_df)

# Select top features based on cumulative sum
area_top_features, area_cum_sum_df = iterate_through_cumsum(area_feature_importance_df, 
                                                            (area_feature_training_sample, area_target_training_sample), 
                                                            (area_feature_testing_data, area_target_testing_data), 
                                                            np.arange(0.6, 1.01, 0.05))

district_top_features, district_cum_sum_df = iterate_through_cumsum(district_feature_importance_df, 
                                                                    (district_feature_training_sample, district_target_training_sample), 
                                                                    (district_feature_testing_data, district_target_testing_data), 
                                                                    np.arange(0.6, 1.01, 0.05))

district_feature_importance_df['cum_sum'] = district_feature_importance_df['importance'].cumsum()

def forward_selection(training_sample: tuple, testing_data: tuple, features: list) -> list:
    """
    Perform forward selection to choose the best features based on model accuracy.

    Args:
        training_sample (tuple): Tuple of training features and target.
        testing_data (tuple): Tuple of testing features and target.
        features (list): List of features to select from.

    Returns:
        list: The best selected features.
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

district_target_features = list(district_feature_importance_df['feature'].head(13).values)
district_top_features = forward_selection((district_feature_training_sample, district_target_training_sample), 
                                          (district_feature_testing_data, district_target_testing_data), 
                                          district_target_features)

# Select the final set of features for the District model
district_feature_training_selected_data = district_feature_training_sample[district_top_features]
district_feature_testing_selected_data = district_feature_testing_data[district_top_features]

# Retest accuracy after feature selection
district_model_selected = RandomForestRegressor(n_jobs=15, verbose=1)
district_model_selected.fit(district_feature_training_selected_data, district_target_training_sample)
evaluate(district_model_selected, district_feature_testing_selected_data, district_target_testing_data, 'District', 'post_feature_selection_residuals')

def graph_parameter(training_samplesets: tuple, testing_datasets: tuple, parameter: str, parameter_start: float, parameter_end: float, parameter_step: float, min_iteration: int, tolerance: float = 0.01, max_tolerable_changes: int = 3) -> tuple:
    """
    Graph the effect of a hyperparameter on model accuracy.

    Args:
        training_samplesets (tuple): Tuple of training features and target.
        testing_datasets (tuple): Tuple of testing features and target.
        parameter (str): The hyperparameter to vary.
        parameter_start (float): The starting value of the hyperparameter.
        parameter_end (float): The ending value of the hyperparameter.
        parameter_step (float): The step size for the hyperparameter.
        min_iteration (int): Minimum number of iterations.
        tolerance (float): Tolerance for detecting no improvement.
        max_tolerable_changes (int): Maximum number of tolerable changes before stopping.

    Returns:
        tuple: The parameter values and corresponding accuracy values.
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

def save_fig(params: list, accs: list, version: str, title: str) -> pd.DataFrame:
    """
    Save a figure displaying the effect of a parameter on model accuracy.

    Args:
        params (list): List of parameter values.
        accs (list): List of accuracy values.
        version (str): The version of the figure.
        title (str): The title of the figure.

    Returns:
        pd.DataFrame: DataFrame containing the parameter values and accuracy values.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(params, accs, marker='o', linestyle='-', color='b')
    plt.title(f'Effect of {title} on Model Accuracy')
    plt.xlabel(title)
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.savefig(f'../results/random_forest/feature_tuning/{version}/{title}.png')
    plt.show()

    return pd.DataFrame(data={'accs': accs, 'params': params})

# Tuning hyperparameters for the District model
district_training_samplesets = (district_feature_training_selected_data, district_target_training_sample)
district_testing_datasets = (district_feature_testing_selected_data, district_target_testing_data)

district_max_sample_params, district_max_sample_accs = graph_parameter(district_training_samplesets, district_testing_datasets, 'max_samples', 0.05, 1.05, 0.05, min_iteration=1)
district_min_samples_split_params, district_min_samples_split_accs = graph_parameter(district_training_samplesets, district_testing_datasets, 'min_samples_split', 2, -1, 2, min_iteration=20)
district_max_leaf_nodes_params, district_max_leaf_nodes_accs = graph_parameter(district_training_samplesets, district_testing_datasets, 'max_leaf_nodes', 1500, -1, 50, min_iteration=100, tolerance=0.05, max_tolerable_changes=25)
district_min_samples_leaf_params, district_min_samples_leaf_accs = graph_parameter(district_training_samplesets, district_testing_datasets, 'min_samples_leaf', 2, -1, 20, min_iteration=20, tolerance=0.01, max_tolerable_changes=5)
district_n_estimators_params, district_n_estimators_accs = graph_parameter(district_training_samplesets, district_testing_datasets, 'n_estimators', 25, -1, 25, min_iteration=20, tolerance=0.01, max_tolerable_changes=5)
district_max_features_params, district_max_features_accs = graph_parameter(district_training_samplesets, district_testing_datasets, 'max_features', 1, -1, 1, min_iteration=15, tolerance=0.01, max_tolerable_changes=5)
district_max_depth_params, district_max_depth_accs = graph_parameter(district_training_samplesets, district_testing_datasets, 'max_depth', 1, -1, 1, min_iteration=25, tolerance=0.01, max_tolerable_changes=3)

district_max_depth_df = save_fig(district_max_depth_params, district_max_depth_accs, title='max_depth', version='district')
district_max_sample_df = save_fig(district_max_sample_params, district_max_sample_accs, title='max_sample', version='district')
district_min_samples_split_df = save_fig(district_min_samples_split_params, district_min_samples_split_accs, title='min_samples_split', version='district')
district_max_features_df = save_fig(district_max_features_params, district_max_features_accs, title='max_features', version='district')
district_max_leaf_nodes_df = save_fig(district_max_leaf_nodes_params, district_max_leaf_nodes_accs, title='max_leaf_nodes', version='district')
district_n_estimators_df = save_fig(district_n_estimators_params, district_n_estimators_accs, title='n_estimators', version='district')
district_min_samples_leaf_df = save_fig(district_min_samples_leaf_params, district_min_samples_leaf_accs, title='min_samples_leaf', version='district')

# Run Grid Search on final parameters for the District model
district_param_grid = {
    'n_estimators': [25, 100, 250],
    'max_features': [6, 10],
    'max_depth': [20, 25, None],
    'min_samples_split': [10, 20],
    'min_samples_leaf': [1],
    'max_samples': [0.6, 0.85, None],
    'max_leaf_nodes': [None]
}

district_model = RandomForestRegressor()
grid_search_district = GridSearchCV(estimator=district_model, param_grid=district_param_grid, cv=5, n_jobs=10, verbose=3, scoring='r2')
grid_search_district.fit(district_feature_training_selected_data, district_target_training_sample)

best_params_district = grid_search_district.best_params_
best_score_district = grid_search_district.best_score_

print(f'Best Parameters for District Model: {best_params_district}')
print(f'Best R2 Score for District Model: {best_score_district:.4f}')

# Train the final District model
final_district_model = grid_search_district.best_estimator_
final_district_model.set_params(n_jobs=15)
final_district_model.set_params(verbose=1)
final_district_model.fit(district_feature_training_selected_data, district_target_training_sample)

# Print final feature importances
for i, feature in enumerate(district_top_features):
    print(f'{feature}: {final_district_model.feature_importances_[i]}')

final_district_importances = final_district_model.feature_importances_
final_district_feature_names = district_top_features
final_district_feature_importance_df = pd.DataFrame({'feature': final_district_feature_names, 'importance': final_district_importances})
final_district_feature_importance_df.sort_values(by='importance', ascending=False, inplace=True)

# Plot final feature importances
plt.figure(figsize=(10, 6))
plt.barh(final_district_feature_importance_df['feature'], final_district_feature_importance_df['importance'], color='b')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importances from Random Forest Model')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('../results/random_forest/final_district_model_feature_importances.png')
plt.show()

evaluate(final_district_model, district_feature_testing_selected_data, district_target_testing_data, 'District', 'final_district_residuals')

# Combine test data and predictions for further analysis
district_df_combined = pd.concat([district_feature_testing_selected_data, pd.DataFrame(district_target_testing_data, columns=['target']), pd.DataFrame(final_district_model.predict(district_feature_testing_selected_data), columns=['predicted'])], axis=1)

district_no_zeros = district_df_combined[district_df_combined['target'] > 0]
district_no_zeros_model = Evaluator(final_district_model, district_no_zeros.drop(['target', 'predicted'], axis=1), district_no_zeros['predicted'])
district_no_zeros_model.set_residuals(district_no_zeros['predicted'] - district_no_zeros['target'])

evaluate(final_district_model, district_no_zeros.drop(['target', 'predicted'], axis=1), district_no_zeros['target'], 'District', 'final_district_no_zero_residuals')

# Evaluate feature importances using permutation importance
from sklearn.inspection import permutation_importance

district_perm_result = permutation_importance(final_district_model, district_feature_training_selected_data, district_target_training_sample, n_repeats=10, random_state=42)

# Print the permutation importance scores
for i in range(len(district_perm_result.importances_mean)):
    print(f"Feature {i}: {district_perm_result.importances_mean[i]}")

district_perm_importance_df = pd.DataFrame({
    'Feature': district_feature_training_selected_data.columns,
    'Importance': district_perm_result.importances_mean
})

district_perm_importance_df.sort_values(by='Importance', ascending=False, inplace=True)

# Plot permutation importance
plt.figure(figsize=(8, 4))
plt.barh(district_perm_importance_df['Feature'], district_perm_importance_df['Importance'], color='skyblue')
plt.xlabel('Permutation Importance')
plt.ylabel('Feature')
plt.title('Permutation Importance of District Features')
plt.tight_layout()
plt.gca().invert_yaxis()
plt.savefig('../results/random_forest/district_permutation_importance_results.png')
plt.show()

district_top_15_features = district_perm_importance_df.head(15)['Feature'].tolist()
print("Top 15 features:", district_top_15_features)

# Generate partial dependence plots for top 15 features
fig, ax = plt.subplots(figsize=(20, 15))
PartialDependenceDisplay.from_estimator(final_district_model, district_feature_training_selected_data, features=district_top_15_features, ax=ax, n_jobs=12, grid_resolution=50, percentiles=(0, 1))

fig.suptitle('District Partial Dependency Plots', fontsize=20)
plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig('../results/random_forest/district_partial_dependence.png')
plt.show()

def feature_ablation(model: RandomForestRegressor, training_sample: tuple, test_data: tuple) -> list:
    """
    Perform feature ablation by removing one feature at a time and evaluating the model.

    Args:
        model (RandomForestRegressor): The trained RandomForestRegressor model.
        training_sample (tuple): Tuple of training features and target.
        test_data (tuple): Tuple of testing features and target.

    Returns:
        list: List of ablation results including feature name and evaluation metrics.
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
        scores = {'MAE': evaluator.mae(), 'R^2': evaluator.r_squared(), 'RMSE': evaluator.rmse(), 'Relative RMSE': evaluator.relative_rmse()}
        accs.append((feature, scores))
        print(f'{feature} ablation has resulting scores of: {scores}')
    
    return accs

def calculate_differences(original_accs: dict, ablation_accs: list) -> dict:
    """
    Calculate differences between original model accuracy and ablation results.

    Args:
        original_accs (dict): Dictionary containing original accuracy metrics.
        ablation_accs (list): List of ablation results including feature name and evaluation metrics.

    Returns:
        dict: Dictionary containing feature names and differences in evaluation metrics.
    """
    out = {}
    scores = ['MAE', 'R^2', 'RMSE', 'Relative RMSE']
    for (feature, accs) in ablation_accs:
        out[feature] = [(score, accs[score] - original_accs[score]) for score in scores]
    return out

# Perform feature ablation on the District model
district_temp_model = RandomForestRegressor()
district_temp_model.set_params(**final_district_model.get_params())
district_feature_ablation_accs = feature_ablation(district_temp_model, (district_feature_training_selected_data, district_target_training_sample), (district_feature_testing_selected_data, district_target_testing_data))

district_evaluator = Evaluator(final_district_model, district_feature_testing_selected_data, district_target_testing_data)
district_evaluator.gather_predictions()
district_scores = {'MAE': district_evaluator.mae(), 'R^2': district_evaluator.r_squared(), 'RMSE': district_evaluator.rmse(), 'Relative RMSE': district_evaluator.relative_rmse()}

district_acc_differences = calculate_differences(district_scores, district_feature_ablation_accs)

# Create DataFrame to display ablation results
rows = []
for feature, metrics in district_acc_differences.items():
    row = {metric[0]: metric[1] for metric in metrics}
    row['Feature'] = feature
    rows.append(row)

df = pd.DataFrame(rows)
df.set_index('Feature', inplace=True)

# Prepare a figure with a grid of subplots
fig = plt.figure(constrained_layout=True, figsize=(8, 4))
spec = gridspec.GridSpec(ncols=len(df.columns), nrows=1, figure=fig)

# Plot each column in a separate subplot
for i, column in enumerate(df.columns):
    ax = fig.add_subplot(spec[0, i])
    sns.heatmap(df[[column]], annot=True, cmap="coolwarm", center=0, ax=ax, cbar=False)
    ax.set_title(column)
    ax.set_ylabel('')
    ax.set_xticklabels([])
    if i != 0:
        ax.set_yticklabels([])

plt.suptitle('District Feature Ablation Differences', fontsize=15)
plt.savefig('../results/random_forest/district_feature_ablation.png')
plt.show()