import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from xgboost import XGBRegressor


# Load data
area_feature_training_sample = pd.read_csv('../../data/pre_training/area_feature_training_sample.csv')
area_target_training_sample = pd.read_csv('../../data/pre_training/area_target_training_sample.csv')
district_feature_training_sample = pd.read_csv('../../data/pre_training/district_feature_training_sample.csv')
district_target_training_sample = pd.read_csv('../../data/pre_training/district_target_training_sample.csv')

area_feature_testing_data = pd.read_csv('../../data/pre_training/area_feature_testing_data.csv')
area_target_testing_data = pd.read_csv('../../data/pre_training/area_target_testing_data.csv')
district_feature_testing_data = pd.read_csv('../../data/pre_training/district_feature_testing_data.csv')
district_target_testing_data = pd.read_csv('../../data/pre_training/district_target_testing_data.csv')

# Drop 'crime_status' column
area_feature_training_sample.drop('crime_status', axis=1, inplace=True)
district_feature_training_sample.drop('crime_status', axis=1, inplace=True)

# Create DMatrix for XGBoost
area_dtrain_reg = xgb.DMatrix(area_feature_training_sample, area_target_training_sample, enable_categorical=True)
area_dtest_reg = xgb.DMatrix(area_feature_testing_data, area_target_testing_data, enable_categorical=True)
district_dtrain_reg = xgb.DMatrix(district_feature_training_sample, district_target_training_sample, enable_categorical=True)
district_dtest_reg = xgb.DMatrix(district_feature_testing_data, district_target_testing_data, enable_categorical=True)


class Evaluator:
    """
    Evaluates the performance of an XGBoost model using various metrics and plots.
    """

    def __init__(self, model: xgb.Booster, testing: xgb.DMatrix):
        self.model = model
        self.test_features = testing
        self.sample_size = len(testing.get_label())
        self.independent_vars = len(testing.feature_names)
        self.predictions = None
        self.residuals = None

    def gather_predictions(self) -> None:
        """Generates model predictions and computes residuals."""
        self.predictions = self.model.predict(self.test_features)
        self.residuals = self.test_features.get_label() - self.predictions

    def mae(self) -> float:
        """Calculates Mean Absolute Error (MAE)."""
        if self.residuals is None:
            raise ValueError("Predictions need to be gathered first")
        return np.mean(np.abs(self.residuals))

    def mse(self) -> float:
        """Calculates Mean Squared Error (MSE)."""
        if self.residuals is None:
            raise ValueError("Predictions need to be gathered first")
        return np.mean(self.residuals ** 2)

    def rmse(self) -> float:
        """Calculates Root Mean Squared Error (RMSE)."""
        return np.sqrt(self.mse())

    def relative_rmse(self) -> float:
        """Calculates the Relative RMSE."""
        return self.rmse() / max(self.test_features.get_label())

    def r_squared(self) -> float:
        """Calculates the R-squared value."""
        ss_res = np.sum(self.residuals ** 2)
        ss_tot = np.sum((self.test_features.get_label() - np.mean(self.test_features.get_label())) ** 2)
        return 1 - (ss_res / ss_tot)

    def adjusted_r_squared(self) -> float:
        """Calculates Adjusted R-squared value."""
        r2 = self.r_squared()
        return 1 - (1 - r2) * ((self.sample_size - 1) / (self.sample_size - self.independent_vars - 1))

    def median_absolute_error(self) -> float:
        """Calculates the Median Absolute Error."""
        return np.median(np.abs(self.residuals))

    def feature_importances(self) -> tuple:
        """
        Retrieves and prints feature importance based on weight, gain, and cover.
        """
        importance_weight = self.model.get_score(importance_type='weight')
        importance_gain = self.model.get_score(importance_type='gain')
        importance_cover = self.model.get_score(importance_type='cover')

        # Sort and display importance
        sorted_weight = sorted(importance_weight.items(), key=lambda x: x[1], reverse=True)
        sorted_gain = sorted(importance_gain.items(), key=lambda x: x[1], reverse=True)
        sorted_cover = sorted(importance_cover.items(), key=lambda x: x[1], reverse=True)

        print("Feature importance (sorted by weight):", sorted_weight)
        print("Feature importance (sorted by gain):", sorted_gain)
        print("Feature importance (sorted by cover):", sorted_cover)

        return importance_weight, importance_gain, importance_cover

    def plot_residuals(self, title: str, save_name: str) -> None:
        """
        Plots residuals as percentage difference between test labels and predicted values.
        """
        percent_diff = (self.residuals / max(self.test_features.get_label())) * 100

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
        plt.savefig(f'../results/xgboost/{save_name}.png')
        plt.show()


def evaluate(model: xgb.Booster, test_labels: xgb.DMatrix, title: str = None, save_name: str = None) -> tuple:
    """
    Evaluates a model's performance using various metrics and prints the results.

    Args:
    - model: The trained XGBoost model.
    - test_labels: DMatrix containing test data.
    - title: Optional plot title.
    - save_name: Optional file name for saving residual plot.

    Returns:
    - Tuple of feature importances and R-squared score.
    """
    ev = Evaluator(model, test_labels)
    ev.gather_predictions()

    print(f"Mean Absolute Error (MAE): {ev.mae():.4f}")
    print(f"Mean Squared Error (MSE): {ev.mse():.4f}")
    print(f"Root Mean Squared Error (RMSE): {ev.rmse():.4f}")
    print(f"Relative RMSE: {ev.relative_rmse():.4f}")
    print(f"R-squared (R²): {ev.r_squared():.4f}")
    print(f"Adjusted R-squared: {ev.adjusted_r_squared():.4f}")
    print(f"Median Absolute Error: {ev.median_absolute_error():.4f}")

    importance_weight, importance_gain, importance_cover = ev.feature_importances()

    if title is not None:
        ev.plot_residuals(title, save_name)

    return importance_weight, importance_gain, importance_cover, ev.r_squared()


# Training Area XGBoost Model
area_params = {"objective": "reg:squarederror", "device": "gpu", "eta": "0.001"}
area_evals = [(area_dtest_reg, "validation"), (area_dtrain_reg, "train")]
area_model = xgb.train(
    params=area_params,
    dtrain=area_dtrain_reg,
    num_boost_round=10000000,
    evals=area_evals,
    verbose_eval=1000,
    early_stopping_rounds=100
)

# Evaluate Area Model
area_importance_weight, area_importance_gain, area_importance_cover, _ = evaluate(area_model, area_dtest_reg, 'Area', 'area_residuals')

# Training District XGBoost Model
district_params = {"objective": "reg:squarederror", "device": "gpu", "eta": "0.001"}
district_evals = [(district_dtest_reg, "validation"), (district_dtrain_reg, "train")]
district_model = xgb.train(
    params=district_params,
    dtrain=district_dtrain_reg,
    num_boost_round=10000000,
    evals=district_evals,
    verbose_eval=1000,
    early_stopping_rounds=100
)

# Evaluate District Model
district_importance_weight, district_importance_gain, district_importance_cover, _ = evaluate(district_model, district_dtest_reg, 'District', 'district_residuals')


# Feature Selection Functions
def normalize_importances(importances: dict) -> dict:
    """Normalizes feature importances to ensure they sum to 1."""
    total = sum(importances.values())
    return {key: val / total for key, val in importances.items()}


def extract_features(importances: dict, type: str, threshold: float) -> list:
    """Extracts features based on importance type and threshold."""
    if type == 'cum_sum':
        sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        cumulative_sum = 0
        selected_features = []
        for feature, importance in sorted_features:
            cumulative_sum += importance
            selected_features.append(feature)
            if cumulative_sum >= threshold:
                break
        return selected_features
    else:
        return [key for key, val in importances.items() if val >= threshold]


def feature_selection(
    model: xgb.Booster, 
    type: str, 
    training_sample: tuple, 
    testing_data: tuple, 
    thresholds: np.ndarray
) -> tuple:
    """
    Perform feature selection based on importance scores (weight, gain, cover).
    
    Args:
    - model: Trained XGBoost model.
    - type: Method for feature selection ('cum_sum' or 'threshold').
    - training_sample: Tuple of (training features, training labels).
    - testing_data: Tuple of (testing features, testing labels).
    - thresholds: Array of thresholds to iterate over for feature selection.
    
    Returns:
    - Best features and their corresponding score.
    """
    weight_importances = normalize_importances(model.get_score(importance_type='weight'))
    gain_importances = normalize_importances(model.get_score(importance_type='gain'))
    cover_importances = normalize_importances(model.get_score(importance_type='cover'))

    agg_importances = {
        feature: np.mean([weight_importances[feature], gain_importances[feature], cover_importances[feature]])
        for feature in weight_importances.keys()
    }

    best_features = []
    best_score = 0

    for threshold in thresholds:
        target_features = extract_features(agg_importances, type, threshold)
        temp_dtrain = xgb.DMatrix(training_sample[0][target_features], training_sample[1], enable_categorical=True)
        temp_dtest = xgb.DMatrix(testing_data[0][target_features], testing_data[1], enable_categorical=True)

        temp_model = xgb.train(
            params={"objective": "reg:squarederror", "device": "gpu", "eta": "0.001"},
            dtrain=temp_dtrain,
            num_boost_round=10000000,
            evals=[(temp_dtest, "validation"), (temp_dtrain, "train")],
            verbose_eval=False,
            early_stopping_rounds=100
        )

        print(f'Threshold of {threshold} resulted in features: {target_features}')
        _, _, _, r_squared = evaluate(temp_model, temp_dtest)

        if r_squared > best_score:
            best_score = r_squared
            best_features = target_features

    return best_features, best_score


# Feature selection for District and Area models
district_top_features, district_top_score = feature_selection(
    district_model, 'cum_sum', (district_feature_training_sample, district_target_training_sample), 
    (district_feature_testing_data, district_target_testing_data), np.arange(0.3, 1.05, 0.05)
)

area_top_features, area_top_score = feature_selection(
    area_model, 'cum_sum', (area_feature_training_sample, area_target_training_sample), 
    (area_feature_testing_data, area_target_testing_data), np.arange(0.3, 1.05, 0.05)
)

# Update the feature set based on selected features
area_feature_training_sample_selected = area_feature_training_sample[area_top_features]
area_feature_testing_data_selected = area_feature_testing_data[area_top_features]
district_feature_training_sample_selected = district_feature_training_sample[district_top_features]
district_feature_testing_data_selected = district_feature_testing_data[district_top_features]

# Update the DMatrix with the selected features
area_dtrain_reg = xgb.DMatrix(area_feature_training_sample_selected, area_target_training_sample, enable_categorical=True)
area_dtest_reg = xgb.DMatrix(area_feature_testing_data_selected, area_target_testing_data, enable_categorical=True)
district_dtrain_reg = xgb.DMatrix(district_feature_training_sample_selected, district_target_training_sample, enable_categorical=True)
district_dtest_reg = xgb.DMatrix(district_feature_testing_data_selected, district_target_testing_data, enable_categorical=True)


# Hyperparameter Tuning
param_grid_district = {
    'n_estimators': [10000],
    'max_depth': np.arange(3, 20, 2),
    'min_child_weight': np.arange(1, 20, 2),
    'gamma': np.linspace(0, 1.0, 10),
    'subsample': np.linspace(0.3, 1.0, 10),
    'colsample_bytree': np.linspace(0.3, 1.0, 10),
    'eta': [0.01],
    'objective': ['reg:squarederror'],
    'device': ['gpu']
}

# Randomized Search for District model
model_district = XGBRegressor()
random_search_district = RandomizedSearchCV(
    estimator=model_district,
    param_distributions=param_grid_district,
    n_iter=200,
    scoring='neg_mean_squared_error',
    cv=3,
    verbose=3
)
random_search_district.fit(district_feature_training_sample, district_target_training_sample)

print("Best Parameters for District Model:", random_search_district.best_params_)
print("Best Score for District Model:", random_search_district.best_score_)

# Final Hyperparameter Grid Search for District Model
param_grid_district = {
    'n_estimators': [10000],
    'max_depth': [19, 21, 23, 25, None],
    'min_child_weight': [1.0],
    'gamma': [0.0],
    'subsample': [0.8, 0.85, 0.9],
    'colsample_bytree': [0.65, 0.7, 0.75],
    'eta': [0.01],
    'objective': ['reg:squarederror'],
    'device': ['gpu']
}

# Grid Search for District Model
grid_search_district = GridSearchCV(
    estimator=model_district,
    param_grid=param_grid_district,
    scoring='neg_mean_squared_error',
    cv=5,
    verbose=3
)
grid_search_district.fit(district_feature_training_sample, district_target_training_sample)

print("Best Parameters for Area Model:", grid_search_district.best_params_)
print("Best Score for Area Model:", grid_search_district.best_score_)


# Train final model with best parameters
final_district_params = grid_search_district.best_params_

learning_rates = [0.01, 0.015, 0.02, 0.05]
final_district_models = {}

for eta in learning_rates:
    print(f"Training model with eta: {eta}")

    params = final_district_params.copy()
    params['eta'] = eta 

    final_district_evals = [(district_dtrain_reg, 'train'), (district_dtest_reg, 'eval')]

    final_district_model = xgb.train(
        params=params,
        dtrain=district_dtrain_reg,
        num_boost_round=10000000,
        evals=final_district_evals,
        verbose_eval=1000,
        early_stopping_rounds=100
    )

    final_district_models[eta] = final_district_model

# Evaluate final district model with eta = 0.05
final_district_params['eta'] = 0.05
final_district_model = xgb.train(
    params=final_district_params,
    dtrain=district_dtrain_reg,
    num_boost_round=51000,
    evals=final_district_evals,
    verbose_eval=1000
)

evaluate(final_district_model, district_dtest_reg, 'District', 'final_district_residuals')