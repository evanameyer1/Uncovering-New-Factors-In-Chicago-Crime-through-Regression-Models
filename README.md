# Identifying Key Infrastructure Factors in Chicago Crime through Regression Models

This project is a deep dive into the complex relationships between urban infrastructure, socio-economic factors, and crime in Chicago. By combining large-scale datasets with advanced data science techniques, the project aims to identify key factors contributing to crime in specific neighborhoods and offer actionable insights for urban planning and law enforcement. Here are some of the key aspects:

Multi-Source Data Integration: The project integrates diverse data sources, including public health indicators, transportation networks, police presence, and public services, to build comprehensive predictive models.

Advanced Machine Learning Models: Utilizes models such as Random Forest and XGBoost, along with sophisticated spatial analysis techniques, to uncover crime hotspots and identify infrastructural vulnerabilities.

Geospatial and Temporal Analysis: Incorporates spatial relationships and temporal trends to provide a more holistic understanding of crime dynamics in urban settings.

Real-Time Application: The ultimate goal is to create a live, dynamic model that the City of Chicago can use for real-time decision-making, resource allocation, and crime reduction strategies.

Replicable Framework: The methodology and approach can be adapted to other cities facing similar urban challenges, making it a significant contribution to the field of urban analytics.

## Repository Structure

```
Custom-LSTM-Model/
│
├── README.md               # Project overview and setup instructions
├── LICENSE                 # Repository license
│
├── docs/                   # Documentation and papers
│   ├── project_proposal.md
│   ├── academic_paper.md
│   └── medium_article.md
│
├── models/                 # Central folder for model-related scripts and notebooks
│   ├── notebooks/          # Jupyter notebooks for exploration and presentation
│   ├── src/                # Source code for models and utilities
│   ├── results/            # Generated analysis results, models, figures, etc.
│   └── tests/              # Test cases for your scripts
│
├── data/                   # Data used in the project (not tracked in Git)
│   ├── raw/                # Raw data, not modified
│   ├── processed/          # Cleaned and pre-processed data
|   ├── simulations/        # Results of model-led simulations
│   └── pre-training/       # Data post feature engineering and feature selection
│
├── requirements.txt        # Python package dependencies
│
└── .gitignore              # Specifies intentionally untracked files to ignore
```

## Data Description

This project utilizes a variety of datasets related to Chicago's infrastructure and public services to analyze their relationship with crime occurrences. Below is a brief description of each dataset used in the analysis:

1. **`raw_vacant_buildings.csv`**: Contains information on vacant buildings in Chicago, including locations, statuses, and dates of vacancy. Used to assess the relationship between vacant buildings and crime rates.

2. **`raw_train_stations.csv`**: Provides details on the locations and types of train stations in Chicago, analyzed to determine the proximity of crimes to train stations.

3. **`raw_train_ridership.csv`**: Records the number of passengers using the train system over time, integrated to explore how fluctuations in ridership correlate with crime patterns.

4. **`raw_streetlights_oneout.csv`**: Contains data on streetlight outages where one light is out, used to examine the impact of inadequate street lighting on crime.

5. **`raw_streetlights_allout.csv`**: Records instances where all streetlights in an area are out, used to assess the broader impact of complete streetlight outages on crime.

6. **`raw_publichealth_indicator.csv`**: Includes various public health indicators such as access to healthcare and socio-economic factors, analyzed to understand their influence on crime rates.

7. **`raw_police_stations.csv`**: Lists all police stations in Chicago with their locations and operational details, examined to study the influence of police presence on crime.

8. **`raw_police_districts.csv`**: Provides data on the boundaries of police districts in Chicago, important for spatial analysis to delineate crime patterns within districts.

9. **`raw_disadvantaged_areas.csv`**: Identifies socio-economically disadvantaged areas in Chicago, used to correlate crime rates with socio-economic disadvantage.

10. **`raw_crime.csv`**: Comprehensive dataset of reported crimes in Chicago, central to the analysis for identifying crime patterns and correlating them with various factors.

11. **`raw_bus_stops.csv`**: Contains locations and details of bus stops throughout Chicago, used to study the relationship between public transportation access and crime.

12. **`raw_bike_trips.csv`**: Records data on bike trips, including locations, times, and durations, analyzed to explore the correlation between bike activity and crime.

13. **`raw_bike_stations.csv`**: Lists bike rental stations in Chicago, examined to assess how proximity to bike stations relates to crime.

14. **`raw_areas.csv`**: Provides data on different areas within Chicago, including demographic information, used for spatial analysis of area-specific factors and crime rates.

15. **`raw_alleylights.csv`**: Contains data on the status of alleylights in Chicago, examined to assess the impact of alleylight conditions on crime in alleys.

## Key Files and Their Purposes

#### 1. `01_data_preprocessing.py`
**Topic**: Data Cleaning and Preprocessing

- Cleans and preprocesses multiple datasets including crime reports, service requests, and public health indicators.
- Handles missing values, formats dates, and standardizes geographical data.

#### 2. `02_geospatial_analysis.py`
**Topic**: Geospatial Data Preparation

- Prepares datasets for crime prediction by encoding and optimizing data for areas and districts.
- Splits data into training and testing sets, performs target and frequency encoding, and conducts stratified sampling.

#### 3. `03_proximity_analysis.py`
**Topic**: Proximity Analysis

- Analyzes spatial proximity of various features (e.g., police stations, bike rides) to crime locations.
- Incorporates temporal dimensions to analyze the relationship between nearby activities and crime occurrences.

#### 4. `04_feature_engineering.py`
**Topic**: Feature Engineering

- Prepares and engineers features for crime prediction, including rolling crime counts and merging datasets like public health indicators.
- Normalizes and organizes data for machine learning workflows.

#### 5. `05_representative_samples.py`
**Topic**: Stratified Sampling

- Balances training and testing datasets for area and district crime prediction through stratified sampling.
- Enhances categorical columns using target and frequency encoding.

#### 6. `06_linear_regression.py`
**Topic**: Linear Regression Modeling

- Trains and evaluates linear regression models for crime prediction.
- Analyzes feature importance and optimizes feature sets using Variance Inflation Factor (VIF).

#### 7. `07_random_forest.py`
**Topic**: Random Forest Modeling

- Builds and tunes Random Forest models, explores feature importance, and performs hyperparameter tuning.
- Uses advanced techniques like feature ablation and permutation importance analysis.

#### 8. `08_xgboost_model.py`
**Topic**: XGBoost Modeling

- Implements and fine-tunes XGBoost models for crime prediction.
- Focuses on feature selection, model evaluation, and hyperparameter tuning using GridSearchCV.

#### 9. `09_factor_analysis.py`
**Topic**: Factor Analysis and Visualization

- Conducts factor analysis to uncover underlying factors in crime data.
- Visualizes these factors and their relationships with crime in different districts using Plotly.

#### 10. 10_simulation.py
**Topic**: Simulation Analysis

- Simulates different scenarios by adjusting factors like bike activity, alleylight and streetlight outages, and recent crime activity.
- Uses XGBoost models to predict crime levels based on the simulations.
- Calculates residuals and compares the impact of simulations across different geographical areas, visualizing the results using heatmaps.

## Requirements

Here is the complete requirements list (found in requirements.txt):

```plaintext
certifi==2024.8.30
charset-normalizer==3.3.2
contourpy==1.3.0
cycler==0.12.1
factor_analyzer==0.5.1
fonttools==4.53.1
geodatasets==2024.8.0
geopandas==1.0.1
idna==3.8
joblib==1.4.2
kiwisolver==1.4.7
matplotlib==3.9.2
mlxtend==0.23.1
numpy==2.1.1
packaging==24.1
pandas==2.2.2
patsy==0.5.6
pillow==10.4.0
platformdirs==4.3.2
pooch==1.8.2
pyogrio==0.9.0
pyparsing==3.1.4
pyproj==3.6.1
python-dateutil==2.9.0.post0
pytz==2024.1
requests==2.32.3
scikit-learn==1.5.1
scipy==1.14.1
seaborn==0.13.2
shapely==2.0.6
six==1.16.0
statsmodels==0.14.2
threadpoolctl==3.5.0
tzdata==2024.1
urllib3==2.2.2
xgboost==2.1.1
```

*Note: Ensure that you have these versions installed to avoid compatibility issues.*

## Getting Started

### Prerequisites

- Python 3.x
- Jupyter Notebook
- Required Python packages (see `requirements.txt`)

### Installation

Clone the repository and install the necessary packages:

```bash
git clone https://github.com/evanameyer1/Custom-LSTM-Model.git
cd Custom-LSTM-Model
pip install -r models/notebooks/requirements.txt
```

## Usage

### Important Notes

- **Jupyter Notebooks**: The `models/notebooks/` directory contains Jupyter notebooks where raw data exploration and Exploratory Data Analysis (EDA) were conducted. These notebooks are **not** organized or commented extensively. They serve as a record of the exploratory process and may contain preliminary analyses and visualizations.

- **Comprehensive Understanding**: For a detailed and well-documented understanding of each step in the project, refer to the Python scripts located in the `models/src/` directory. These `.py` files contain function definitions, descriptions, and comments that explain the logic and purpose behind each process.

- **Repository Intention**: This repository is **not** intended for others to clone and run all the files directly. Due to the large size of the datasets (approximately 30GB) and the extensive processing involved (handling hundreds of millions of rows), running the scripts can be time-consuming and resource-intensive. The primary purpose of this repository is to provide insight into the code and methodologies used for this research project, allowing others to understand and potentially build upon the work. Please be aware of the computational requirements and processing times involved.

## Next Steps

Moving forward, the project aims to collaborate with the City of Chicago to implement a live ensemble model that integrates multiple advanced modeling techniques:

1. **Ensemble Models**:
   - **Random Forest and XGBoost**: These models will form the base of the ensemble, leveraging their strengths in handling complex, non-linear relationships within the data.
   - **LSTM (Long Short-Term Memory) Networks**: To capture seasonality and temporal patterns in crime data, LSTM models will be incorporated.
   - **Geographically Weighted Regression (GWR)**: To account for spatial relationships and geographic heterogeneity, GWR will be used to model georelationships effectively.

2. **Live Hosted Model**:
   - Utilizing the City of Chicago's internal data API, a live hosted model will be developed. This model will continuously update with real-time data, ensuring that crime predictions are based on the most current information available.
   - The live

 ensemble model will provide dynamic and accurate crime forecasting, aiding in proactive law enforcement and resource allocation.

3. **Implementation and Deployment**:
   - Collaborate with city officials and data teams to integrate the model into existing systems.
   - Ensure scalability and robustness of the model to handle real-time data streams and large-scale computations.

## Documentation

Detailed documentation can be found in the `docs/` directory, including the project proposal, academic paper, and a Medium article summarizing the findings.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Acknowledgments

This project was inspired by the need to understand and predict crime patterns in urban areas. Special thanks to contributors and open-source communities for their support and resources.
