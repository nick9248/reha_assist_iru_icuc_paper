import os
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from sklearn.impute import KNNImputer
from sklearn.neural_network import MLPRegressor

# Load environment variables
load_dotenv()
BASE_DATASET = os.getenv("BASE_DATASET")
OUTPUT_FOLDER = os.getenv("OUTPUT_FOLDER")
LOG_FOLDER = os.getenv("LOG_FOLDER")
PLOTS = os.getenv("PLOTS")
PLOT_FOLDER = os.path.join(PLOTS,"step1")
IMPUTED_FOLDER = os.path.join(OUTPUT_FOLDER, "imputed_datasets")
STEP1_LOG_FOLDER = os.path.join(LOG_FOLDER, "step1")

# Ensure output directories exist
os.makedirs(IMPUTED_FOLDER, exist_ok=True)
os.makedirs(PLOT_FOLDER, exist_ok=True)
os.makedirs(STEP1_LOG_FOLDER, exist_ok=True)

# Configure logging
log_file = os.path.join(STEP1_LOG_FOLDER, "ingestion_preprocessing_log.txt")
logging.basicConfig(filename=log_file, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def log_and_print(message):
    print(message)
    logging.info(message)


# Load and preprocess data
def load_and_filter_data(file_path):
    log_and_print("Loading dataset...")
    data = pd.read_excel(file_path)
    log_and_print(f"Initial data shape: {data.shape}")

    # Filter dataset
    filtered_data = data[~data['Telefonat'].isin([2, 4])]
    log_and_print(f"Filtered data shape: {filtered_data.shape}")

    # Check missing values
    missing_values = filtered_data.isnull().sum()
    log_and_print(f"Missing values before imputation:\n{missing_values}")

    return filtered_data


# Imputation Methods
def mean_imputation(data, column_name):
    data[column_name] = data[column_name].fillna(data[column_name].mean())
    return data


def median_imputation(data, column_name):
    data[column_name] = data[column_name].fillna(data[column_name].median())
    return data


def mode_imputation(data, column_name):
    data[column_name] = data[column_name].fillna(data[column_name].mode()[0])
    return data


def knn_imputation(data, column_name, n_neighbors=5):
    imputer = KNNImputer(n_neighbors=n_neighbors)
    data[[column_name]] = imputer.fit_transform(data[[column_name]])
    return data


def hot_deck_imputation(data, column_name):
    data[column_name] = data[column_name].ffill().bfill()
    return data


def autoencoder_imputation(data, column_name):
    known = data[data[column_name].notnull()]
    unknown = data[data[column_name].isnull()]

    X_train = known.drop(columns=[column_name])
    y_train = known[column_name]
    X_test = unknown.drop(columns=[column_name])

    autoencoder = MLPRegressor(hidden_layer_sizes=(10, 5, 10), max_iter=1000, random_state=42)
    autoencoder.fit(X_train, y_train)
    predictions = autoencoder.predict(X_test)
    data.loc[data[column_name].isnull(), column_name] = predictions
    return data


def knn_imputation_binary(data, column_name, n_neighbors=5):
    data = knn_imputation(data, column_name, n_neighbors)
    data[column_name] = (data[column_name] > 0.5).astype(int)
    return data


def autoencoder_imputation_binary(data, column_name):
    data = autoencoder_imputation(data, column_name)
    data[column_name] = (data[column_name] > 0.5).astype(int)
    return data


# Apply imputation and save results
def apply_imputation_and_save(data, method_name, impute_func, **kwargs):
    log_and_print(f"Applying {method_name} imputation...")
    imputed_data = impute_func(data.copy(), 'Verlauf_entspricht_NBE', **kwargs)
    output_file = os.path.join(IMPUTED_FOLDER, f"imputed_dataset_{method_name}.xlsx")
    imputed_data.to_excel(output_file, index=False)

    # Check for missing values after imputation
    missing_values_after = imputed_data['Verlauf_entspricht_NBE'].isnull().sum()
    log_and_print(f"Missing values after {method_name} imputation: {missing_values_after}")

    # Check if all values are binary (0 or 1)
    unique_values = imputed_data['Verlauf_entspricht_NBE'].unique()
    log_and_print(f"Unique values after {method_name} imputation: {unique_values}")

    return imputed_data


# Load and filter dataset
filtered_data = load_and_filter_data(BASE_DATASET)
filtered_data = filtered_data.drop(columns=['TelHVID', 'Schadennummer'])

# Run imputations
imputed_datasets = {
    'mean': apply_imputation_and_save(filtered_data, 'mean', mean_imputation),
    'median': apply_imputation_and_save(filtered_data, 'median', median_imputation),
    'mode': apply_imputation_and_save(filtered_data, 'mode', mode_imputation),
    'knn': apply_imputation_and_save(filtered_data, 'knn', knn_imputation_binary, n_neighbors=5),
    'hot_deck': apply_imputation_and_save(filtered_data, 'hot_deck', hot_deck_imputation),
    'autoencoder': apply_imputation_and_save(filtered_data, 'autoencoder', autoencoder_imputation_binary)
}

# Identify the best imputation method
valid_methods = []
for method, data in imputed_datasets.items():
    if set(data['Verlauf_entspricht_NBE'].unique()).issubset({0, 1}):
        valid_methods.append(method)

best_method = valid_methods[0] if valid_methods else "None"
log_and_print(f"Best imputation method: {best_method}")


# Evaluate imputations
def evaluate_imputations(datasets, column_name):
    log_and_print("Evaluating imputation methods...")
    distributions = {method: data[column_name].value_counts(normalize=True) for method, data in datasets.items()}

    # Create and save evaluation plot
    distributions_df = pd.DataFrame(distributions).T
    plot_path = os.path.join(PLOT_FOLDER, "imputation_evaluation.png")
    distributions_df.plot(kind='bar', figsize=(12, 8))
    plt.title('Frequency Distribution of Imputed Values')
    plt.xlabel('Imputation Method')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.legend(title='Value', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(plot_path)
    log_and_print(f"Saved imputation evaluation plot to {plot_path}")


evaluate_imputations(imputed_datasets, 'Verlauf_entspricht_NBE')

log_and_print("All imputations and evaluations completed successfully.")