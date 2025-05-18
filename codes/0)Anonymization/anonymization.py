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
PLOT_FOLDER = os.path.join(PLOTS, "step1")
IMPUTED_FOLDER = os.path.join(OUTPUT_FOLDER, "imputed_datasets")
STEP1_LOG_FOLDER = os.path.join(LOG_FOLDER, "step1")

# Define anonymization folder inside the BASE_DATASET directory
BASE_DATASET_DIR = os.path.dirname(BASE_DATASET)
ANONYMIZATION_FOLDER = os.path.join(BASE_DATASET_DIR, "step0_anonymization")
ANONYMIZATION_LOG_FOLDER = os.path.join(LOG_FOLDER, "step0_anonymization")
MAPPING_FILE = os.path.join(ANONYMIZATION_FOLDER, "schadennummer_mapping.xlsx")
ANONYMIZED_DATASET_FILE = os.path.join(ANONYMIZATION_FOLDER, "anonymized_dataset.xlsx")

# Ensure output directories exist
os.makedirs(IMPUTED_FOLDER, exist_ok=True)
os.makedirs(PLOT_FOLDER, exist_ok=True)
os.makedirs(STEP1_LOG_FOLDER, exist_ok=True)
os.makedirs(ANONYMIZATION_FOLDER, exist_ok=True)
os.makedirs(ANONYMIZATION_LOG_FOLDER, exist_ok=True)

# Configure logging
log_file = os.path.join(ANONYMIZATION_LOG_FOLDER, "anonymization_log.txt")
logging.basicConfig(filename=log_file, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def log_and_print(message):
    print(message)
    logging.info(message)


# Load and anonymize data
def anonymize_schadennummer(data):
    unique_ids = data["Schadennummer"].unique()
    mapping = {id_: f"patient_{i + 1}" for i, id_ in enumerate(unique_ids)}

    # Save mapping file
    mapping_df = pd.DataFrame(list(mapping.items()), columns=["Schadennummer", "Anonymized_ID"])
    mapping_df.to_excel(MAPPING_FILE, index=False)
    log_and_print(f"Schadennummer mapping saved to {MAPPING_FILE}")

    # Replace Schadennummer with anonymized ID
    data.loc[:, "Schadennummer"] = data["Schadennummer"].map(mapping)
    return data


# Load and preprocess data
def load_and_filter_data(file_path):
    log_and_print("Loading dataset...")
    data = pd.read_excel(file_path)
    log_and_print(f"Initial data shape: {data.shape}")

    # Anonymize Schadennummer before filtering
    data = anonymize_schadennummer(data)

    # Filter dataset
    filtered_data = data[~data['Telefonat'].isin([2, 4])]
    log_and_print(f"Filtered data shape: {filtered_data.shape}")

    # Save anonymized dataset
    filtered_data.to_excel(ANONYMIZED_DATASET_FILE, index=False)
    log_and_print(f"Anonymized dataset saved to {ANONYMIZED_DATASET_FILE}")

    return filtered_data


# Load and filter dataset
filtered_data = load_and_filter_data(BASE_DATASET)

log_and_print("Data anonymization completed successfully.")
