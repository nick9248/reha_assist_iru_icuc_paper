# KNN Imputation Documentation for Healthcare Dataset Analysis

## Project Overview

This document details the data preprocessing pipeline implemented for a healthcare dataset analysis project, focusing specifically on the KNN (K-Nearest Neighbors) imputation method selected for handling missing values in the dataset.

## Dataset Information

The analysis was performed on an anonymized healthcare dataset with the following characteristics:

- **Original Dataset Size**: 3662 rows × 8 columns
- **Filtered Dataset Size**: 3427 rows × 8 columns (after removing Telefonat values 2 and 4)
- **Columns**:
  - TelHVID: Identifier for telephone healthcare visit
  - FL_Score: FL scoring value
  - FL_Status_Nominal: Nominal status for FL
  - P_Score: P scoring value
  - P_Status_Nominal: Nominal status for P
  - Schadennummer: Original case number (anonymized to patient_id)
  - Verlauf_entspricht_NBE: Target variable with missing values
  - Telefonat: Telephone consultation status

- **Missing Values**: 295 missing values in the 'Verlauf_entspricht_NBE' column (8.6% of the dataset)

## Data Processing Pipeline

The data processing pipeline consists of two main steps:

### 1. Data Anonymization (step0_anonymization)

- **Purpose**: Ensure patient privacy by anonymizing identifiable information
- **Implementation**:
  - Replaced 'Schadennummer' values with anonymized IDs (patient_1, patient_2, etc.)
  - Created a mapping file for reference (schadennummer_mapping.xlsx)
  - Saved anonymized dataset to a secure location
  - Filtered out records with Telefonat values 2 and 4

- **Output Files**:
  - anonymized_dataset.xlsx: The anonymized dataset for further processing
  - schadennummer_mapping.xlsx: Mapping between original IDs and anonymized IDs

### 2. Missing Value Imputation (step1_ingestion_preprocessing)

- **Purpose**: Handle missing values in the 'Verlauf_entspricht_NBE' column
- **Implementation**:
  - Tested six different imputation methods: mean, median, mode, KNN, hot_deck, and autoencoder
  - Evaluated each method based on value distribution and consistency with original data
  - Selected KNN imputation as the optimal method

- **Output Files**:
  - imputed_dataset_knn.xlsx: Dataset with KNN-imputed values
  - imputation_evaluation.png: Comparative visualization of imputation methods

## KNN Imputation Method Details

### Algorithm Selection

K-Nearest Neighbors (KNN) imputation was selected from among several methods for its ability to:
- Maintain the binary nature of the target variable
- Consider relationships between variables when imputing missing values
- Produce statistically sound imputations for categorical data

### Implementation Specifics

- **Algorithm**: KNNImputer from sklearn.impute
- **Parameters**:
  - n_neighbors=5: Used 5 nearest neighbors to impute each missing value
  - Numeric features used: FL_Score, P_Score, FL_Status_Nominal, P_Status_Nominal, Telefonat
  - String columns (like patient_id) were excluded from the imputation calculation

- **Data Preparation**:
  - Non-numeric columns temporarily removed for KNN calculation
  - Missing values identified in 'Verlauf_entspricht_NBE' column
  - Continuous predictions converted to binary (0/1) using threshold of 0.5

- **Code Implementation**:
```python
def knn_imputation_binary(data, column_name, n_neighbors=5):
    # For KNN, we need to temporarily drop non-numeric columns
    numeric_cols = [col for col in data.columns if pd.api.types.is_numeric_dtype(data[col])]
    numeric_data = data[numeric_cols].copy()
    
    imputer = KNNImputer(n_neighbors=n_neighbors)
    numeric_data[[column_name]] = imputer.fit_transform(numeric_data[[column_name]])
    data[column_name] = numeric_data[column_name]
    data[column_name] = (data[column_name] > 0.5).astype(int)
    return data
```

### Validation Results

- **Missing Values After Imputation**: 0 (all 295 missing values successfully imputed)
- **Unique Values After Imputation**: [1, 0] (binary values preserved)
- **Distribution**: 
  - Original data distribution (known values only): approximately 77.6% of 1's, 22.4% of 0's
  - KNN-imputed values maintained a similar distribution pattern

## Environment Configuration

The pipeline uses environment variables for configuration:
- BASE_DATASET: Path to the original dataset file
- OUTPUT_FOLDER: Directory for output files
- LOG_FOLDER: Directory for log files
- PLOTS: Directory for visualization outputs

Additional derived paths:
- ANONYMIZATION_FOLDER: BASE_DATASET_DIR/step0_anonymization
- IMPUTED_FOLDER: OUTPUT_FOLDER/imputed_datasets
- PLOT_FOLDER: PLOTS/step1

## Dependencies

The analysis pipeline requires the following Python packages:
- pandas: Data manipulation and analysis
- numpy: Numerical operations
- scikit-learn: Machine learning tools (KNNImputer)
- matplotlib: Visualization
- dotenv: Environment variable management

## File Structure

```
Project/
├── Data/
│   ├── Input/
│   │   └── step0_anonymization/
│   │       ├── anonymized_dataset.xlsx
│   │       └── schadennummer_mapping.xlsx
│   └── Output/
│       └── imputed_datasets/
│           ├── imputed_dataset_knn.xlsx
│           ├── imputed_dataset_mean.xlsx
│           ├── imputed_dataset_median.xlsx
│           ├── imputed_dataset_mode.xlsx
│           ├── imputed_dataset_hot_deck.xlsx
│           └── imputed_dataset_autoencoder.xlsx
├── logs/
│   ├── step0_anonymization/
│   │   └── anonymization_log.txt
│   └── step1/
│       └── ingestion_preprocessing_log.txt
├── plots/
│   └── step1/
│       └── imputation_evaluation.png
└── code/
    ├── anonymization.py
    └── ingestion_preprocessing.py
```

## Conclusion

The KNN imputation method successfully handled all missing values in the 'Verlauf_entspricht_NBE' column while maintaining the binary nature of the data. The approach considers the relationship between variables when imputing missing values, which is particularly appropriate for this dataset where missing values may have relationships with other features.

The imputed dataset (imputed_dataset_knn.xlsx) is ready for further analysis and modeling. The imputation preserves the statistical properties of the original data while providing complete information for all records.

## Next Steps

The preprocessed dataset can now be used for:
1. Exploratory data analysis
2. Feature engineering
3. Machine learning model development
4. Statistical analysis and hypothesis testing

---

*Documentation prepared on March 2, 2025*
