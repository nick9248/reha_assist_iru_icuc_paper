import os
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from scipy import stats
import missingno as msno

# Load environment variables
load_dotenv()
BASE_DATASET = os.getenv("BASE_DATASET")
OUTPUT_FOLDER = os.getenv("OUTPUT_FOLDER")
LOG_FOLDER = os.getenv("LOG_FOLDER")
PLOTS = os.getenv("PLOTS")

# Define paths based on anonymization paths
BASE_DATASET_DIR = os.path.dirname(BASE_DATASET)
ANONYMIZATION_FOLDER = os.path.join(BASE_DATASET_DIR, "step0_anonymization")
ANONYMIZED_DATASET_FILE = os.path.join(ANONYMIZATION_FOLDER, "anonymized_dataset.xlsx")

# Define imputed dataset path (using KNN imputation)
IMPUTED_FOLDER = os.path.join(OUTPUT_FOLDER, "imputed_datasets")
KNN_IMPUTED_DATASET = os.path.join(IMPUTED_FOLDER, "imputed_dataset_knn.xlsx")

# Define step2 output paths
STEP2_LOG_FOLDER = os.path.join(LOG_FOLDER, "step2")
STEP2_PLOT_FOLDER = os.path.join(PLOTS, "step2")
STEP2_OUTPUT_FOLDER = os.path.join(OUTPUT_FOLDER, "step2_quality_check")

# Ensure output directories exist
os.makedirs(STEP2_LOG_FOLDER, exist_ok=True)
os.makedirs(STEP2_PLOT_FOLDER, exist_ok=True)
os.makedirs(STEP2_OUTPUT_FOLDER, exist_ok=True)

# Configure logging
log_file = os.path.join(STEP2_LOG_FOLDER, "quality_check_log.txt")
logging.basicConfig(filename=log_file, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def log_and_print(message):
    print(message)
    logging.info(message)


# Column information dictionary
column_info = {
    'TelHVID': 'Telephone healthcare visit identifier',
    'FL_Score': 'Function limitation score (0-best to 4-worst)',
    'FL_Status_Nominal': 'Function limitation status compared to previous assessment (0-better, 1-no change, 2-worse)',
    'P_Score': 'Pain score (0-best to 4-worst)',
    'P_Status_Nominal': 'Pain status compared to previous assessment (0-better, 1-no change, 2-worse)',
    'Schadennummer': 'Anonymized patient ID',
    'Verlauf_entspricht_NBE': 'Within Nachbehandlungsempfehlungen period (1-good, 0-bad)',
    'Telefonat': 'Contact type (0-Erstkontakt, 1-Folgekontakt, 2-nicht erreicht, 3-Fallabschluss, 4-Komplikationsbesprechung)'
}


def load_data():
    """Load the KNN-imputed dataset"""
    log_and_print("Loading KNN-imputed dataset...")
    try:
        data = pd.read_excel(KNN_IMPUTED_DATASET)
        log_and_print(f"Dataset loaded successfully: {data.shape[0]} rows, {data.shape[1]} columns")
        return data
    except Exception as e:
        log_and_print(f"Error loading dataset: {str(e)}")
        return None


def generate_custom_profile_report(data):
    """Generate a custom profile report without pandas-profiling"""
    log_and_print("Generating custom profile report...")

    profile_data = []

    for col in data.columns:
        # Basic statistics
        count = data[col].count()
        missing = data[col].isnull().sum()
        unique_values = data[col].nunique()

        # Type-specific statistics
        if pd.api.types.is_numeric_dtype(data[col]):
            data_type = str(data[col].dtype)
            min_val = data[col].min()
            max_val = data[col].max()
            mean_val = data[col].mean()
            std_val = data[col].std()
            median_val = data[col].median()
            q1_val = data[col].quantile(0.25)
            q3_val = data[col].quantile(0.75)

            # Most common values
            value_counts = data[col].value_counts().head(5).to_dict()
            most_common = ', '.join([f"{val}: {count}" for val, count in value_counts.items()])

            # Histogram
            plt.figure(figsize=(10, 6))
            sns.histplot(data[col], kde=True)
            plt.title(f'Distribution of {col}')
            plt.savefig(os.path.join(STEP2_PLOT_FOLDER, f"profile_hist_{col}.png"))
            plt.close()

            profile_data.append({
                'Column': col,
                'Description': column_info.get(col, "No description available"),
                'Data Type': data_type,
                'Count': count,
                'Missing': missing,
                'Missing (%)': round(missing / len(data) * 100, 2),
                'Unique Values': unique_values,
                'Unique (%)': round(unique_values / count * 100, 2) if count > 0 else 0,
                'Min': min_val,
                'Max': max_val,
                'Mean': mean_val,
                'Std Dev': std_val,
                'Median': median_val,
                'Q1 (25%)': q1_val,
                'Q3 (75%)': q3_val,
                'Most Common Values': most_common
            })
        else:
            data_type = str(data[col].dtype)

            # Most common values
            value_counts = data[col].value_counts().head(5).to_dict()
            most_common = ', '.join([f"{val}: {count}" for val, count in value_counts.items()])

            # Bar chart for categorical
            plt.figure(figsize=(10, 6))
            data[col].value_counts().sort_index().plot(kind='bar')
            plt.title(f'Distribution of {col}')
            plt.savefig(os.path.join(STEP2_PLOT_FOLDER, f"profile_bar_{col}.png"))
            plt.close()

            profile_data.append({
                'Column': col,
                'Description': column_info.get(col, "No description available"),
                'Data Type': data_type,
                'Count': count,
                'Missing': missing,
                'Missing (%)': round(missing / len(data) * 100, 2),
                'Unique Values': unique_values,
                'Unique (%)': round(unique_values / count * 100, 2) if count > 0 else 0,
                'Most Common Values': most_common
            })

    # Create profile dataframe
    profile_df = pd.DataFrame(profile_data)

    # Save profile report
    profile_file = os.path.join(STEP2_OUTPUT_FOLDER, "custom_profile_report.xlsx")
    profile_df.to_excel(profile_file, index=False)
    log_and_print(f"Custom profile report saved to {profile_file}")

    # Create missing values visualization
    plt.figure(figsize=(12, 8))
    msno.matrix(data)
    plt.title('Missing Values Matrix')
    plt.savefig(os.path.join(STEP2_PLOT_FOLDER, "missing_values_matrix.png"))
    plt.close()

    # Generate HTML report
    html_content = ["<html><head><title>Dataset Profile Report</title>",
                    "<style>table {border-collapse: collapse; width: 100%;}",
                    "th, td {border: 1px solid #ddd; padding: 8px; text-align: left;}",
                    "th {background-color: #f2f2f2;}",
                    "tr:nth-child(even) {background-color: #f9f9f9;}",
                    "h1, h2, h3 {color: #333;}</style></head><body>",
                    "<h1>Dataset Profile Report</h1>",
                    f"<p>Dataset Shape: {data.shape[0]} rows, {data.shape[1]} columns</p>",
                    "<h2>Column Profiles</h2>"]

    # Add tables for each column
    for col in data.columns:
        html_content.append(f"<h3>{col}</h3>")
        html_content.append(f"<p><strong>Description:</strong> {column_info.get(col, 'No description available')}</p>")
        html_content.append("<table>")

        if pd.api.types.is_numeric_dtype(data[col]):
            html_content.append("<tr><th>Statistic</th><th>Value</th></tr>")
            html_content.append(f"<tr><td>Data Type</td><td>{data[col].dtype}</td></tr>")
            html_content.append(f"<tr><td>Count</td><td>{data[col].count()}</td></tr>")
            html_content.append(
                f"<tr><td>Missing</td><td>{data[col].isnull().sum()} ({data[col].isnull().sum() / len(data) * 100:.2f}%)</td></tr>")
            html_content.append(f"<tr><td>Unique Values</td><td>{data[col].nunique()}</td></tr>")
            html_content.append(f"<tr><td>Min</td><td>{data[col].min()}</td></tr>")
            html_content.append(f"<tr><td>Max</td><td>{data[col].max()}</td></tr>")
            html_content.append(f"<tr><td>Mean</td><td>{data[col].mean()}</td></tr>")
            html_content.append(f"<tr><td>Median</td><td>{data[col].median()}</td></tr>")
            html_content.append(f"<tr><td>Std Dev</td><td>{data[col].std()}</td></tr>")

            # Add histogram image
            html_content.append("</table>")
            html_content.append(
                f"<img src='../../../{STEP2_PLOT_FOLDER}/profile_hist_{col}.png' alt='Histogram of {col}' style='max-width:800px;'>")
        else:
            html_content.append("<tr><th>Statistic</th><th>Value</th></tr>")
            html_content.append(f"<tr><td>Data Type</td><td>{data[col].dtype}</td></tr>")
            html_content.append(f"<tr><td>Count</td><td>{data[col].count()}</td></tr>")
            html_content.append(
                f"<tr><td>Missing</td><td>{data[col].isnull().sum()} ({data[col].isnull().sum() / len(data) * 100:.2f}%)</td></tr>")
            html_content.append(f"<tr><td>Unique Values</td><td>{data[col].nunique()}</td></tr>")

            # Add top 5 most common values
            html_content.append("<tr><td>Most Common Values</td><td>")
            for val, count in data[col].value_counts().head(5).items():
                html_content.append(f"{val}: {count} ({count / len(data) * 100:.2f}%)<br>")
            html_content.append("</td></tr>")

            # Add bar chart image
            html_content.append("</table>")
            html_content.append(
                f"<img src='../../../{STEP2_PLOT_FOLDER}/profile_bar_{col}.png' alt='Bar chart of {col}' style='max-width:800px;'>")

    # Add correlation matrix for numeric columns
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_cols) > 1:
        html_content.append("<h2>Correlation Matrix</h2>")

        # Generate correlation matrix
        plt.figure(figsize=(12, 10))
        corr_matrix = data[numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(STEP2_PLOT_FOLDER, "profile_correlation_matrix.png"))
        plt.close()

        html_content.append(
            f"<img src='../../../{STEP2_PLOT_FOLDER}/profile_correlation_matrix.png' alt='Correlation Matrix' style='max-width:800px;'>")

    html_content.append("</body></html>")

    # Write HTML report
    html_file = os.path.join(STEP2_OUTPUT_FOLDER, "custom_profile_report.html")
    with open(html_file, 'w') as f:
        f.write("\n".join(html_content))

    log_and_print(f"HTML profile report saved to {html_file}")


def analyze_data_types(data):
    """Analyze and visualize data types"""
    log_and_print("Analyzing data types...")

    # Get data types
    dtypes = data.dtypes
    dtype_summary = pd.DataFrame({
        'Column': dtypes.index,
        'Data Type': dtypes.values,
        'Description': [column_info.get(col, "No description available") for col in dtypes.index]
    })

    # Save data type summary
    dtype_file = os.path.join(STEP2_OUTPUT_FOLDER, "data_types.xlsx")
    dtype_summary.to_excel(dtype_file, index=False)
    log_and_print(f"Data type summary saved to {dtype_file}")

    # Log data types
    log_and_print("Data type summary:")
    for i, row in dtype_summary.iterrows():
        log_and_print(f"  {row['Column']}: {row['Data Type']} - {row['Description']}")

    # Visualize data types
    plt.figure(figsize=(10, 6))
    dtype_count = dtype_summary['Data Type'].astype(str).value_counts()
    dtype_count.plot(kind='bar', color='skyblue')
    plt.title('Data Types Distribution')
    plt.xlabel('Data Type')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(STEP2_PLOT_FOLDER, "data_types_distribution.png"))
    plt.close()


def check_column_quality(data):
    """Check quality of each column"""
    log_and_print("Checking column quality...")

    quality_metrics = []

    for col in data.columns:
        # Calculate basic metrics
        count = data[col].count()
        n_missing = data[col].isnull().sum()
        pct_missing = n_missing / len(data) * 100
        n_unique = data[col].nunique()
        pct_unique = n_unique / count * 100 if count > 0 else 0

        # Check for duplicates in ID column
        is_id_col = col in ['TelHVID', 'Schadennummer']
        duplicates = data[col].duplicated().sum() if is_id_col else 'N/A'

        # Range for numeric columns
        if pd.api.types.is_numeric_dtype(data[col]):
            min_val = data[col].min()
            max_val = data[col].max()
            mean_val = data[col].mean()
            median_val = data[col].median()
            std_dev = data[col].std()

            # Check if values match expected range for specific columns
            if col == 'FL_Score' or col == 'P_Score':
                in_range = (data[col] >= 0) & (data[col] <= 4)
                pct_in_range = in_range.mean() * 100
            elif col == 'FL_Status_Nominal' or col == 'P_Status_Nominal':
                in_range = (data[col] >= 0) & (data[col] <= 2)
                pct_in_range = in_range.mean() * 100
            elif col == 'Verlauf_entspricht_NBE':
                in_range = (data[col] >= 0) & (data[col] <= 1)
                pct_in_range = in_range.mean() * 100
            elif col == 'Telefonat':
                in_range = (data[col] >= 0) & (data[col] <= 4)
                pct_in_range = in_range.mean() * 100
            else:
                pct_in_range = 'N/A'
        else:
            min_val = 'N/A'
            max_val = 'N/A'
            mean_val = 'N/A'
            median_val = 'N/A'
            std_dev = 'N/A'
            pct_in_range = 'N/A'

        quality_metrics.append({
            'Column': col,
            'Count': count,
            'Missing': n_missing,
            'Missing (%)': round(pct_missing, 2),
            'Unique Values': n_unique,
            'Unique (%)': round(pct_unique, 2),
            'Duplicates': duplicates,
            'Min': min_val,
            'Max': max_val,
            'Mean': mean_val,
            'Median': median_val,
            'StdDev': std_dev,
            'Values in Expected Range (%)': pct_in_range,
            'Description': column_info.get(col, "No description available")
        })

    # Create quality metrics dataframe
    quality_df = pd.DataFrame(quality_metrics)

    # Save quality metrics
    quality_file = os.path.join(STEP2_OUTPUT_FOLDER, "column_quality_metrics.xlsx")
    quality_df.to_excel(quality_file, index=False)
    log_and_print(f"Column quality metrics saved to {quality_file}")

    return quality_df


def visualize_column_distributions(data):
    """Create visualizations for column distributions"""
    log_and_print("Visualizing column distributions...")

    # Visualize numeric columns
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns

    # Histograms for each numeric column
    for col in numeric_cols:
        plt.figure(figsize=(10, 6))
        sns.histplot(data[col], kde=True)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(STEP2_PLOT_FOLDER, f"distribution_{col}.png"))
        plt.close()

    # Visualize categorical columns
    categorical_cols = data.select_dtypes(exclude=['int64', 'float64']).columns

    # Bar plots for each categorical column
    for col in categorical_cols:
        plt.figure(figsize=(12, 6))
        value_counts = data[col].value_counts().sort_index()
        value_counts.plot(kind='bar', color='skyblue')
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(STEP2_PLOT_FOLDER, f"distribution_{col}.png"))
        plt.close()

    # Create correlation matrix for numeric columns
    if len(numeric_cols) > 1:
        plt.figure(figsize=(12, 10))
        corr_matrix = data[numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(STEP2_PLOT_FOLDER, "correlation_matrix.png"))
        plt.close()

        # Save correlation matrix
        corr_file = os.path.join(STEP2_OUTPUT_FOLDER, "correlation_matrix.xlsx")
        corr_matrix.to_excel(corr_file)
        log_and_print(f"Correlation matrix saved to {corr_file}")


def check_for_outliers(data):
    """Check for outliers in numeric columns"""
    log_and_print("Checking for outliers...")

    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
    outlier_summary = []

    for col in numeric_cols:
        # Skip binary columns
        if data[col].nunique() <= 2:
            continue

        # Calculate Z-scores
        z_scores = np.abs(stats.zscore(data[col]))
        outliers_z = data[z_scores > 3]

        # Calculate IQR
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers_iqr = data[(data[col] < (Q1 - 1.5 * IQR)) | (data[col] > (Q3 + 1.5 * IQR))]

        # Box plot to visualize outliers
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=data[col])
        plt.title(f'Box Plot for {col} showing potential outliers')
        plt.tight_layout()
        plt.savefig(os.path.join(STEP2_PLOT_FOLDER, f"outliers_{col}.png"))
        plt.close()

        outlier_summary.append({
            'Column': col,
            'Z-score Outliers Count': len(outliers_z),
            'Z-score Outliers (%)': round(len(outliers_z) / len(data) * 100, 2),
            'IQR Outliers Count': len(outliers_iqr),
            'IQR Outliers (%)': round(len(outliers_iqr) / len(data) * 100, 2),
            'Min Value': data[col].min(),
            'Max Value': data[col].max()
        })

    # Create outlier summary dataframe
    outlier_df = pd.DataFrame(outlier_summary)

    # Save outlier summary
    if not outlier_df.empty:
        outlier_file = os.path.join(STEP2_OUTPUT_FOLDER, "outlier_summary.xlsx")
        outlier_df.to_excel(outlier_file, index=False)
        log_and_print(f"Outlier summary saved to {outlier_file}")
    else:
        log_and_print("No appropriate numeric columns for outlier detection")

    return outlier_df


def analyze_relationships(data):
    """Analyze relationships between key variables"""
    log_and_print("Analyzing relationships between variables...")

    # Relationship between FL_Score and P_Score
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='FL_Score', y='P_Score', data=data)
    plt.title('Relationship between Function Limitation Score and Pain Score')
    plt.xlabel('Function Limitation Score (0-best to 4-worst)')
    plt.ylabel('Pain Score (0-best to 4-worst)')
    plt.savefig(os.path.join(STEP2_PLOT_FOLDER, "relationship_fl_p_score.png"))
    plt.close()

    # Relationship between FL_Status_Nominal and P_Status_Nominal
    plt.figure(figsize=(10, 6))
    sns.heatmap(pd.crosstab(data['FL_Status_Nominal'], data['P_Status_Nominal'],
                            normalize='index'), annot=True, cmap='Blues', fmt='.2f')
    plt.title('Heatmap of Function Limitation Status vs Pain Status')
    plt.xlabel('Pain Status (0-better, 1-no change, 2-worse)')
    plt.ylabel('Function Limitation Status (0-better, 1-no change, 2-worse)')
    plt.savefig(os.path.join(STEP2_PLOT_FOLDER, "heatmap_fl_p_status.png"))
    plt.close()

    # Relationship with Verlauf_entspricht_NBE
    target_col = 'Verlauf_entspricht_NBE'
    numeric_predictors = ['FL_Score', 'P_Score']
    categorical_predictors = ['FL_Status_Nominal', 'P_Status_Nominal', 'Telefonat']

    # For numeric predictors
    for col in numeric_predictors:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=target_col, y=col, data=data)
        plt.title(f'Relationship between {target_col} and {col}')
        plt.xlabel('Within NBE Period (1-yes/good, 0-no/bad)')
        plt.ylabel(column_info.get(col, col))
        plt.savefig(os.path.join(STEP2_PLOT_FOLDER, f"relationship_{target_col}_{col}.png"))
        plt.close()

    # For categorical predictors
    for col in categorical_predictors:
        plt.figure(figsize=(10, 6))
        cross_tab = pd.crosstab(data[col], data[target_col], normalize='index')
        cross_tab.plot(kind='bar', stacked=True)
        plt.title(f'Relationship between {col} and {target_col}')
        plt.xlabel(column_info.get(col, col))
        plt.ylabel('Proportion')
        plt.legend(title='Within NBE Period', labels=['No/Bad (0)', 'Yes/Good (1)'])
        plt.savefig(os.path.join(STEP2_PLOT_FOLDER, f"relationship_{col}_{target_col}.png"))
        plt.close()


def check_special_considerations(data):
    """Check for any special considerations or business rules"""
    log_and_print("Checking for special considerations and business rules...")

    # Check if FL_Score and P_Score are within expected range
    score_range_check = {
        'FL_Score': (data['FL_Score'] >= 0) & (data['FL_Score'] <= 4),
        'P_Score': (data['P_Score'] >= 0) & (data['P_Score'] <= 4)
    }

    for col, check in score_range_check.items():
        invalid_count = (~check).sum()
        if invalid_count > 0:
            log_and_print(f"WARNING: {invalid_count} records have invalid {col} values outside expected range 0-4")

    # Check if FL_Status_Nominal and P_Status_Nominal are within expected range
    status_range_check = {
        'FL_Status_Nominal': (data['FL_Status_Nominal'] >= 0) & (data['FL_Status_Nominal'] <= 2),
        'P_Status_Nominal': (data['P_Status_Nominal'] >= 0) & (data['P_Status_Nominal'] <= 2)
    }

    for col, check in status_range_check.items():
        invalid_count = (~check).sum()
        if invalid_count > 0:
            log_and_print(f"WARNING: {invalid_count} records have invalid {col} values outside expected range 0-2")

    # Check if Telefonat is within expected range
    telefonat_check = (data['Telefonat'] >= 0) & (data['Telefonat'] <= 4)
    invalid_telefonat = (~telefonat_check).sum()
    if invalid_telefonat > 0:
        log_and_print(f"WARNING: {invalid_telefonat} records have invalid Telefonat values outside expected range 0-4")

    # Check if Verlauf_entspricht_NBE is binary
    nbe_check = (data['Verlauf_entspricht_NBE'] == 0) | (data['Verlauf_entspricht_NBE'] == 1)
    invalid_nbe = (~nbe_check).sum()
    if invalid_nbe > 0:
        log_and_print(
            f"WARNING: {invalid_nbe} records have invalid Verlauf_entspricht_NBE values outside expected binary values 0-1")

    # If no issues found
    if all(check.all() for check in list(score_range_check.values()) + list(
            status_range_check.values())) and telefonat_check.all() and nbe_check.all():
        log_and_print("All columns conform to expected value ranges")


def create_summary_report(data, quality_df, outlier_df=None):
    """Create a summary report with key findings"""
    log_and_print("Creating summary report...")

    summary = []

    # Dataset summary
    summary.append("## Dataset Summary")
    summary.append(f"- Total records: {len(data)}")
    summary.append(f"- Total columns: {len(data.columns)}")

    # Column types summary
    dtypes = data.dtypes
    dtype_counts = dtypes.groupby(dtypes.astype(str)).size()
    summary.append("\n## Column Types")
    for dtype, count in dtype_counts.items():
        summary.append(f"- {dtype}: {count} columns")

    # Quality issues summary
    summary.append("\n## Data Quality Summary")
    # Missing values
    missing_cols = quality_df[quality_df['Missing'] > 0]
    if not missing_cols.empty:
        summary.append("\n### Missing Values")
        for _, row in missing_cols.iterrows():
            summary.append(f"- {row['Column']}: {row['Missing']} missing values ({row['Missing (%)']}%)")
    else:
        summary.append("\n### Missing Values")
        summary.append("- No missing values found in any column")

    # Outliers
    if outlier_df is not None and not outlier_df.empty:
        summary.append("\n### Outliers")
        for _, row in outlier_df.iterrows():
            summary.append(
                f"- {row['Column']}: {row['Z-score Outliers Count']} outliers based on Z-score ({row['Z-score Outliers (%)']}%)")

    # Range violations
    summary.append("\n### Range Violations")
    range_cols = {
        'FL_Score': [0, 4],
        'P_Score': [0, 4],
        'FL_Status_Nominal': [0, 2],
        'P_Status_Nominal': [0, 2],
        'Verlauf_entspricht_NBE': [0, 1],
        'Telefonat': [0, 4]
    }

    for col, [min_val, max_val] in range_cols.items():
        if col in data.columns:
            out_of_range = ((data[col] < min_val) | (data[col] > max_val)).sum()
            if out_of_range > 0:
                summary.append(f"- {col}: {out_of_range} values outside expected range [{min_val}-{max_val}]")

    # Value distribution summary
    summary.append("\n## Value Distribution Highlights")

    # FL_Score and P_Score distribution
    summary.append("\n### Function Limitation and Pain Scores")
    for col in ['FL_Score', 'P_Score']:
        if col in data.columns:
            value_counts = data[col].value_counts().sort_index()
            summary.append(f"\n{col} distribution:")
            for val, count in value_counts.items():
                summary.append(f"- Value {val}: {count} records ({count/len(data)*100:.1f}%)")

    # Status distribution
    summary.append("\n### Status Distributions")
    for col in ['FL_Status_Nominal', 'P_Status_Nominal']:
        if col in data.columns:
            value_counts = data[col].value_counts().sort_index()
            summary.append(f"\n{col} distribution:")
            for val, count in value_counts.items():
                label = 'Better' if val == 0 else 'No Change' if val == 1 else 'Worse'
                summary.append(f"- {label} ({val}): {count} records ({count/len(data)*100:.1f}%)")

    # Telefonat distribution
    if 'Telefonat' in data.columns:
        summary.append("\n### Contact Type Distribution")
        telefonat_labels = {
            0: 'Erstkontakt',
            1: 'Folgekontakt',
            2: 'nicht erreicht',
            3: 'Fallabschluss',
            4: 'Komplikationsbesprechung'
        }
        value_counts = data['Telefonat'].value_counts().sort_index()
        for val, count in value_counts.items():
            label = telefonat_labels.get(val, f"Unknown ({val})")
            summary.append(f"- {label}: {count} records ({count/len(data)*100:.1f}%)")

    # Target variable distribution
    if 'Verlauf_entspricht_NBE' in data.columns:
        summary.append("\n### Target Variable Distribution")
        value_counts = data['Verlauf_entspricht_NBE'].value_counts().sort_index()
        for val, count in value_counts.items():
            label = 'Not within NBE (Bad)' if val == 0 else 'Within NBE (Good)'
            summary.append(f"- {label}: {count} records ({count/len(data)*100:.1f}%)")

    # Correlation highlights
    if len(data.select_dtypes(include=['int64', 'float64']).columns) > 1:
        corr_matrix = data.select_dtypes(include=['int64', 'float64']).corr()
        # Get the top 5 correlations (excluding self-correlations)
        corrs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corrs.append((corr_matrix.index[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))

        # Sort by absolute correlation
        corrs = sorted(corrs, key=lambda x: abs(x[2]), reverse=True)

        if corrs:
            summary.append("\n## Top Correlations")
            for col1, col2, corr in corrs[:5]:
                summary.append(f"- {col1} and {col2}: {corr:.3f}")

    # Key insights
    summary.append("\n## Key Insights")
    # Calculate some insights
    if all(col in data.columns for col in ['FL_Score', 'P_Score', 'Verlauf_entspricht_NBE']):
        # Average scores by target variable
        avg_by_target = data.groupby('Verlauf_entspricht_NBE')[['FL_Score', 'P_Score']].mean()

        if len(avg_by_target) == 2:
            summary.append("\n### Average Scores by NBE Status")
            for target, row in avg_by_target.iterrows():
                label = 'Not within NBE (Bad)' if target == 0 else 'Within NBE (Good)'
                summary.append(f"- {label}:")
                summary.append(f"  - Average Function Limitation Score: {row['FL_Score']:.2f}")
                summary.append(f"  - Average Pain Score: {row['P_Score']:.2f}")

    # Write summary to file
    summary_text = "\n".join(summary)
    summary_file = os.path.join(STEP2_OUTPUT_FOLDER, "data_quality_summary.md")

    with open(summary_file, 'w') as f:
        f.write(summary_text)

    log_and_print(f"Summary report saved to {summary_file}")


def main():
    """Main function to run all quality checks"""
    log_and_print("Starting data quality check...")

    # Load data
    data = load_data()
    if data is None:
        log_and_print("Error: Could not load dataset. Exiting.")
        return

    # Column profiling
    generate_custom_profile_report(data)

    # Data types analysis
    analyze_data_types(data)

    # Column quality check
    quality_df = check_column_quality(data)

    # Visualize distributions
    visualize_column_distributions(data)

    # Check for outliers
    outlier_df = check_for_outliers(data)

    # Analyze relationships
    analyze_relationships(data)

    # Check special considerations
    check_special_considerations(data)

    # Create summary report
    create_summary_report(data, quality_df, outlier_df)

    log_and_print("Data quality check completed successfully.")


if __name__ == "__main__":
    main()