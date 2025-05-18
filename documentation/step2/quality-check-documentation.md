# Healthcare Dataset Quality Check Documentation

## Project Overview

This document provides a comprehensive overview of the data quality check process implemented for the healthcare dataset analysis project. The quality check was performed after data anonymization and missing value imputation.

## Data Pipeline Context

The data quality check represents the third stage in our data pipeline:

1. **Step 0: Data Anonymization** - Patient identifiers (Schadennummer) were anonymized to protect privacy.
2. **Step 1: Data Preprocessing & Imputation** - Missing values were handled using KNN imputation.
3. **Step 2: Data Quality Check** - Comprehensive quality assessment and profiling of the dataset.

## Dataset Information

The healthcare dataset contains information about patient assessments with the following characteristics:

- **Size**: 3,427 records, 7 columns
- **Variables**:
  - **FL_Score**: Function limitation score (0-best to 4-worst)
  - **FL_Status_Nominal**: Function limitation status compared to previous assessment (0-better, 1-no change, 2-worse)
  - **P_Score**: Pain score (0-best to 4-worst)
  - **P_Status_Nominal**: Pain status compared to previous assessment (0-better, 1-no change, 2-worse)
  - **Schadennummer**: Anonymized patient ID
  - **Verlauf_entspricht_NBE**: Within Nachbehandlungsempfehlungen period (1-good, 0-bad)
  - **Telefonat**: Contact type (0-Erstkontakt, 1-Folgekontakt, 2-nicht erreicht, 3-Fallabschluss, 4-Komplikationsbesprechung)

## Quality Check Methodology

The quality check was implemented in Python using a custom script that performed multiple types of analysis:

### 1. Data Profiling

A custom profiling approach was implemented that:
- Generated descriptive statistics for each column
- Created visualizations of distributions
- Produced detailed Excel and HTML reports

### 2. Data Type Analysis

- Cataloged all column data types
- Documented appropriate data types for each variable
- Created summary visualizations of data type distribution

### 3. Column Quality Metrics

For each column, we calculated:
- Count of non-null values
- Count and percentage of missing values
- Count and percentage of unique values
- Duplicates (for ID columns)
- Statistical measures (min, max, mean, median, standard deviation)
- Range validation (percentage of values within expected ranges)

### 4. Distribution Analysis

- Generated histograms for numeric columns
- Created bar charts for categorical columns
- Analyzed the distribution patterns in key variables

### 5. Outlier Detection

Two methods were used to identify outliers:
- **Z-score method**: Identified values more than 3 standard deviations from the mean
- **IQR method**: Identified values below Q1 - 1.5*IQR or above Q3 + 1.5*IQR

Results were visualized using box plots and summarized in an outlier report.

### 6. Relationship Analysis

- Created correlation matrix for numeric variables
- Generated scatter plots between key variables
- Created heatmaps for categorical relationships
- Analyzed the relationship between predictors and the target variable

### 7. Business Rule Validation

Performed checks to ensure data conforms to expected value ranges:
- FL_Score and P_Score: 0-4
- FL_Status_Nominal and P_Status_Nominal: 0-2
- Verlauf_entspricht_NBE: 0-1
- Telefonat: 0-4

## Key Findings

### Data Quality Issues

- **Missing Values**: None (successfully imputed in Step 1)
- **Outliers**: 26 outliers detected in P_Score (0.76% of data) corresponding to patients with the highest pain score of 4
- **Range Violations**: None - all values are within their expected ranges

### Distribution Insights

- **FL_Score**: Most common value is 2 (35.5% of records), followed by 1 (29.9%)
- **P_Score**: Most common value is 1 (45.1% of records), followed by 2 (31.6%)
- **FL_Status_Nominal**: Majority (60.2%) show worsening status, only 2.0% improving
- **P_Status_Nominal**: Majority (62.9%) show worsening status, only 3.6% improving
- **Telefonat**: Mostly follow-up contacts (54.2%) and initial contacts (30.2%)
- **Verlauf_entspricht_NBE**: 79.5% within NBE period (good outcome), 20.5% not within NBE period

### Relationship Insights

- Strong correlation (0.642) between FL_Status_Nominal and P_Status_Nominal
- Moderate correlation (0.478) between FL_Score and P_Score
- Patients within NBE period have better function scores (1.93 vs 2.11) and pain scores (1.32 vs 1.67)

## Outputs Generated

### Reports & Documents

1. **Custom Profile Report**:
   - Excel version: `custom_profile_report.xlsx`
   - HTML version: `custom_profile_report.html`
   - Contains comprehensive statistics for each column

2. **Data Type Summary**:
   - Excel file: `data_types.xlsx`
   - Documents data types and descriptions

3. **Column Quality Metrics**:
   - Excel file: `column_quality_metrics.xlsx`
   - Contains detailed quality metrics for each column

4. **Correlation Matrix**:
   - Excel file: `correlation_matrix.xlsx`
   - Shows correlations between numeric variables

5. **Outlier Summary**:
   - Excel file: `outlier_summary.xlsx`
   - Lists identified outliers by column

6. **Data Quality Summary**:
   - Markdown file: `data_quality_summary.md`
   - Executive summary of all quality findings

### Visualizations

1. **Distribution Plots**:
   - Histograms and bar charts for each variable
   - File format: PNG
   - Location: `plots/step2/`

2. **Correlation Heatmap**:
   - Visualization of correlations between numeric variables
   - File: `correlation_matrix.png`

3. **Outlier Box Plots**:
   - Box plots showing distribution and outliers
   - Files: `outliers_*.png`

4. **Relationship Plots**:
   - Scatter plots and heatmaps showing variable relationships
   - Files: `relationship_*.png`

## Implementation Details

### Environment Configuration

The quality check script uses environment variables for configuration:
- BASE_DATASET: Path to the original dataset file
- OUTPUT_FOLDER: Directory for output files
- LOG_FOLDER: Directory for log files
- PLOTS: Directory for visualization outputs

Additional derived paths:
- STEP2_LOG_FOLDER: LOG_FOLDER/step2
- STEP2_PLOT_FOLDER: PLOTS/step2
- STEP2_OUTPUT_FOLDER: OUTPUT_FOLDER/step2_quality_check

### Dependencies

The analysis was implemented using the following Python libraries:
- pandas: Data manipulation and analysis
- numpy: Numerical operations
- matplotlib & seaborn: Visualization
- scipy: Statistical functions
- missingno: Missing value visualization
- dotenv: Environment variable management

### Code Architecture

The script is organized into modular functions:
1. `load_data()`: Load the KNN-imputed dataset
2. `generate_custom_profile_report()`: Generate comprehensive profile reports
3. `analyze_data_types()`: Analyze and document data types
4. `check_column_quality()`: Calculate quality metrics for each column
5. `visualize_column_distributions()`: Create distribution visualizations
6. `check_for_outliers()`: Detect and document outliers
7. `analyze_relationships()`: Analyze relationships between variables
8. `check_special_considerations()`: Validate business rules
9. `create_summary_report()`: Generate summary report with key findings
10. `main()`: Orchestrate the entire quality check process

## Conclusion

The quality check process revealed a dataset with good overall quality:
- No missing values (after imputation)
- No range violations
- Minimal outliers (only in P_Score)
- Expected distributions based on the healthcare context

The only notable outliers were the 26 patients with a P_Score of 4, representing 0.76% of the dataset. These are valid values within the expected range but statistically rare in this population.

The data quality is suitable for proceeding to further analysis, modeling, and visualization stages.


*Documentation prepared on March 3, 2025*
