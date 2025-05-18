import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.power import TTestIndPower, tt_ind_solve_power
from scipy import stats
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OUTPUT_FOLDER = os.getenv("OUTPUT_FOLDER")
LOG_FOLDER = os.getenv("LOG_FOLDER")
PLOTS = os.getenv("PLOTS")

# Define output paths
SAMPLE_SIZE_LOG_FOLDER = os.path.join(LOG_FOLDER, "step3")
SAMPLE_SIZE_PLOT_FOLDER = os.path.join(PLOTS, "step3")
SAMPLE_SIZE_OUTPUT_FOLDER = os.path.join(OUTPUT_FOLDER, "step3_sample_size")

# Ensure output directories exist
os.makedirs(SAMPLE_SIZE_LOG_FOLDER, exist_ok=True)
os.makedirs(SAMPLE_SIZE_PLOT_FOLDER, exist_ok=True)
os.makedirs(SAMPLE_SIZE_OUTPUT_FOLDER, exist_ok=True)

# Configure logging
log_file = os.path.join(SAMPLE_SIZE_LOG_FOLDER, "sample_size_calculation_log.txt")
logging.basicConfig(filename=log_file, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def log_and_print(message):
    print(message)
    logging.info(message)


def cohen_d(group1, group2):
    """
    Calculate Cohen's d for independent samples.

    Parameters:
    - group1: First group data
    - group2: Second group data

    Returns:
    - cohen_d: Calculated effect size
    """
    # Ensure we have numeric data
    group1 = pd.to_numeric(group1, errors='coerce')
    group2 = pd.to_numeric(group2, errors='coerce')

    # Remove NaN values
    group1 = group1.dropna()
    group2 = group2.dropna()

    # Check if we have enough data
    if len(group1) < 2 or len(group2) < 2:
        return np.nan

    # Calculate Cohen's d
    diff_mean = np.mean(group1) - np.mean(group2)
    pooled_std = np.sqrt(((len(group1) - 1) * np.std(group1, ddof=1) ** 2 +
                          (len(group2) - 1) * np.std(group2, ddof=1) ** 2) /
                         (len(group1) + len(group2) - 2))

    # Guard against division by zero
    if pooled_std == 0:
        return np.nan

    return diff_mean / pooled_std


def calculate_sample_size(effect_size, alpha=0.05, power=0.80, ratio=1):
    """
    Calculate the required sample size for a two-sample t-test.

    Parameters:
    - effect_size: Expected effect size (Cohen's d)
    - alpha: Significance level
    - power: Desired power
    - ratio: Ratio of sample sizes in two groups

    Returns:
    - sample_size: Required sample size per group
    """
    # Handle invalid effect sizes
    if np.isnan(effect_size) or effect_size == 0:
        return np.nan

    # Use absolute value of effect size (direction doesn't matter for sample size)
    effect_size = abs(effect_size)

    power_analysis = TTestIndPower()
    sample_size = power_analysis.solve_power(
        effect_size=effect_size,
        alpha=alpha,
        power=power,
        ratio=ratio
    )

    return sample_size


def create_power_curve_plot(effect_sizes, group_name, output_file):
    """
    Create power curve plot for different effect sizes and sample sizes.

    Parameters:
    - effect_sizes: Dictionary of effect sizes {variable_name: effect_size}
    - group_name: Name of the comparison groups
    - output_file: Path to save the plot
    """
    plt.figure(figsize=(12, 8))

    sample_sizes = np.arange(20, 500, 10)
    power_analysis = TTestIndPower()

    for var_name, effect_size in effect_sizes.items():
        if not np.isnan(effect_size) and effect_size != 0:
            # Calculate power for range of sample sizes
            power = [power_analysis.power(effect_size=abs(effect_size),
                                          nobs1=n,
                                          alpha=0.05,
                                          ratio=1) for n in sample_sizes]
            plt.plot(sample_sizes, power, label=f"{var_name} (d={effect_size:.2f})")

    # Add reference lines
    plt.axhline(y=0.8, linestyle='--', color='gray', alpha=0.7, label="80% Power")
    plt.axhline(y=0.9, linestyle='--', color='darkgray', alpha=0.7, label="90% Power")

    plt.xlabel('Sample Size (per group)')
    plt.ylabel('Power (1-β)')
    plt.title(f'Power Analysis for {group_name} Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_file)
    plt.close()


def main():
    log_and_print("Starting sample size calculation...")

    # Load the KNN-imputed dataset
    knn_dataset_path = os.path.join(OUTPUT_FOLDER, "imputed_datasets", "imputed_dataset_knn.xlsx")
    log_and_print(f"Loading KNN-imputed dataset from {knn_dataset_path}")

    try:
        data = pd.read_excel(knn_dataset_path)
        log_and_print(f"Dataset loaded successfully: {data.shape[0]} rows, {data.shape[1]} columns")
    except Exception as e:
        log_and_print(f"Error loading dataset: {str(e)}")
        return

    # Define the group comparisons we want to analyze
    group_comparisons = [
        {
            'name': 'Initial vs Follow-up Contact',
            'group1_filter': data['Telefonat'] == 0,  # Initial contact (Erstkontakt)
            'group2_filter': data['Telefonat'] == 1,  # Follow-up contact (Folgekontakt)
            'group1_name': 'Initial Contact',
            'group2_name': 'Follow-up Contact'
        },
        {
            'name': 'Initial vs Case Closure',
            'group1_filter': data['Telefonat'] == 0,  # Initial contact (Erstkontakt)
            'group2_filter': data['Telefonat'] == 3,  # Case closure (Fallabschluss)
            'group1_name': 'Initial Contact',
            'group2_name': 'Case Closure'
        },
        {
            'name': 'Within NBE vs Outside NBE',
            'group1_filter': data['Verlauf_entspricht_NBE'] == 1,  # Within NBE
            'group2_filter': data['Verlauf_entspricht_NBE'] == 0,  # Outside NBE
            'group1_name': 'Within NBE',
            'group2_name': 'Outside NBE'
        }
    ]

    # Variables to analyze
    analysis_variables = ['P_Score', 'FL_Score', 'FL_Status_Nominal', 'P_Status_Nominal']

    # Create a results dataframe
    results = []

    # Perform analysis for each comparison
    for comparison in group_comparisons:
        log_and_print(f"\nAnalyzing {comparison['name']}...")

        group1 = data[comparison['group1_filter']]
        group2 = data[comparison['group2_filter']]

        log_and_print(f"Group 1 ({comparison['group1_name']}): {len(group1)} records")
        log_and_print(f"Group 2 ({comparison['group2_name']}): {len(group2)} records")

        effect_sizes = {}

        # Calculate effect sizes and sample sizes for each variable
        for var in analysis_variables:
            # Calculate Cohen's d
            effect_size = cohen_d(group1[var], group2[var])
            effect_sizes[var] = effect_size

            # Calculate required sample size
            sample_size = calculate_sample_size(effect_size)

            # Calculate actual power with current sample sizes
            if not np.isnan(effect_size) and effect_size != 0:
                actual_power = TTestIndPower().power(
                    effect_size=abs(effect_size),
                    nobs1=min(len(group1), len(group2)),
                    alpha=0.05,
                    ratio=max(len(group1), len(group2)) / min(len(group1), len(group2))
                )
            else:
                actual_power = np.nan

            # Perform t-test to get p-value
            if len(group1) > 1 and len(group2) > 1:  # Need at least 2 samples for t-test
                t_stat, p_value = stats.ttest_ind(
                    group1[var].dropna(),
                    group2[var].dropna(),
                    equal_var=False  # Welch's t-test
                )
            else:
                p_value = np.nan

            # Store results
            results.append({
                'Comparison': comparison['name'],
                'Variable': var,
                'Group1': comparison['group1_name'],
                'Group2': comparison['group2_name'],
                'Group1_Size': len(group1),
                'Group2_Size': len(group2),
                'Group1_Mean': group1[var].mean(),
                'Group2_Mean': group2[var].mean(),
                'Mean_Difference': group1[var].mean() - group2[var].mean(),
                'Effect_Size': effect_size,
                'Required_Sample_Size_Per_Group': sample_size,
                'Current_Power': actual_power,
                'P_Value': p_value,
                'Statistically_Significant': p_value < 0.05 if not np.isnan(p_value) else np.nan,
                'Sample_Size_Sufficient': min(len(group1), len(group2)) >= sample_size if not np.isnan(
                    sample_size) else np.nan
            })

            log_and_print(f"  {var}:")
            log_and_print(f"    - Effect size (Cohen's d): {effect_size:.4f}")
            log_and_print(f"    - Required sample size per group: {sample_size:.2f}")
            log_and_print(f"    - Current power: {actual_power:.4f}" if not np.isnan(
                actual_power) else "    - Current power: N/A")
            log_and_print(f"    - P-value: {p_value:.4f}" if not np.isnan(p_value) else "    - P-value: N/A")

        # Create power curve plot
        plot_file = os.path.join(SAMPLE_SIZE_PLOT_FOLDER,
                                 f"power_curve_{comparison['name'].replace(' ', '_').lower()}.png")
        create_power_curve_plot(effect_sizes, comparison['name'], plot_file)
        log_and_print(f"  Power curve plot saved to {plot_file}")

    # Convert results to DataFrame and save
    results_df = pd.DataFrame(results)
    results_file = os.path.join(SAMPLE_SIZE_OUTPUT_FOLDER, "sample_size_analysis_results.xlsx")
    results_df.to_excel(results_file, index=False)
    log_and_print(f"\nResults saved to {results_file}")

    # Create a summary table with recommendations
    summary = []
    for index, row in results_df.iterrows():
        if pd.notna(row['Sample_Size_Sufficient']):
            if row['Sample_Size_Sufficient']:
                recommendation = "Sample size is sufficient for detecting the observed effect."
            else:
                recommendation = f"Sample size is insufficient. Need {row['Required_Sample_Size_Per_Group']:.0f} samples per group."
        else:
            recommendation = "Cannot determine sample sufficiency due to invalid effect size."

        summary.append({
            'Comparison': row['Comparison'],
            'Variable': row['Variable'],
            'Effect_Size': row['Effect_Size'],
            'Current_Sample_Size': min(row['Group1_Size'], row['Group2_Size']),
            'Required_Sample_Size': row['Required_Sample_Size_Per_Group'],
            'Is_Sufficient': row['Sample_Size_Sufficient'],
            'P_Value': row['P_Value'],
            'Is_Significant': row['Statistically_Significant'],
            'Recommendation': recommendation
        })

    summary_df = pd.DataFrame(summary)
    summary_file = os.path.join(SAMPLE_SIZE_OUTPUT_FOLDER, "sample_size_summary.xlsx")
    summary_df.to_excel(summary_file, index=False)
    log_and_print(f"Summary with recommendations saved to {summary_file}")

    # Create summary visualization
    plt.figure(figsize=(12, 8))

    valid_rows = summary_df[pd.notna(summary_df['Required_Sample_Size']) &
                            pd.notna(summary_df['Current_Sample_Size'])]

    # Skip visualization if no valid data
    if not valid_rows.empty:
        max_sample_size = max(valid_rows['Required_Sample_Size'].max(),
                              valid_rows['Current_Sample_Size'].max()) * 1.1

        bar_width = 0.35
        x = np.arange(len(valid_rows))

        fig, ax = plt.subplots(figsize=(14, 10))
        current = ax.bar(x - bar_width / 2, valid_rows['Current_Sample_Size'],
                         bar_width, label='Current Sample Size', color='skyblue')
        required = ax.bar(x + bar_width / 2, valid_rows['Required_Sample_Size'],
                          bar_width, label='Required Sample Size', color='lightcoral')

        # Add labels and customization
        ax.set_ylabel('Sample Size')
        ax.set_title('Current vs. Required Sample Size')
        ax.set_xticks(x)
        ax.set_xticklabels([f"{row['Comparison']}\n{row['Variable']}"
                            for _, row in valid_rows.iterrows()], rotation=45, ha='right')
        ax.legend()

        # Add a line at y=0
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

        # Add value labels on the bars
        for i, bar in enumerate(current):
            height = bar.get_height()
            ax.annotate(f'{height:.0f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

        for i, bar in enumerate(required):
            height = bar.get_height()
            ax.annotate(f'{height:.0f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

        # Adjust layout and save
        fig.tight_layout()
        summary_plot_file = os.path.join(SAMPLE_SIZE_PLOT_FOLDER, "sample_size_comparison.png")
        fig.savefig(summary_plot_file)
        plt.close(fig)

        log_and_print(f"Summary visualization saved to {summary_plot_file}")
    else:
        log_and_print("No valid data for visualization.")

    log_and_print("\nSample size calculation completed successfully.")


if __name__ == "__main__":
    main()