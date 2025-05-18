# Sample Size Calculation Documentation

## Overview

This document provides comprehensive documentation for the sample size calculation analysis performed on the healthcare dataset. The analysis evaluates whether the current dataset has sufficient statistical power for detecting meaningful differences between various patient groups.

## Purpose

The primary purpose of this analysis was to:

1. Calculate effect sizes between different patient groups for key outcome variables
2. Determine required sample sizes for each comparison
3. Assess whether the current dataset has sufficient statistical power
4. Provide guidance on the reliability of potential findings

## Methodology

### Data Source
- Analysis was performed on the KNN-imputed dataset (`imputed_dataset_knn.xlsx`)
- Total sample size: 3,427 patients
- The data was previously anonymized and quality-checked

### Statistical Approach
- **Effect Size Calculation**: Cohen's d was used to quantify the magnitude of differences between groups
- **Sample Size Determination**: Power analysis for two-sample t-tests
- **Significance Level (α)**: 0.05 (conventional)
- **Statistical Power (1-β)**: 0.80 (conventional)
- **Analysis Tool**: Python's `statsmodels.stats.power.TTestIndPower`

### Group Comparisons Analyzed
Three key group comparisons were evaluated:

1. **Initial vs. Follow-up Contact**
   - Group 1: Initial contact (Telefonat = 0, n = 1,036)
   - Group 2: Follow-up contact (Telefonat = 1, n = 1,857)

2. **Initial vs. Case Closure**
   - Group 1: Initial contact (Telefonat = 0, n = 1,036) 
   - Group 2: Case closure (Telefonat = 3, n = 534)

3. **Within NBE vs. Outside NBE**
   - Group 1: Within NBE period (Verlauf_entspricht_NBE = 1, n = 2,726)
   - Group 2: Outside NBE period (Verlauf_entspricht_NBE = 0, n = 701)

### Outcome Variables
Four key outcome variables were analyzed for each comparison:

1. **P_Score**: Pain score (0-best to 4-worst)
2. **FL_Score**: Function limitation score (0-best to 4-worst)
3. **FL_Status_Nominal**: Function limitation status compared to previous assessment (0-better, 1-no change, 2-worse)
4. **P_Status_Nominal**: Pain status compared to previous assessment (0-better, 1-no change, 2-worse)

## Results

### Effect Sizes and Required Sample Sizes

#### Initial vs. Follow-up Contact

| Variable | Effect Size (Cohen's d) | Required Sample Size per Group | Current Sample Size | Current Power | p-value | Sufficient? |
|----------|-------------------------|-------------------------------|---------------------|---------------|---------|-------------|
| P_Score | 0.1872 | 449 | 1,036 | 0.9979 | <0.0001 | Yes |
| FL_Score | 0.4307 | 86 | 1,036 | 1.0000 | <0.0001 | Yes |
| FL_Status_Nominal | -0.0962 | 1,698 | 1,036 | 0.6984 | 0.0121 | No |
| P_Status_Nominal | 0.1906 | 433 | 1,036 | 0.9984 | <0.0001 | Yes |

#### Initial vs. Case Closure

| Variable | Effect Size (Cohen's d) | Required Sample Size per Group | Current Sample Size | Current Power | p-value | Sufficient? |
|----------|-------------------------|-------------------------------|---------------------|---------------|---------|-------------|
| P_Score | 1.0275 | 16 | 534 | 1.0000 | <0.0001 | Yes |
| FL_Score | 1.4075 | 9 | 534 | 1.0000 | <0.0001 | Yes |
| FL_Status_Nominal | -0.8198 | 24 | 534 | 1.0000 | <0.0001 | Yes |
| P_Status_Nominal | -0.3757 | 112 | 534 | 1.0000 | <0.0001 | Yes |

#### Within NBE vs. Outside NBE

| Variable | Effect Size (Cohen's d) | Required Sample Size per Group | Current Sample Size | Current Power | p-value | Sufficient? |
|----------|-------------------------|-------------------------------|---------------------|---------------|---------|-------------|
| P_Score | -0.4247 | 88 | 701 | 1.0000 | <0.0001 | Yes |
| FL_Score | -0.1801 | 485 | 701 | 0.9890 | <0.0001 | Yes |
| FL_Status_Nominal | 0.5842 | 47 | 701 | 1.0000 | <0.0001 | Yes |
| P_Status_Nominal | 0.6494 | 38 | 701 | 1.0000 | <0.0001 | Yes |

### Key Findings

1. **Initial vs. Follow-up Contact**
   - Three of four variables have sufficient sample sizes and high statistical power
   - FL_Status_Nominal comparison is underpowered (power = 69.8%), requiring 1,698 samples per group
   - The smaller effect size for FL_Status_Nominal (-0.0962) indicates more subtle changes in function limitation status between initial and follow-up contacts

2. **Initial vs. Case Closure**
   - All variables show very large effect sizes (0.38-1.41)
   - Required sample sizes are extremely small (9-112 per group)
   - Current sample sizes far exceed requirements, resulting in 100% power
   - The large effect sizes indicate substantial clinical changes by case closure

3. **Within NBE vs. Outside NBE**
   - All comparisons have sufficient statistical power with current sample sizes
   - Effect sizes range from small (-0.18 for FL_Score) to medium-large (0.65 for P_Status_Nominal)
   - All p-values are highly significant (<0.0001)

## Power Curve Analysis

Power curve plots were generated for each comparison to visualize the relationship between sample size and statistical power:

1. **Initial vs. Follow-up Contact**
   - FL_Score reaches 80% power at relatively small sample sizes (~86 per group)
   - P_Score and P_Status_Nominal require moderate sample sizes (~449 and ~433 respectively)
   - FL_Status_Nominal requires a much larger sample size (~1,698 per group)

2. **Initial vs. Case Closure**
   - All variables reach 80% power with very small sample sizes
   - FL_Score has the steepest power curve, reaching 80% power with just 9 samples per group
   - P_Status_Nominal requires the largest sample size but still achieves 80% power with only 112 per group

3. **Within NBE vs. Outside NBE**
   - P_Status_Nominal and FL_Status_Nominal reach 80% power quickly (38 and 47 samples)
   - P_Score requires 88 samples per group
   - FL_Score has the shallowest curve, requiring 485 samples per group

## Technical Implementation

### Software and Libraries
- Python 3.x
- NumPy and Pandas for data manipulation
- Matplotlib and Seaborn for visualization
- StatsModels for power analysis
- SciPy for statistical tests
- Logging for tracking execution
- dotenv for environment variables

### Output Files Generated

1. **Excel Reports**
   - `sample_size_analysis_results.xlsx`: Detailed results for all comparisons
   - `sample_size_summary.xlsx`: Summary with recommendations

2. **Visualization Plots**
   - `power_curve_initial_vs_follow-up_contact.png`: Power curves for Initial vs. Follow-up
   - `power_curve_initial_vs_case_closure.png`: Power curves for Initial vs. Case Closure
   - `power_curve_within_nbe_vs_outside_nbe.png`: Power curves for Within NBE vs. Outside NBE
   - `sample_size_comparison.png`: Bar chart comparing current vs. required sample sizes

3. **Log File**
   - `sample_size_calculation_log.txt`: Detailed execution log

### Key Functions

1. **cohen_d()**: Calculates Cohen's d effect size between two groups
   - Handles missing values
   - Provides error checking for empty groups and division by zero

2. **calculate_sample_size()**: Determines required sample size based on effect size
   - Uses absolute effect size (direction doesn't matter for sample size)
   - Returns NaN for invalid effect sizes

3. **create_power_curve_plot()**: Generates visualization of power vs. sample size
   - Shows power curves for each variable in a comparison
   - Includes reference lines for 80% and 90% power

4. **main()**: Orchestrates the entire analysis process
   - Loads data
   - Defines group comparisons
   - Calculates effect sizes and required sample sizes
   - Performs t-tests for p-values
   - Generates reports and visualizations

## Interpretation and Recommendations

### Dataset Reliability Assessment

Based on the analysis, the dataset's reliability can be assessed as follows:

1. **Highly Reliable Comparisons**
   - Initial vs. Case Closure: All metrics show large effects with 100% power
   - Within NBE vs. Outside NBE: All metrics have sufficient sample sizes and power

2. **Mostly Reliable Comparison**
   - Initial vs. Follow-up: Three metrics have sufficient power, but FL_Status_Nominal is underpowered

### Recommendations

1. **For Current Analysis**
   - Proceed with confidence for most comparisons, as sample sizes are sufficient
   - Exercise caution when interpreting FL_Status_Nominal differences between initial and follow-up contacts
   - Consider the clinical significance of the large effect sizes observed between initial contact and case closure

2. **For Future Data Collection**
   - To achieve 80% power for all comparisons, aim to collect approximately 1,698 samples per group for the Initial vs. Follow-up comparison
   - No additional data collection is needed for Initial vs. Case Closure or Within NBE vs. Outside NBE comparisons

3. **For Future Analyses**
   - When examining FL_Status_Nominal between initial and follow-up contacts, consider using more sensitive statistical methods or combining data with other variables
   - The large effect sizes in the Initial vs. Case Closure comparison suggest this might be the most informative comparison for clinical insights

## Conclusion

The sample size calculation analysis demonstrates that the current dataset is generally reliable for most of the intended analyses. The only notable limitation is in detecting small differences in function limitation status (FL_Status_Nominal) between initial and follow-up contacts, which would require a larger sample size to achieve adequate statistical power.

The extremely large effect sizes observed between initial contacts and case closure indicate substantial clinical changes that are easily detected with the current sample size. This suggests that the dataset is particularly well-suited for analyses focused on treatment outcomes at case closure.

Overall, the dataset provides a solid foundation for healthcare analyses, with 11 out of 12 key comparisons having sufficient statistical power to detect meaningful differences.

---

*Document prepared on March 3, 2025*
