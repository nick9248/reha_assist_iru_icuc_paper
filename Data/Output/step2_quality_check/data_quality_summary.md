## Dataset Summary
- Total records: 3427
- Total columns: 7

## Column Types
- int64: 6 columns
- object: 1 columns

## Data Quality Summary

### Missing Values
- No missing values found in any column

### Outliers
- FL_Score: 0 outliers based on Z-score (0.0%)
- FL_Status_Nominal: 0 outliers based on Z-score (0.0%)
- P_Score: 26 outliers based on Z-score (0.76%)
- P_Status_Nominal: 0 outliers based on Z-score (0.0%)
- Telefonat: 0 outliers based on Z-score (0.0%)

### Range Violations

## Value Distribution Highlights

### Function Limitation and Pain Scores

FL_Score distribution:
- Value 0: 162 records (4.7%)
- Value 1: 1024 records (29.9%)
- Value 2: 1215 records (35.5%)
- Value 3: 811 records (23.7%)
- Value 4: 215 records (6.3%)

P_Score distribution:
- Value 0: 456 records (13.3%)
- Value 1: 1547 records (45.1%)
- Value 2: 1083 records (31.6%)
- Value 3: 315 records (9.2%)
- Value 4: 26 records (0.8%)

### Status Distributions

FL_Status_Nominal distribution:
- Better (0): 69 records (2.0%)
- No Change (1): 1295 records (37.8%)
- Worse (2): 2063 records (60.2%)

P_Status_Nominal distribution:
- Better (0): 125 records (3.6%)
- No Change (1): 1146 records (33.4%)
- Worse (2): 2156 records (62.9%)

### Contact Type Distribution
- Erstkontakt: 1036 records (30.2%)
- Folgekontakt: 1857 records (54.2%)
- Fallabschluss: 534 records (15.6%)

### Target Variable Distribution
- Not within NBE (Bad): 701 records (20.5%)
- Within NBE (Good): 2726 records (79.5%)

## Top Correlations
- FL_Status_Nominal and P_Status_Nominal: 0.642
- FL_Score and P_Score: 0.478
- FL_Score and FL_Status_Nominal: -0.427
- FL_Score and Telefonat: -0.419
- P_Score and P_Status_Nominal: -0.368

## Key Insights

### Average Scores by NBE Status
- Not within NBE (Bad):
  - Average Function Limitation Score: 2.11
  - Average Pain Score: 1.67
- Within NBE (Good):
  - Average Function Limitation Score: 1.93
  - Average Pain Score: 1.32