import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Set the random seed for reproducibility
np.random.seed(42)

# Define the number of samples to generate
num_samples = 5000

# 1. Credit History: Number of late payments or defaults in the past (range: 0 to 10, where higher is riskier)
credit_history = np.random.poisson(1, num_samples)

# 2. Income Stability (Proxy): Employment stability score (range: 0 to 1, higher means more stable)
income_stability = np.clip(np.random.normal(0.7, 0.2, num_samples), 0, 1)

# 3. Debt-to-Income Ratio: Percentage of income used to pay debt (range: 0 to 1, where 1 is very high)
debt_to_income_ratio = np.clip(np.random.normal(0.4, 0.2, num_samples), 0, 1)

# 4. Loan-to-Value Ratio: Ratio of loan amount to the value of collateral (range: 0 to 1, where higher is riskier)
loan_to_value_ratio = np.clip(np.random.normal(0.6, 0.25, num_samples), 0, 1)

# 5. Credit Utilization Rate: Percentage of available credit being used (range: 0 to 1, where 1 means maxed out)
credit_utilization_rate = np.clip(np.random.normal(0.5, 0.3, num_samples), 0, 1)

# 6. Recent Credit Inquiries: Number of recent credit inquiries in the last 6 months (range: 0 to 5, where more is riskier)
recent_credit_inquiries = np.random.poisson(1, num_samples)

# 7. Payment History for Utilities/Bills: Number of late utility/bill payments (range: 0 to 10, where higher is riskier)
payment_history_utilities = np.random.poisson(1, num_samples)

# 8. Financial Stability Indicator: Indicator derived from spending/saving patterns (range: 0 to 1, where lower is riskier)
financial_stability_indicator = np.clip(np.random.normal(0.75, 0.15, num_samples), 0, 1)

# Additional Features for Enhanced Robustness

# 9. Savings or Cash Reserves: Amount of cash savings (in arbitrary units)
savings_cash_reserves = np.random.exponential(5, num_samples)  # Most people have lower reserves, but some may have high amounts

# 10. Credit Mix: A score indicating the variety of credit types (0 - no mix, 1 - moderate mix, 2 - diverse mix)
credit_mix = np.random.choice([0, 1, 2], num_samples, p=[0.3, 0.5, 0.2])

# 11. Employment Type: 1 for salaried (stable), 0 for self-employed (less stable)
employment_type = np.random.choice([0, 1], num_samples, p=[0.3, 0.7])  # Majority salaried for this dataset

# 12. Monthly Cash Flow / Expense Ratio: Ratio of income after expenses (range: 0 to 1, where higher is better)
monthly_cash_flow_ratio = np.clip(np.random.normal(0.5, 0.2, num_samples), 0, 1)

# Target Variable: Risk Level (1 - High risk, 0 - Low risk)
risk_score = (
    (credit_history * 0.25) +
    (1 - income_stability) * 0.2 +
    debt_to_income_ratio * 0.15 +
    loan_to_value_ratio * 0.1 +
    credit_utilization_rate * 0.1 +
    recent_credit_inquiries * 0.05 +
    payment_history_utilities * 0.05 +
    (1 - financial_stability_indicator) * 0.05 +
    (1 - monthly_cash_flow_ratio) * 0.05 +
    (credit_mix == 0) * 0.05  # Penalize applicants with no credit mix
)
risk_level = (risk_score > risk_score.mean()).astype(int)  # Binary target based on risk score

# Combine all features into a DataFrame
data = pd.DataFrame({
    'Credit_History': credit_history,
    'Income_Stability': income_stability,
    'Debt_to_Income_Ratio': debt_to_income_ratio,
    'Loan_to_Value_Ratio': loan_to_value_ratio,
    'Credit_Utilization_Rate': credit_utilization_rate,
    'Recent_Credit_Inquiries': recent_credit_inquiries,
    'Payment_History_Utilities': payment_history_utilities,
    'Financial_Stability_Indicator': financial_stability_indicator,
    'Savings_Cash_Reserves': savings_cash_reserves,
    'Credit_Mix': credit_mix,
    'Employment_Type': employment_type,
    'Monthly_Cash_Flow_Ratio': monthly_cash_flow_ratio,
    'Risk_Level': risk_level  # Target variable
})

# Split the data into train, validation, and test sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)  # 80% train + validation, 20% test
train_data, val_data = train_test_split(train_data, test_size=0.25, random_state=42)  # 60% train, 20% validation

# Save the datasets to CSV files
train_data.to_csv("synthetic_credit_risk_train.csv", index=False)
val_data.to_csv("synthetic_credit_risk_val.csv", index=False)
test_data.to_csv("synthetic_credit_risk_test.csv", index=False)

print("Datasets saved as 'synthetic_credit_risk_train.csv', 'synthetic_credit_risk_val.csv', and 'synthetic_credit_risk_test.csv'")
