
# Synthetic Data Generator for Credit Risk Lending Model

This project generates synthetic data for building a credit risk scoring model, which can be used to predict the likelihood of loan default for applicants. The dataset includes features that capture various aspects of an applicant's financial behavior and loan repayment ability, without relying on personal information like age or marital status.

## Features

The synthetic dataset includes the following features:

1. **Credit_History**: The number of late payments or defaults in the past. Higher values indicate a higher risk of default.

2. **Income_Stability**: A score (0 to 1) representing employment stability, with higher values indicating more stability. Applicants with stable employment are generally seen as lower risk.

3. **Debt_to_Income_Ratio**: The percentage of an applicant’s income used for debt payments. Higher ratios imply a higher debt burden and, therefore, higher risk.

4. **Loan_to_Value_Ratio**: The ratio of the loan amount to the collateral's value (for secured loans). A higher ratio implies more risk to the lender in case of default.

5. **Credit_Utilization_Rate**: The percentage of available credit being used. High utilization rates suggest financial stress, increasing the risk of default.

6. **Recent_Credit_Inquiries**: The number of recent credit applications, indicating credit-seeking behavior. More inquiries could signal financial instability.

7. **Payment_History_Utilities**: The number of late payments on utility bills. Frequent late payments may indicate poor financial management and increased risk.

8. **Financial_Stability_Indicator**: A score (0 to 1) derived from spending and saving patterns. Higher scores indicate better financial stability, lowering the perceived risk.

### Additional Features for Enhanced Robustness

9. **Savings_Cash_Reserves**: The amount of savings available, which serves as a financial buffer. Applicants with higher reserves are typically lower risk.

10. **Credit_Mix**: A score representing the diversity of credit types the applicant has. A higher credit mix indicates experience with credit, which can lower perceived risk.

11. **Employment_Type**: A binary feature (1 for salaried, 0 for self-employed). Salaried employment is seen as more stable and lowers risk.

12. **Monthly_Cash_Flow_Ratio**: The ratio of income left after expenses (0 to 1), with higher values indicating more disposable income and lower risk.

13. **Risk_Level (Target Variable)**: A binary variable (1 for high risk, 0 for low risk) representing the predicted risk of default. This variable is derived from a combination of the other features.

## Dataset Structure

The dataset is split into three parts:
- **Training Set**: 60% of the data, used to train the model.
- **Validation Set**: 20% of the data, used to fine-tune the model's hyperparameters.
- **Test Set**: 20% of the data, used to evaluate model performance on unseen data.

## Getting Started

### Requirements

- Python 3.x
- Pandas
- NumPy
- scikit-learn

### Usage

Run the script `synthetic_data_generator.py` to generate the dataset:

```bash
python synthetic_data_generator.py
```

This will create three CSV files:
- `synthetic_credit_risk_train.csv`
- `synthetic_credit_risk_val.csv`
- `synthetic_credit_risk_test.csv`

Each file contains the synthetic data split into training, validation, and testing sets.

## Metrics Explanation

When building a credit risk model, these evaluation metrics are important:

1. **Accuracy**: The percentage of correct predictions, useful when the classes (high-risk vs. low-risk) are balanced.

2. **Precision**: The percentage of true high-risk predictions out of all high-risk predictions. Helps minimize false positives (approving risky borrowers).

3. **Recall**: The percentage of actual high-risk applicants correctly identified. This minimizes false negatives (missing risky borrowers).

4. **F1 Score**: The harmonic mean of precision and recall, providing a balanced metric when false positives and false negatives are equally important.

5. **AUC-ROC**: Measures the model's ability to distinguish between high-risk and low-risk applicants. A higher AUC indicates better model performance.

6. **Confusion Matrix**: Shows the counts of true positives, false positives, true negatives, and false negatives, giving a full picture of model performance.

## License

This project is licensed under the MIT License.

## Contributing

Feel free to open issues or submit pull requests to improve the synthetic data generator or enhance the feature set.
