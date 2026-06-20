# Customer Churn Prediction Using Decision Tree

## Overview

This project predicts whether a telecom customer is likely to churn (leave the company) or stay based on customer account and billing information. A Decision Tree Classifier is used to analyze customer behavior and make predictions.

---

## Objective

The objective of this project is to identify customers who are at risk of leaving the telecom service, helping businesses improve customer retention and reduce churn.

---

## Dataset

Dataset: Telco Customer Churn Dataset

Source: Kaggle

---

## Features Used

- Tenure
- Contract Type
- Monthly Charges
- Total Charges

---

## Target Variable

- Churn
  - 0 = Customer Will Stay
  - 1 = Customer Will Churn

---

## Technologies Used

- Python
- Pandas
- Scikit-Learn
- Matplotlib

---

## Project Workflow

1. Load Dataset
2. Data Cleaning and Preprocessing
3. Handle Missing Values
4. Encode Categorical Variables
5. Feature Selection
6. Train-Test Split
7. Train Decision Tree Classifier
8. Predict Customer Churn
9. Evaluate Model Performance
10. Visualize ROC Curve

---

## Model Evaluation

The model performance is evaluated using:

- Accuracy Score
- ROC Curve
- AUC Score

---

## Sample Output

```text
========== CUSTOMER CHURN PREDICTION ==========

Accuracy Score : 72.60%
AUC Score      : 0.66

Customer Information

   tenure  Contract  MonthlyCharges  TotalCharges
0      34         1           56.95        1889.5

Stay Probability : 100.00%
Churn Probability: 0.00%

Prediction: Customer Will Stay

==============================================
```

---

## ROC Curve

The ROC Curve is used to visualize the model's ability to distinguish between customers who will churn and those who will stay.

---

## Learning Outcomes

- Data Preprocessing
- Feature Engineering
- Classification using Decision Tree
- Model Evaluation
- ROC Curve Analysis
- Customer Churn Prediction

---

## Future Improvements

- Random Forest Classifier
- XGBoost Classifier
- Streamlit Web Application
- Interactive Dashboard
- Real-Time Customer Prediction System

---

## Author

Suprajaa Sankari G

Information Technology Student | Aspiring Data Scientist
