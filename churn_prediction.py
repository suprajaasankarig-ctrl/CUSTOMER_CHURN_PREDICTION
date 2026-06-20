# IMPORT LIBRARIES

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_curve,
    roc_auc_score
)

# LOAD DATASET

df = pd.read_csv(
    r"C:\Users\supra\Documents\Customer_Churn_Prediction\data\WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

# REMOVE CUSTOMER ID

df.drop("customerID", axis=1, inplace=True)

# CONVERT TOTALCHARGES TO NUMERIC

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# FILL MISSING VALUES

df["TotalCharges"] = df["TotalCharges"].fillna(
    df["TotalCharges"].median()
)

# ENCODE CONTRACT COLUMN

contract_map = {
    "Month-to-month": 0,
    "One year": 1,
    "Two year": 2
}

df["Contract"] = df["Contract"].map(contract_map)

# ENCODE CHURN COLUMN

churn_map = {
    "No": 0,
    "Yes": 1
}

df["Churn"] = df["Churn"].map(churn_map)

# FEATURES (INPUTS)

X = df[
    [
        "tenure",
        "Contract",
        "MonthlyCharges",
        "TotalCharges"
    ]
]

# TARGET (OUTPUT)

y = df["Churn"]

# SPLIT DATASET

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# CREATE MODEL

model = DecisionTreeClassifier(
    random_state=42
)

# TRAIN MODEL

model.fit(X_train, y_train)

# PREDICT TEST DATA

y_pred = model.predict(X_test)

# ACCURACY

accuracy = accuracy_score(
    y_test,
    y_pred
)

# AUC SCORE

y_prob = model.predict_proba(X_test)[:, 1]

auc_score = roc_auc_score(
    y_test,
    y_prob
)

# NEW CUSTOMER

customer = pd.DataFrame(
    [[34, 1, 56.95, 1889.50]],
    columns=[
        "tenure",
        "Contract",
        "MonthlyCharges",
        "TotalCharges"
    ]
)

# CUSTOMER PREDICTION

prediction = model.predict(customer)

probability = model.predict_proba(customer)

stay_prob = probability[0][0] * 100
churn_prob = probability[0][1] * 100

# OUTPUT

print("\n========== CUSTOMER CHURN PREDICTION ==========")

print(f"\nAccuracy Score : {accuracy * 100:.2f}%")
print(f"AUC Score      : {auc_score:.2f}")

print("\nCustomer Information")
print(customer)

print(f"\nStay Probability : {stay_prob:.2f}%")
print(f"Churn Probability: {churn_prob:.2f}%")

if prediction[0] == 1:
    print("\nPrediction: Customer Will Churn")
else:
    print("\nPrediction: Customer Will Stay")

print("\n==============================================")

# ROC CURVE

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_prob
)

plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    label=f"AUC = {auc_score:.2f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Customer Churn Prediction")
plt.legend()

plt.show()
