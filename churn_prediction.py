# IMPORT LIBRARIES
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

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

# ENCODE CONTRACT COLUMN MANUALLY
contract_map = {
    "Month-to-month": 0,
    "One year": 1,
    "Two year": 2
}

df["Contract"] = df["Contract"].map(contract_map)

# ENCODE CHURN COLUMN MANUALLY
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

# CREATE DECISION TREE MODEL
model = DecisionTreeClassifier(
    random_state=42
)

# TRAIN MODEL
model.fit(X_train, y_train)

# PREDICT TEST DATA
y_pred = model.predict(X_test)

# ACCURACY
print("Accuracy:", accuracy_score(y_test, y_pred))

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

# PREDICT CUSTOMER
prediction = model.predict(customer)

print("\nCustomer Details")
print(customer)

if prediction[0] == 1:
    print("\nPrediction: Customer Will Churn")
else:
    print("\nPrediction: Customer Will Stay")