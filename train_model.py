import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load data
data = pd.read_csv("data/churn.csv")

# ===== ENCODING TEXT TO NUMBER =====

# Gender
data["gender"] = data["gender"].map({
    "Female": 0,
    "Male": 1
})

# Yes / No Columns
yes_no_cols = ["Partner", "Dependents", "Churn"]

for col in yes_no_cols:
    data[col] = data[col].map({
        "No": 0,
        "Yes": 1
    })

# Convert TotalCharges to number
data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")

# Fill missing values
data = data.fillna(0)

# ===== SELECT FEATURES =====
X = data[[
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]]

y = data["Churn"]

# ===== TRAIN MODEL =====
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier()
model.fit(X_train, y_train)

pickle.dump(model, open("model/churn_model.pkl", "wb"))

print("Model trained successfully with encoded data!")

