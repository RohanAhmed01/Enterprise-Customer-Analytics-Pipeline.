

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler


# STEP 1: RAW DATA INGESTION (Simulating a messy client database dump)
print(" Step 1: Ingesting raw corporate data rows...")

raw_corporate_data = {
    "Account_ID": ["ACC_101", "ACC_102", "ACC_103", "ACC_104", "ACC_105", "ACC_106"],
    "Ad_Spend": ["$5,000", "$2,500", "$12,000", "$8,000", np.nan, "$15,000"],
    "Platform_Engagement_Hours": [120, 45, 310, 190, 85, 420],
    "Support_Tickets_Logged": [1, 5, 0, 2, 6, 1],
    "Quarterly_Revenue": [25000, 11000, 58000, 39000, 15000, 72000]
}

df = pd.DataFrame(raw_corporate_data)


# STEP 2: ENTERPRISE DATA CLEANING & PREPROCESSING
print(" Step 2: Running data sanitization and median imputation...")

# Clean the "Ad_Spend" column (Remove '$' and ',', convert to float numeric type)
df["Ad_Spend"] = pd.to_numeric(df["Ad_Spend"].astype(str).str.replace(r'[\$,]', '', regex=True), errors='coerce')

# Calculate the median of the valid ad spend numbers and fill the missing NaN value
median_spend = df["Ad_Spend"].median()
df["Ad_Spend"] = df["Ad_Spend"].fillna(median_spend)

# STEP 3: FEATURE ENGINEERING (Creating targets for ML models)
print("⚙️ Step 3: Engineering corporate target labels...")

# Create a binary classification label column named "Is_High_Value".
# If "Quarterly_Revenue" is greater than $30,000, assign it a 1, otherwise 0.
df["Is_High_Value"] = np.where(df["Quarterly_Revenue"] > 30000, 1, 0)

# STEP 4: REGRESSION PIPELINE (Predicting Next-Quarter Revenue)
print(" Step 4: Deploying Multi-Variable Linear Regression Model...")

# Features for predicting revenue: Ad_Spend and Platform_Engagement_Hours
X_reg = df[["Ad_Spend", "Platform_Engagement_Hours"]].values
y_reg = df["Quarterly_Revenue"].values

# Initialize and fit the Linear Regression model
reg_model = LinearRegression()
reg_model.fit(X_reg, y_reg)

# Calculate the R^2 Score of this regression model
reg_r2 = reg_model.score(X_reg, y_reg)



# STEP 5: CLASSIFICATION PIPELINE (Predicting VIP High-Value Accounts)
print(" Step 5: Deploying Logistic Regression Classifier...")

# Features for classification: Platform_Engagement_Hours and Support_Tickets_Logged
X_clf = df[["Platform_Engagement_Hours", "Support_Tickets_Logged"]].values
y_clf = df["Is_High_Value"].values

# Initialize and fit the Logistic Regression model
clf_model = LogisticRegression()
clf_model.fit(X_clf, y_clf)

# Calculate the accuracy score of the classifier
clf_accuracy = clf_model.score(X_clf, y_clf)

# STEP 6: PRODUCTION FORECASTING & DEPLOYMENT TEST
print("\n Step 6: Pipeline operational! Running production forecast query...")

# A new prospective client comes in:
# They will spend $7,500 on ads and have 175 engagement hours.
# They have logged 2 support tickets.
new_client_reg_features = np.array([[7500, 175]])
new_client_clf_features = np.array([[175, 2]])

# Run the regression model prediction for this client
predicted_revenue = reg_model.predict(new_client_reg_features)

# Run the classification model prediction and probability for this client
predicted_class = clf_model.predict(new_client_clf_features)
predicted_proba = clf_model.predict_proba(new_client_clf_features)[0]



# STEP 7: EXECUTIVE DASHBOARD OUTPUT

print("\n" + "="*60)
print("              GLOBAL DATA OPERATIONS REPORT                  ")
print("="*60)
print(f" Regression Model Reliability (R² Score): {reg_r2:.4f}")
print(f" Classification Model Accuracy Score:    {clf_accuracy * 100:.1f}%")
print("-" * 60)
print(" PROSPECTIVE CLIENT ASSIGNMENT ANALYSIS:")
print(f"   -> Predicted Revenue Yield:            ${predicted_revenue[0]:,.2f}")

if predicted_class[0] == 1:
    print(f"   -> Status Flag:                         [ HIGH-VALUE ACCOUNT]")
    print(f"   -> Assignment Confidence:               {predicted_proba[1] * 100:.1f}% probability")
else:
    print(f"   -> Status Flag:                         [ STANDARD ACCOUNT]")
    print(f"   -> Assignment Confidence:               {predicted_proba[0] * 100:.1f}% probability")
print("="*60 + "\n")
