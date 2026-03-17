import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load data
try:
    df = pd.read_csv('50_startups_dataset.csv')
    print("Data loaded successfully!")
except FileNotFoundError:
    print(" Error: '50_startups_dataset.csv' not found in this folder.")
    exit()

# Preprocessing
df_encoded = pd.get_dummies(df, columns=['State'], drop_first=True)
X = df_encoded.drop('Profit', axis=1)
y = df_encoded['Profit']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training
model = LinearRegression()
model.fit(X_train, y_train)

# Output Results
y_pred = model.predict(X_test)
print(f"\n--- MODEL RESULTS ---")
print(f"R-squared Score: {r2_score(y_test, y_pred):.4f}")
print(f"Model Coefficients: {dict(zip(X.columns, model.coef_.round(2)))}")