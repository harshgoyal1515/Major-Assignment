# src/predict.py

import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# Load model
model = joblib.load("model.joblib")

# Load data
data = fetch_california_housing()
X, y = data.data, data.target
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Predict
predictions = model.predict(X_test)

# Show sample predictions
print("Sample predictions:")
print(predictions[:5])
