# src/train.py

import joblib
import os
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

def train_model():
    # Load dataset
    data = fetch_california_housing()
    X, y = data.data, data.target

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    # Save model
    model_path = "model.joblib"
    joblib.dump(model, model_path)

    # Compute model size
    model_size_kb = os.path.getsize(model_path) / 1024

    # Output results
    print("\n✅ Model Evaluation")
    print(f"R² Score:             {r2:.4f}")
    print(f"MSE:                  {mse:.4f}")
    print(f"Model Size:           {model_size_kb:.1f} KB")

    return model, r2, mse

if __name__ == "__main__":
    train_model()
