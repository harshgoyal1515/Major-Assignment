# tests/test_train.py

import os
import joblib
from sklearn.linear_model import LinearRegression
from src.train import train_model

def test_model_training_and_saving():
    # Run training and get outputs
    model, r2, mse = train_model()

    # Test 1: R² score should be reasonable
    assert r2 > 0.5, f"R² score too low: {r2}"

    # Test 2: Model file should exist
    assert os.path.exists("model.joblib"), "model.joblib was not created"

    # Test 3: Model should be a LinearRegression instance
    assert isinstance(model, LinearRegression), "Model is not LinearRegression"

    # Test 4: Model should have learned coefficients
    assert hasattr(model, "coef_"), "Model has no 'coef_' attribute"
    assert hasattr(model, "intercept_"), "Model has no 'intercept_' attribute"

    # Optional: Reload from file and verify it again
    reloaded_model = joblib.load("model.joblib")
    assert isinstance(reloaded_model, LinearRegression), "Reloaded model is invalid"
