# tests/test_train.py

import sys
import os

#  This line adds the parent directory to Python's module path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import joblib
from sklearn.linear_model import LinearRegression
from src.train import train_model

def test_model_training_and_saving():
    r2 = train_model()
    assert r2 > 0.5, f"R² score too low: {r2}"

    assert os.path.exists("model.joblib"), "Model file not found"
    model = joblib.load("model.joblib")

    assert isinstance(model, LinearRegression), "Model is not LinearRegression"
    assert hasattr(model, "coef_"), "Model lacks coef_ attribute"
