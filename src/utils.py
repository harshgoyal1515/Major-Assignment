# src/utils.py

import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np


def load_dataset():
    """Load California housing dataset and split into train/test sets."""
    data = fetch_california_housing()
    X, y = data.data, data.target
    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_linear_regression(X_train, y_train):
    """Train LinearRegression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate the model and return R2 and MSE."""
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    return r2, mse


def save_model(model, path="model.joblib"):
    """Save model to disk."""
    joblib.dump(model, path)


def load_model(path="model.joblib"):
    """Load model from disk."""
    return joblib.load(path)


def quantize_params(coefs, intercept):
    """Quantize model parameters to uint8 and return all artifacts."""
    min_val = coefs.min()
    max_val = coefs.max()
    scale = 255 / (max_val - min_val)

    quant_coefs = ((coefs - min_val) * scale).astype(np.uint8)
    quant_intercept = int((intercept - min_val) * scale)

    return quant_coefs, quant_intercept, scale, min_val


def dequantize_params(quant_coefs, quant_intercept, scale, min_val):
    """Dequantize model parameters back to float32."""
    coefs = quant_coefs.astype(np.float32) / scale + min_val
    intercept = quant_intercept / scale + min_val
    return coefs, intercept