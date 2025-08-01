# src/predict.py

import joblib
import numpy as np

def predict(X):
    """
    Predict using the trained linear regression model.

    Parameters:
    - X: ndarray of shape (n_samples, n_features)

    Returns:
    - predictions: ndarray of shape (n_samples,)
    """
    model = joblib.load("model.joblib")
    return model.predict(X)

if __name__ == "__main__":
    # Example usage for manual testing
    X_sample = np.random.rand(5, 8)  # Random 5 samples, 8 features (California housing format)
    preds = predict(X_sample)
    print("Sample predictions:")
    print(preds)