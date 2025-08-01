import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


def load_dataset():
    """
    Load the California housing dataset and split into train/test sets.

    Returns:
        X_train (ndarray): Training features
        X_test (ndarray): Testing features
        y_train (ndarray): Training targets
        y_test (ndarray): Testing targets
    """
    data = fetch_california_housing()
    X, y = data.data, data.target
    return train_test_split(X, y, test_size=0.2, random_state=42)


def create_model():
    """
    Create and return a Linear Regression model.

    Returns:
        model (LinearRegression): Untrained Linear Regression model
    """
    return LinearRegression()


def save_model(model, filename):
    """
    Save a trained model to a file using joblib.

    Args:
        model: Trained model object
        filename (str): Path to save the model
    """
    joblib.dump(model, filename)


def load_model(filename):
    """
    Load a trained model from a file using joblib.

    Args:
        filename (str): Path to the saved model file

    Returns:
        model: Loaded model object
    """
    return joblib.load(filename)


def calculate_metrics(y_true, y_pred):
    """
    Calculate R² score and Mean Squared Error for predictions.

    Args:
        y_true (ndarray): Ground truth values
        y_pred (ndarray): Predicted values

    Returns:
        r2 (float): R-squared score
        mse (float): Mean squared error
    """
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    return r2, mse