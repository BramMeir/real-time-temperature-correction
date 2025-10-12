"""
Functionality:
- evaluate_forecasts: Function to compute evaluation metrics (MAE and MSE) for forecasts.
"""
from sklearn.metrics import mean_absolute_error, mean_squared_error


def evaluate_forecasts(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred)
    }
