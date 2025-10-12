import numpy as np


def create_dataset(series, lookback=4320, horizon=6):
    """
    series: Time series data
    lookback: Number of past points to consider
    horizon: Number of future points to predict
    """
    if len(series) < lookback + horizon:
        raise ValueError("Series is too short for the specified lookback and horizon")

    X, y = [], []

    # Divide the series into non-overlapping segments of length lookback + horizon
    for i in range(0, len(series) - lookback - horizon + 1, lookback + horizon):
        X.append(series[i: i + lookback])
        y.append(series[i + lookback: i + lookback + horizon])

    return np.array(X), np.array(y)
