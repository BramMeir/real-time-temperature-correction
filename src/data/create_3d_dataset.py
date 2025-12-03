import pandas as pd
import numpy as np


def create_3d_dataset(df, target_station, previous_time_steps=24, exog_cols=None):
    """
    Creates a 3D dataset for TCN models: (Samples, Timesteps, Features).

    Input
    -----
    df: DataFrame with time series data, indexed by datetime.
    target_station: Name of the target station column to predict.
    previous_time_steps: The lookback window (L) for the TCN (max number of past time steps to use).
    exog_cols: List of exogenous columns (neighboring stations).

    Output
    ------
    X: 3D Numpy Array of shape (Samples, Timesteps, Features)
    y: 1D Numpy Array of shape (Samples,)
    indices: The datetime index corresponding to the targets y
    """
    # 1. Prepare the Feature Matrix
    # We want a DataFrame that contains ALL features for a single time step
    feature_df = pd.DataFrame()

    # A. Add the target station (autoregressive feature)
    feature_df[target_station] = df[target_station]

    # B. Add exogenous features (other stations)
    if exog_cols:
        for col in exog_cols:
            feature_df[col] = df[col]

    # C. Add cyclical time encoding (calculated for the whole index at once)
    hour_of_day = df.index.hour
    feature_df['sin_time'] = np.sin(2 * np.pi * hour_of_day / 24)
    feature_df['cos_time'] = np.cos(2 * np.pi * hour_of_day / 24)

    # Convert to numpy array for faster slicing
    data_values = feature_df.values
    target_values = df[target_station].values

    # 2. Create the Sliding Window (3D Structure)
    X, y, indices = [], [], []

    # Iterate starting from 'previous_time_steps' up to the end
    # We stop at len(df) because we are predicting the current value using past data
    for i in range(previous_time_steps, len(df)):

        # The input X is the history from (i - L) up to (i) NOT including i
        # Shape: (previous_time_steps, number_of_features)
        X_window = data_values[i - previous_time_steps:i]

        # The target y is the value at the current step i
        y_value = target_values[i]

        X.append(X_window)
        y.append(y_value)
        indices.append(df.index[i])

    # Convert lists to numpy arrays
    X = np.array(X)
    y = np.array(y)
    indices = pd.to_datetime(indices)

    print("First rows of created 3D dataset:")
    print("X[0]:", X[0])
    print("y[0]:", y[0])

    print(f"Dataset created. Shape X: {X.shape}, Shape y: {y.shape}")
    return X, y, indices
