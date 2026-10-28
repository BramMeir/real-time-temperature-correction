import pandas as pd
import numpy as np


def create_supervised_dataset(df, target_station, previous_time_steps=24, exog_cols=None):
    """
    Build the fixed input-output pairs of the tabular models (RF, MLP): for every time step, the target's
    previous time steps, the exogenous columns at that time step and the hour of day as a sine and a cosine.

    Input
    -----
    df: DataFrame with time series data, indexed by datetime
    target_station: Name of the target station column to predict
    previous_time_steps: Number of previous time steps to include as features (default is 24)
    exog_cols: List of additional exogenous columns to include as features, at the current time step only
      (default is None)

    Output
    ------
    X: DataFrame with features (lagged values and exogenous variables)
    y: Series with target variable to predict
    """
    X_rows = []
    y_rows = []
    idx = []

    for i in range(previous_time_steps, len(df)):
        row = []

        # Target lags
        target_lags = df[target_station].iloc[i - previous_time_steps:i].values
        row.extend(target_lags)

        # Exogenous features at the current time step
        if exog_cols:
            for col in exog_cols:
                row.append(df[col].iloc[i])

        # Add a cyclical encoding for hour of day
        hour_of_day = df.index[i].hour
        row.append(np.sin(2 * np.pi * hour_of_day / 24))
        row.append(np.cos(2 * np.pi * hour_of_day / 24))

        X_rows.append(row)
        y_rows.append(df[target_station].iloc[i])
        idx.append(df.index[i])

    # Column names
    columns = [f"{target_station}_lag_{j}" for j in range(previous_time_steps, 0, -1)]
    if exog_cols:
        columns += exog_cols

    columns += ['hour_of_day_sin', 'hour_of_day_cos']

    X = pd.DataFrame(X_rows, index=idx, columns=columns)
    y = pd.Series(y_rows, index=idx, name=target_station)

    return X, y
