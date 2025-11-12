"""
Script: create_supervised.py
Description: Create a supervised learning dataset from time series data. This is necessary for models
             like Random Forests that require fixed input-output pairs (for example, previous 24 hours to predict next hour).

Example usage:
python -m src.data.preprocessing --input ./data/Turku_1H_LI.csv --output ./data/Turku_preprocessed.csv --mode turku
"""
import pandas as pd
import numpy as np


def create_supervised_dataset(df, target_station, previous_time_steps=24, exog_cols=None, exog_lags=0):
    """
    Input
    -----
    df: DataFrame with time series data, indexed by datetime
    target_station: Name of the target station column to predict
    previous_time_steps: Number of previous time steps to include as features (default is 24)
    exog_cols: List of additional exogenous columns to include as features (default is None)
    exog_lags: Number of previous time steps to include for exogenous features (default is 0)

    Output
    ------
    X: DataFrame with features (lagged values and exogenous variables)
    y: Series with target variable to predict
    """
    X_rows = []
    y_rows = []
    idx = []

    for i in range(max(previous_time_steps, exog_lags), len(df)):
        row = []

        # Target lags
        target_lags = df[target_station].iloc[i - previous_time_steps:i].values
        row.extend(target_lags)

        # Exogenous features (current + lags)
        if exog_cols:
            for col in exog_cols:
                # Current value
                row.append(df[col].iloc[i])

                # Lags of this exogenous variable
                if exog_lags > 0:
                    exog_lag_values = df[col].iloc[i - exog_lags:i].values
                    row.extend(exog_lag_values)

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
        for col in exog_cols:
            # Current value
            columns.append(col)

            if exog_lags > 0:
                columns += [f"{col}_{j}" for j in range(exog_lags, 0, -1)]

    columns += ['hour_of_day_sin', 'hour_of_day_cos']

    X = pd.DataFrame(X_rows, index=idx, columns=columns)
    y = pd.Series(y_rows, index=idx, name=target_station)

    return X, y
