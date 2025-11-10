"""
Script: create_supervised.py
Description: Create a supervised learning dataset from time series data. This is necessary for models
             like Random Forests that require fixed input-output pairs (for example, previous 24 hours to predict next hour).

Example usage:
python -m src.data.preprocessing --input ./data/Turku_1H_LI.csv --output ./data/Turku_preprocessed.csv --mode turku
"""
import pandas as pd


def create_supervised_dataset(df, target_station, previous_time_steps=24, exog_cols=None):
    """
    Input
    -----
    df: DataFrame with time series data, indexed by datetime
    target_station: Name of the target station column to predict
    previous_time_steps: Number of previous time steps to include as features (default is 24)
    exog_cols: List of additional exogenous columns to include as features (default is None)

    Output
    ------
    X: DataFrame with features (lagged values and exogenous variables)
    y: Series with target variable to predict
    """
    X_rows = []
    y_rows = []
    idx = []

    for i in range(previous_time_steps, len(df)):
        lags = df[target_station].iloc[i - previous_time_steps:i].values
        row = list(lags)
        if exog_cols:
            exog = df[exog_cols].iloc[i].values
            row.extend(exog)

        X_rows.append(row)
        y_rows.append(df[target_station].iloc[i])
        idx.append(df.index[i])

    X = pd.DataFrame(X_rows, index=idx,
                     columns=[f"lag_{j}" for j in range(previous_time_steps, 0, -1)] + (exog_cols or []))
    y = pd.Series(y_rows, index=idx, name=target_station)
    return X, y
