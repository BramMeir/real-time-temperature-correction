import numpy as np
import pandas as pd


def add_time_features(exog_df):
    """
    Expands the exogenous DataFrame with time-based features (hour of day and day of year) using sine and cosine transformations.
    """
    index = exog_df.index

    # Extract hour of day and day of year
    hour = index.hour
    day_of_year = index.dayofyear

    # Create sine and cosine transformations for hour of day and day of year
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)

    day_sin = np.sin(2 * np.pi * day_of_year / 365)
    day_cos = np.cos(2 * np.pi * day_of_year / 365)

    # Create a new DataFrame for the time features
    time_features = pd.DataFrame({
        'hour_sin': hour_sin,
        'hour_cos': hour_cos,
        'day_sin': day_sin,
        'day_cos': day_cos
    }, index=index)

    # Concatenate the time features with the original exogenous DataFrame
    exog_df_expanded = pd.concat([exog_df, time_features], axis=1)

    return exog_df_expanded
