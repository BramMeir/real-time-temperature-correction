"""
- generate_forecast_start function to create a reproducible random forecast start date for the experiments.
  This function ensures that the forecast start date is chosen such that there is enough historical data for training
  and enough future data for forecasting, based on the specified maximum history and horizon.
"""
import numpy as np
import pandas as pd


def generate_forecast_start(series, seed, repeat_id, max_history_days, max_horizon):
    """
    Generate a reproducible random forecast start date for a given time series, ensuring
    that there is enough historical data for training and enough future data for forecasting.

    Input
    -----
    series: Target time series with datetime index.
    seed: Base seed for reproducibility.
    repeat_id: Repeat number to ensure different samples.
    max_history_days: Maximum number of days of historical data to use for training.
    max_horizon: Largest forecast horizon (hours).

    Output
    ------
    forecast_start: Start timestamp of the forecast period.
    """
    # Create a random number generator with a seed that combines the base seed and repeat ID
    rng = np.random.default_rng(seed + repeat_id)

    # Minimum date is the earliest timestamp plus the maximum number of training days, to ensure enough history for training
    # Maximum date is the latest timestamp minus the maximum forecast horizon, to ensure enough future data for forecasting
    min_date = series.index.min() + pd.Timedelta(days=max_history_days)
    max_date = series.index.max() - pd.Timedelta(hours=max_horizon)

    if max_date <= min_date:
        raise ValueError("Series too short for chosen configuration")

    random_fraction = rng.random()

    forecast_start = min_date + (max_date - min_date) * random_fraction
    forecast_start = forecast_start.floor('h')

    return forecast_start
