import numpy as np
import pandas as pd


def generate_forecast_start(usable, seed, repeat_id, max_history_days, max_horizon):
    """
    Generate a reproducible random forecast start date for a target, ensuring that there is enough historical data for
    training and enough future data for forecasting. A start is drawn again until the outage covers only validated
    observations of the target and the window, history and outage together, holds no long gap of any station.

    Input
    -----
    usable: DataFrame with the hourly index of the target's data and the boolean columns validated and complete, as
      returned by usable_hours in src/utils/datasets.py.
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
    history = pd.Timedelta(days=max_history_days)
    horizon = pd.Timedelta(hours=max_horizon)
    min_date = usable.index.min() + history
    max_date = usable.index.max() - horizon

    if max_date <= min_date:
        raise ValueError("Series too short for chosen configuration")

    # A dataset without validation flags or long gaps accepts the first start, so its starts stay as before
    while True:
        forecast_start = (min_date + (max_date - min_date) * rng.random()).floor('h')
        outage = usable.loc[forecast_start:forecast_start + horizon]
        window = usable.loc[forecast_start - history:forecast_start + horizon]
        if outage["validated"].all() and window["complete"].all():
            return forecast_start
