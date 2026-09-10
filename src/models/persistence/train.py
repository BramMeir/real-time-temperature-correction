def train_persistence(series):
    """
    "Train" a persistence baseline model. Persistence has no parameters to fit: training simply means
    taking the most recently observed value of the training window, which is then used as the constant
    forecast for the whole horizon.

    Input
    -----
    series: Series with the training observations for the target station, indexed by datetime

    Output
    ------
    last_value: The last observed value of the training window (float)
    """
    if series.empty:
        raise ValueError("Cannot train a persistence model on an empty series")

    last_value = series.iloc[-1]

    return float(last_value)
