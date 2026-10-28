import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def forecast_errors(observed, predicted):
    """
    MAE and MSE of a reconstruction, over the hours at which the target was observed.

    Input
    -----
    observed: Series with the observed values over the outage, indexed by datetime
    predicted: Series with the reconstructed values, on the same index

    Output
    ------
    mae: Mean Absolute Error
    mse: Mean Squared Error
    """
    # The target's own record has gaps inside some outages, and those hours cannot be scored
    combined = pd.concat([observed.rename("observed"), predicted.rename("predicted")], axis=1).dropna()

    return (mean_absolute_error(combined["observed"], combined["predicted"]),
            mean_squared_error(combined["observed"], combined["predicted"]))
