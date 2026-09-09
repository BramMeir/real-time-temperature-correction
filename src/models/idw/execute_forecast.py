import pandas as pd
from src.models.idw.train import train_idw
from src.evaluation.evaluate_baseline_forecast import evaluate_baseline_forecast


def run_single_forecast(df, target_station, model=None, exog_cols=None, start=None, train_end=None,
                        test_end=None, mode="forecast", coords=None, power=2, k=None):
    """
    Runs a single forecast using inverse distance weighting (IDW). The forecast for each timestamp is the weighted
    sum of the neighbour stations' observed values at that timestamp, with fixed weights based on the distance
    between each neighbour and the target station. This assumes the neighbour observations are known over the
    forecast horizon, the same assumption the exogenous variables make for ARIMAX and the Random Forest.

    IDW does not learn from the training window, so start and train_end are only used to define where the test
    window begins (the day after train_end): the test index is df.loc[train_end:test_end] for the target station,
    excluding train_end itself. mode is kept only for interface compatibility with the other models: IDW has no
    hyperparameter search, so every mode behaves like "forecast".

    Input
    -----
    df: DataFrame with a datetime index, one column per station (target station plus neighbour columns)
    target_station: The target station for forecasting
    model: Optional pre-built weights dict, as returned by train_idw (the comparison experiment passes this, so
        coords does not need to be given in that case). If None, a weights dict is built from coords.
    exog_cols: List of neighbour station column names (the stations that get interpolated)
    start: Unused by IDW, kept for interface compatibility with the other models
    train_end: End date of the training window; the forecast starts right after this date
    test_end: End date of the test set
    mode: Unused by IDW, kept for interface compatibility with the other models
    coords: Dict mapping station_name to a (latitude, longitude) tuple, required if model is None
    power: Power parameter of the inverse distance weighting, used if model is None (default is 2)
    k: Number of nearest neighbour stations to use, used if model is None (default is None, meaning all of them)

    Output
    ------
    mae: Mean Absolute Error on the test set
    mse: Mean Squared Error on the test set
    best_hp: Always None, kept for interface compatibility with the other models
    """
    if model is None:
        if coords is None:
            raise ValueError("IDW needs the station coordinates: pass either a pre-built `model` weights dict "
                             "or `coords` to build one from")
        model = train_idw(coords, target_station, exog_cols, power=power, k=k)

    # Test window starts one step after train_end, so the forecast does not use the training window itself
    y_test = df.loc[train_end:test_end, target_station].iloc[1:]
    exog_test = df.loc[train_end:test_end, list(model.keys())].iloc[1:]

    # Renormalise the weights per row over the neighbours available at that timestamp, so a partially
    # missing row still yields a prediction
    available = exog_test.notna()
    weights_row = available.mul(pd.Series(model), axis=1)
    weights_row = weights_row.div(weights_row.sum(axis=1), axis=0)

    predictions = (exog_test.fillna(0) * weights_row).sum(axis=1)
    predictions[weights_row.sum(axis=1) == 0] = float("nan")

    mae, rmse = evaluate_baseline_forecast(predictions, y_test, model_name="idw", plot=False)
    mse = rmse ** 2

    return mae, mse, None
