from src.models.climatology.train import train_hourly_climatology
from src.evaluation.evaluate_baseline_forecast import evaluate_baseline_forecast


def run_single_forecast(df, target_station, model=None, exog_cols=None, start=None, train_end=None,
                        test_end=None, mode="forecast"):
    """
    Runs a single forecast using the hourly climatology baseline: every test timestamp is predicted
    with the mean observed temperature for its hour of day over the training window.

    The climatology has no hyperparameter search, so every value of `mode` behaves like "forecast" -
    the argument is only kept for interface compatibility with the other models.

    Input
    -----
    df: DataFrame with the complete dataset, one column per station, indexed by datetime
    target_station: The target station for forecasting
    model: Optional pre-trained climatology (Series indexed by hour of day) to reuse (if None, a new
        one will be trained)
    exog_cols: List of neighbour station column names, unused by the climatology (kept for interface
        compatibility)
    start: Start date of the training window
    train_end: End date of the training window / start of the forecast horizon
    test_end: End date of the forecast horizon
    mode: Kept only for interface compatibility with the other models; has no effect

    Output
    ------
    mae: Mean Absolute Error on the test set
    mse: Mean Squared Error on the test set
    best_hp: Always None, keeps the return shape of the other models
    """
    # Print the date range being used
    print(f"Running forecast from {start} to {test_end} with training until {train_end}")

    y_train = df.loc[start:train_end, target_station]

    if model is None:
        model = train_hourly_climatology(y_train)

    # Forecast horizon starts one step after the training window
    y_test = df.loc[train_end:test_end, target_station].iloc[1:]

    predictions = model.reindex(y_test.index.hour)
    predictions.index = y_test.index

    mae, rmse = evaluate_baseline_forecast(predictions, y_test, y_train=y_train, model_name="climatology", plot=False)
    mse = rmse ** 2

    return mae, mse, None
