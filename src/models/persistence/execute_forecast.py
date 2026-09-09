import pandas as pd
from src.models.persistence.train import train_persistence
from src.evaluation.evaluate_baseline_forecast import evaluate_baseline_forecast


def run_single_forecast(df, target_station, model=None, exog_cols=None, start=None, train_end=None,
                        test_end=None, mode="forecast"):
    """
    Runs a single forecast using the persistence baseline: the last observed value of the training window
    is held constant over the whole forecast horizon. Persistence has no hyperparameters to search, so
    `mode` is kept only for interface compatibility with the other models and every mode behaves like
    "forecast".

    Input
    -----
    df: DataFrame with the complete dataset, one column per station (datetime index, hourly frequency)
    target_station: The target station for forecasting
    model: Optional pre-trained model to use for forecasting (if None, a new model will be trained). For
        persistence, the "model" is simply the last observed value of the training window (a float)
    exog_cols: List of neighbour station column names (unused, accepted only for interface compatibility)
    start: Start date for the training window
    train_end: End date for the training window / start of the forecast horizon
    test_end: End date for the test set
    mode: Kept only for interface compatibility with the other models (unused, always behaves like "forecast")

    Output
    ------
    mae: Mean Absolute Error on the test set
    mse: Mean Squared Error on the test set
    best_hp: Always None, keeps the return shape consistent with the other models
    """
    # Print the date range being used
    print(f"Running forecast from {start} to {test_end} with training until {train_end}")

    if model is None:
        y_train = df.loc[start:train_end, target_station]
        model = train_persistence(y_train)

    # The test window starts one step after the training window, so exclude train_end itself
    y_test = df.loc[train_end:test_end, target_station].iloc[1:]

    predictions = pd.Series(model, index=y_test.index)

    mae, rmse = evaluate_baseline_forecast(predictions, y_test, model_name="persistence", plot=False)
    mse = rmse ** 2

    return mae, mse, None
