import pandas as pd
from src.models.regsarima.train import train_regression_sarima_errors, RESIDUAL_ORDER, RESIDUAL_SEASONAL_ORDER
from src.evaluation.metrics import forecast_errors


def forecast_two_stage(model, X):
    """
    Forecast a trained (regression, residual_model) over the hours of X, the neighbours' observations
    from the first hour after the training window on.

    Output
    ------
    predictions: Series with the forecast for every hour of X
    spatial: Series with the regression part alone, the forecast without the residual model
    """
    regression, residual_model = model

    # Spatial part: the OLS regression evaluated on the neighbours' observations
    spatial = pd.Series(regression.predict(X), index=X.index)

    # Local part: the SARIMA forecast of the residuals, which mean-reverts to zero over the horizon
    local = residual_model.get_forecast(steps=len(X)).predicted_mean.values

    return spatial + local, spatial


def run_single_forecast(df, target_station, model=None, exog_cols=None, start=None, train_end=None,
                        test_end=None, arima_order=RESIDUAL_ORDER, seasonal_order=RESIDUAL_SEASONAL_ORDER, mode="forecast"):
    """
    Runs a single forecast with a regression on the neighbouring stations plus a SARIMA model of the
    regression residuals. As for SARIMAX and the random forest, the neighbours' observations over the
    forecast horizon are assumed to be known.

    This model has no hyperparameter search, so every mode behaves the same as "forecast" — the mode
    argument is only kept for interface compatibility with the other models.

    Input
    -----
    df: DataFrame with a datetime index, one column per station, hourly frequency
    target_station: The target station for forecasting
    model: Optional pre-trained (regression, residual_model) tuple (if None, a new model will be trained)
    exog_cols: List of neighbouring station column names used as regressors (if None, every column
              of df except the target station is used)
    start: Start date for the training window
    train_end: End date for the training window / start of the forecast horizon
    test_end: End date for the forecast horizon
    arima_order: Tuple specifying the (p, d, q) parameters for the residual SARIMA (default is RESIDUAL_ORDER)
    seasonal_order: Tuple specifying the (P, D, Q, s) parameters for the residual SARIMA (default is RESIDUAL_SEASONAL_ORDER)
    mode: Mode of operation (kept only for interface compatibility, unused)

    Output
    ------
    mae: Mean Absolute Error on the test set
    mse: Mean Squared Error on the test set
    best_hp: Always None, this model has no hyperparameters to search
    predictions: Series with the forecast for every hour of the test window
    """
    # Print the date range being used
    print(f"Running forecast from {start} to {test_end} with training until {train_end}")

    if exog_cols is None:
        exog_cols = [col for col in df.columns if col != target_station]

    if model is None:
        model = train_regression_sarima_errors(
            df.loc[start:train_end], target_station, exog_cols, arima_order, seasonal_order
        )

    # Test index excludes train_end so the forecast starts one step after the training window
    test_index = df.loc[train_end:test_end].index[1:]

    # The neighbour observations are assumed to be known over the forecast horizon; interpolate
    # any short gaps so a missing regressor value does not crash the prediction
    X_test = df.loc[test_index, exog_cols].interpolate(limit_direction="both")
    y_test = df.loc[test_index, target_station]

    predictions, _ = forecast_two_stage(model, X_test)

    mae, mse = forecast_errors(y_test, predictions)

    return mae, mse, None, predictions
