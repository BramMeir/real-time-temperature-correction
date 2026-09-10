"""
Script: execute_forecast.py

Forecasting with the two-stage regression with SARIMA errors: the spatial part is evaluated on the
neighbours' observations over the horizon, the local part is the SARIMA forecast of the regression
residuals. The local part decays to zero within a few hours, after which the forecast reduces to the
OLS regression of the neighbouring stations.
"""
import pandas as pd
from src.models.regression_sarima_errors.train import train_regression_sarima_errors
from src.evaluation.evaluate_baseline_forecast import evaluate_baseline_forecast


def run_single_forecast(df, target_station, model=None, exog_cols=None, start=None, train_end=None,
                        test_end=None, arima_order=(2, 0, 0), seasonal_order=(1, 0, 1, 24), mode="forecast"):
    """
    Runs a single forecast with a regression on the neighbouring stations plus a SARIMA model of the
    regression residuals. As for ARIMAX and the Random Forest, the neighbours' observations over the
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
    arima_order: Tuple specifying the (p, d, q) parameters for the residual SARIMA (default is (2, 0, 0))
    seasonal_order: Tuple specifying the (P, D, Q, s) parameters for the residual SARIMA (default is (1, 0, 1, 24))
    mode: Mode of operation (kept only for interface compatibility, unused)

    Output
    ------
    mae: Mean Absolute Error on the test set
    mse: Mean Squared Error on the test set
    best_hp: Always None, this model has no hyperparameters to search
    """
    # Print the date range being used
    print(f"Running forecast from {start} to {test_end} with training until {train_end}")

    if exog_cols is None:
        exog_cols = [col for col in df.columns if col != target_station]

    if model is None:
        model = train_regression_sarima_errors(
            df.loc[start:train_end], target_station, exog_cols, arima_order, seasonal_order
        )

    regression, residual_model = model

    # Test index excludes train_end so the forecast starts one step after the training window
    test_index = df.loc[train_end:test_end].index[1:]

    # The neighbour observations are assumed to be known over the forecast horizon; interpolate
    # any short gaps so a missing regressor value does not crash the prediction
    X_test = df.loc[test_index, exog_cols].interpolate(limit_direction="both")
    y_test = df.loc[test_index, target_station]

    # Spatial part: the OLS regression evaluated on the neighbours' observations
    spatial = regression.predict(X_test)

    # Local part: the SARIMA forecast of the residuals, which mean-reverts to zero over the horizon
    local = residual_model.get_forecast(steps=len(test_index)).predicted_mean.values

    predictions = pd.Series(spatial + local, index=test_index)

    mae, rmse = evaluate_baseline_forecast(
        predictions, y_test, model_name="regression_sarima_errors", plot=False
    )
    mse = rmse ** 2

    return mae, mse, None
