"""
Script: confidence_score.py

Confidence scoring for the two-stage regression with SARIMA errors model, analogous to
src/models/arima/confidence_score.py for ARIMAX but derived empirically instead of from
statsmodels' conf_int(): the spatial stage is a plain sklearn LinearRegression, which exposes
no prediction-interval machinery, and even for the SARIMA stage a parametric interval on the
residual alone would ignore the spatial stage's error. Both stages' error is instead measured
jointly on a held-out calibration window and turned into an empirical prediction interval.

The width comes from the conformal rank of the absolute calibration errors, not from bootstrapped
percentiles. Experiment P1.5 (src/scripts/interval_coverage_experiment.py) measured the bootstrap
at 0.87 coverage against a nominal 0.95: resampling pins down the percentile of the calibration
sample precisely, but that percentile is itself biased narrow as a bound on a new error. The rank
instead leaves k of the n + 1 gaps between the sorted errors inside the band, which is the
quantity that actually governs coverage.

Functions:
- conformal_interval_width: Symmetric empirical prediction interval from an error sample.
- two_stage_forecast_with_confidence_score: Fit the two-stage model, holding out a calibration
  window to derive a MAE-based confidence score and an interval width, then forecast the actual
  test window.
"""
import numpy as np
import pandas as pd
from src.evaluation.interval_coverage import conformal_rank
from src.models.regression_sarima_errors.train import train_regression_sarima_errors
from src.models.regression_sarima_errors.execute_forecast import run_single_forecast


def conformal_interval_width(errors, ci_level=0.95):
    """
    Symmetric empirical prediction interval from a sample of forecast errors, at the conformal
    rank of their absolute values.

    Input
    -----
    errors: Array-like of forecast errors (actual - predicted) from a held-out calibration window
    ci_level: Confidence level of the interval (default is 0.95)

    Output
    ------
    width: Width of the interval
    lower, upper: Interval bounds relative to the forecast
    """
    absolute_errors = np.sort(np.abs(np.asarray(errors)))
    half_width = absolute_errors[conformal_rank(absolute_errors.size, ci_level) - 1]

    return 2 * half_width, -half_width, half_width


def two_stage_forecast_with_confidence_score(
        df, target_station, exog_cols=None, hours_to_forecast=48, arima_order=(2, 0, 0),
        seasonal_order=(1, 0, 1, 24), max_iter=1000, calibration_days=3, ci_level=0.95
):
    """
    Fit the two-stage regression with SARIMA errors model, deriving a confidence score and a
    prediction-interval width from a held-out calibration window, then forecast the actual test
    window.

    Input
    -----
    df: DataFrame with a datetime index and one column per station, hourly frequency
    target_station: Name of the target station column to predict
    exog_cols: List of neighbouring station column names used as regressors (default is None,
      which uses every other column)
    hours_to_forecast: Number of hours in the test window (default is 48)
    arima_order: Tuple specifying the (p, d, q) parameters for the residual SARIMA
    seasonal_order: Tuple specifying the (P, D, Q, s) parameters for the residual SARIMA
    max_iter: Maximum number of iterations for fitting the residual SARIMA
    calibration_days: Length of the held-out calibration window, carved out of the training data
      immediately before the test window (default is 3, as for ARIMAX)
    ci_level: Confidence level of the interval (default is 0.95)

    Output
    ------
    result: Dictionary with the test-window MAE/MSE, the calibration-window confidence score and
      interval width/bounds, and the test-window predictions and calibration errors
    """
    if exog_cols is None:
        exog_cols = [c for c in df.columns if c != target_station]

    split_date = df.index.max() - pd.DateOffset(hours=hours_to_forecast)
    calibration_period = pd.DateOffset(days=calibration_days)
    calibration_end = split_date
    train_end = split_date - calibration_period

    # Fit on the training window with the calibration period held out, then forecast it to build
    # an empirical, two-stage-combined error sample
    calibration_model = train_regression_sarima_errors(
        df[df.index <= train_end], target_station, exog_cols, arima_order, seasonal_order, max_iter
    )
    _, _, _, calibration_predictions = run_single_forecast(
        df, target_station, model=calibration_model, exog_cols=exog_cols,
        train_end=train_end, test_end=calibration_end
    )
    calibration_actual = df.loc[calibration_predictions.index, target_station]
    calibration_errors = calibration_actual - calibration_predictions

    confidence_score = 1 / (1 + calibration_errors.abs().mean())
    interval_width, lower_quantile, upper_quantile = conformal_interval_width(
        calibration_errors, ci_level
    )

    # Refit with the calibration period folded back into training, then forecast the actual test window
    test_model = train_regression_sarima_errors(
        df[df.index <= calibration_end], target_station, exog_cols, arima_order, seasonal_order, max_iter
    )
    mae, mse, _, test_predictions = run_single_forecast(
        df, target_station, model=test_model, exog_cols=exog_cols,
        train_end=calibration_end, test_end=df.index.max()
    )

    return {
        "mae": mae,
        "mse": mse,
        "confidence_score": confidence_score,
        "interval_width": interval_width,
        "lower_quantile": lower_quantile,
        "upper_quantile": upper_quantile,
        "calibration_errors": calibration_errors,
        "test_predictions": test_predictions
    }
