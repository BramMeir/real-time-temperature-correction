"""
Script: repeat_forecast.py

Repeat the two-stage model's confidence scoring (src/models/regression_sarima_errors/confidence_score.py)
on random segments of one station's data, mirroring src/models/arima/repeat_forecast.py so the two
models' reliability indicators are directly comparable, and so the MAE realised on a later, independent
segment can be checked against the reliability indicators computed on the internal calibration window
(as in Fig. 8 of the paper, for ARIMAX).

Functions:
- repeat_forecasts: Run the confidence-scored forecast on n_repeats random segments and average the
  realised error and the reliability indicators.
"""
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from src.models.regression_sarima_errors.confidence_score import two_stage_forecast_with_confidence_score


def _run_single_forecast(
        i, df, target_station, exog_cols, start, end, hours_to_forecast, arima_order,
        seasonal_order, calibration_days, ci_level, max_iter
):
    """Helper function to run a single confidence-scored forecast on a sub-window, for parallel execution."""
    print(f"Run {i + 1}: using data from {start} to {end}")

    sub_df = df[(df.index >= start) & (df.index <= end)]

    start_time = pd.Timestamp.now()
    result = two_stage_forecast_with_confidence_score(
        sub_df, target_station, exog_cols=exog_cols, hours_to_forecast=hours_to_forecast,
        arima_order=arima_order, seasonal_order=seasonal_order, max_iter=max_iter,
        calibration_days=calibration_days, ci_level=ci_level
    )
    duration = (pd.Timestamp.now() - start_time).total_seconds()

    return result["mae"], result["mse"], result["confidence_score"], result["interval_width"], duration


def repeat_forecasts(
        df, target_station, exog_cols=None, weeks=8, hours_to_forecast=48, arima_order=(2, 0, 0),
        seasonal_order=(1, 0, 1, 24), calibration_days=3, ci_level=0.95,
        n_repeats=15, random_seed=47, max_iter=1000, n_jobs=4
):
    """
    Run the two-stage model's confidence-scored forecast on n_repeats random segments of one
    station's data, and average the realised test-window error and the reliability indicators
    (confidence score, interval width) computed on each segment's calibration window.

    Input
    -----
    df: DataFrame with a datetime index and one column per station, hourly frequency
    target_station: Name of the target station column to predict
    exog_cols: List of neighbouring station column names used as regressors (default is None,
      which uses every other column)
    weeks: Number of weeks of training data per segment, before the calibration window (default is 8)
    hours_to_forecast: Number of hours in the test window (default is 48)
    arima_order, seasonal_order: SARIMA parameters for the residual model
    calibration_days: Length of the held-out calibration window (default is 3, as for ARIMAX)
    ci_level: Confidence level of the interval (default is 0.95)
    n_repeats: Number of random segments to test (default is 15)
    random_seed: Seed for the random segment starts (default is 47)
    max_iter: Maximum number of iterations for fitting the residual SARIMA
    n_jobs: Number of parallel jobs to run (default is 4)

    Output
    ------
    Dictionary with the average MAE/MSE, confidence score and interval width across the segments
    """
    if exog_cols is None:
        exog_cols = [c for c in df.columns if c != target_station]

    segment_length = pd.Timedelta(weeks=weeks) + pd.Timedelta(days=calibration_days) \
        + pd.Timedelta(hours=hours_to_forecast)
    max_start = df.index.max() - segment_length

    possible_starts = df.index[(df.index >= df.index.min()) & (df.index <= max_start)]
    if len(possible_starts) == 0:
        raise ValueError("Series too short for chosen window length and forecast horizon.")

    np.random.seed(random_seed)
    start_dates = np.random.choice(possible_starts, size=n_repeats, replace=False)
    date_ranges = [(start, start + segment_length) for start in start_dates]

    mae_scores, mse_scores, confidence_scores, interval_widths, durations = [], [], [], [], []

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = [
            executor.submit(
                _run_single_forecast, i, df, target_station, exog_cols, start, end,
                hours_to_forecast, arima_order, seasonal_order, calibration_days, ci_level,
                max_iter
            )
            for i, (start, end) in enumerate(date_ranges)
        ]

        for f in as_completed(futures):
            mae, mse, confidence_score, interval_width, duration = f.result()
            mae_scores.append(mae)
            mse_scores.append(mse)
            confidence_scores.append(confidence_score)
            interval_widths.append(interval_width)
            durations.append(duration)

    return {
        "mae_mean": np.mean(mae_scores),
        "mse_mean": np.mean(mse_scores),
        "confidence_score_mean": np.mean(confidence_scores),
        "interval_width_mean": np.mean(interval_widths),
        "duration_mean_seconds": np.mean(durations),
    }
