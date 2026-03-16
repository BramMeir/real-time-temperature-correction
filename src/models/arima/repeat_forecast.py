"""
Module: repeat_forecast.py

Description:
This module contains functions to perform repeated ARIMA forecasts on random segments of a time series.

Functionality:
- _run_single_forecast: A helper function to run a single forecast on a sub-series, which can be executed in parallel.
- repeat_forecasts: A function to perform multiple forecasts on random segments of the data.
"""
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from src.models.arima.sarima_forecast import sarima_forecast
from src.models.arima.confidence_score import sarima_forecast_with_confidence_score


def _run_single_forecast(i, series, exog_df, model, start_date, end_date, hours_to_forecast,
                         arima_order, seasonal_order, confidence_score, max_iter, plot=False):
    """
    Helper function to run a single forecast on a sub-series.

    Input
    -----
    i: Index of the current run (for logging purposes)
    series: Pandas Series with the time series data
    exog_df: DataFrame with exogenous variables
    model: Optional pre-trained SARIMA model (if None, a new model will be trained)
    start_date: Start date for the sub-series
    end_date: End date for the sub-series
    hours_to_forecast: Number of hours to forecast into the future
    arima_order: Tuple specifying the (p, d, q) parameters for the ARIMA model
    seasonal_order: Tuple specifying the (P, D, Q, S) parameters for the SARIMA model
    max_iter: Maximum number of iterations for model fitting
    plot: Whether to plot the forecast results (default is False)

    Output
    ------
    Returns a tuple of (MAE, MSE) for the forecast on the sub-series.
    """
    # Extract the sub-series for the current run
    sub_series = series[(series.index >= start_date) & (series.index <= end_date)]
    sub_exog = None

    if exog_df is not None:
        sub_exog = exog_df.loc[sub_series.index]

    print(f"\n🔹 Run {i + 1}: using data from {start_date} to {end_date}")

    start_time = pd.Timestamp.now()

    if confidence_score:
        errors, importance, confidence_score_value, avg_conf_interval_size = sarima_forecast_with_confidence_score(
            sub_series,
            exog_df=sub_exog,
            hours_to_forecast=hours_to_forecast,
            arima_order=arima_order,
            seasonal_order=seasonal_order,
            max_iter=max_iter,
            plot=plot
        )

        end_time = pd.Timestamp.now()
        duration = end_time - start_time

        return errors['MAE'], errors['MSE'], importance, duration, confidence_score_value, avg_conf_interval_size

    else:
        errors, importance = sarima_forecast(
            sub_series,
            exog_df=sub_exog,
            model=model,
            hours_to_forecast=hours_to_forecast,
            arima_order=arima_order,
            seasonal_order=seasonal_order,
            max_iter=max_iter,
            plot=plot
        )

        end_time = pd.Timestamp.now()
        duration = end_time - start_time

        return errors['MAE'], errors['MSE'], importance, duration, None, None


def repeat_forecasts(series, exog_df=None, weeks=2, hours_to_forecast=48, arima_order=(10, 0, 1),
                     seasonal_order=(0, 0, 0, 0), confidence_score=False, n_repeats=5, random_seed=42,
                     max_iter=1000, n_jobs=4, plot=False, verbose=False):
    """
    Perform multiple forecasts on random n-month segments of the data.

    Input
    -----
    series: Pandas Series with the time series data
    exog_df: DataFrame with exogenous variables (can be None)
    weeks: Number of weeks of data to include in each segment (default is 2 weeks)
    hours_to_forecast: Number of hours to forecast into the future (default is 48)
    arima_order: Tuple specifying the (p, d, q) parameters for the ARIMA model (default is (10, 0, 1))
    seasonal_order: Tuple specifying the (P, D, Q, S) parameters for the SARIMA model (default is (0, 0, 0, 0))
    confidence_score: Whether to calculate confidence scores for the forecasts (default is False)
    n_repeats: Number of random segments to test (default is 5)
    random_seed: Seed for random number generator for reproducibility (default is 42)
    max_iter: Maximum number of iterations for model fitting (default is 1000)
    n_jobs: Number of parallel jobs to run (default is 4)
    plot: Whether to plot the forecast results for each run (default is False)
    verbose: Whether to print verbose output (default is False)

    Output
    ------
    Returns a dictionary with all the average metrics and feature importance across the runs.
    """
    np.random.seed(random_seed)
    weeks_offset = pd.DateOffset(weeks=weeks)
    max_start = series.index.max() - weeks_offset

    possible_starts = series.index[(series.index >= series.index.min()) & (series.index <= max_start)]
    if len(possible_starts) == 0:
        raise ValueError("Series too short for chosen window length and forecast horizon.")

    # Pre-generate all start and end dates
    start_dates = np.random.choice(possible_starts, size=n_repeats, replace=False)
    date_ranges = [(start, start + weeks_offset) for start in start_dates]

    mae_scores, mse_scores = [], []
    importances = []
    durations = []
    confidence_scores = []
    avg_conf_interval_sizes = []

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = [
            executor.submit(
                _run_single_forecast, i, series, exog_df, None, start, end,
                hours_to_forecast, arima_order, seasonal_order, confidence_score, max_iter, plot
            )
            for i, (start, end) in enumerate(date_ranges)
        ]

        for f in as_completed(futures):
            mae, mse, importance, duration, confidence_score_value, avg_conf_interval_size = f.result()
            mae_scores.append(mae)
            mse_scores.append(mse)
            importances.append(importance)
            durations.append(duration)
            if confidence_score:
                confidence_scores.append(confidence_score_value)
                avg_conf_interval_sizes.append(avg_conf_interval_size)

    if verbose:
        print(f"\nAverage MAE across {len(mae_scores)} runs: {np.mean(mae_scores):.3f} ± {np.std(mae_scores):.3f}")
        print(f"Average MSE across {len(mse_scores)} runs: {np.mean(mse_scores):.3f} ± {np.std(mse_scores):.3f}")
        print(f"Average Duration per run: {np.mean(durations).total_seconds():.2f} seconds")

    # Print the average feature importance if exogenous variables were used
    if exog_df is not None and importances:
        avg_importance = pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False)

        if verbose:
            print("\nAverage Feature Importance across runs:")
            print(avg_importance)

    # Print the confidence score statistics if calculated
    if confidence_score and confidence_scores and verbose:
        print(f"\nAverage Confidence Score across {len(confidence_scores)} runs: {np.mean(confidence_scores):.3f}")
        print(f"Average Confidence Interval Size across {len(avg_conf_interval_sizes)} runs: {np.mean(avg_conf_interval_sizes):.3f}")

    return {
        "mae_mean": np.mean(mae_scores),
        "mse_mean": np.mean(mse_scores),
        "importance": avg_importance,
        "duration_mean_seconds": np.mean(durations).total_seconds(),
        "confidence_score_mean": np.mean(confidence_scores) if confidence_scores else 0,
        "avg_conf_interval_size": np.mean(avg_conf_interval_sizes) if avg_conf_interval_sizes else 0,
    }
