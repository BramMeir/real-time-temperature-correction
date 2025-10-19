"""
Script: main.py

Entry point to run ARIMA and SARIMA forecasting, grid search, diagnostics, and repeated forecasts.

Functions:
- repeat_forecasts: Perform multiple forecasts on random segments of the data for reliability testing.

Example usage:
python -m src.models.arima.main --mode forecast --model arima --months 6 --resample 1h --hours_to_forecast 48
python -m src.models.arima.main --mode grid_search
"""

import argparse
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from src.models.arima.arima_forecast import arima_forecast
from src.models.arima.sarima_forecast import sarima_forecast
from src.models.arima.grid_search import arima_grid_search
from src.models.arima.plot_diagnositcs import arima_plot_diagnostics


def _run_single_forecast(i, series, start_date, end_date, hours_to_forecast,
                         arima_order, seasonal_order, max_iter):
    """
    Helper function to run a single forecast on a sub-series.

    Input
    -----
    i: Index of the current run (for logging purposes)
    series: Pandas Series with the time series data
    start_date: Start date for the sub-series
    end_date: End date for the sub-series
    hours_to_forecast: Number of hours to forecast into the future
    arima_order: Tuple specifying the (p, d, q) parameters for the ARIMA model
    seasonal_order: Tuple specifying the (P, D, Q, S) parameters for the SARIMA model
    max_iter: Maximum number of iterations for model fitting

    Output
    ------
    Returns a tuple of (MAE, MSE) for the forecast on the sub-series.
    """
    # Extract the sub-series for the current run
    sub_series = series[(series.index >= start_date) & (series.index <= end_date)]

    print(f"\n🔹 Run {i+1}: using data from {start_date} to {end_date}")
    errors = sarima_forecast(
        sub_series,
        hours_to_forecast=hours_to_forecast,
        arima_order=arima_order,
        seasonal_order=seasonal_order,
        max_iter=max_iter,
        plot=False
    )

    return errors['MAE'], errors['MSE']


def repeat_forecasts(series, months=6, hours_to_forecast=48, arima_order=(10, 0, 1),
                     seasonal_order=(0, 0, 0, 0), n_repeats=5, random_seed=42,
                     max_iter=1000, n_jobs=4):
    """
    Perform multiple forecasts on random n-month segments of the data.
    Returns list of MSE values for scientific reliability testing.

    Input
    -----
    series: Pandas Series with the time series data
    months: Number of months of data to include in each segment (default is 6)
    hours_to_forecast: Number of hours to forecast into the future (default is 48)
    arima_order: Tuple specifying the (p, d, q) parameters for the ARIMA model (default is (10, 0, 1))
    seasonal_order: Tuple specifying the (P, D, Q, S) parameters for the SARIMA model (default is (0, 0, 0, 0))
    n_repeats: Number of random segments to test (default is 5)
    random_seed: Seed for random number generator for reproducibility (default is 42)
    max_iter: Maximum number of iterations for model fitting (default is 1000)
    n_jobs: Number of parallel jobs to run (default is 4)

    Output
    ------
    Returns two lists: mae_scores and mse_scores containing the MAE and MSE for each repeat.
    """
    np.random.seed(random_seed)
    months_offset = pd.DateOffset(months=months)
    max_start = series.index.max() - months_offset

    possible_starts = series.index[(series.index >= series.index.min()) & (series.index <= max_start)]
    if len(possible_starts) == 0:
        raise ValueError("Series too short for chosen window length and forecast horizon.")

    # Pre-generate all start and end dates
    start_dates = np.random.choice(possible_starts, size=n_repeats, replace=False)
    date_ranges = [(start, start + months_offset) for start in start_dates]

    mae_scores, mse_scores = [], []

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = [
            executor.submit(
                _run_single_forecast, i, series, start, end,
                hours_to_forecast, arima_order, seasonal_order, max_iter
            )
            for i, (start, end) in enumerate(date_ranges)
        ]

        for f in as_completed(futures):
            mae, mse = f.result()
            mae_scores.append(mae)
            mse_scores.append(mse)

    print(f"\nAverage MAE across {len(mae_scores)} runs: {np.mean(mae_scores):.3f}")
    print(f"Average MSE across {len(mse_scores)} runs: {np.mean(mse_scores):.3f}")

    return mae_scores, mse_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ARIMA forecast utilities.")
    parser.add_argument("--mode", choices=["forecast", "repeat_forecast", "grid_search", "diagnostics", "auto_arima"],
                        default="forecast", help="Select which ARIMA task to run.")
    parser.add_argument("--model", choices=["arima", "sarima"],
                        default="arima", help="Choose between ARIMA and SARIMA model (default: ARIMA).")
    parser.add_argument("--months", type=int, default=6,
                        help="Number of months of data to include (default: 6).")
    parser.add_argument("--resample", type=str, default="1h",
                        help="Resampling interval, e.g. '10min', '30min', '1h' (default: '1h').")
    parser.add_argument("--hours_to_forecast", type=int, default=48,
                        help="Number of hours to forecast into the future (default: 48).")
    args = parser.parse_args()

    # Read the preprocessed data
    df = pd.read_csv("data/preprocessed.csv")

    # Select data for a specific station
    station = "STABROEK AWS"
    station_data = df[df['station_name'] == station]

    # Create a time series with a datetime index
    series = pd.Series(
        station_data['temp_dry_avg_2m'].values,
        index=pd.to_datetime(station_data['datetime'])
    ).asfreq('10min').dropna()

    # Select random n months of the data (better simulates real-world forecasting)
    if args.mode != "repeat_forecast":
        total_months = (series.index.max().year - series.index.min().year) * 12 + \
                       (series.index.max().month - series.index.min().month)

        if total_months > args.months:
            start_month = np.random.randint(0, total_months - args.months + 1)
            start_date = series.index.min() + pd.DateOffset(months=start_month)
            end_date = start_date + pd.DateOffset(months=args.months)
            series = series[(series.index >= start_date) & (series.index < end_date)]

    # Resample data by taking the mean
    series = series.resample(args.resample).mean()

    # Dispatch based on argument
    if args.mode == "forecast":
        if args.model == "sarima":
            sarima_forecast(series, hours_to_forecast=args.hours_to_forecast, arima_order=(10, 0, 1),
                            seasonal_order=(3, 0, 1, 24), max_iter=1000)
        else:
            arima_forecast(series, hours_to_forecast=args.hours_to_forecast, arima_order=(25, 0, 1), max_iter=1000)

    elif args.mode == "repeat_forecast":
        if args.model == "sarima":
            repeat_forecasts(series, months=args.months, hours_to_forecast=args.hours_to_forecast,
                             arima_order=(10, 0, 1), seasonal_order=(3, 0, 1, 24),
                             n_repeats=10, random_seed=47, max_iter=1000, n_jobs=5)
        else:
            repeat_forecasts(series, months=args.months, hours_to_forecast=args.hours_to_forecast,
                             arima_order=(25, 0, 1), seasonal_order=(0, 0, 0, 0),
                             n_repeats=10, random_seed=47, max_iter=1000, n_jobs=4)

    elif args.mode == "grid_search":
        arima_grid_search(series, p_values=range(0, 31, 1), d_values=[0], q_values=range(0, 2), max_workers=6)

    elif args.mode == "diagnostics":
        arima_plot_diagnostics(series)
