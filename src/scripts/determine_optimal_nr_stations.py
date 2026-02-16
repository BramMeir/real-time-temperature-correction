"""
Script: determine_optimal_nr_stations.py

Description:
This script tries to determine the optimal number of stations to include as exogenous variables in an ARIMA forecasting model
for a target station. It does this by:
1. Ranking all the stations based on their importance for forecasting the target station.
2. Iteratively testing the forecast performance (MAE, MSE) of the ARIMA model using the top k stations as exogenous variables.

Example usage:
python -m src.scripts.determine_optimal_nr_stations --target_station "Sint_Baafs_Gent" --input_file "data/Synthetic/temperature_data.csv""
"""

import argparse
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from src.models.arima.sarima_forecast import sarima_forecast
from src.models.arima.confidence_score import sarima_forecast_with_confidence_score


def _run_single_forecast(i, series, exog_df, start_date, end_date, hours_to_forecast,
                         arima_order, seasonal_order, confidence_score, max_iter):
    """
    Helper function to run a single forecast on a sub-series.

    Input
    -----
    i: Index of the current run (for logging purposes)
    series: Pandas Series with the time series data
    exog_df: DataFrame with exogenous variables
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
            plot=False
        )

        end_time = pd.Timestamp.now()
        duration = end_time - start_time

        return errors['MAE'], errors['MSE'], importance, duration, confidence_score_value, avg_conf_interval_size

    else:
        errors, importance = sarima_forecast(
            sub_series,
            exog_df=sub_exog,
            hours_to_forecast=hours_to_forecast,
            arima_order=arima_order,
            seasonal_order=seasonal_order,
            max_iter=max_iter,
            plot=False
        )

        end_time = pd.Timestamp.now()
        duration = end_time - start_time

        return errors['MAE'], errors['MSE'], importance, duration, None, None


def repeat_forecasts(series, exog_df=None, weeks=2, hours_to_forecast=48, arima_order=(10, 0, 1),
                     seasonal_order=(0, 0, 0, 0), confidence_score=False, n_repeats=5, random_seed=42,
                     max_iter=1000, n_jobs=4):
    """
    Perform multiple forecasts on random n-month segments of the data.
    Returns list of MSE values for scientific reliability testing.

    Input
    -----
    series: Pandas Series with the time series data
    exog_df: DataFrame with exogenous variables (can be None)
    weeks: Number of weeks of data to include in each segment (default is 2 weeks)
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
                _run_single_forecast, i, series, exog_df, start, end,
                hours_to_forecast, arima_order, seasonal_order, confidence_score, max_iter
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

    # Print the average feature importance if exogenous variables were used
    if exog_df is not None and importances:
        avg_importance = pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False)
    else:
        avg_importance = None

    return {
        "mae_mean": np.mean(mae_scores),
        "mse_mean": np.mean(mse_scores),
        "importance": avg_importance,
        "duration_mean_seconds": np.mean(durations).total_seconds(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ARIMA forecast utilities.")
    parser.add_argument("--weeks", type=int, default=2,
                        help="Number of weeks of data to include (default: 2).")
    parser.add_argument("--resample", type=str, default="1h",
                        help="Resampling interval, e.g. '10min', '30min', '1h' (default: '1h').")
    parser.add_argument("--hours_to_forecast", type=int, default=48,
                        help="Number of hours to forecast into the future (default: 48).")
    parser.add_argument("--target_station", type=str, default="Melle AWS",
                        help="Name of the target weather station (default: 'Melle AWS').")
    parser.add_argument("--input_file", type=str, default="data/Part_AWS/preprocessed.csv",
                        help="Path to the preprocessed data CSV file (default: 'data/Part_AWS/preprocessed.csv').")
    args = parser.parse_args()

    # Read the preprocessed data
    df = pd.read_csv(args.input_file)

    # Select target station
    station_data = df[df['station_name'] == args.target_station]

    # Create target time series with a datetime index
    series = pd.Series(
        station_data['temp_dry_avg_2m'].values,
        index=pd.to_datetime(station_data['datetime'])
    ).asfreq('1h').dropna()

    # Create exogenous DataFrame
    other_stations = [s for s in df['station_name'].unique() if s != args.target_station]
    exog_data = df[df['station_name'].isin(other_stations)]

    # Pivot to get each station as a separate column
    exog_pivot = exog_data.pivot_table(
        index='datetime', columns='station_name', values='temp_dry_avg_2m'
    )

    # Create datetime index
    exog_pivot.index = pd.to_datetime(exog_pivot.index)

    # Remove the missing timestamps and align with the main series
    exog_df = exog_pivot.reindex(series.index)

    # Resample data by taking the mean
    series = series.resample(args.resample).mean().interpolate(limit_direction="both")
    exog_df = exog_df.resample(args.resample).mean().interpolate(limit_direction="both")

    # Step 1: Rank all the stations based on their importance for forecasting the target station
    print("Step 1: Ranking stations based on importance for forecasting the target station...")

    results_all = repeat_forecasts(
        series,
        exog_df=exog_df,
        weeks=args.weeks,
        hours_to_forecast=args.hours_to_forecast,
        arima_order=(2, 0, 0),
        seasonal_order=(1, 0, 1, 24),
        confidence_score=False,
        n_repeats=20,
        random_seed=47,
        max_iter=1000,
        n_jobs=10
    )

    importance_ranking = results_all['importance']

    print("\nStation ranking based on average importance:")
    print(importance_ranking)

    # Step 2: Determine the optimal number of stations to include
    print("\nStep 2: Determining the optimal number of stations to include...")

    ordered_stations = importance_ranking.index.tolist()
    sub_stations_results = []

    # Keep track of best performance to implement an early stopping criterion
    best_mse = float("inf")

    for k in range(1, len(ordered_stations) + 1):
        selected_stations = ordered_stations[:k]

        print(f"\nTesting top {k} stations:")
        print(selected_stations)

        sub_exog_df = exog_df[selected_stations]

        results_k = repeat_forecasts(
            series,
            exog_df=sub_exog_df,
            weeks=args.weeks,
            hours_to_forecast=args.hours_to_forecast,
            arima_order=(2, 0, 0),
            seasonal_order=(1, 0, 1, 24),
            confidence_score=False,
            n_repeats=20,
            random_seed=47,
            max_iter=1000,
            n_jobs=10
        )

        sub_stations_results.append({
            "num_stations": k,
            "stations": selected_stations,
            "mae_mean": results_k['mae_mean'],
            "mse_mean": results_k['mse_mean'],
            "duration_mean_seconds": results_k['duration_mean_seconds']
        })

    # Step 3: Analyze results to find the optimal number of stations
    final_df = pd.DataFrame(sub_stations_results)
    sorted_df = final_df.sort_values(by="mse_mean")

    top_3 = sorted_df.head(3)
    print("\nTop 3 configurations based on MSE:")

    for idx, row in top_3.iterrows():
        print(f"\nNumber of stations: {row['num_stations']}")
        print(f"Stations: {row['stations']}")
        print(f"MAE: {row['mae_mean']:.4f}")
        print(f"MSE: {row['mse_mean']:.4f}")
        print(f"Average duration (seconds): {row['duration_mean_seconds']:.2f}")
