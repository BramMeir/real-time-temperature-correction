import argparse
import pandas as pd
import numpy as np
from src.models.arima.arima_forecast import arima_forecast
from src.models.arima.sarima_forecast import sarima_forecast
from src.models.arima.grid_search import arima_grid_search
from src.models.arima.plot_diagnositcs import arima_plot_diagnostics
from src.models.arima.auto_arima import execute_auto_arima


def repeat_forecasts(series, months=6, hours_to_forecast=48, arima_order=(10, 0, 1),
                     n_repeats=5, random_seed=42, max_iter=1000):
    """
    Perform multiple ARIMA forecasts on random 6-month segments of the data.
    Returns list of MSE values for scientific reliability testing.
    """
    np.random.seed(random_seed)
    mse_scores = []
    mae_scores = []

    six_months = pd.DateOffset(months=months)
    max_start = series.index.max() - six_months

    possible_starts = series.index[(series.index >= series.index.min()) & (series.index <= max_start)]
    if len(possible_starts) == 0:
        raise ValueError("Series too short for chosen window length and forecast horizon.")

    for i in range(n_repeats):
        # Randomly select start date
        start_date = np.random.choice(possible_starts)
        end_date = start_date + six_months
        sub_series = series[(series.index >= start_date) & (series.index <= end_date)]

        print(f"\n🔹 Run {i+1}: using data from {start_date} to {end_date}")
        errors = arima_forecast(
            sub_series,
            hours_to_forecast=hours_to_forecast,
            arima_order=arima_order,
            max_iter=max_iter
        )
        mae_scores.append(errors['MAE'])
        mse_scores.append(errors['MSE'])

    if len(mse_scores) > 0:
        print(f"\nAverage MAE across {len(mae_scores)} runs: {np.mean(mae_scores):.3f}")
        print(f"\nAverage MSE across {len(mse_scores)} runs: {np.mean(mse_scores):.3f}")

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
    station = "Melle AWS"
    station_data = df[df['station_name'] == station]

    # Create a time series with a datetime index
    series = pd.Series(
        station_data['temp_dry_avg_2m'].values,
        index=pd.to_datetime(station_data['datetime'])
    ).asfreq('10min').dropna()

    # Focus on the last n months
    if args.mode != "repeat_forecast":
        last_month_start = series.index.max() - pd.DateOffset(months=args.months)
        series = series[(series.index >= last_month_start)]

    # Resample to hourly data by taking the mean
    series = series.resample(args.resample).mean()

    # Dispatch based on argument
    if args.mode == "forecast":
        if args.model == "sarima":
            sarima_forecast(series, hours_to_forecast=args.hours_to_forecast, arima_order=(25, 0, 1),
                            seasonal_order=(0, 0, 0, 0), max_iter=1000)
        else:
            arima_forecast(series, hours_to_forecast=args.hours_to_forecast, arima_order=(2, 0, 0), max_iter=1000)
    elif args.mode == "repeat_forecast":
        repeat_forecasts(series, months=args.months, hours_to_forecast=args.hours_to_forecast,
                         arima_order=(25, 0, 1), n_repeats=5, random_seed=47, max_iter=1000)
    elif args.mode == "grid_search":
        arima_grid_search(series, p_values=range(0, 31, 1), d_values=[0], q_values=range(0, 2), max_workers=6)
    elif args.mode == "auto_arima":
        execute_auto_arima(series)
    elif args.mode == "diagnostics":
        arima_plot_diagnostics(series)
