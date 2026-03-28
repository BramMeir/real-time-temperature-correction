"""
Script: main.py

Entry point to run ARIMA and SARIMA forecasting, grid search, diagnostics, and repeated forecasts.

Functions:
- repeat_forecasts: Perform multiple forecasts on random segments of the data for reliability testing.

Example usage:
python -m src.models.arima.main --mode forecast --model arima --weeks 6 --resample 1h --hours_to_forecast 48
python -m src.models.arima.main --mode grid_search
"""
import os
import argparse
import pandas as pd
import numpy as np
from src.models.arima.repeat_forecast import repeat_forecasts
from src.models.arima.sarima_forecast import sarima_forecast
from src.models.arima.grid_search import arima_grid_search, sarima_grid_search
from src.models.arima.plot_diagnositcs import arima_plot_diagnostics
from src.models.arima.simulate_real_forecast import repeat_simulate_forecast, experiment_retrain_frequency
from src.models.arima.bayes_search import sarima_bayes_search
from src.models.arima.experiment_exog_scaling import experiment_exog_scaling, plot_exog_scaling_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ARIMA forecast utilities.")
    parser.add_argument("--mode", choices=["forecast", "repeat_forecast", "grid_search", "bayes_search", "diagnostics",
                                           "simulate_real_forecast", "experiment_retrain_frequency", "confidence_score",
                                           "experiment_exog_scaling"],
                        default="forecast", help="Select which ARIMA task to run.")
    parser.add_argument("--model", choices=["arima", "sarima"],
                        default="arima", help="Choose between ARIMA and SARIMA model (default: ARIMA).")
    parser.add_argument("--weeks", type=int, default=2,
                        help="Number of weeks of data to include (default: 2).")
    parser.add_argument("--resample", type=str, default="1h",
                        help="Resampling interval, e.g. '10min', '30min', '1h' (default: '1h').")
    parser.add_argument("--hours_to_forecast", type=int, default=48,
                        help="Number of hours to forecast into the future (default: 48).")
    parser.add_argument("--exog", action="store_true",
                        help="Include exogenous variables from other stations if set.")
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

    # Create exogenous DataFrame if requested
    exog_df = None
    if args.exog:
        other_stations = [s for s in df['station_name'].unique() if s != args.target_station]

        exog_data = df[df['station_name'].isin(other_stations)]

        # Pivot to get each station as a separate column
        exog_pivot = exog_data.pivot_table(
            index='datetime', columns='station_name', values='temp_dry_avg_2m'
        )

        # Create datetime index
        exog_pivot.index = pd.to_datetime(exog_pivot.index)

        # Remove the missing timestamps and align with the main series
        exog_pivot = exog_pivot.reindex(series.index)

        # exog_df = exog_pivot.interpolate(limit_direction="both")
        exog_df = exog_pivot

    # Shorten the data to the specified number of weeks (if in given modes)
    if args.mode in ["forecast", "grid_search", "bayes_search", "diagnostics"]:
        total_weeks = (series.index.max().year - series.index.min().year) * 52 + \
                      (series.index.max().month - series.index.min().month) * 4 + \
                      (series.index.max().day - series.index.min().day) // 7

        if total_weeks > args.weeks:
            start_weeks = np.random.randint(0, total_weeks - args.weeks + 1)
            start_date = series.index.min() + pd.DateOffset(weeks=start_weeks)
            end_date = start_date + pd.DateOffset(weeks=args.weeks)
            series = series[(series.index >= start_date) & (series.index < end_date)]

            if exog_df is not None:
                exog_df = exog_df[(exog_df.index >= start_date) & (exog_df.index < end_date)]

    # Resample data by taking the mean
    series = series.resample(args.resample).mean().interpolate(limit_direction="both")

    # Resample exogenous data if provided
    if exog_df is not None:
        exog_df = exog_df.resample(args.resample).mean().interpolate(limit_direction="both")

    # Dispatch based on argument
    if args.mode == "forecast":
        if args.model == "sarima":
            sarima_forecast(series, exog_df=exog_df, hours_to_forecast=args.hours_to_forecast, arima_order=(10, 0, 1),
                            seasonal_order=(1, 0, 1, 24), max_iter=1000)
        else:
            sarima_forecast(series, exog_df=exog_df, hours_to_forecast=args.hours_to_forecast, arima_order=(2, 0, 0),
                            seasonal_order=(1, 0, 1, 24), max_iter=1000)

    elif args.mode == "repeat_forecast":
        if args.model == "sarima":
            repeat_forecasts(series, exog_df=exog_df, weeks=args.weeks, hours_to_forecast=args.hours_to_forecast,
                             arima_order=(15, 0, 0), seasonal_order=(1, 0, 1, 24), confidence_score=False,
                             n_repeats=30, random_seed=47, max_iter=1000, n_jobs=10, plot=True, verbose=True)
        else:
            repeat_forecasts(series, exog_df=exog_df, weeks=args.weeks, hours_to_forecast=args.hours_to_forecast,
                             arima_order=(2, 0, 0), seasonal_order=(1, 0, 1, 24), confidence_score=False,
                             n_repeats=30, random_seed=47, max_iter=1000, n_jobs=10, plot=True, verbose=True)

    elif args.mode == "grid_search":
        if args.model == "sarima":
            sarima_grid_search(series, exog_df=exog_df, p_values=[15], d_values=[0], q_values=[0],
                               P_values=range(1, 4), D_values=[0], Q_values=range(1, 4), S=24, max_workers=10)
        else:
            arima_grid_search(series, exog_df=exog_df, p_values=range(10, 41, 10), d_values=[0], q_values=[0], max_workers=10)

    elif args.mode == "bayes_search":
        sarima_bayes_search(series, exog_df=exog_df, S=24, p_range=(0, 50), d_range=(0, 2), q_range=(0, 2),
                            P_range=(0, 2), D_range=(0, 1), Q_range=(0, 2))

    elif args.mode == "simulate_real_forecast":
        repeat_simulate_forecast(series, exog_df=exog_df, weeks=args.weeks, hours_to_forecast=args.hours_to_forecast,
                                 arima_order=(25, 0, 0), seasonal_order=(0, 0, 0, 0),
                                 n_repeats=30, random_seed=47, max_iter=1000, n_jobs=10)

    elif args.mode == "experiment_retrain_frequency":
        experiment_retrain_frequency(series, exog_df=exog_df, weeks=args.weeks, hours_to_forecast=args.hours_to_forecast,
                                     arima_order=(25, 0, 0), seasonal_order=(0, 0, 0, 0),
                                     n_repeats=30, random_seed=57, max_iter=1000, n_jobs=10)

    elif args.mode == "confidence_score":
        repeat_forecasts(series, exog_df=exog_df, weeks=args.weeks, hours_to_forecast=args.hours_to_forecast,
                         arima_order=(2, 0, 0), seasonal_order=(1, 0, 1, 24), confidence_score=True,
                         n_repeats=50, random_seed=47, max_iter=1000, n_jobs=10, plot=False, verbose=True)

    elif args.mode == "diagnostics":
        arima_plot_diagnostics(series)

    elif args.mode == "experiment_exog_scaling":
        df_results = experiment_exog_scaling(
            series,
            full_exog_df=exog_df,
            k_values=list(range(1, exog_df.shape[1] + 1, 2)),
            weeks=args.weeks,
            hours_to_forecast=args.hours_to_forecast,
            arima_order=(2, 0, 0),
            seasonal_order=(1, 0, 1, 24),
            n_repeats=30,
            max_iter=1000,
            n_jobs=10
        )

        # Make sure the output directory exists
        os.makedirs("output/experiment_exog_scaling", exist_ok=True)

        # Save the results to a CSV file
        df_results.to_csv(f"output/experiment_exog_scaling/{args.target_station}.csv", index=False)

        # Plot the results
        plot_exog_scaling_results(
            csv_path=f"output/experiment_exog_scaling/{args.target_station}.csv",
            save_path=f"output/experiment_exog_scaling/{args.target_station}_training_time_versus_nr_exog_stations.png",
            log_scale=False
        )
