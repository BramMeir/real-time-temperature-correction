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
import os
import pandas as pd
from src.utils.select_LASSO_stations import select_LASSO_stations
from src.models.arima.repeat_forecast import repeat_forecasts


def run_station_experiment(
    df,
    target_station,
    weeks,
    resample,
    hours_to_forecast
):
    """
    Executes the experiment to determine the LASSO performance compared to using all stations.

    Input:
    -----
    df: pandas DataFrame containing the preprocessed data with columns 'datetime', 'station_name', and 'temp_dry_avg_2m'.
    target_station: Name of the target weather station.
    weeks: Number of weeks of data to include.
    resample: Resampling interval.
    hours_to_forecast: Number of hours to forecast into the future.

    Output:
    ------
    A dictionary containing the results of the experiment.
    """
    # Create target time series with a datetime index
    station_data = df[df['station_name'] == target_station]

    series = pd.Series(
        station_data['temp_dry_avg_2m'].values,
        index=pd.to_datetime(station_data['datetime'])
    ).asfreq('1h').dropna()

    # Create exogenous DataFrame
    other_stations = [s for s in df['station_name'].unique() if s != target_station]
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

    # Step 1: Determine the performance of using all stations as exogenous variables
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

    mae_all = results_all["mae_mean"]
    mse_all = results_all["mse_mean"]
    duration_all = results_all["duration_mean_seconds"]

    # Step 2: Determine the performance of the model using subsets of stations selected by LASSO
    # Timing for LASSO ranking
    start_time = pd.Timestamp.now()

    # Execute LASSO variable selection
    top_exog_df, _, ranking = select_LASSO_stations(series, exog_df)

    # Timing for LASSO ranking
    duration_lasso_selection = (pd.Timestamp.now() - start_time).total_seconds()

    lasso_result = repeat_forecasts(
        series,
        exog_df=top_exog_df,
        weeks=weeks,
        hours_to_forecast=hours_to_forecast,
        arima_order=(2, 0, 0),
        seasonal_order=(1, 0, 1, 24),
        confidence_score=False,
        n_repeats=20,
        random_seed=47,
        max_iter=1000,
        n_jobs=10
    )

    mae_lasso = lasso_result["mae_mean"]
    mse_lasso = lasso_result["mse_mean"]
    duration_lasso_forecast = lasso_result["duration_mean_seconds"]

    return {
        "target_station": target_station,
        "nr_stations": len(ranking),
        "optimal_mse": mse_lasso,
        "all_mse": mse_all,
        "mse_difference": mse_all - mse_lasso,
        "optimal_mae": mae_lasso,
        "all_mae": mae_all,
        "mae_difference": mae_all - mae_lasso,
        "duration_all_seconds": duration_all,
        "duration_lasso_selection_seconds": duration_lasso_selection,
        "duration_lasso_forecast_seconds": duration_lasso_forecast
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ARIMA forecast utilities.")
    parser.add_argument("--weeks", type=int, default=2,
                        help="Number of weeks of data to include (default: 2).")
    parser.add_argument("--resample", type=str, default="1h",
                        help="Resampling interval, e.g. '10min', '30min', '1h' (default: '1h').")
    parser.add_argument("--hours_to_forecast", type=int, default=48,
                        help="Number of hours to forecast into the future (default: 48).")
    parser.add_argument("--station_index", type=int, default=0,
                        help="Index of the target station to run the experiment on (default: 0).")
    parser.add_argument("--input_file", type=str, default="data/Part_AWS/preprocessed.csv",
                        help="Path to the preprocessed data CSV file (default: 'data/Part_AWS/preprocessed.csv').")
    args = parser.parse_args()

    # Read the preprocessed data
    df = pd.read_csv(args.input_file)

    # Get the list of unique stations and select the target station based on the provided index
    stations = sorted(df['station_name'].unique())
    target_station = stations[args.station_index]

    # Run the experiment for each station and save results
    result = run_station_experiment(
        df,
        target_station,
        weeks=args.weeks,
        resample=args.resample,
        hours_to_forecast=args.hours_to_forecast
    )

    # Make sure the output directory exists
    os.makedirs("output/LASSO_stations", exist_ok=True)

    # Save the result to a CSV file
    pd.DataFrame([result]).to_csv(f"output/LASSO_stations/performance_{args.station_index}.csv", index=False)
