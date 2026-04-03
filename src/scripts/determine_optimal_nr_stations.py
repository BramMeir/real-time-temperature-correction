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
import signal
import sys
from src.models.arima.repeat_forecast import repeat_forecasts

# Global variable to store the final results
final_result = None
args_global = None  # store args to use in signal handler


def save_partial_results(signal_number=None, frame=None):
    """
    Save results to CSV even if the job is killed by a timeout or manual interrupt.
    """
    if final_result is not None and args_global is not None:
        os.makedirs("output/optimal_nr_stations_full", exist_ok=True)
        pd.DataFrame([final_result]).to_csv(
            f"output/optimal_nr_stations_full/optimal_station_{args_global.station_index}.csv",
            index=False
        )
    sys.exit(0)


# Register the signal handler for SIGTERM (Slurm timeout) and SIGINT (manual kill)
signal.signal(signal.SIGTERM, save_partial_results)
signal.signal(signal.SIGINT, save_partial_results)


def run_station_experiment(
    df,
    target_station,
    weeks,
    resample,
    hours_to_forecast
):
    """
    Executes the experiment to determine the optimal number of stations for forecasting a target station.

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

    # Step 1: Rank all the stations based on their importance for forecasting the target station
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

    importance_ranking = results_all["importance"]
    ordered_stations = importance_ranking.index.tolist()

    mae_all = results_all["mae_mean"]
    mse_all = results_all["mse_mean"]
    duration_all = results_all["duration_mean_seconds"]

    # Step 2: Determine the optimal number of stations by iteratively testing
    # the performance using the top k stations as exogenous variables
    subset_results = []

    for k in range(1, len(ordered_stations)):
        selected = ordered_stations[:k]

        res_k = repeat_forecasts(
            series,
            exog_df=exog_df[selected],
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

        duration_k = res_k["duration_mean_seconds"]

        subset_results.append({
            "k": k,
            "mse": res_k["mse_mean"],
            "mae": res_k["mae_mean"],
            "duration_seconds": duration_k
        })

        # Save intermediate results in case of timeout
        global final_result
        final_result = pd.DataFrame(subset_results).iloc[-1].to_dict()

    subset_df = pd.DataFrame(subset_results)

    best_row_mse = subset_df.loc[subset_df["mse"].idxmin()]
    best_row_mae = subset_df.loc[subset_df["mae"].idxmin()]

    final_result = {
        "target_station": target_station,
        "optimal_k_mse": int(best_row_mse["k"]),
        "optimal_mse": best_row_mse["mse"],
        "optimal_k_mae": int(best_row_mae["k"]),
        "optimal_mae": best_row_mae["mae"],
        "all_mse": mse_all,
        "mse_difference": mse_all - best_row_mse["mse"],
        "all_mae": mae_all,
        "mae_difference": mae_all - best_row_mae["mae"],
        "duration_all_seconds": duration_all,
        "duration_mae_optimal_seconds": best_row_mae["duration_seconds"],
        "duration_mse_optimal_seconds": best_row_mse["duration_seconds"],
        "nr_stations": len(ordered_stations)
    }

    return final_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ARIMA forecast utilities.")
    parser.add_argument("--weeks", type=int, default=2,
                        help="Number of weeks of data to include (default: 2).")
    parser.add_argument("--resample", type=str, default="1h",
                        help="Resampling interval, e.g. '10min', '30min', '1h' (default: '1h').")
    parser.add_argument("--hours_to_forecast", type=int, default=48,
                        help="Number of hours to forecast into the future (default: 48).")
    parser.add_argument("--station_index", type=int, default=0,
                        help="Index of the target station to analyze (default: 0).")
    parser.add_argument("--input_file", type=str, default="data/Part_AWS/preprocessed.csv",
                        help="Path to the preprocessed data CSV file (default: 'data/Part_AWS/preprocessed.csv').")
    args = parser.parse_args()

    args_global = args  # store globally for signal handler

    # Read the preprocessed data
    df = pd.read_csv(args.input_file)

    # Get the list of unique stations and select the target station based on the provided index
    stations = sorted(df['station_name'].unique())
    target_station = stations[args.station_index]

    # Run the experiment
    result = run_station_experiment(
        df,
        target_station,
        weeks=args.weeks,
        resample=args.resample,
        hours_to_forecast=args.hours_to_forecast
    )

    # Make sure the output directory exists
    os.makedirs("output/optimal_nr_stations_full", exist_ok=True)

    # Save the result to a CSV file
    pd.DataFrame([result]).to_csv(f"output/optimal_nr_stations_full/optimal_station_{args.station_index}.csv", index=False)
