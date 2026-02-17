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
from src.models.arima.repeat_forecast import repeat_forecasts


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
    print(f"MAE: {results_all['mae_mean']:.4f}")
    print(f"MSE: {results_all['mse_mean']:.4f}")
    print(f"Average duration (seconds): {results_all['duration_mean_seconds']:.2f}")

    # Step 2: Determine the optimal number of stations to include
    print("\nStep 2: Determining the optimal number of stations to include...")

    ordered_stations = importance_ranking.index.tolist()
    sub_stations_results = []

    # Keep track of best performance to implement an early stopping criterion
    best_mse = float("inf")

    for k in range(1, max(len(ordered_stations) + 1, 30)):  # Test up to all stations or a maximum of 30
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
