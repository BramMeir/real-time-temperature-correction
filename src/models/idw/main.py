"""
Main script for the IDW baseline model.

Example usage:
python -m src.models.idw.main --input_file ./data/preprocessed.csv --dataset KMI
"""
import argparse
import numpy as np
import pandas as pd
from src.utils.load_station_coordinates import load_station_coordinates
from src.models.idw.execute_forecast import run_single_forecast


def repeat_task(df, target_station, coords, random_seed=42, n_repeats=10, weeks=8, hours_to_forecast=48):
    """
    Repeats the evaluation of the IDW model over random training/forecast periods.

    Input
    -----
    df: DataFrame with time series data, indexed by datetime, one column per station
    target_station: Name of the target station column to predict
    coords: Dict mapping station_name to a (latitude, longitude) tuple
    random_seed: Random seed for reproducibility (default is 42)
    n_repeats: Number of repetitions for the task (default is 10)
    weeks: Number of weeks for the training period (default is 8)
    hours_to_forecast: Number of hours to forecast in each repetition (default is 48)

    Output
    ------
    Prints the average MAE and MSE over all repetitions.
    """
    exog_cols = [col for col in df.columns if col != target_station]

    np.random.seed(random_seed)
    weeks_offset = pd.DateOffset(weeks=weeks)
    hours_offset = pd.DateOffset(hours=hours_to_forecast)
    max_start = df.index.max() - weeks_offset - hours_offset

    possible_starts = df.index[df.index <= max_start]
    start_dates = np.random.choice(possible_starts, size=n_repeats, replace=False)

    mae_scores, mse_scores = [], []

    for start in start_dates:
        train_end = start + weeks_offset
        test_end = train_end + hours_offset

        mae, mse, _ = run_single_forecast(df, target_station, exog_cols=exog_cols, start=start,
                                          train_end=train_end, test_end=test_end, coords=coords)
        mae_scores.append(mae)
        mse_scores.append(mse)

    print(f"\nRunning with weeks={weeks} and hours_to_forecast={hours_to_forecast}")
    print(f"Average MAE over {n_repeats} runs: {np.mean(mae_scores):.4f} ± {np.std(mae_scores):.4f}")
    print(f"Average MSE over {n_repeats} runs: {np.mean(mse_scores):.4f} ± {np.std(mse_scores):.4f}")
    return np.mean(mae_scores), np.mean(mse_scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IDW Baseline Model Script")
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input CSV file')
    parser.add_argument("--dataset", choices=["KMI", "TURKU", "SYNTHETIC"], required=True,
                        help="Dataset name, used to load the station coordinates")
    args = parser.parse_args()

    df = pd.read_csv(args.input_file, index_col='datetime', parse_dates=True)

    # Pivot the DataFrame to have every station as a separate column
    df_pivot = df.pivot_table(index='datetime', columns='station_name', values='temp_dry_avg_2m')
    df_pivot.columns.name = None

    df_pivot = df_pivot.sort_index()
    df_pivot = df_pivot.resample('1h').mean()

    coords = load_station_coordinates(args.dataset)

    # Define the target station
    target_station = "MELLE"

    repeat_task(df_pivot, target_station, coords, random_seed=47, n_repeats=30, weeks=8, hours_to_forecast=48)
