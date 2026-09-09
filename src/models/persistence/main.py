"""
Main script for the persistence baseline model.

Example usage:
python -m src.models.persistence.main --input_file ./data/preprocessed.csv
"""
import argparse
import numpy as np
import pandas as pd
from src.models.persistence.execute_forecast import run_single_forecast


def repeat_task(df, target_station, exog_cols, random_seed=42, n_repeats=10, weeks=2, hours_to_forecast=48):
    """
    Repeats the training and evaluation of the persistence baseline model.

    Input
    -----
    df: DataFrame with time series data, indexed by datetime
    target_station: Name of the target station column to predict
    exog_cols: List of additional station columns (unused, accepted only for interface compatibility)
    random_seed: Random seed for reproducibility (default is 42)
    n_repeats: Number of repetitions for the task (default is 10)
    weeks: Number of weeks for the training period (default is 2)
    hours_to_forecast: Number of hours to forecast in each repetition (default is 48)

    Output
    ------
    Prints the average MAE and MSE over all repetitions.
    """
    # Select random n_repeats parts with `weeks`-weeks training and `hours_to_forecast`-hours test from the df
    np.random.seed(random_seed)
    weeks_offset = pd.DateOffset(weeks=weeks)
    hours_offset = pd.DateOffset(hours=hours_to_forecast)
    max_start = df.index.max() - weeks_offset - hours_offset

    possible_starts = df.index[df.index <= max_start]

    start_dates = np.random.choice(possible_starts, size=n_repeats, replace=False)
    date_ranges = [(start, start + weeks_offset, start + weeks_offset + hours_offset)
                   for start in start_dates]

    mae_scores, mse_scores = [], []

    # Persistence is instantaneous to train/forecast, so no need for parallel execution
    for (start, train_end, end) in date_ranges:
        df_slice = df.loc[start:end].copy()
        mae, mse, _ = run_single_forecast(df_slice, target_station, exog_cols=exog_cols,
                                          start=start, train_end=train_end, test_end=end, mode="forecast")
        mae_scores.append(mae)
        mse_scores.append(mse)

    print(f"\nRunning with weeks={weeks} and hours_to_forecast={hours_to_forecast}")
    print(f"Average MAE over {n_repeats} runs: {np.mean(mae_scores):.4f} ± {np.std(mae_scores):.4f}")
    print(f"Average MSE over {n_repeats} runs: {np.mean(mse_scores):.4f} ± {np.std(mse_scores):.4f}")
    return np.mean(mae_scores), np.mean(mse_scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Persistence Baseline Model Script")
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input CSV file')
    args = parser.parse_args()

    # Read the dataset
    df = pd.read_csv(args.input_file, index_col='datetime', parse_dates=True)

    # Pivot the DataFrame to have every station as a separate column
    df_pivot = df.pivot_table(index='datetime', columns='station_name', values='temp_dry_avg_2m')

    # Remove the multi-index on columns if present (to simplify column access)
    df_pivot.columns.name = None

    # Sort by datetime index
    df_pivot = df_pivot.sort_index()

    # Resample to hourly frequency if not already
    df_pivot = df_pivot.resample('1h').mean()

    # Define the target station
    target_station = "MELLE"

    # Define the other stations as exogenous variables
    exog_cols = [col for col in df_pivot.columns if col != target_station]

    # Repeat over random training/forecast periods
    repeat_task(df_pivot, target_station, exog_cols=exog_cols, random_seed=47, n_repeats=30,
               weeks=8, hours_to_forecast=48)
