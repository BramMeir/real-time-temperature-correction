"""
Main script for Linear Regression (of the neighbouring stations) baseline training.

Example usage:
python -m src.models.linear_regression.main --input_file ./data/preprocessed.csv
"""
import argparse
import numpy as np
import pandas as pd
from src.models.linear_regression.execute_forecast import run_single_forecast


def repeat_task(df, target_station, exog_cols, random_seed=42, n_repeats=10, weeks=8, hours_to_forecast=48):
    """
    Repeats the training and evaluation of the linear regression baseline over random training/forecast periods.

    Input
    -----
    df: DataFrame with time series data, indexed by datetime
    target_station: Name of the target station column to predict
    exog_cols: List of neighbouring station column names used as regressors
    random_seed: Random seed for reproducibility (default is 42)
    n_repeats: Number of repetitions for the task (default is 10)
    weeks: Number of weeks for the training period (default is 8)
    hours_to_forecast: Number of hours to forecast in each repetition (default is 48)

    Output
    ------
    Prints the average MAE and MSE, and the average coefficients over all repetitions.
    """
    # Select random n_repeats training/forecast periods from the df
    np.random.seed(random_seed)
    weeks_offset = pd.DateOffset(weeks=weeks)
    hours_offset = pd.DateOffset(hours=hours_to_forecast)
    max_start = df.index.max() - weeks_offset - hours_offset

    possible_starts = df.index[df.index <= max_start]

    start_dates = np.random.choice(possible_starts, size=n_repeats, replace=False)
    date_ranges = [(start, start + weeks_offset, start + weeks_offset + hours_offset)
                  for start in start_dates]

    mae_scores, mse_scores, coefficients_list = [], [], []

    for (start, train_end, test_end) in date_ranges:
        df_slice = df.loc[start:test_end].copy()

        mae, mse, best_hp, coefficients = run_single_forecast(
            df_slice, target_station, exog_cols=exog_cols, start=start, train_end=train_end, test_end=test_end
        )

        mae_scores.append(mae)
        mse_scores.append(mse)
        coefficients_list.append(coefficients)

    all_coefficients = pd.concat(coefficients_list)
    avg_coefficients = all_coefficients.groupby('Feature').mean().sort_values(by='Coefficient', ascending=False,
                                                                              key=abs)
    print("Average coefficients over all runs:")
    print(avg_coefficients)

    print(f"\nRunning with weeks={weeks}")
    print(f"Average MAE over {n_repeats} runs: {np.mean(mae_scores):.4f} ± {np.std(mae_scores):.4f}")
    print(f"Average MSE over {n_repeats} runs: {np.mean(mse_scores):.4f} ± {np.std(mse_scores):.4f}")
    return np.mean(mae_scores), np.mean(mse_scores)


if __name__ == "__main__":
    # Define the arguments for the main script
    parser = argparse.ArgumentParser(description="Linear Regression Baseline Training Script")
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

    repeat_task(df_pivot, target_station, exog_cols=exog_cols, random_seed=47, n_repeats=30,
                weeks=8, hours_to_forecast=48)
