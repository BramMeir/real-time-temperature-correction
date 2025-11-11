"""
Main script for Random Forest model training.

Example usage:
python -m src.models.random_forest.main --input ./data/preprocessed.csv
"""
import argparse
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.data.create_supervised import create_supervised_dataset
from src.models.random_forest.train import train_random_forest
from src.models.random_forest.evaluate_forecast import evaluate_forecast


def _run_single_forecast(df, target_station, previous_time_steps, exog_cols, seed, start, train_end, end):
    """
    Runs a single training and evaluation of the Random Forest model.

    Input
    -----
    X_train: Training features DataFrame
    y_train: Training target Series
    X_test: Test features DataFrame
    y_test: Test target Series
    seed: Random seed for reproducibility (default is 42)

    """
    # Print the date range being used
    print(f"Running forecast from {start} to {end} with training until {train_end}")

    # Create supervised dataset
    X, y = create_supervised_dataset(df, target_station=target_station,
                                     previous_time_steps=previous_time_steps,
                                     exog_cols=exog_cols)

    # Select the data based on the provided date ranges
    X_train, y_train = X.loc[start:train_end], y.loc[start:train_end]
    X_test, y_test = X.loc[train_end:end], y.loc[train_end:end]

    # Train the model
    model, _ = train_random_forest(X_train, y_train, X_test, y_test, random_seed=seed)

    # Evaluate using recursive multi-step forecasting
    mae, rmse = evaluate_forecast(model, y_train, X_test, y_test, plot=False)
    mse = rmse ** 2

    print(f"Completed forecast from {start} to {end}: MAE={mae:.4f}, MSE={mse:.4f}")

    return mae, mse


def repeat_forecasts(df, target_station, previous_time_steps, exog_cols, random_seed=42, n_repeats=10,
                     weeks=2, hours_to_forecast=48):
    """
    Repeats the training and evaluation of the Random Forest model.

    Input
    -----
    df: DataFrame with time series data, indexed by datetime
    target_station: Name of the target station column to predict
    previous_time_steps: Number of previous time steps to include as features
    exog_cols: List of additional exogenous columns to include as features
    seed: Random seed for reproducibility (default is 42)

    """
    # Select random n_repeast parts with 2-weeks training and 2 days test from the df
    np.random.seed(random_seed)
    weeks_offset = pd.DateOffset(weeks=weeks)
    hours_offset = pd.DateOffset(hours=hours_to_forecast)
    max_start = df.index.max() - weeks_offset - hours_offset

    possible_starts = df.index[df.index <= max_start]

    # Pregenerate all the start and end dates
    start_dates = np.random.choice(possible_starts, size=n_repeats, replace=False)
    date_ranges = [(start, start + weeks_offset, start + weeks_offset + hours_offset)
                   for start in start_dates]

    mae_scores, mse_scores = [], []

    with ThreadPoolExecutor() as executor:
        futures = []

        for (start, train_end, end) in date_ranges:
            df_slice = df.loc[start:end].copy()
            futures.append(
                executor.submit(
                    _run_single_forecast, df_slice, target_station, previous_time_steps, exog_cols, random_seed, start, train_end, end
                )
            )

        for f in as_completed(futures):
            print("A forecast run has completed.")
            mae, mse = f.result()
            mae_scores.append(mae)
            mse_scores.append(mse)

    print(f"Average MAE over {n_repeats} runs: {np.mean(mae_scores):.4f} ± {np.std(mae_scores):.4f}")
    print(f"Average MSE over {n_repeats} runs: {np.mean(mse_scores):.4f} ± {np.std(mse_scores):.4f}")


if __name__ == "__main__":
    # Define the arguments for the main script
    parser = argparse.ArgumentParser(description="Random Forest Model Training Script")
    parser.add_argument('--input', type=str, required=True, help='Path to the input CSV file')
    args = parser.parse_args()

    # Read the dataset
    df = pd.read_csv(args.input, index_col='datetime', parse_dates=True)

    # Pivot the DataFrame to have every station as a separate column
    df_pivot = df.pivot_table(index='datetime', columns='station_name', values='temp_dry_avg_2m')

    # Remove the multi-index on columns if present (to simplify column access)
    df_pivot.columns.name = None

    # Sort by datetime index
    df_pivot = df_pivot.sort_index()

    # Resample to hourly frequency if not already
    df_pivot = df_pivot.resample('1h').mean()

    # Define the target station
    target_station = "Melle AWS"

    # Define the other stations as exogenous variables
    exog_cols = [col for col in df_pivot.columns if col != target_station]

    # Run repeated forecasts
    repeat_forecasts(df_pivot, target_station, previous_time_steps=24, exog_cols=exog_cols,
                     random_seed=53, n_repeats=30, weeks=2, hours_to_forecast=48)
