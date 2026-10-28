import argparse
import numpy as np
import pandas as pd
import collections
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.models.MLP.execute_forecast import run_single_forecast


def tune(df, target_station, previous_time_steps, exog_cols, random_seed=42, n_repeats=10,
         weeks=2, hours_to_forecast=48):
    """
    Runs the Bayesian hyperparameter search of the MLP model on n_repeats random training periods.

    Input
    -----
    df: DataFrame with time series data, indexed by datetime
    target_station: Name of the target station column to predict
    previous_time_steps: Number of previous time steps to include as features
    exog_cols: List of additional exogenous columns to include as features
    random_seed: Random seed for reproducibility (default is 42)
    n_repeats: Number of repetitions for the task (default is 10)
    weeks: Number of weeks for the training period (default is 2)
    hours_to_forecast: Number of hours to forecast in each repetition (default is 48)

    Output
    ------
    The average MAE and MSE over all repetitions, and the most frequent hyperparameters.
    """
    # Select random n_repeats parts with 2-weeks training and 2 days test from the df
    np.random.seed(random_seed)
    weeks_offset = pd.DateOffset(weeks=weeks)
    hours_offset = pd.DateOffset(hours=hours_to_forecast)
    max_start = df.index.max() - weeks_offset - hours_offset

    possible_starts = df.index[df.index <= max_start]

    # Pregenerate all the start and end dates
    start_dates = np.random.choice(possible_starts, size=n_repeats, replace=False)
    date_ranges = [(start, start + weeks_offset, start + weeks_offset + hours_offset)
                   for start in start_dates]

    mae_scores, mse_scores, best_hp_list = [], [], []

    with ThreadPoolExecutor() as executor:
        futures = []

        for (start, train_end, end) in date_ranges:
            df_slice = df.loc[start:end].copy()
            futures.append(
                executor.submit(
                    run_single_forecast, df_slice, target_station, None, previous_time_steps, exog_cols, start, train_end, end, "bayes_search"
                )
            )

        for f in as_completed(futures):
            mae, mse, best_hp = f.result()
            mae_scores.append(mae)
            mse_scores.append(mse)

            if best_hp is not None:
                best_hp_list.append(best_hp)

    print("Best hyperparameters from Bayesian search (most frequent/average values):")

    # Filter out None values
    filtered_hp = [hp for hp in best_hp_list if hp is not None]

    if filtered_hp:
        # Collect all values per hyperparameter
        hp_values = collections.defaultdict(list)
        for hp in filtered_hp:
            for key, value in hp.items():
                hp_values[key].append(value)

        # Compute most frequent values
        most_frequent_hp = {}
        for key, values in hp_values.items():
            most_frequent_hp[key] = collections.Counter(values).most_common(1)[0][0]

        print("\nMost frequent hyperparameters:")
        for k, v in most_frequent_hp.items():
            print(f"  {k}: {v}")

    print(f"\nRunning with previous_time_steps={previous_time_steps} and weeks={weeks}")
    print(f"Average MAE over {n_repeats} runs: {np.mean(mae_scores):.4f} ± {np.std(mae_scores):.4f}")
    print(f"Average MSE over {n_repeats} runs: {np.mean(mse_scores):.4f} ± {np.std(mse_scores):.4f}")

    return np.mean(mae_scores), np.mean(mse_scores), most_frequent_hp


if __name__ == "__main__":
    # Define the arguments for the main script
    parser = argparse.ArgumentParser(description="MLP hyperparameter search")
    parser.add_argument('--input_file', type=str, default='data/RMI/observations.csv',
                        help='Path to the input CSV file (default: data/RMI/observations.csv)')
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

    with open(f"output/mlp_bayes_search_{target_station}.csv", "w") as f:
        f.write("previous_time_steps,weeks,mae,mse,most_frequent_hp\n")

        for repeat_step in [3, 8, 12, 24]:
            for weeks in [2, 4, 8, 12]:
                mae, mse, most_frequent_hp = tune(df_pivot, target_station, previous_time_steps=repeat_step,
                                                  exog_cols=exog_cols, random_seed=47, n_repeats=10, weeks=weeks,
                                                  hours_to_forecast=48)
                f.write(f"{repeat_step},{weeks},{mae:.4f},{mse:.4f},{most_frequent_hp}\n")
