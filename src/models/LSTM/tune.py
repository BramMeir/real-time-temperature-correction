import argparse
import numpy as np
import pandas as pd
import collections
from concurrent.futures import ProcessPoolExecutor, as_completed
from src.utils.datasets import load_station_data
from src.models.LSTM.execute_forecast import run_single_forecast


def tune(df, target_station, previous_time_steps, random_seed=42, n_repeats=10,
         weeks=2, hours_to_forecast=48):
    """
    Runs the Bayesian hyperparameter search of the LSTM model on n_repeats random training periods.

    Input
    -----
    df: DataFrame with time series data, indexed by datetime
    target_station: Name of the target station column to predict
    previous_time_steps: Number of previous time steps to include as features
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

    with ProcessPoolExecutor() as executor:
        futures = []

        for i, (start, train_end, end) in enumerate(date_ranges):
            df_slice = df.loc[start:end].copy()
            futures.append(
                executor.submit(
                    run_single_forecast, df_slice, i, target_station, None, previous_time_steps, start, train_end, end, "bayes_search"
                )
            )

        for f in as_completed(futures):
            mae, mse, best_hp = f.result()
            mae_scores.append(mae)
            mse_scores.append(mse)
            best_hp_list.append(best_hp)

    print("Best hyperparameters from Bayesian search (most frequent/average values):")

    # Filter out None values
    filtered_hp = [hp for hp in best_hp_list if hp is not None]

    if filtered_hp:
        # Collect all values per hyperparameter
        hp_values = collections.defaultdict(list)
        for hp in filtered_hp:
            for key, value in hp.values.items():
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
    parser = argparse.ArgumentParser(description="LSTM hyperparameter search")
    parser.add_argument('--input_file', type=str, default='data/RMI/observations.csv',
                        help='Path to the input CSV file (default: data/RMI/observations.csv)')
    args = parser.parse_args()

    # Define the target station
    target_station = "MELLE"

    # The hourly data of the experiments: the observation at each full hour, with gaps interpolated
    df_pivot = load_station_data(pd.read_csv(args.input_file), target_station)

    with open(f"output/lstm_bayes_search_{target_station}.csv", "w") as f:
        f.write("previous_time_steps,weeks,mae,mse,most_frequent_hp\n")

        for repeat_step in [8, 12, 24]:
            for weeks in [4, 8, 12]:
                mae, mse, most_frequent_hp = tune(df_pivot, target_station, previous_time_steps=repeat_step,
                                                  random_seed=47, n_repeats=10, weeks=weeks,
                                                  hours_to_forecast=48)
                f.write(f"{repeat_step},{weeks},{mae:.4f},{mse:.4f},{most_frequent_hp}\n")
