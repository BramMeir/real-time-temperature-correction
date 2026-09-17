"""
Main script for XGBoost model training and hyperparameter search.

The search is run at the settings the model comparison uses (24 lags, 8 training weeks), so the
resulting hyperparameters are the ones the comparison will actually run with.

python -m src.models.xgboost_model.main --input_file ./data/Full_AWS/preprocessed_10m_2022_2025.csv --mode bayes_search
"""
import argparse
import collections
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.models.xgboost_model.execute_forecast import run_single_forecast

# Settings of the model comparison, so the search optimizes for the setting that is reported
PREVIOUS_TIME_STEPS = 24
TRAINING_WEEKS = 8
HOURS_TO_FORECAST = 48


def repeat_task(df, target_station, previous_time_steps, exog_cols, random_seed=47, n_repeats=10,
                weeks=8, hours_to_forecast=48, mode="repeat_forecast"):
    """
    Repeats the training and evaluation of the XGBoost model over random training periods.

    Input
    -----
    df: DataFrame with time series data, indexed by datetime
    target_station: Name of the target station column to predict
    previous_time_steps: Number of previous time steps to include as features
    exog_cols: List of additional exogenous columns to include as features
    random_seed: Random seed for reproducibility (default is 47)
    n_repeats: Number of repetitions for the task (default is 10)
    weeks: Number of weeks for the training period (default is 8)
    hours_to_forecast: Number of hours to forecast in each repetition (default is 48)
    mode: Mode of operation for the task (default is "repeat_forecast")

    Output
    ------
    mae, mse: Averages over all repetitions
    most_frequent_hp: Most frequent hyperparameters over the repetitions (only for "bayes_search")
    """
    np.random.seed(random_seed)
    weeks_offset = pd.DateOffset(weeks=weeks)
    hours_offset = pd.DateOffset(hours=hours_to_forecast)
    max_start = df.index.max() - weeks_offset - hours_offset

    possible_starts = df.index[df.index <= max_start]

    # Pregenerate all the start and end dates
    start_dates = np.random.choice(possible_starts, size=n_repeats, replace=False)
    date_ranges = [(start, start + weeks_offset, start + weeks_offset + hours_offset)
                   for start in start_dates]

    mae_scores, mse_scores, best_hp_list, importances_list = [], [], [], []

    with ThreadPoolExecutor() as executor:
        futures = []

        for (start, train_end, end) in date_ranges:
            df_slice = df.loc[start:end].copy()
            futures.append(
                executor.submit(
                    run_single_forecast, df_slice, target_station, None, previous_time_steps,
                    exog_cols, start, train_end, end, mode
                )
            )

        for f in as_completed(futures):
            mae, mse, best_hp, importances = f.result()
            mae_scores.append(mae)
            mse_scores.append(mse)

            if best_hp is not None:
                best_hp_list.append(best_hp)

            if importances is not None:
                importances_list.append(importances)

    if importances_list:
        all_importances = pd.concat(importances_list)
        avg_importances = all_importances.groupby('Feature').mean().sort_values(by='Importance', ascending=False)
        print("Average Feature Importances over all runs:")
        print(avg_importances.head(10))

    most_frequent_hp = None
    if mode == "bayes_search":
        # Collect all values per hyperparameter and keep the most frequent one over the repetitions
        hp_values = collections.defaultdict(list)
        for hp in best_hp_list:
            for key, value in hp.items():
                hp_values[key].append(value)

        most_frequent_hp = {key: collections.Counter(values).most_common(1)[0][0]
                            for key, values in hp_values.items()}

        print("\nMost frequent hyperparameters:")
        for k, v in most_frequent_hp.items():
            print(f"  {k}: {v}")

    print(f"\nRunning with previous_time_steps={previous_time_steps} and weeks={weeks}")
    print(f"Average MAE over {n_repeats} runs: {np.mean(mae_scores):.4f} ± {np.std(mae_scores):.4f}")
    print(f"Average MSE over {n_repeats} runs: {np.mean(mse_scores):.4f} ± {np.std(mse_scores):.4f}")
    return np.mean(mae_scores), np.mean(mse_scores), most_frequent_hp


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XGBoost Model Training Script")
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input CSV file')
    parser.add_argument("--mode", choices=["repeat_forecast", "bayes_search"],
                        default="repeat_forecast", help="Select which XGBoost task to run.")
    parser.add_argument("--target_station", type=str, default="MELLE", help="Station to tune on")
    args = parser.parse_args()

    df = pd.read_csv(args.input_file, index_col='datetime', parse_dates=True)

    # Pivot the DataFrame to have every station as a separate column
    df_pivot = df.pivot_table(index='datetime', columns='station_name', values='temp_dry_avg_2m')
    df_pivot.columns.name = None
    df_pivot = df_pivot.sort_index().resample('1h').mean()

    exog_cols = [col for col in df_pivot.columns if col != args.target_station]

    mae, mse, most_frequent_hp = repeat_task(
        df_pivot, args.target_station, previous_time_steps=PREVIOUS_TIME_STEPS, exog_cols=exog_cols,
        random_seed=47, n_repeats=10, weeks=TRAINING_WEEKS, hours_to_forecast=HOURS_TO_FORECAST,
        mode=args.mode
    )

    if args.mode == "bayes_search":
        with open(f"output/xgboost_bayes_search_{args.target_station}.csv", "w") as f:
            f.write("previous_time_steps,weeks,mae,mse,most_frequent_hp\n")
            f.write(f"{PREVIOUS_TIME_STEPS},{TRAINING_WEEKS},{mae:.4f},{mse:.4f},\"{most_frequent_hp}\"\n")
