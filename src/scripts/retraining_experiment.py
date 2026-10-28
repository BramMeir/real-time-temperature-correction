import argparse
import pandas as pd
from src.models.neighbour_regression.train import train_neighbour_regression
from src.models.regsarima.train import stage_one_residuals, fit_residual_sarima
from src.scripts.aging_experiment import (
    SEED, STATION_SEED_STRIDE, RESIDUAL_ORDER, RESIDUAL_SEASONAL_ORDER,
    train_two_stage, age_two_stage, forecast_two_stage
)
from src.utils.datasets import DATASETS, load_station_data
from src.utils.generate_forecast_start import generate_forecast_start
from src.utils.experiment_utils import get_worker_df_complete, run_tasks_for_station, write_results_csv

# Every retraining strategy is simulated for this long, forecasting FORECAST_HORIZON_HOURS ahead every
# STEP_DAYS
SIMULATION_DAYS = 60
FORECAST_HORIZON_HOURS = 24
STEP_DAYS = 3

NUMBER_OF_REPEATS = 10

RETRAIN_GAP_DAYS = 3


def full_retrain_two_stage(model, df_window, station, exog_cols):
    """Full retrain from scratch on the trailing training window; the current model is unused."""
    return train_two_stage(df_window, station, exog_cols)


def retrain_lr_only(model, df_window, station, exog_cols):
    """Refit the regression on the training window; the residual SARIMA keeps its parameters."""
    _, residual_model = model
    regression = train_neighbour_regression(df_window, station, exog_cols)

    # The state is only advanced when no retraining happens, so bring it up to date here on the new
    # regression's residuals, without refitting (as for the aging arm)
    _, window_residuals = stage_one_residuals(df_window, station, exog_cols, regression=regression)
    residual_model = residual_model.apply(endog=window_residuals, refit=False)

    return (regression, residual_model), exog_cols


def retrain_sarima_only(model, df_window, station, exog_cols):
    """Refit the residual SARIMA on the training window, under the frozen, original regression."""
    regression, _ = model

    _, window_residuals = stage_one_residuals(df_window, station, exog_cols, regression=regression)
    residual_model = fit_residual_sarima(window_residuals, RESIDUAL_ORDER, RESIDUAL_SEASONAL_ORDER, max_iter=1000)

    return (regression, residual_model), exog_cols


# (model_type, retrain_gap_days, retrain_fn). Every arm retrains from scratch on the same
# trailing training window at the same frequency, differing only in which component(s) get refit
MODEL_CONFIGS = [
    ("no_retrain", None, None),
    ("lr_only", RETRAIN_GAP_DAYS, retrain_lr_only),
    ("sarima_only", RETRAIN_GAP_DAYS, retrain_sarima_only),
    ("both", RETRAIN_GAP_DAYS, full_retrain_two_stage),
]


def _run_single_experiment(task, training_weeks):
    """
    Simulate every retraining strategy of MODEL_CONFIGS for one station and training period.

    Input
    -----
    task: A tuple containing (dataset_name, repeat_id, station, station_index)
    training_weeks: Number of weeks of training data to use

    Output
    ------
    A list of rows, one per (model_type, forecast step):
        [dataset_name, station, repeat_id, model_type, current_time, mae, mse, retrain_event,
         retraining_duration]
    """
    dataset_name, repeat_id, station, station_index = task

    df_complete = get_worker_df_complete()
    series = df_complete[station]
    exog_cols = [c for c in df_complete.columns if c != station]

    forecast_start = generate_forecast_start(
        series=series,
        seed=SEED + station_index * STATION_SEED_STRIDE,
        repeat_id=repeat_id,
        max_history_days=training_weeks * 7,
        max_horizon=SIMULATION_DAYS * 24 + FORECAST_HORIZON_HOURS
    )

    results = []

    for model_type, retrain_gap, retrain_fn in MODEL_CONFIGS:
        train_start = forecast_start - pd.Timedelta(weeks=training_weeks)
        train_end = forecast_start
        model, exog_cols = train_two_stage(df_complete[train_start:train_end], station, exog_cols)

        last_retrain_time = forecast_start
        last_update_time = forecast_start
        current_time = forecast_start
        end_time = forecast_start + pd.Timedelta(days=SIMULATION_DAYS)

        while current_time < end_time:
            retrain_event = 0
            retraining_duration = 0

            if model_type != "no_retrain" and (current_time - last_retrain_time) >= pd.Timedelta(days=retrain_gap):
                retrain_event = 1
                start_time = pd.Timestamp.now()

                new_train_start = current_time - pd.Timedelta(weeks=training_weeks)
                model, exog_cols = retrain_fn(model, df_complete[new_train_start:current_time], station, exog_cols)
                train_start = new_train_start

                retraining_duration = (pd.Timestamp.now() - start_time).total_seconds()
                last_retrain_time = current_time

            # If no retraining happened, still advance the model's state with the new observations
            if not retrain_event and current_time > last_update_time:
                update_start = last_update_time + pd.Timedelta(hours=1)
                model = age_two_stage(model, df_complete[update_start:current_time], station, exog_cols)

            test_begin = current_time
            test_end = current_time + pd.Timedelta(hours=FORECAST_HORIZON_HOURS)
            mae, mse = forecast_two_stage(repeat_id, df_complete, station, model, exog_cols, train_start, test_begin, test_end)

            results.append([
                dataset_name, station, repeat_id, model_type, current_time, mae, mse,
                retrain_event, retraining_duration
            ])

            last_update_time = current_time
            current_time += pd.Timedelta(days=STEP_DAYS)

    return results


def run_retraining_experiment(dataset_name, training_weeks, output_path, max_workers=None):
    """
    Run the retraining-strategy experiment for every station of one dataset and save the results to a
    CSV file.

    Input
    -----
    dataset_name: Name of the dataset to run the experiment on (must be a key of DATASETS)
    training_weeks: Number of weeks of training data to use
    output_path: Path of the CSV file to write the results to
    max_workers: Number of worker processes (default is None, which uses one per core)
    """
    results_all = []
    dataset_info = DATASETS[dataset_name]

    df = pd.read_csv(dataset_info["file"])

    for station_index, station in enumerate(dataset_info["stations"]):
        print(f"Running retraining experiments for dataset {dataset_name}, station {station}...")

        df_complete = load_station_data(df, station)

        tasks = [
            (dataset_name, repeat_id, station, station_index)
            for repeat_id in range(NUMBER_OF_REPEATS)
        ]

        results_all.extend(run_tasks_for_station(
            df_complete, tasks, _run_single_experiment, (training_weeks,), max_workers=max_workers
        ))

    write_results_csv(output_path, [
        "dataset",
        "station",
        "repeat_id",
        "model_type",
        "current_time",
        "mae",
        "mse",
        "retrain_event",
        "retraining_duration"
    ], results_all)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Determine which part of RegSARIMA needs retraining (the"
                                                 " regression, the residual SARIMA, or both) across all"
                                                 " stations of one dataset. The results are saved to a CSV"
                                                 " file for later analysis.")
    parser.add_argument("--dataset", type=str, choices=DATASETS.keys(), required=True,
                        help="The dataset to run the experiment on.")
    parser.add_argument("--max_workers", type=int, default=None,
                        help="Number of worker processes (default: one per core). Lower it if the"
                             " full core count runs into memory pressure")
    args = parser.parse_args()

    run_retraining_experiment(
        dataset_name=args.dataset,
        training_weeks=8,
        output_path=f"output/retraining_results_regression_sarima_errors/results_{args.dataset}.csv",
        max_workers=args.max_workers
    )
