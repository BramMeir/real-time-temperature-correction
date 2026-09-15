"""
Module: retraining_experiment.py

Generic engine for the "retraining strategy" experiment: simulate a long deployment, comparing how
different retraining strategies keep forecast error under control over time.

Each strategy is a (model_type, retrain_gap_days, window_kind, retrain_fn) config. window_kind is
"trailing" (retrain_fn refits on the trailing training window, e.g. from scratch) or "delta" (retrain_fn
warm-starts from the window since the last update). Shared by compare_retraining_strategies.py (ARIMAX)
and compare_retraining_strategies_RegressionSARIMAErrors.py (the two-stage model); reuses DATASETS/SEED
and station loading from aging_experiment.py, and the pool/CSV helpers from experiment_utils.py.
"""
import pandas as pd
from src.utils.generate_forecast_start import generate_forecast_start
from src.evaluation.aging_experiment import DATASETS, SEED, STATION_SEED_STRIDE, _load_station_data
from src.evaluation.experiment_utils import get_worker_df_complete, run_tasks_for_station, write_results_csv

# Every retraining strategy is simulated for this long, forecasting FORECAST_HORIZON_HOURS ahead every
# STEP_DAYS, exactly as in the paper's retraining-strategy comparison for ARIMAX
SIMULATION_DAYS = 60
FORECAST_HORIZON_HOURS = 24
STEP_DAYS = 3

NUMBER_OF_REPEATS = 10


def _run_single_experiment(task, model_configs, train_fn, age_fn, forecast_fn, training_weeks):
    """
    Simulate every retraining strategy for one station and training period.

    Input
    -----
    task: A tuple containing (dataset_name, repeat_id, station, station_index)
    model_configs: List of (model_type, retrain_gap_days, window_kind, retrain_fn) tuples, see the
      module docstring. "no_retrain" has retrain_gap_days/window_kind/retrain_fn all None
    train_fn, age_fn, forecast_fn: Same hooks as aging_experiment.py's _run_single_experiment

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

    for model_type, retrain_gap, window_kind, retrain_fn in model_configs:
        train_start = forecast_start - pd.Timedelta(weeks=training_weeks)
        train_end = forecast_start
        model, exog_cols = train_fn(df_complete[train_start:train_end], station, exog_cols)

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

                if window_kind == "trailing":
                    new_train_start = current_time - pd.Timedelta(weeks=training_weeks)
                    model, exog_cols = retrain_fn(model, df_complete[new_train_start:current_time], station, exog_cols)
                    train_start = new_train_start
                elif window_kind == "delta":
                    update_start = last_update_time + pd.Timedelta(hours=1)
                    model = retrain_fn(model, df_complete[update_start:current_time], station, exog_cols)

                retraining_duration = (pd.Timestamp.now() - start_time).total_seconds()
                last_retrain_time = current_time

            # If no retraining happened, still advance the model's state with the new observations
            if not retrain_event and current_time > last_update_time:
                update_start = last_update_time + pd.Timedelta(hours=1)
                model = age_fn(model, df_complete[update_start:current_time], station, exog_cols)

            test_begin = current_time
            test_end = current_time + pd.Timedelta(hours=FORECAST_HORIZON_HOURS)
            mae, mse = forecast_fn(repeat_id, df_complete, station, model, exog_cols, train_start, test_begin, test_end)

            results.append([
                dataset_name, station, repeat_id, model_type, current_time, mae, mse,
                retrain_event, retraining_duration
            ])

            last_update_time = current_time
            current_time += pd.Timedelta(days=STEP_DAYS)

    return results


def run_retraining_experiment(dataset_name, training_weeks, train_fn, age_fn, forecast_fn, model_configs,
                              output_path, number_of_repeats=NUMBER_OF_REPEATS, max_workers=None):
    """
    Run the retraining-strategy experiment for every station of one dataset and save the results to a
    CSV file. See _run_single_experiment for the model-specific hooks and model_configs format.
    """
    results_all = []
    dataset_info = DATASETS[dataset_name]

    for station_index, station in enumerate(dataset_info["stations"]):
        print(f"Running retraining experiments for dataset {dataset_name}, station {station}...")

        df_complete = _load_station_data(dataset_info, station)

        tasks = [
            (dataset_name, repeat_id, station, station_index)
            for repeat_id in range(number_of_repeats)
        ]

        results_all.extend(run_tasks_for_station(
            df_complete, tasks, _run_single_experiment,
            (model_configs, train_fn, age_fn, forecast_fn, training_weeks),
            max_workers=max_workers
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
