import argparse
import pandas as pd
from src.models.regsarima.train import train_regression_sarima_errors, stage_one_residuals
from src.models.regsarima.execute_forecast import run_single_forecast as run_regression_sarima_errors
from src.utils.generate_forecast_start import generate_forecast_start
from src.utils.datasets import DATASETS as SHARED_DATASETS, load_station_data
from src.utils.experiment_utils import get_worker_df_complete, run_tasks_for_station, write_results_csv


# The aging and retraining experiments add three synthetic targets to those of the other experiments
DATASETS = {
    **SHARED_DATASETS,
    "SYNTHETIC": {
        "file": SHARED_DATASETS["SYNTHETIC"]["file"],
        "stations": SHARED_DATASETS["SYNTHETIC"]["stations"] + ["Etalle", "Centrum_Malmedy", "Markt_Hoei"]
    }
}

# Seed that is used to define the training periods (for reproducibility)
SEED = 47

# Seed offset per station, so every station draws its own forecast starts instead of every station of
# a network repeating the same weather. generate_forecast_start adds the repeat id to the seed, so the
# stride has to exceed the number of repeats to keep the streams apart
STATION_SEED_STRIDE = 10_000

MAX_AGING_GAP_DAYS = 90

NUMBER_OF_REPEATS = 100

FORECAST_HORIZON_HOURS = 48

# Orders of the residual SARIMA, selected on the stage-one residuals
RESIDUAL_ORDER, RESIDUAL_SEASONAL_ORDER = (3, 0, 0), (1, 0, 0, 24)


def train_two_stage(df_window, station, exog_cols):
    """
    Train RegSARIMA on the training window, on every neighbouring station.

    Output
    ------
    model: The trained (regression, residual_model)
    exog_cols: The neighbouring stations the model uses, the ones it was given
    """
    model = train_regression_sarima_errors(
        df=df_window,
        target_station=station,
        exog_cols=exog_cols,
        arima_order=RESIDUAL_ORDER,
        seasonal_order=RESIDUAL_SEASONAL_ORDER,
        max_iter=1000
    )
    return model, exog_cols


def age_two_stage(model, df_gap, station, exog_cols):
    """
    Advance the residual SARIMA's state with the gap's true stage-one residuals, without refitting. The
    regression has no state of its own to advance, it only needs the current neighbour readings.
    """
    regression, residual_model = model

    _, gap_residuals = stage_one_residuals(df_gap, station, exog_cols, regression=regression)

    return regression, residual_model.append(endog=gap_residuals, refit=False)


def forecast_two_stage(repeat_id, df_complete, station, model, exog_cols, train_start, test_begin, test_end):
    """Reconstruct the target over [test_begin, test_end] with a trained model, returning its MAE and MSE."""
    result = run_regression_sarima_errors(
        df=df_complete,
        target_station=station,
        model=model,
        exog_cols=exog_cols,
        start=train_start,
        train_end=test_begin,
        test_end=test_end,
        arima_order=RESIDUAL_ORDER,
        seasonal_order=RESIDUAL_SEASONAL_ORDER,
        mode="forecast"
    )
    return result[:2]


def _run_single_experiment(task, training_weeks):
    """
    Run a single aging experiment: train once, then re-forecast at every age gap from a state that
    has been advanced (never retrained) with the true observations up to that point.

    Input
    -----
    task: A tuple containing (dataset_name, repeat_id, station, station_index)
    training_weeks: Number of weeks of training data to use

    Output
    ------
    A list containing the results of the experiment, one row per age gap:
        [dataset_name, station, training_weeks, train_start, train_end, age_gap, forecast_horizon,
         mae, mse, training_duration]
    """
    dataset_name, repeat_id, station, station_index = task

    df_complete = get_worker_df_complete()
    series = df_complete[station]
    exog_cols = [c for c in df_complete.columns if c != station]

    # Generate random training period (start and end date) for the given dataset and station. Every
    # station gets its own seed so stations within a dataset don't all sample the same forecast starts
    forecast_start = generate_forecast_start(
        series=series,
        seed=SEED + station_index * STATION_SEED_STRIDE,
        repeat_id=repeat_id,
        max_history_days=training_weeks * 7,
        max_horizon=MAX_AGING_GAP_DAYS * 24 + FORECAST_HORIZON_HOURS
    )

    train_start = forecast_start - pd.Timedelta(days=training_weeks * 7)
    train_end = forecast_start

    # Train the model once on the training period to reuse for every age gap
    start_time = pd.Timestamp.now()
    model, exog_cols = train_two_stage(df_complete[train_start:train_end], station, exog_cols)
    training_duration = pd.Timestamp.now() - start_time

    # Evaluate the model for all the 'aging gaps' (0 to MAX_AGING_GAP_DAYS, step 3) to see how
    # performance degrades with aging
    results = []
    for age_gap in range(0, MAX_AGING_GAP_DAYS + 1, 3):
        # Set the test period as [train_end + age_gap, train_end + age_gap + forecast horizon]
        test_begin = train_end + pd.Timedelta(days=age_gap)
        test_end = test_begin + pd.Timedelta(hours=FORECAST_HORIZON_HOURS)

        aged_model = model

        if age_gap > 0:
            # The regression/coefficients are never refit, so they are just as frozen as the aged
            # model's parameters: only its internal state is advanced with the true observations
            gap_start = train_end + pd.Timedelta(hours=1)  # Right after the training period
            aged_model = age_two_stage(model, df_complete[gap_start:test_begin], station, exog_cols)

        mae, mse = forecast_two_stage(
            repeat_id, df_complete, station, aged_model, exog_cols, train_start, test_begin, test_end
        )

        results.append([
            dataset_name,
            station,
            training_weeks,
            train_start,
            train_end,
            age_gap,
            FORECAST_HORIZON_HOURS,
            mae,
            mse,
            training_duration
        ])

    return results


def run_aging_experiment(dataset_name, training_weeks, output_path, max_workers=None):
    """
    Run the aging experiment for every station of one dataset and save the results to a CSV file.

    Input
    -----
    dataset_name: Name of the dataset to run the experiment on (must be a key of DATASETS)
    training_weeks: Number of weeks of training data to use
    output_path: Path of the CSV file to write the results to
    max_workers: Number of worker processes (default is None, which uses one per core). Lower it if
      the full core count runs into memory pressure
    """
    results_all = []
    dataset_info = DATASETS[dataset_name]

    df = pd.read_csv(dataset_info["file"])

    for station_index, station in enumerate(dataset_info["stations"]):
        print(f"Running experiments for dataset {dataset_name}, station {station}...")

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
        "training_weeks",
        "train_start",
        "train_end",
        "age_gap",
        "forecast_horizon",
        "mae",
        "mse",
        "training_duration"
    ], results_all)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the aging experiment for RegSARIMA, across all stations"
                                                 " and age gaps of one dataset. The results are saved to a"
                                                 " CSV file for later analysis.")
    parser.add_argument("--dataset", type=str, choices=DATASETS.keys(), required=True,
                        help="The dataset to run the experiment on.")
    parser.add_argument("--max_workers", type=int, default=None,
                        help="Number of worker processes (default: one per core). Lower it if the"
                             " full core count runs into memory pressure")
    args = parser.parse_args()

    run_aging_experiment(
        dataset_name=args.dataset,
        training_weeks=8,
        output_path=f"output/aging_model_regression_sarima_errors/results_{args.dataset}_100.csv",
        max_workers=args.max_workers
    )
