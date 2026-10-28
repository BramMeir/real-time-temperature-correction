import argparse
import pandas as pd
from src.models.regsarima.train import RESIDUAL_ORDER, RESIDUAL_SEASONAL_ORDER
from src.utils.datasets import DATASETS, load_station_data
from src.utils.experiment_utils import get_worker_df_complete, run_tasks_for_station, write_results_csv
from src.utils.select_LASSO_stations import select_LASSO_stations
from src.utils.generate_forecast_start import generate_forecast_start
from src.models.sarima.train import train_sarima_model
from src.models.sarima.execute_forecast import run_single_forecast as run_sarimax
from src.models.regsarima.train import train_regression_sarima_errors
from src.models.regsarima.execute_forecast import run_single_forecast as run_regression_sarima_errors

DATASET_NAME = "SYNTHETIC"

# Seed that is used to define the training periods (for reproducibility)
SEED = 42

# Seed offset per station, so every station draws its own training periods. The stride has to exceed the
# number of repeats to keep the streams of the stations apart
STATION_SEED_STRIDE = 10_000

# Forecasting horizons in hours (1h, 4h, 12h, 1D, 2D, 4D, 7D, 14D, 21D, 30D), as in the model comparison
HORIZONS = [1, 4, 12, 24, 48, 96, 168, 336, 504, 720]

# Orders of the two estimators, as used in the model comparison
SARIMAX_ORDER, SARIMAX_SEASONAL_ORDER = (2, 0, 0), (1, 0, 1, 24)

OUTPUT_DIR = "output/LASSO_selection_estimators"

COLUMNS = [
    "dataset",
    "station",
    "repeat_id",
    "train_start",
    "train_end",
    "estimator",
    "arm",
    "nr_stations",
    "lasso_selection_duration",
    "training_duration",
    "horizon",
    "mae",
    "mse"
]

def fit_and_evaluate(estimator, stations, station, series, exog_df, df_complete, train_start, train_end):
    """
    Fit one estimator on the given stations and evaluate it on every horizon.

    Input
    -----
    estimator: "SARIMAX" or "RegSARIMA"
    stations: Names of the neighbouring stations used as inputs
    station: Name of the target station
    series, exog_df, df_complete: Target series, the other stations, and both in one DataFrame
    train_start, train_end: Training window

    Output
    ------
    training_duration: Seconds spent fitting the model
    errors: List of (horizon, mae, mse) tuples
    """
    start_time = pd.Timestamp.now()

    if estimator == "SARIMAX":
        model = train_sarima_model(
            series=series[train_start:train_end],
            exog_df=exog_df.loc[train_start:train_end, stations],
            arima_order=SARIMAX_ORDER,
            seasonal_order=SARIMAX_SEASONAL_ORDER,
            max_iter=1000
        )
    else:
        model = train_regression_sarima_errors(
            df=df_complete[train_start:train_end],
            target_station=station,
            exog_cols=stations,
            arima_order=RESIDUAL_ORDER,
            seasonal_order=RESIDUAL_SEASONAL_ORDER,
            max_iter=1000
        )

    training_duration = (pd.Timestamp.now() - start_time).total_seconds()

    errors = []

    for horizon in HORIZONS:
        test_end = train_end + pd.Timedelta(hours=horizon)

        if estimator == "SARIMAX":
            result = run_sarimax(
                0,
                series=series,
                exog_df=exog_df[stations],
                model=model,
                start_date=train_start,
                end_date=test_end,
                hours_to_forecast=horizon,
                arima_order=SARIMAX_ORDER,
                seasonal_order=SARIMAX_SEASONAL_ORDER,
                max_iter=1000
            )
        else:
            result = run_regression_sarima_errors(
                df=df_complete,
                target_station=station,
                model=model,
                exog_cols=stations,
                start=train_start,
                train_end=train_end,
                test_end=test_end,
                arima_order=RESIDUAL_ORDER,
                seasonal_order=RESIDUAL_SEASONAL_ORDER,
                mode="forecast"
            )

        errors.append((horizon, result[0], result[1]))

    return training_duration, errors


def run_single_experiment(task, training_weeks):
    """
    Run one training period of one target: select the stations with LASSO, then fit both estimators on
    all stations and on the selection, and evaluate every fit on every horizon.

    Input
    -----
    task: A tuple containing (station, station_index, repeat_id)
    training_weeks: Number of training weeks

    Output
    ------
    A list of result rows, one per (estimator, arm, horizon), in the order of COLUMNS
    """
    station, station_index, repeat_id = task

    df_complete = get_worker_df_complete()
    series = df_complete[station]
    exog_df = df_complete.drop(columns=station)

    # Generate random training period (start and end date) for the given station. Every station gets its
    # own seed so stations don't all sample the same forecast starts
    forecast_start = generate_forecast_start(
        series=series,
        seed=SEED + station_index * STATION_SEED_STRIDE,
        repeat_id=repeat_id,
        max_history_days=104 * 7,  # Leaves room for two years of history, which fixes the reported training periods
        max_horizon=max(HORIZONS)
    )

    train_start = forecast_start - pd.Timedelta(days=training_weeks * 7)
    train_end = forecast_start

    all_stations = exog_df.columns.tolist()

    # Determine the LASSO selection on only the training period to avoid data leakage. A single core,
    # like every fit in this experiment, so the selection time is comparable with the training times
    start_time = pd.Timestamp.now()

    _, lasso_stations, _ = select_LASSO_stations(
        series=series[train_start:train_end],
        exog_df=exog_df[train_start:train_end],
        n_jobs=1
    )

    lasso_selection_duration = (pd.Timestamp.now() - start_time).total_seconds()

    rows = []

    for estimator in ["SARIMAX", "RegSARIMA"]:
        for arm, arm_stations in [("all", all_stations), ("lasso", lasso_stations)]:
            training_duration, errors = fit_and_evaluate(
                estimator, arm_stations, station, series, exog_df, df_complete, train_start, train_end
            )

            print(DATASET_NAME, station, repeat_id, estimator, arm, f"{training_duration:.1f} s")

            for horizon, mae, mse in errors:
                rows.append([
                    DATASET_NAME,
                    station,
                    repeat_id,
                    train_start,
                    train_end,
                    estimator,
                    arm,
                    len(arm_stations),
                    lasso_selection_duration,
                    training_duration,
                    horizon,
                    mae,
                    mse
                ])

    return rows


def run_all_experiments(training_weeks, repeats, max_workers):
    """
    Run every training period of every target station and write the results to a CSV file.

    Input
    -----
    training_weeks: Number of training weeks
    repeats: Number of training periods per target station
    max_workers: Number of worker processes (default is None, which uses one per core)
    """
    df = pd.read_csv(DATASETS[DATASET_NAME]["file"])
    rows = []

    for station_index, station in enumerate(sorted(df["station_name"].unique())):
        tasks = [(station, station_index, repeat_id) for repeat_id in range(repeats)]
        rows.extend(run_tasks_for_station(
            load_station_data(df, station), tasks, run_single_experiment, (training_weeks,), max_workers=max_workers
        ))

    write_results_csv(f"{OUTPUT_DIR}/results_{training_weeks}_weeks.csv", COLUMNS, rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare the effect of LASSO station selection on the"
                                                 " training time and accuracy of the one-stage SARIMAX fit"
                                                 " and the two-stage regression with SARIMA errors model on"
                                                 " the Synthetic dataset. The results are saved to a CSV"
                                                 " file for later analysis.")
    parser.add_argument("--training_weeks", type=int, default=8,
                        help="Number of training weeks to use (default: 8)")
    parser.add_argument("--repeats", type=int, default=15,
                        help="Number of training periods per target station (default: 15)")
    parser.add_argument("--max_workers", type=int, default=None,
                        help="Number of worker processes (default: one per core)")
    args = parser.parse_args()

    run_all_experiments(args.training_weeks, args.repeats, args.max_workers)
