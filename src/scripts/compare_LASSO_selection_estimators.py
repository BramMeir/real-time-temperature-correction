"""
Main script to compare what LASSO station selection does to the cost and accuracy of the two estimators
of the regression with SARIMA errors model: the one-stage SARIMAX fit, which estimates the station
coefficients jointly with the error model, and the two-stage model, which estimates them by OLS first.
For every training period LASSO selects the stations once on the training window, and both estimators
are then fit twice on that window, once on all neighbouring stations and once on the selected subset,
so the four fits share exactly the same selection and the same forecast periods. Runs over every target
station of the Synthetic dataset (49 stations, so 48 candidate neighbours per target), the largest
network available, and writes every result to a CSV file as soon as its training period is done.

Supersedes the timings of determine_LASSO_performance_RegressionSARIMAErrors.py (measured on a
workstation) and determine_LASSO_performance.py (earlier ARIMAX setup): run on the HPC with the same
allocation as the model comparison, so the durations are comparable with that experiment.

python -m src.scripts.compare_LASSO_selection_estimators [--training_weeks 8] [--repeats 15]
"""
import os
import argparse
import csv
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from src.utils.select_LASSO_stations import select_LASSO_stations
from src.utils.generate_forecast_start import generate_forecast_start
from src.models.arima.train import train_sarima_model
from src.models.arima.repeat_forecast import _run_single_forecast as run_sarimax
from src.models.regression_sarima_errors.train import train_regression_sarima_errors
from src.models.regression_sarima_errors.execute_forecast import run_single_forecast as run_regression_sarima_errors


DATASET_NAME = "SYNTHETIC"
DATASET_FILE = "data/Synthetic/temperature_data.csv"

# Seed that is used to define the training periods (for reproducibility)
SEED = 42

# Seed offset per station, as in determine_LASSO_performance_RegressionSARIMAErrors.py, so the first
# repeats reuse the training periods of that earlier run. The stride has to exceed the number of repeats
# to keep the streams of the stations apart
STATION_SEED_STRIDE = 10_000

# Forecasting horizons in hours (1h, 4h, 12h, 1D, 2D, 4D, 7D, 14D, 21D, 30D), as in the model comparison
HORIZONS = [1, 4, 12, 24, 48, 96, 168, 336, 504, 720]

# Orders of the two estimators, as used in the model comparison
SARIMAX_ORDER, SARIMAX_SEASONAL_ORDER = (2, 0, 0), (1, 0, 1, 24)
RESIDUAL_ORDER, RESIDUAL_SEASONAL_ORDER = (3, 0, 0), (1, 0, 0, 24)

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

# Network data of the worker process, loaded once per worker by load_network so a task only has to carry
# the station name and the repeat id
NETWORK = None


def load_network(file):
    """
    Read the dataset and pivot it to one column per station, once per worker process.

    Input
    -----
    file: Path to the preprocessed dataset CSV file
    """
    global NETWORK

    df = pd.read_csv(file)

    NETWORK = df.pivot_table(index="datetime", columns="station_name", values="temp_dry_avg_2m")
    NETWORK.index = pd.to_datetime(NETWORK.index)


def station_data(station):
    """
    Build the target series and the neighbouring stations of one target, with the same preprocessing
    as determine_LASSO_performance_RegressionSARIMAErrors.py.

    Input
    -----
    station: Name of the target station

    Output
    ------
    series: Hourly target series
    exog_df: Hourly series of all other stations, aligned with the target
    df_complete: Target and neighbouring stations in one DataFrame, as required by the two-stage model
    """
    series = NETWORK[station].asfreq("1h").dropna()
    exog_df = NETWORK.drop(columns=station).reindex(series.index)

    series = series.resample("1h").mean().interpolate(limit_direction="both")
    exog_df = exog_df.resample("1h").mean().interpolate(limit_direction="both")

    df_complete = series.to_frame(name=station).join(exog_df)

    return series, exog_df, df_complete


def fit_and_evaluate(estimator, stations, station, series, exog_df, df_complete, train_start, train_end):
    """
    Fit one estimator on the given stations and evaluate it on every horizon.

    Input
    -----
    estimator: "SARIMAX" or "RegSARIMA"
    stations: Names of the neighbouring stations used as inputs
    station: Name of the target station
    series, exog_df, df_complete: Data of the target, as returned by station_data
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
                confidence_score=False,
                use_LASSO_selection=False,
                max_iter=1000,
                plot=False
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


def run_single_experiment(task):
    """
    Run one training period of one target: select the stations with LASSO, then fit both estimators on
    all stations and on the selection, and evaluate every fit on every horizon.

    Input
    -----
    task: A tuple containing (station, station_index, repeat_id, training_weeks)

    Output
    ------
    A list of result rows, one per (estimator, arm, horizon), in the order of COLUMNS
    """
    station, station_index, repeat_id, training_weeks = task

    series, exog_df, df_complete = station_data(station)

    # Generate random training period (start and end date) for the given station. Every station gets its
    # own seed so stations don't all sample the same forecast starts
    forecast_start = generate_forecast_start(
        series=series,
        seed=SEED + station_index * STATION_SEED_STRIDE,
        repeat_id=repeat_id,
        max_history_days=104 * 7,  # Max of 2 years of history that is tested
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


def run_all_experiments(training_weeks, repeats):
    """
    Run every training period of every target station in one pool of workers, one per reserved core,
    and append the results of each training period to the CSV file as soon as it is done.

    Input
    -----
    training_weeks: Number of training weeks
    repeats: Number of training periods per target station
    """
    stations = sorted(pd.read_csv(DATASET_FILE, usecols=["station_name"])["station_name"].unique())

    tasks = [
        (station, station_index, repeat_id, training_weeks)
        for station_index, station in enumerate(stations)
        for repeat_id in range(repeats)
    ]

    # One worker per core reserved for the job, as in the model comparison
    max_workers = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()

    with open(f"{OUTPUT_DIR}/results_{training_weeks}_weeks.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)

        with ProcessPoolExecutor(
            max_workers=max_workers, initializer=load_network, initargs=(DATASET_FILE,)
        ) as executor:
            futures = [executor.submit(run_single_experiment, t) for t in tasks]

            for done, future in enumerate(as_completed(futures), start=1):
                writer.writerows(future.result())
                f.flush()

                print(f"Finished {done} / {len(tasks)} training periods")


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
    args = parser.parse_args()

    # Make sure the output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    run_all_experiments(args.training_weeks, args.repeats)
