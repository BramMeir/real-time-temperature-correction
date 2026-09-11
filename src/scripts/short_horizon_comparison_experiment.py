"""
Main script to compare ARIMAX, the neighbour regression and the two-stage regression with SARIMA
errors over short forecast horizons, where the three models actually differ. Runs over all datasets,
stations and training periods, and saves the hourly forecast and observation of every forecast path
to one CSV file per model, so any horizon can be derived afterwards.

python -m src.scripts.short_horizon_comparison_experiment --models LinearRegression RegressionSARIMAErrors
"""
import os
import csv
import argparse
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from src.utils.generate_forecast_start import generate_forecast_start
from src.scripts.determine_residual_sarima_order import DATASETS, SEED, load_station_series
from src.models.arima.train import train_sarima_model
from src.models.arima.repeat_forecast import _run_single_forecast as run_arimax
from src.models.linear_regression.train import train_neighbour_regression
from src.models.linear_regression.execute_forecast import run_single_forecast as run_linear_regression
from src.models.regression_sarima_errors.train import train_regression_sarima_errors
from src.models.regression_sarima_errors.execute_forecast import run_single_forecast as run_regression_sarima_errors

# The three models being compared, cheapest first so a long ARIMAX run never blocks the other two
MODELS = ["LinearRegression", "RegressionSARIMAErrors", "ARIMAX"]

# Length of every forecast path in hours, covering the leads where the models still differ. The
# analysis derives its horizons from the hourly rows, so this is the only horizon the script knows
MAX_LEAD_HOURS = 96

# Number of training periods to evaluate per station
NUMBER_OF_REPEATS = 10

# History reserved when sampling a forecast start, matching the longest training window on offer
MAX_HISTORY_DAYS = 8 * 7

# Seed offset per station, so every station draws its own forecast starts instead of every station of
# a network repeating the same weather. generate_forecast_start adds the repeat id to the seed, so the
# stride has to exceed the number of repeats to keep the streams apart
STATION_SEED_STRIDE = 10_000

# Orders used by the full comparison experiment, ARIMAX selected on the raw series and the two-stage
# model on the stage-one residuals
ARIMAX_ORDER, ARIMAX_SEASONAL_ORDER = (2, 0, 0), (1, 0, 1, 24)
RESIDUAL_ORDER, RESIDUAL_SEASONAL_ORDER = (3, 0, 0), (1, 0, 0, 24)


def train_model(model_name, station, series, exog_df, df_complete, train_start, train_end):
    """
    Train one model on the given training window.

    Input
    -----
    model_name: Name of the model to train, one of MODELS
    station: Name of the target station
    series: Target time series with an hourly datetime index
    exog_df: DataFrame with the neighbouring stations as columns
    df_complete: DataFrame with the target station and the neighbouring stations as columns
    train_start: Start of the training window
    train_end: End of the training window

    Output
    ------
    model: The trained model
    duration: Training time in seconds
    """
    start_time = pd.Timestamp.now()

    if model_name == "ARIMAX":
        model = train_sarima_model(
            series=series[train_start:train_end],
            exog_df=exog_df[train_start:train_end],
            arima_order=ARIMAX_ORDER,
            seasonal_order=ARIMAX_SEASONAL_ORDER,
            max_iter=1000
        )
    elif model_name == "LinearRegression":
        model = train_neighbour_regression(
            df=df_complete[train_start:train_end],
            target_station=station,
            exog_cols=exog_df.columns.tolist()
        )
    else:
        model = train_regression_sarima_errors(
            df=df_complete[train_start:train_end],
            target_station=station,
            exog_cols=exog_df.columns.tolist(),
            arima_order=RESIDUAL_ORDER,
            seasonal_order=RESIDUAL_SEASONAL_ORDER,
            max_iter=1000
        )

    return model, (pd.Timestamp.now() - start_time).total_seconds()


def forecast_model(model_name, model, repeat_id, station, series, exog_df, df_complete,
                   train_start, train_end):
    """
    Forecast the whole path with an already trained model.

    Input
    -----
    model_name: Name of the model to forecast with, one of MODELS
    model: The trained model returned by train_model
    repeat_id: Index of the training period, only used by the ARIMAX runner
    station: Name of the target station
    series: Target time series with an hourly datetime index
    exog_df: DataFrame with the neighbouring stations as columns
    df_complete: DataFrame with the target station and the neighbouring stations as columns
    train_start: Start of the training window
    train_end: End of the training window and start of the forecast

    Output
    ------
    predictions: Series with the forecast for every hour of the path
    duration: Forecast time in seconds
    """
    test_end = train_end + pd.Timedelta(hours=MAX_LEAD_HOURS)
    start_time = pd.Timestamp.now()

    if model_name == "ARIMAX":
        result = run_arimax(
            repeat_id,
            series=series,
            exog_df=exog_df,
            model=model,
            start_date=train_start,
            end_date=test_end,
            hours_to_forecast=MAX_LEAD_HOURS,
            arima_order=ARIMAX_ORDER,
            seasonal_order=ARIMAX_SEASONAL_ORDER,
            confidence_score=False,
            use_LASSO_selection=False,
            max_iter=1000,
            plot=False
        )
    else:
        forecast_fn = run_linear_regression if model_name == "LinearRegression" else run_regression_sarima_errors
        result = forecast_fn(
            df=df_complete,
            target_station=station,
            model=model,
            exog_cols=exog_df.columns.tolist(),
            start=train_start,
            train_end=train_end,
            test_end=test_end,
            mode="forecast"
        )

    # Every forecast runner returns the hourly forecast as the last element of its result tuple
    predictions = result[-1]

    return predictions, (pd.Timestamp.now() - start_time).total_seconds()


def run_single_experiment(task):
    """
    Train one model on one training period and forecast the whole path.

    Input
    -----
    task: Tuple of (dataset_name, model_name, station, station_index, repeat_id, training_days,
          series, exog_df, df_complete)

    Output
    ------
    Returns a list of result rows, one per forecast hour, matching the CSV header.
    """
    (dataset_name, model_name, station, station_index, repeat_id, training_days,
     series, exog_df, df_complete) = task

    # Drawn per station so every station samples its own weather, and identical for every model so the
    # three are always compared on exactly the same window
    forecast_start = generate_forecast_start(
        series=series,
        seed=SEED + station_index * STATION_SEED_STRIDE,
        repeat_id=repeat_id,
        max_history_days=max(MAX_HISTORY_DAYS, training_days),
        max_horizon=MAX_LEAD_HOURS
    )

    train_start = forecast_start - pd.Timedelta(days=training_days)
    train_end = forecast_start

    print(dataset_name, station, model_name, train_start, train_end)

    model, train_duration = train_model(
        model_name, station, series, exog_df, df_complete, train_start, train_end
    )

    predictions, forecast_duration = forecast_model(
        model_name, model, repeat_id, station, series, exog_df, df_complete, train_start, train_end
    )

    observations = series.reindex(predictions.index)

    return [
        [
            dataset_name, station, model_name, train_start, train_end,
            int((timestamp - train_end).total_seconds() // 3600),
            forecast, observations[timestamp], train_duration, forecast_duration
        ]
        for timestamp, forecast in predictions.items()
    ]


def run_comparison(model_name, training_weeks, repeats, output_file, datasets=None, max_workers=None):
    """
    Run one model over all datasets, stations and training periods, and save the results to a CSV file.

    Input
    -----
    model_name: Name of the model to run, one of MODELS
    training_weeks: Number of weeks of training data
    repeats: Number of training periods to evaluate per station
    output_file: Path of the CSV file to write the results to
    datasets: List of dataset names to evaluate (default is None, which uses all of them)
    max_workers: Number of worker processes (default is None, which uses one per core). Every worker
      holds its own copy of the station data, so lowering it trades runtime for memory
    """
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Dataset", "Station", "Model", "Train_start", "Train_end", "Lead_hours",
            "Forecast", "Observation", "Train_duration_seconds", "Forecast_duration_seconds"
        ])

        for dataset_name in (datasets or DATASETS):
            dataset_info = DATASETS[dataset_name]
            df = pd.read_csv(dataset_info["file"])

            for station_index, station in enumerate(dataset_info["stations"]):
                series, df_complete, exog_cols = load_station_series(df, station)
                exog_df = df_complete[exog_cols]

                tasks = [
                    (dataset_name, model_name, station, station_index, repeat_id, training_weeks * 7,
                     series, exog_df, df_complete)
                    for repeat_id in range(repeats)
                ]

                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    futures = [executor.submit(run_single_experiment, task) for task in tasks]

                    for future in as_completed(futures):
                        for row in future.result():
                            writer.writerow(row)

    print(f"Results for {model_name} written to {output_file}")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Compare ARIMAX, the neighbour regression and the two-stage "
                                                 "regression with SARIMA errors over short forecast horizons, "
                                                 "across datasets, stations and training periods. The hourly "
                                                 "forecasts are saved to one CSV file per model.")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS,
                        help="Models to run (default: all three)")
    parser.add_argument("--training_weeks", type=int, default=8,
                        help="Number of weeks to use for training (default: 8, as in the comparison experiment)")
    parser.add_argument("--repeats", type=int, default=NUMBER_OF_REPEATS,
                        help=f"Number of training periods per station (default: {NUMBER_OF_REPEATS})")
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS),
                        help="Datasets to evaluate (default: all of them)")
    parser.add_argument("--max_workers", type=int, default=None,
                        help="Number of worker processes (default: one per core). Lower it when a "
                             "memory hungry model such as ARIMAX exhausts the available memory")
    args = parser.parse_args()

    # Make sure the output directory exists
    os.makedirs("output/short_horizon_comparison", exist_ok=True)

    for model in args.models:
        run_comparison(
            model_name=model,
            training_weeks=args.training_weeks,
            repeats=args.repeats,
            output_file=f"output/short_horizon_comparison/{model}_{args.training_weeks}_weeks.csv",
            datasets=args.datasets,
            max_workers=args.max_workers
        )
