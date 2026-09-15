"""
Main script to determine whether LASSO station selection still helps the two-stage regression with
SARIMA errors model, or whether it is no longer relevant now that beta is estimated by OLS instead of
jointly with the SARIMA errors. For every training period, the model is fit twice: once on all
neighbouring stations, once on the subset LASSO selects on the training window, and both are evaluated
on the same forecast horizons. Mirrors determine_LASSO_performance.py, which ran this comparison for
ARIMAX. Runs over every target station of the Synthetic dataset (49 stations, so 48 candidate
neighbours per target), the largest network available, and saves the results in a CSV file.

python -m src.scripts.determine_LASSO_performance_RegressionSARIMAErrors
"""
import os
import argparse
import csv
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from src.utils.select_LASSO_stations import select_LASSO_stations
from src.models.regression_sarima_errors.train import train_regression_sarima_errors
from src.models.regression_sarima_errors.execute_forecast import run_single_forecast as run_regression_sarima_errors
from src.utils.generate_forecast_start import generate_forecast_start


# Metadata about the used dataset. Stations is None so every unique station in the file is used as a
# target in turn, giving the largest available candidate pool of neighbours (48 per target)
DATASETS = {
    "SYNTHETIC": {
        "file": "data/Synthetic/temperature_data.csv",
        "stations": None
    }
}

# Seed that is used to define the training periods (for reproducibility)
SEED = 42

# Seed offset per station, so every station draws its own forecast starts instead of every station of
# a network repeating the same weather. generate_forecast_start adds the repeat id to the seed, so the
# stride has to exceed the number of repeats to keep the streams apart
STATION_SEED_STRIDE = 10_000

# Forecasting horizons in hours (4h, 12h, 1D, 2D, 4D, 7D, 14D, 21D, 30D)
HORIZONS = [4, 12, 24, 48, 96, 168, 336, 504, 720]

NUMBER_OF_REPEATS = 10

# Orders selected on the stage-one residuals, as used in short_horizon_comparison_experiment.py
RESIDUAL_ORDER, RESIDUAL_SEASONAL_ORDER = (3, 0, 0), (1, 0, 0, 24)


def run_single_experiment(task):
    """
    Run a single experiment with the given parameters: fit the two-stage model once on all
    neighbouring stations and once on the LASSO-selected subset, and evaluate both on every horizon.

    Input
    -----
    task: A tuple containing (dataset_name, repeat_id, station, station_index, training_weeks, series,
          exog_df, df_complete)

    Output
    ------
    A list containing the results of the experiment, one row per (arm, horizon):
        [dataset_name, station, repeat_id, train_start, train_end, arm, nr_stations,
         lasso_selection_duration, training_duration, horizon, mae, mse]
    """
    dataset_name, repeat_id, station, station_index, training_weeks, series, exog_df, df_complete = task

    # Generate random training period (start and end date) for the given dataset and station. Every
    # station gets its own seed so stations within a dataset don't all sample the same forecast starts
    forecast_start = generate_forecast_start(
        series=series,
        seed=SEED + station_index * STATION_SEED_STRIDE,
        repeat_id=repeat_id,
        max_history_days=104 * 7,  # Max of 2 years of history that is tested
        max_horizon=max(HORIZONS)
    )

    # Determine the required training period for the model
    training_days = training_weeks * 7

    train_start = forecast_start - pd.Timedelta(days=training_days)
    train_end = forecast_start

    all_stations = exog_df.columns.tolist()

    # Determine the LASSO selection on only the training period to avoid data leakage
    start_time = pd.Timestamp.now()

    _, lasso_stations, _ = select_LASSO_stations(
        series=df_complete.loc[train_start:train_end, station],
        exog_df=df_complete.loc[train_start:train_end, all_stations]
    )

    lasso_selection_duration = (pd.Timestamp.now() - start_time).total_seconds()

    results = []

    for arm, arm_stations in [("all", all_stations), ("lasso", lasso_stations)]:
        # Start the timing of the model training
        start_time = pd.Timestamp.now()

        model = train_regression_sarima_errors(
            df=df_complete[train_start:train_end],
            target_station=station,
            exog_cols=arm_stations,
            arima_order=RESIDUAL_ORDER,
            seasonal_order=RESIDUAL_SEASONAL_ORDER,
            max_iter=1000
        )

        training_duration = (pd.Timestamp.now() - start_time).total_seconds()

        # Evaluate the model for each forecast horizon and save the results
        for horizon in HORIZONS:
            test_end = train_end + pd.Timedelta(hours=horizon)

            print(
                dataset_name,
                station,
                arm,
                horizon
            )

            result = run_regression_sarima_errors(
                df=df_complete,
                target_station=station,
                model=model,
                exog_cols=arm_stations,
                start=train_start,
                train_end=train_end,
                test_end=test_end,
                arima_order=RESIDUAL_ORDER,
                seasonal_order=RESIDUAL_SEASONAL_ORDER,
                mode="forecast"
            )

            mae, mse = result[:2]

            results.append([
                dataset_name,
                station,
                repeat_id,
                train_start,
                train_end,
                arm,
                len(arm_stations),
                lasso_selection_duration,
                training_duration,
                horizon,
                mae,
                mse
            ])

    return results


def run_all_experiments(training_weeks):
    """
    Main function to run all experiments across datasets, stations, training periods, and forecast horizons. The results are
    saved to a CSV file for later analysis.
    """
    results_all = []

    # Loop through all combinations of datasets, station, and training period
    for dataset_name, dataset_info in DATASETS.items():
        # Read the preprocessed data
        df = pd.read_csv(dataset_info["file"])

        # Use every unique station in the file as a target when none are specified
        stations = dataset_info["stations"] or sorted(df["station_name"].unique().tolist())

        for station_index, station in enumerate(stations):
            # Create the pandas DataFrame for the dataset with hourly frequency and datetime index
            # Select target station
            station_data = df[df['station_name'] == station]

            # Create target time series with a datetime index
            series = pd.Series(
                station_data['temp_dry_avg_2m'].values,
                index=pd.to_datetime(station_data['datetime'])
            ).asfreq('1h').dropna()

            # Create exogenous DataFrame
            other_stations = [s for s in df['station_name'].unique() if s != station]

            exog_data = df[df['station_name'].isin(other_stations)]

            # Pivot to get each station as a separate column
            exog_pivot = exog_data.pivot_table(
                index='datetime', columns='station_name', values='temp_dry_avg_2m'
            )

            # Create datetime index
            exog_pivot.index = pd.to_datetime(exog_pivot.index)

            # Remove the missing timestamps and align with the main series
            exog_df = exog_pivot.reindex(series.index)

            # Resample data by taking the mean
            series = series.resample('1h').mean().interpolate(limit_direction="both")
            exog_df = exog_df.resample('1h').mean().interpolate(limit_direction="both")

            # Combine the target and neighbouring stations into one DataFrame, as required by the
            # two-stage model's regression stage
            df_complete = series.to_frame(name=station).join(exog_df)

            # Build a list of taks to run in parallel
            tasks = []

            # Generate NUMBER_OF_REPEATS random training periods for the given dataset and station
            for repeat_id in range(NUMBER_OF_REPEATS):
                tasks.append((
                    dataset_name,
                    repeat_id,
                    station,
                    station_index,
                    training_weeks,
                    series,
                    exog_df,
                    df_complete
                ))

            with ProcessPoolExecutor() as executor:
                futures = [executor.submit(run_single_experiment, t) for t in tasks]

                for f in as_completed(futures):
                    results_all.extend(f.result())

    # Save all results to a CSV file for later analysis
    with open(
        f"output/LASSO_performance_regression_sarima_errors/results_{training_weeks}_weeks.csv",
        "w", newline="", encoding="utf-8"
    ) as f:
        # Create CSV writer and write header
        writer = csv.writer(f)

        writer.writerow([
            "dataset",
            "station",
            "repeat_id",
            "train_start",
            "train_end",
            "arm",
            "nr_stations",
            "lasso_selection_duration",
            "training_duration",
            "horizon",
            "mae",
            "mse"
        ])

        writer.writerows(results_all)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Compare the two-stage regression with SARIMA errors model"
                                                 " trained on all neighbouring stations against the same model"
                                                 " trained on the LASSO-selected subset, across all datasets,"
                                                 " stations, and forecast horizons. The results are saved to a"
                                                 " CSV file for later analysis.")
    parser.add_argument("--training_weeks", type=int, default=8,
                        help="Number of training weeks to use (default: 8)")
    args = parser.parse_args()

    # Make sure the output directory exists
    os.makedirs("output/LASSO_performance_regression_sarima_errors", exist_ok=True)

    # Run all experiments and save results to CSV
    run_all_experiments(args.training_weeks)
