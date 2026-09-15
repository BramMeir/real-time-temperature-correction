"""
Main script to determine how much the performance of the two-stage regression with SARIMA errors
model degrades with aging, i.e. when it keeps forecasting without ever being retrained. Mirrors
test_aging_model.py, which ran this experiment for ARIMAX. The stage-one regression has no internal
state to age (it only ever needs the current neighbour readings), so aging is only applied to the
residual SARIMA: for every "age gap", the true stage-one residuals over the gap are appended to the
residual SARIMA (refit=False, updating its state without re-estimating its parameters) before
forecasting 48h ahead from that aged state. Since nothing is ever retrained, this still captures any
staleness in the stage-one regression weights too, exactly as it does for ARIMAX's coefficients.
Runs over multiple datasets and stations. The results are saved in a CSV file for further analysis.

python -m src.scripts.test_aging_model_RegressionSARIMAErrors --dataset SYNTHETIC
"""
import os
import argparse
import csv
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from src.models.regression_sarima_errors.train import train_regression_sarima_errors, stage_one_residuals
from src.models.regression_sarima_errors.execute_forecast import run_single_forecast as run_regression_sarima_errors
from src.utils.generate_forecast_start import generate_forecast_start


# Metadata about the used datasets and stations to evaluate on, identical to test_aging_model.py so
# the two models' aging curves are directly comparable
DATASETS = {
    "KMI": {
        "file": "data/Full_AWS/preprocessed_10m_2022_2025.csv",
        "stations": ["MELLE", "DIEPENBEEK", "HUMAIN", "SINT-KATELIJNE-WAVER", "ERNAGE"]
    },
    "TURKU": {
        "file": "data/Turku/Turku_preprocessed.csv",
        "stations": ["Betel", "Virastotalo", "Ylijoki", "Kurala"]
    },
    "SYNTHETIC": {
        "file": "data/Synthetic/temperature_data.csv",
        "stations": [
            "Stadhuis_Brussel_Grote_Markt",
            "Stadhuis_Antwerpen_Grote_Markt",
            "Grote_Markt_Kortrijk",
            "Tielt",
            "Gembloux",
            "Slag_om_Ardennen_Museum_La_Roche_en_Ardenne",
            "Abdij_Tongerlo",
            "Etalle",
            "Centrum_Malmedy",
            "Markt_Hoei"
        ]
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

# Orders selected on the stage-one residuals, as used in short_horizon_comparison_experiment.py
RESIDUAL_ORDER, RESIDUAL_SEASONAL_ORDER = (3, 0, 0), (1, 0, 0, 24)


def run_single_experiment(task):
    """
    Run a single experiment with the given parameters.

    Input
    -----
    task: A tuple containing (dataset_name, repeat_id, station, station_index, training_weeks,
          series, exog_df, df_complete)

    Output
    ------
    A list containing the results of the experiment:
        [dataset_name, station, training_weeks, train_start, train_end, age_gap, forecast_horizon,
         mae, mse, training_duration]
    """
    dataset_name, repeat_id, station, station_index, training_weeks, series, exog_df, df_complete = task

    all_stations = exog_df.columns.tolist()

    # Generate random training period (start and end date) for the given dataset and station. Every
    # station gets its own seed so stations within a dataset don't all sample the same forecast starts
    forecast_start = generate_forecast_start(
        series=series,
        seed=SEED + station_index * STATION_SEED_STRIDE,
        repeat_id=repeat_id,
        max_history_days=8 * 7,                        # Max of 8 weeks of history to train the model on
        max_horizon=MAX_AGING_GAP_DAYS * 24 + 48         # Max aging gap in hours + 2 days of forecast to evaluate on
    )

    # Determine the required training period for the model
    training_days = training_weeks * 7

    train_start = forecast_start - pd.Timedelta(days=training_days)
    train_end = forecast_start

    # Start the timing of the model training
    start_time = pd.Timestamp.now()

    # Train the model once on the training period to reuse for every age gap
    regression, residual_model = train_regression_sarima_errors(
        df=df_complete[train_start:train_end],
        target_station=station,
        exog_cols=all_stations,
        arima_order=RESIDUAL_ORDER,
        seasonal_order=RESIDUAL_SEASONAL_ORDER,
        max_iter=1000
    )

    end_time = pd.Timestamp.now()
    training_duration = end_time - start_time

    # Evaluate the model for all the 'aging gaps' (0 to MAX_AGING_GAP_DAYS, step 3)
    #  to see how performance degrades with aging (forecast horizon = 48h)
    results = []
    for age_gap in range(0, MAX_AGING_GAP_DAYS + 1, 3):
        # Set the test period as [train_end + age_gap, train_end + age_gap + 48h]
        test_begin = train_end + pd.Timedelta(days=age_gap)
        test_end = test_begin + pd.Timedelta(days=2)

        gap_start = train_end + pd.Timedelta(hours=1)  # Start the gap right after the training period
        gap_end = test_begin                           # End the gap right before the test period

        updated_residual_model = residual_model

        if age_gap > 0:
            # The regression itself is never refit, so its stage-one weights are just as frozen as
            # the residual SARIMA's parameters: only the SARIMA's internal state is kept current
            _, gap_residuals = stage_one_residuals(
                df_complete[gap_start:gap_end], station, all_stations, regression=regression
            )

            updated_residual_model = residual_model.append(
                endog=gap_residuals,
                refit=False
            )

        result = run_regression_sarima_errors(
            df=df_complete,
            target_station=station,
            model=(regression, updated_residual_model),
            exog_cols=all_stations,
            start=train_start,
            train_end=test_begin,
            test_end=test_end,
            arima_order=RESIDUAL_ORDER,
            seasonal_order=RESIDUAL_SEASONAL_ORDER,
            mode="forecast"
        )

        mae, mse = result[:2]

        results.append([
            dataset_name,
            station,
            training_weeks,
            train_start,
            train_end,
            age_gap,
            48,  # Forecast horizon is always 48h in this experiment
            mae,
            mse,
            training_duration
        ])

    return results


def run_all_experiments(training_weeks, dataset_name, max_workers=None):
    """
    Main function to run all experiments across stations and age gaps for one dataset. The results are
    saved to a CSV file for later analysis.

    Input
    -----
    training_weeks: Number of weeks of training data to use
    dataset_name: Name of the dataset to run the experiment on
    max_workers: Number of worker processes (default is None, which uses one per core). Lower it if
      the full core count runs into memory pressure
    """
    results_all = []

    dataset_info = DATASETS[dataset_name]

    # Read the preprocessed data
    df = pd.read_csv(dataset_info["file"])

    for station_index, station in enumerate(dataset_info["stations"]):
        print(f"Running experiments for dataset {dataset_name}, station {station}...")

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

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(run_single_experiment, t) for t in tasks]

            for f in as_completed(futures):
                results_all.extend(f.result())

    # Save all results to a CSV file for later analysis
    with open(
        f"output/aging_model_regression_sarima_errors/results_{dataset_name}_100.csv",
        "w", newline="", encoding="utf-8"
    ) as f:
        # Create CSV writer and write header
        writer = csv.writer(f)

        writer.writerow([
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
        ])

        writer.writerows(results_all)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run the aging experiment for the two-stage regression"
                                                 " with SARIMA errors model, across all stations and age"
                                                 " gaps of one dataset. The results are saved to a CSV"
                                                 " file for later analysis.")
    parser.add_argument("--dataset", type=str, choices=DATASETS.keys(), required=True,
                        help="The dataset to run the experiment on.")
    parser.add_argument("--max_workers", type=int, default=None,
                        help="Number of worker processes (default: one per core). Lower it if the"
                             " full core count runs into memory pressure")
    args = parser.parse_args()

    # Make sure the output directory exists
    os.makedirs("output/aging_model_regression_sarima_errors", exist_ok=True)

    # Run all experiments and save results to CSV
    nr_training_weeks = 8
    run_all_experiments(nr_training_weeks, dataset_name=args.dataset, max_workers=args.max_workers)
