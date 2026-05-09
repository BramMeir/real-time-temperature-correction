"""
Main script to determine the how much the performance of the ARIMAX model degrades with aging. This script runs the experiment
over multiple datasets and stations. The results are saved in a CSV file for further analysis.

python -m src.scripts.test_aging_model
"""
import os
import argparse
import csv
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from src.utils.select_LASSO_stations import select_LASSO_stations
from src.models.arima.train import train_sarima_model
from src.evaluation.evaluate_forecasts import evaluate_forecasts
from src.utils.generate_forecast_start import generate_forecast_start


# Metadata about the used datasets and stations to evaluate on
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
            "Tielt",
            "Etalle",
            "Markt_Hoei",
            "Stadhuis_Antwerpen_Grote_Markt",
            "Gembloux",
            "Slag_om_Ardennen_Museum_La_Roche_en_Ardenne",
            "Grote_Markt_Kortrijk",
        ]
    }
}

MODEL_CONFIGS = [
    ("no_retrain", None),

    ("full_3d", 3),
    ("full_15d", 15),

    ("inc_3d", 3),
    ("inc_15d", 15),

    ("full_9d", 9),
    ("inc_9d", 9),
]

# Seed that is used to define the training periods (for reproducibility)
SEED = 47


def run_single_experiment(task):
    """
    Run a single experiment with the given parameters.

    Input
    -----
    task: A tuple containing (dataset_name, repeat_id, station, training_weeks, series, exog_df)

    Output
    ------
    A list containing the results of the experiment:
        [dataset_name, station, training_weeks, train_start, train_end, horizon, mae, mse]
    """
    dataset_name, repeat_id, station, training_weeks, series, exog_df = task

    # Make sure there is exogenous data for the given station, otherwise throw an error
    if exog_df is None:
        raise ValueError(f"No exogenous data found for station {station} in dataset {dataset_name}")

    # Generate random training period (start and end date) for the given dataset and station
    forecast_start = generate_forecast_start(
        series=series,
        seed=SEED,
        repeat_id=repeat_id,
        max_history_days=8 * 7,        # Max of 8 weeks of history to train the model on
        max_horizon=60 * 24 + 24       # Max 60 days of simulation + 1 day of forecast to evaluate on
    )

    # Determine the required training period for the model
    training_days = training_weeks * 7

    train_start = forecast_start - pd.Timedelta(days=training_days)
    train_end = forecast_start

    # Determine the LASSO selection on only the training period to avoid data leakage
    if dataset_name == "SYNTHETIC":
        _, selected_stations, _ = select_LASSO_stations(series[train_start:train_end], exog_df[train_start:train_end])
        exog_df = exog_df[selected_stations]

    # Evaluate all the different retraining strategies for the given training period and forecast start
    output_results = []
    for model_type, retrain_gap in MODEL_CONFIGS:

        # Initial training of the model on the training period
        train_start = forecast_start - pd.Timedelta(weeks=training_weeks)
        train_end = forecast_start

        results = train_sarima_model(
            series=series[train_start:train_end],
            exog_df=exog_df[train_start:train_end],
            arima_order=(2, 0, 0),
            seasonal_order=(1, 0, 1, 24),
            max_iter=1000
        )

        last_retrain_time = forecast_start
        last_update_time = forecast_start

        current_time = forecast_start
        end_time = forecast_start + pd.Timedelta(days=60)  # Simulate for 60 days

        # Rolling simulation
        while current_time < end_time:
            retrain_event = 0

            # Check for retraining
            if model_type != "no_retrain" and (current_time - last_retrain_time) >= pd.Timedelta(days=retrain_gap):
                retrain_event = 1

                # Start the timing of the retraining
                start_retraining_time = pd.Timestamp.now()

                # Determine what retraining strategy to use
                if model_type.startswith("full"):
                    # Full retraining
                    new_train_start = current_time - pd.Timedelta(weeks=training_weeks)
                    new_train_end = current_time

                    results = train_sarima_model(
                        series=series[new_train_start:new_train_end],
                        exog_df=exog_df[new_train_start:new_train_end],
                        arima_order=(2, 0, 0),
                        seasonal_order=(1, 0, 1, 24),
                        max_iter=1000
                    )

                    train_start = new_train_start

                elif model_type.startswith("inc"):
                    # Incremental retraining
                    update_start = last_update_time + pd.Timedelta(hours=1)
                    update_end = current_time

                    # Refit=True so model parameters can be updated based on the new data
                    results = results.append(
                        endog=series[update_start:update_end],
                        exog=exog_df[update_start:update_end],
                        refit=True
                    )

                end_retraining_time = pd.Timestamp.now()
                retraining_duration = end_retraining_time - start_retraining_time

                last_retrain_time = current_time

            # If the model was not retrained, still update the model state with the new data
            if not retrain_event and current_time > last_update_time:
                update_start = last_update_time + pd.Timedelta(hours=1)
                update_end = current_time

                results = results.append(
                    endog=series[update_start:update_end],
                    exog=exog_df[update_start:update_end],
                    refit=False
                )

            # Forecast the next 24 hours
            forecast_end = current_time + pd.Timedelta(days=1)
            pred = results.get_forecast(steps=24, exog=exog_df[current_time + pd.Timedelta(hours=1):forecast_end])
            test_series = series[current_time + pd.Timedelta(hours=1):forecast_end]

            # Create index for forecasted values (same as test index)
            y_forecasted = pd.Series(pred.predicted_mean.values, index=test_series.index)
            errors = evaluate_forecasts(test_series, y_forecasted)
            print(errors)

            output_results.append([
                dataset_name,
                station,
                repeat_id,
                model_type,
                current_time,
                errors["MAE"],
                errors["MSE"],
                retrain_event,
                retraining_duration.total_seconds() if retrain_event else 0
            ])

            # Move to the next step
            last_update_time = current_time
            current_time += pd.Timedelta(days=3)  # Step of 3 days between forecasts

    return output_results


def run_all_experiments(repeat_id, training_weeks, dataset_name=None):
    """
    Main function to run all experiments across datasets, stations, training periods and age gaps. The results are
    saved to a CSV file for later analysis.
    """
    results_all = []

    # Read the preprocessed data
    dataset_info = DATASETS[dataset_name] if dataset_name else None
    df = pd.read_csv(dataset_info["file"])

    # Build a list of taks to run in parallel
    tasks = []

    for station in dataset_info["stations"]:
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

        # Generate NUMBER_OF_REPEATS random training periods for the given dataset and station
        tasks.append((
            dataset_name,
            repeat_id,
            station,
            training_weeks,
            series,
            exog_df
        ))

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_single_experiment, t) for t in tasks]

        for f in as_completed(futures):
            results_all.extend(f.result())

    # Save all results to a CSV file for later analysis
    with open(f"output/retraining_results/results_{dataset_name}_{repeat_id}.csv", "w", newline="") as f:
        # Create CSV writer and write header
        writer = csv.writer(f)

        writer.writerow([
            "dataset",
            "station",
            "repeat_id",
            "model_type",
            "current_time",
            "mae",
            "mse",
            "retrain_event",
            "retraining_duration"
        ])

        writer.writerows(results_all)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run the full experiment with the given number of training weeks"
                                                 " across all datasets, stations, and age gaps. The results"
                                                 " are saved to a CSV file for later analysis.")
    parser.add_argument("--dataset", type=str, choices=DATASETS.keys(), default=None,
                        help="The dataset to run the experiment on.")
    parser.add_argument("--repeat_id", type=int, default=0,
                        help="The repeat ID to use for generating the training periods (for reproducibility).")
    args = parser.parse_args()

    # Make sure the output directory exists
    os.makedirs("output/retraining_results", exist_ok=True)

    # Run all experiments and save results to CSV
    nr_training_weeks = 8
    run_all_experiments(args.repeat_id, nr_training_weeks, dataset_name=args.dataset)
