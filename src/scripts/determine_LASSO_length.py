"""
Experiment to determine the optimal amount of historical data to use for LASSO-based station selection
before the fixed model training period. The ARIMAX model always uses the fixed training period.
"""

import os
import argparse
import csv
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from src.utils.select_LASSO_stations import select_LASSO_stations
from src.models.arima.train import train_sarima_model
from src.models.arima.repeat_forecast import _run_single_forecast as run_arimax
from src.utils.generate_forecast_start import generate_forecast_start

DATASETS = {
    "SYNTHETIC": {
        "file": "data/Synthetic/temperature_data.csv",
        "stations": [
            "Stadhuis_Brussel_Grote_Markt",
            "Stadhuis_Antwerpen_Grote_Markt",
            "Grote_Markt_Kortrijk",
            "Tielt",
            "Gembloux",
            "Slag_om_Ardennen_Museum_La_Roche_en_Ardenne",
            "Abdij_Tongerlo"
        ]
    }
}

# Seed that is used to define the training periods (for reproducibility)
SEED = 42

# Forecasting horizons in hours (4h, 12h, 1D, 2D, 4D, 7D, 14D, 21D, 30D)
HORIZONS = [4, 12, 24, 48, 96, 168, 336, 504, 720]

NUMBER_OF_REPEATS = 10


def run_single_experiment(dataset_name, repeat_id, station, training_weeks, lasso_weeks, series, exog_df):
    """
    Run a single experiment with the given parameters.

    Input
    -----
    dataset_name, repeat_id, station, training_weeks, lasso_weeks, series, exog_df

    Output
    ------
    A list containing the results of the experiment:
        [dataset_name, station, training_weeks, train_start, train_end, horizon, mae, mse]
    """

    # Make sure there is exogenous data for the given station, otherwise throw an error
    if exog_df is None:
        raise ValueError(f"No exogenous data found for station {station} in dataset {dataset_name}")

    # Generate random training period (start and end date) for the given dataset and station
    forecast_start = generate_forecast_start(
        series=series,
        seed=SEED,
        repeat_id=repeat_id,
        max_history_days=52 * 7,  # Max of 1 year of history that is tested
        max_horizon=max(HORIZONS)
    )

    # Fixed model training period
    train_start = forecast_start - pd.Timedelta(weeks=training_weeks)
    train_end = forecast_start

    # Historical data for LASSO selection
    lasso_start = train_end - pd.Timedelta(weeks=lasso_weeks)
    lasso_end = train_end

    # Time the LASSO selection process
    lasso_start_time = pd.Timestamp.now()

    # Apply LASSO selection on this historical window
    _, selected_stations, _ = select_LASSO_stations(series[lasso_start:lasso_end], exog_df[lasso_start:lasso_end])
    exog_selected = exog_df[selected_stations]
    nr_selected_stations = len(selected_stations)

    # Determine how long the LASSO selection process took
    lasso_end_time = pd.Timestamp.now()
    lasso_duration = (lasso_end_time - lasso_start_time).total_seconds()

    # Train the model once on the training period to reuse for all horizons
    model = train_sarima_model(
        series=series[train_start:train_end],
        exog_df=exog_selected[train_start:train_end],
        arima_order=(2, 0, 0),
        seasonal_order=(1, 0, 1, 24),
        max_iter=10000
    )

    results = []
    for horizon in HORIZONS:
        test_end = train_end + pd.Timedelta(hours=horizon)

        result = run_arimax(
            repeat_id,
            series=series,
            exog_df=exog_selected,
            model=model,
            start_date=train_start,
            end_date=test_end,
            hours_to_forecast=horizon,
            arima_order=(2, 0, 0),
            seasonal_order=(1, 0, 1, 24),
            confidence_score=False,
            use_LASSO_selection=False,
            max_iter=1000,
            plot=False
        )

        mae, mse = result[:2]

        results.append([dataset_name, station, training_weeks, lasso_weeks,
                        train_start, train_end, horizon, mae, mse, lasso_duration, nr_selected_stations])

    return results


def run_all_experiments(training_weeks, lasso_weeks):
    """
    Main function to run all experiments across datasets, stations, training periods, and forecast horizons. The results are
    saved to a CSV file for later analysis.
    """
    results_all = []

    # Loop over datasets and stations
    for dataset_name, dataset_info in DATASETS.items():
        # Read the preprocessed data
        df = pd.read_csv(dataset_info["file"])

        for station in dataset_info["stations"]:
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

            # Build a list of taks to run in parallel
            tasks = [(dataset_name, repeat_id, station, training_weeks, lasso_weeks, series, exog_df)
                     for repeat_id in range(NUMBER_OF_REPEATS)]

            with ProcessPoolExecutor() as executor:
                futures = [executor.submit(run_single_experiment, *t) for t in tasks]

                for f in as_completed(futures):
                    results_all.extend(f.result())

    # Save all results to a CSV file for later analysis
    os.makedirs("output/optimal_lasso_weeks", exist_ok=True)
    output_file = f"output/optimal_lasso_weeks/results_train{training_weeks}_lasso{lasso_weeks}.csv"

    with open(output_file, "w", newline="") as f:
        # Create CSV writer and write header
        writer = csv.writer(f)

        writer.writerow([
            "dataset",
            "station",
            "training_weeks",
            "lasso_weeks",
            "train_start",
            "train_end",
            "horizon",
            "mae",
            "mse",
            "lasso_duration",
            "nr_selected_stations"
        ])
        writer.writerows(results_all)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Determine optimal amount of historical data to build LASSO on.")
    parser.add_argument("--lasso_weeks", type=int, required=True, help="Number of historical weeks to use for LASSO selection")
    args = parser.parse_args()

    training_weeks = 8  # Fixed training period of 8 weeks for all experiments

    run_all_experiments(training_weeks, args.lasso_weeks)
