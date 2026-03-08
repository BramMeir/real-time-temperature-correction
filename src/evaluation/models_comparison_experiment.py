"""
Main script to run the full experiment comparing all models across all datasets, different stations, training periods,
and forecast horizons. The results are saved to a CSV file for later analysis.

python -m src.evaluation.models_comparison_experiment
"""
import csv
import pandas as pd
import numpy as np
from src.models.arima.repeat_forecast import _run_single_forecast as run_arimax
from src.models.random_forest.execute_forecast import run_single_forecast as run_rf
from src.models.LSTM.execute_forecast import run_single_forecast as run_lstm

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
        "file": "data/Synthetic/temperatue_data.csv",
        "stations": ["Stadhuis_Brussel_Grote_Markt", "Stadhuis_Antwerpen_Grote_Markt", "Grote_Markt_Kortrijk",
                     "Tielt", "Gembloux", "Slag_om_Ardennen_Museum_La_Roche_en_Ardenne", "Abdij_Tongerlo"]
    }
}

# Seed that is used to define the training periods (for reproducibility)
SEED = 42

# Forecasting horizons in hours (24h, 48h, 1 week, 2 weeks)
HORIZONS = [24, 48, 168, 336]
# HORIZONS = [336]

OUTPUT_FILE = "output/models_comparison_experiment.csv"
NUMBER_OF_REPEATS = 10

MODELS = {
    "ARIMAX": run_arimax,
    "LSTM": run_lstm,
    "RF": run_rf
}

MODEL_TRAINING_DAYS = {
    "ARIMAX": 2 * 7,  # 2 weeks of hourly data (336 hours)
    "LSTM": 8 * 7,    # 8 weeks of hourly data (1344 hours)
    "RF": 8 * 7       # 8 weeks of hourly data (1344 hours)
}


def run_all_experiments():
    """
    Main function to run all experiments across datasets, stations, training periods, and forecast horizons. The results are
    saved to a CSV file for later analysis.
    """
    with open(OUTPUT_FILE, "w", newline="") as f:
        # Create CSV writer and write header
        writer = csv.writer(f)

        writer.writerow([
            "dataset",
            "station",
            "model",
            "train_start",
            "train_end",
            "horizon",
            "mae",
            "mse"
        ])

        # Loop through all combinations of dataset, station, training period, forecast horizon, and model
        for dataset_name, dataset_info in DATASETS.items():
            # Retrieve the file path for the dataset
            dataset_file = dataset_info["file"]

            for station in dataset_info["stations"]:
                # Create the pandas DataFrame for the dataset with hourly frequency and datetime index
                # Read the preprocessed data
                df = pd.read_csv(dataset_file)

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

                # Create also a complete version of the DataFrame for some of the models that require it (e.g., LSTM, Transformer)
                df_complete = series.to_frame(name=station).join(exog_df)

                # Generate NUMBER_OF_REPEATS random training periods for the given dataset and station
                for i in range(NUMBER_OF_REPEATS):
                    # Generate random training period (start and end date) for the given dataset and station
                    forecast_start = generate_forecast_start(
                        series=series,
                        seed=SEED,
                        repeat_id=i,
                        max_history_days=max(MODEL_TRAINING_DAYS.values()),
                        max_horizon=max(HORIZONS)
                    )

                    for horizon in HORIZONS:
                        for model_name, model_fn in MODELS.items():
                            # Determine the required training period for the model
                            training_days = MODEL_TRAINING_DAYS[model_name]
                            train_start = forecast_start - pd.Timedelta(days=training_days)
                            train_end = forecast_start
                            test_end = train_end + pd.Timedelta(hours=horizon)

                            print(
                                dataset_name,
                                station,
                                model_name,
                                train_start,
                                train_end,
                                horizon
                            )

                            if model_name == "ARIMAX":
                                mae, mse, _, _, _, _ = model_fn(
                                    i,
                                    series=series,
                                    exog_df=exog_df,
                                    start_date=train_start,
                                    end_date=test_end,
                                    hours_to_forecast=horizon,
                                    arima_order=(2, 0, 0),
                                    seasonal_order=(1, 0, 1, 24),
                                    confidence_score=False,
                                    max_iter=1000,
                                    plot=False
                                )

                            elif model_name in ["LSTM", "Transformer"]:
                                mae, mse, _ = model_fn(
                                    df=df_complete,
                                    number=i,
                                    target_station=station,
                                    start=train_start,
                                    train_end=train_end,
                                    test_end=test_end,
                                    mode="forecast"
                                )

                            else:
                                mae, mse, _, _ = model_fn(
                                    df=df_complete,
                                    target_station=station,
                                    exog_cols=exog_df.columns.tolist(),
                                    start=train_start,
                                    train_end=train_end,
                                    test_end=test_end,
                                    mode="forecast"
                                )

                            writer.writerow([
                                dataset_name,
                                station,
                                model_name,
                                train_start,
                                train_end,
                                horizon,
                                mae,
                                mse
                            ])


def generate_forecast_start(series, seed, repeat_id, max_history_days, max_horizon):
    """
    Generate a reproducible random forecast start date for a given time series, ensuring
    that there is enough historical data for training and enough future data for forecasting.

    Input
    -----
    series: Target time series with datetime index.
    seed: Base seed for reproducibility.
    repeat_id: Repeat number to ensure different samples.
    max_history_days: Maximum number of days of historical data to use for training.
    max_horizon: Largest forecast horizon (hours).

    Output
    ------
    forecast_start: Start timestamp of the forecast period.
    """
    # Create a random number generator with a seed that combines the base seed and repeat ID
    rng = np.random.default_rng(seed + repeat_id)

    # Minimum date is the earliest timestamp plus the maximum number of training days, to ensure enough history for training
    # Maximum date is the latest timestamp minus the maximum forecast horizon, to ensure enough future data for forecasting
    min_date = series.index.min() + pd.Timedelta(days=max_history_days)
    max_date = series.index.max() - pd.Timedelta(hours=max_horizon)

    if max_date <= min_date:
        raise ValueError("Series too short for chosen configuration")

    random_fraction = rng.random()

    forecast_start = min_date + (max_date - min_date) * random_fraction
    forecast_start = forecast_start.floor('h')

    return forecast_start


if __name__ == "__main__":
    # Run all experiments and save results to CSV
    run_all_experiments()
