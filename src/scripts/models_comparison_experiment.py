"""
Main script to run the full experiment comparing all models across all datasets, different stations, training periods,
and forecast horizons. The results are saved to a CSV file for later analysis.

python -m src.scripts.models_comparison_experiment --model RF
"""
import csv
import pandas as pd
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from src.utils.generate_forecast_start import generate_forecast_start
from src.models.arima.train import train_sarima_model
from src.models.LSTM.train import train_LSTM_model
from src.models.MLP.train import train_mlp_model
from src.models.transformer.train import train_transformer_model
from src.models.random_forest.train import train_random_forest
from src.models.TCN.train import train_tcn_model
from src.models.arima.repeat_forecast import _run_single_forecast as run_arimax
from src.models.random_forest.execute_forecast import run_single_forecast as run_rf
from src.models.LSTM.execute_forecast import run_single_forecast as run_lstm
from src.models.MLP.execute_forecast import run_single_forecast as run_mlp
from src.models.transformer.execute_forecast import run_single_forecast as run_transformer
from src.models.TCN.execute_forecast import run_single_forecast as run_tcn
from src.data.create_supervised import create_supervised_dataset
from src.data.create_3d_dataset import create_3d_dataset

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

TRAIN_MODELS = {
    "ARIMA": train_sarima_model,
    "ARIMAX": train_sarima_model,
    "LSTM": train_LSTM_model,
    "RF": train_random_forest,
    "MLP": train_mlp_model,
    "Transformer": train_transformer_model,
    "TCN": train_tcn_model
}

FORECAST_MODELS = {
    "ARIMA": run_arimax,
    "ARIMAX": run_arimax,
    "LSTM": run_lstm,
    "RF": run_rf,
    "MLP": run_mlp,
    "Transformer": run_transformer,
    "TCN": run_tcn
}

MODEL_TRAINING_DAYS = {
    "ARIMA": 2 * 7,         # 2 weeks of hourly data (336 hours)
    "ARIMAX": 8 * 7,        # 8 weeks of hourly data (1344 hours)
    "LSTM": 8 * 7,          # 8 weeks of hourly data (1344 hours)
    "RF": 8 * 7,            # 8 weeks of hourly data (1344 hours)
    "MLP": 8 * 7,           # 8 weeks of hourly data (1344 hours)
    "Transformer": 8 * 7,   # 8 weeks of hourly data (1344 hours)
    "TCN": 8 * 7            # 8 weeks of hourly data (1344 hours)
}


def run_single_experiment(task):
    """
    Run a single experiment with the given parameters.

    Input
    -----
    task: A tuple containing (dataset_name, repeat_id, model_name, station, series, exog_df, df_complete)

    Output
    ------
    A list containing the results of the experiment:
        [dataset_name, station, model_name, train_start, train_end, horizon, mae, mse]
    """
    dataset_name, repeat_id, model_name, station, series, exog_df, df_complete = task

    # Generate random training period (start and end date) for the given dataset and station
    forecast_start = generate_forecast_start(
        series=series,
        seed=SEED,
        repeat_id=repeat_id,
        max_history_days=max(MODEL_TRAINING_DAYS.values()),
        max_horizon=max(HORIZONS)
    )

    # Determine the required training period for the model
    training_days = MODEL_TRAINING_DAYS[model_name]

    train_start = forecast_start - pd.Timedelta(days=training_days)
    train_end = forecast_start

    # Train the model once on the training period to reuse for all horizons
    train_model_fn = TRAIN_MODELS[model_name]

    if model_name in ["ARIMA", "ARIMAX"]:
        model = train_model_fn(
            series=series[train_start:train_end],
            exog_df=exog_df[train_start:train_end] if model_name == "ARIMAX" else None,
            arima_order=(2, 0, 0),
            seasonal_order=(1, 0, 1, 24),
            max_iter=1000
        )
    elif model_name == "LSTM":
        model = train_model_fn(
            df=df_complete[train_start:train_end],
            target_station=station,
            previous_time_steps=24,
        )
    elif model_name == "Transformer":
        model, x_scaler, y_scaler = train_model_fn(
            df=df_complete[train_start:train_end],
            target_station=station,
            previous_time_steps=8,
        )
    elif model_name in ["RF", "MLP"]:
        # Create supervised dataset
        X, y = create_supervised_dataset(
            df_complete, target_station=station, previous_time_steps=24, exog_cols=exog_df.columns.tolist(), exog_lags=0
        )

        # Select the data based on the training period
        X_train, y_train = X.loc[train_start:train_end], y.loc[train_start:train_end]

        # Train the model
        train_result = train_model_fn(X_train, y_train)
        model = train_result[0] if model_name == "RF" else train_result
    elif model_name == "TCN":
        # Create supervised dataset
        X, y, dates = create_3d_dataset(
            df_complete, target_station=station, previous_time_steps=24 * 3, exog_cols=exog_df.columns.tolist()
        )

        # Select the data based on the training period
        train_mask = (dates >= train_start) & (dates <= train_end)
        X_train, y_train = X[train_mask], y[train_mask]

        # Train the TCN model
        model = train_model_fn(X_train, y_train)

    # Evaluate the model for each forecast horizon and save the results
    results = []
    for horizon in HORIZONS:
        test_end = train_end + pd.Timedelta(hours=horizon)

        forecast_model_fn = FORECAST_MODELS[model_name]

        print(
            dataset_name,
            station,
            model_name,
            train_start,
            train_end,
            horizon
        )

        if model_name in ["ARIMA", "ARIMAX"]:
            result = forecast_model_fn(
                repeat_id,
                series=series,
                exog_df=exog_df if model_name == "ARIMAX" else None,
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
        elif model_name == "LSTM":
            result = forecast_model_fn(
                df=df_complete,
                number=repeat_id,
                target_station=station,
                model=model,
                start=train_start,
                train_end=train_end,
                test_end=test_end,
                mode="forecast"
            )
        elif model_name == "Transformer":
            result = forecast_model_fn(
                df=df_complete,
                number=repeat_id,
                target_station=station,
                model=model,
                x_scaler=x_scaler,
                y_scaler=y_scaler,
                start=train_start,
                train_end=train_end,
                test_end=test_end,
                mode="forecast"
            )
        elif model_name in ["RF", "MLP", "TCN"]:
            result = forecast_model_fn(
                df=df_complete,
                target_station=station,
                model=model,
                exog_cols=exog_df.columns.tolist(),
                start=train_start,
                train_end=train_end,
                test_end=test_end,
                mode="forecast"
            )

        mae, mse = result[:2]

        results.append((horizon, mae, mse))

    return [[
        dataset_name,
        station,
        model_name,
        train_start,
        train_end,
        horizon,
        mae,
        mse
    ] for horizon, mae, mse in results]


def run_all_experiments(model_name):
    """
    Main function to run all experiments across datasets, stations, training periods, and forecast horizons. The results are
    saved to a CSV file for later analysis.

    Input
    -----
    model_name: Name of the model to run (must be a key in the MODELS dictionary)
    """
    with open(f"output/models_comparison_{model_name}_expanded_8_weeks.csv", "w", newline="") as f:
        # Create CSV writer and write header
        writer = csv.writer(f)

        writer.writerow([
            "Dataset",
            "Station",
            "Model",
            "Train_start",
            "Train_end",
            "Horizon",
            "MAE",
            "MSE"
        ])

        # Loop through all combinations of dataset, station, training period, forecast horizon, and model
        for dataset_name, dataset_info in DATASETS.items():
            # Retrieve the file path for the dataset
            dataset_file = dataset_info["file"]

            # Read the preprocessed data
            df = pd.read_csv(dataset_file)

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

                # Create also a complete version of the DataFrame for some of the models that require it (e.g., LSTM, Transformer)
                df_complete = series.to_frame(name=station).join(exog_df)

                # Build a list of taks to run in parallel
                tasks = []

                # Generate NUMBER_OF_REPEATS random training periods for the given dataset and station
                for i in range(NUMBER_OF_REPEATS):
                    tasks.append((
                        dataset_name,
                        i,
                        model_name,
                        station,
                        series,
                        exog_df,
                        df_complete
                    ))

                # If on GPU (for LSTM + Transformer), do not use too much parallelism to avoid out-of-memory errors,
                # so we run sequentially
                if model_name in ["LSTM", "Transformer", "TCN"]:
                    for task in tasks:
                        results = run_single_experiment(task)

                        # Write each result to the CSV file
                        for result in results:
                            writer.writerow(result)
                else:
                    with ProcessPoolExecutor() as executor:
                        futures = [executor.submit(run_single_experiment, task) for task in tasks]

                        for future in as_completed(futures):
                            results = future.result()

                            # Write each result to the CSV file
                            for result in results:
                                writer.writerow(result)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run the full experiment comparing all models across all datasets, "
                                                 "different stations, training periods, and forecast horizons. The results "
                                                 "are saved to a CSV file for later analysis.")
    parser.add_argument("--model", choices=TRAIN_MODELS.keys(), required=True,
                        help="Name of the model to run (must be a key in the MODELS dictionary)")
    args = parser.parse_args()

    # Run all experiments and save results to CSV
    run_all_experiments(args.model)
