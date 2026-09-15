"""
Module: aging_experiment.py

Generic engine for the "model aging" experiment: train a model once on a training window, then check
how far its forecast quality degrades the longer it goes without ever being retrained. For a range of
"age gaps" the model's internal state (never its parameters) is advanced with the true intervening
observations, and a fixed-horizon forecast is evaluated from that aged state.

Shared by test_aging_model.py (ARIMAX) and test_aging_model_RegressionSARIMAErrors.py (the two-stage
regression with SARIMA errors model): the training/aging/forecasting mechanics only differ in what
"train", "age" and "forecast" mean for a given model, so those three steps are passed in as functions.

Functions:
- run_aging_experiment: Run the aging experiment for every station of one dataset and save the
  results to a CSV file.
"""
import pandas as pd
from src.utils.generate_forecast_start import generate_forecast_start
from src.evaluation.experiment_utils import get_worker_df_complete, run_tasks_for_station, write_results_csv


# Metadata about the used datasets and stations to evaluate on, shared by every aging experiment so
# different models' aging curves are directly comparable
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

FORECAST_HORIZON_HOURS = 48


def _run_single_experiment(task, train_fn, age_fn, forecast_fn, training_weeks):
    """
    Run a single aging experiment: train once, then re-forecast at every age gap from a state that
    has been advanced (never retrained) with the true observations up to that point.

    Input
    -----
    task: A tuple containing (dataset_name, repeat_id, station, station_index)
    train_fn: (df_train_window, station, exog_cols) -> (model, exog_cols). The returned exog_cols lets
      a model narrow its own candidate stations during training (e.g. ARIMAX's LASSO selection on
      large networks); most models just return the exog_cols they were given
    age_fn: (model, df_gap_window, station, exog_cols) -> aged_model
    forecast_fn: (repeat_id, df_complete, station, model, exog_cols, train_start, test_begin, test_end)
      -> (mae, mse)
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
    model, exog_cols = train_fn(df_complete[train_start:train_end], station, exog_cols)
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
            aged_model = age_fn(model, df_complete[gap_start:test_begin], station, exog_cols)

        mae, mse = forecast_fn(
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


def _load_station_data(dataset_info, station):
    """
    Build the wide, hourly (target + neighbouring stations) DataFrame for one station, from the
    preprocessed long-format dataset file.

    Input
    -----
    dataset_info: One DATASETS entry, with "file" and "stations" keys
    station: Name of the target station

    Output
    ------
    df_complete: DataFrame with a datetime index, the target station and every neighbouring station
      as columns, resampled to hourly frequency with short gaps interpolated
    """
    df = pd.read_csv(dataset_info["file"])
    station_data = df[df['station_name'] == station]

    series = pd.Series(
        station_data['temp_dry_avg_2m'].values,
        index=pd.to_datetime(station_data['datetime'])
    ).asfreq('1h').dropna()

    other_stations = [s for s in df['station_name'].unique() if s != station]
    exog_data = df[df['station_name'].isin(other_stations)]

    exog_pivot = exog_data.pivot_table(
        index='datetime', columns='station_name', values='temp_dry_avg_2m'
    )
    exog_pivot.index = pd.to_datetime(exog_pivot.index)
    exog_df = exog_pivot.reindex(series.index)

    series = series.resample('1h').mean().interpolate(limit_direction="both")
    exog_df = exog_df.resample('1h').mean().interpolate(limit_direction="both")

    return series.to_frame(name=station).join(exog_df)


def run_aging_experiment(dataset_name, training_weeks, train_fn, age_fn, forecast_fn, output_path,
                         max_workers=None):
    """
    Run the aging experiment for every station of one dataset and save the results to a CSV file.

    Input
    -----
    dataset_name: Name of the dataset to run the experiment on (must be a key of DATASETS)
    training_weeks: Number of weeks of training data to use
    train_fn, age_fn, forecast_fn: Model-specific hooks, see _run_single_experiment
    output_path: Path of the CSV file to write the results to
    max_workers: Number of worker processes (default is None, which uses one per core). Lower it if
      the full core count runs into memory pressure
    """
    results_all = []
    dataset_info = DATASETS[dataset_name]

    for station_index, station in enumerate(dataset_info["stations"]):
        print(f"Running experiments for dataset {dataset_name}, station {station}...")

        df_complete = _load_station_data(dataset_info, station)

        tasks = [
            (dataset_name, repeat_id, station, station_index)
            for repeat_id in range(NUMBER_OF_REPEATS)
        ]

        results_all.extend(run_tasks_for_station(
            df_complete, tasks, _run_single_experiment, (train_fn, age_fn, forecast_fn, training_weeks),
            max_workers=max_workers
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
