"""
Script: interval_coverage_experiment.py

Experiment P1.5: empirical coverage of the two-stage model's prediction intervals, as a function of
outage duration.

Per onset: fit with the last three days of the training window held out, reconstruct them to build
the calibration sample, refit on the full window, then forecast once out to the longest horizon and
slice. The residual SARIMA forecast is deterministic given the fitted model, so slicing 720 steps
is identical to forecasting h steps and one forecast serves every horizon.

The onsets come from the same SEED and history length as models_comparison_experiment.py, so the
coverage numbers are paired with the MAE numbers already reported. Keep the constants below in sync.

Run:
    python -m src.scripts.interval_coverage_experiment
"""
import argparse
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from src.evaluation.experiment_utils import write_results_csv
from src.evaluation.interval_coverage import conformal_bounds, empirical_bounds, summarise_interval
from src.models.regression_sarima_errors.train import train_regression_sarima_errors
from src.utils.generate_forecast_start import generate_forecast_start

# Datasets and stations, mirroring models_comparison_experiment.py
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

# Must match models_comparison_experiment.py to keep the onsets paired
SEED = 42

# Forecasting horizons in hours (4h, 12h, 1D, 2D, 4D, 7D, 14D, 21D, 30D)
HORIZONS = [4, 12, 24, 48, 96, 168, 336, 504, 720]

NUMBER_OF_REPEATS = 10

# Residual SARIMA orders as used elsewhere for this model
RESIDUAL_ORDER, RESIDUAL_SEASONAL_ORDER = (3, 0, 0), (1, 0, 0, 24)

# Held-out calibration windows. 3 days is what regression_sarima_errors/confidence_score.py uses;
# the longer ones test how much of the under-coverage is simply too small a calibration sample.
# The total history stays at training_weeks, so a longer holdout leaves the calibration fit less
# data - that is the real operational trade-off.
DEFAULT_CALIBRATION_DAYS = [3, 7, 14]

# 0.95 is the level the paper claims; the others come free from the same forecast and give the
# calibration curve. 0.99 is left out: 72 calibration hours cannot pin down that tail.
DEFAULT_LEVELS = [0.5, 0.8, 0.9, 0.95]

# Both bands come from the same calibration errors, so comparing them costs no extra fits
BAND_METHODS = {"quantile": empirical_bounds, "conformal": conformal_bounds}


def two_stage_predict(df, target_station, exog_cols, model, test_index):
    """
    Point forecast of the two-stage model, as in regression_sarima_errors/execute_forecast.py: the
    OLS regression on the neighbours plus the SARIMA forecast of its residuals.

    Input
    -----
    df: Wide DataFrame, one column per station, hourly
    target_station: Target station column
    exog_cols: Neighbour station columns used as regressors
    model: (regression, residual_model) tuple from train_regression_sarima_errors
    test_index: Window to forecast

    Output
    ------
    predictions: Series with the forecast over test_index
    """
    regression, residual_model = model

    # The neighbour observations are assumed known over the horizon; interpolate short gaps so a
    # missing regressor value does not crash the prediction
    X_test = df.loc[test_index, exog_cols].interpolate(limit_direction="both")

    spatial = regression.predict(X_test)
    local = residual_model.get_forecast(steps=len(test_index)).predicted_mean.values

    return pd.Series(spatial + local, index=test_index)


def run_intervals(df_complete, target_station, exog_cols, train_start, forecast_start, test_index,
                  calibration_days, levels):
    """
    Interval bounds over the reconstruction window, for every calibration length, band method and
    level. The reconstruction forecast is fitted once and shared, so the cost is one test fit plus
    one calibration fit per calibration length.

    Input
    -----
    df_complete: Wide DataFrame, one column per station, hourly
    target_station: Target station column
    exog_cols: Neighbour station columns used as regressors
    train_start, forecast_start: Training window boundaries
    test_index: Reconstruction window
    calibration_days: Holdout lengths in days
    levels: Nominal coverage levels

    Output
    ------
    bounds: Dict mapping (days, method, level) to a (lower, upper) tuple
    """
    # Fitted on the full history and forecast once; only the band depends on the holdout
    test_model = train_regression_sarima_errors(
        df_complete[train_start:forecast_start], target_station, exog_cols,
        RESIDUAL_ORDER, RESIDUAL_SEASONAL_ORDER, max_iter=1000
    )
    test_predictions = two_stage_predict(
        df_complete, target_station, exog_cols, test_model, test_index
    )

    bounds = {}

    for days in calibration_days:
        calibration_start = forecast_start - pd.Timedelta(days=days)

        # Starts one step after the boundary, as in the model's execute_forecast module
        calibration_index = df_complete.loc[calibration_start:forecast_start].index[1:]

        # Fit with the holdout carved out, then reconstruct it to measure real forecast errors
        calibration_model = train_regression_sarima_errors(
            df_complete[train_start:calibration_start], target_station, exog_cols,
            RESIDUAL_ORDER, RESIDUAL_SEASONAL_ORDER, max_iter=1000
        )
        calibration_predictions = two_stage_predict(
            df_complete, target_station, exog_cols, calibration_model, calibration_index
        )
        calibration_errors = (
            df_complete.loc[calibration_index, target_station] - calibration_predictions
        )

        for method, build_bounds in BAND_METHODS.items():
            for level in levels:
                bounds[(days, method, level)] = build_bounds(
                    test_predictions, calibration_errors, level
                )

    return bounds


def run_single_experiment(task):
    """
    Coverage rows for one (dataset, station, onset).

    Input
    -----
    task: Tuple of (dataset_name, repeat_id, station, series, exog_df, df_complete, training_days,
      calibration_days, levels)

    Output
    ------
    rows: One row per (calibration length, method, level, horizon)
    """
    (dataset_name, repeat_id, station, series, exog_df, df_complete, training_days,
     calibration_days, levels) = task

    # Same onset as the accuracy experiment, since SEED and the history length match
    forecast_start = generate_forecast_start(
        series=series, seed=SEED, repeat_id=repeat_id,
        max_history_days=training_days, max_horizon=max(HORIZONS)
    )

    train_start = forecast_start - pd.Timedelta(days=training_days)

    # Starts one step after the onset, as in the model's execute_forecast module
    test_index = df_complete.loc[forecast_start:].index[1:max(HORIZONS) + 1]

    print(f"{dataset_name} {station} onset {forecast_start} (repeat {repeat_id})")

    bounds = run_intervals(
        df_complete, station, exog_df.columns.tolist(), train_start, forecast_start, test_index,
        calibration_days, levels
    )

    actual = df_complete.loc[test_index, station]

    rows = []
    for (days, method, level), (lower, upper) in bounds.items():
        for horizon in HORIZONS:
            # Cumulative over hours 1..horizon, matching how MAE is reported per horizon
            window = test_index[:horizon]
            summary = summarise_interval(
                lower.loc[window], upper.loc[window], actual.loc[window], level
            )

            rows.append([
                dataset_name,
                station,
                repeat_id,
                train_start,
                forecast_start,
                days,
                method,
                level,
                horizon,
                summary["n_total"],
                summary["n_inside"],
                summary["coverage"],
                summary["mean_width"],
                summary["mean_interval_score"]
            ])

    return rows


def prepare_station_data(df, station):
    """
    Build the hourly series and neighbour frame for one station, as in
    models_comparison_experiment.py.

    Input
    -----
    df: Long-format DataFrame with station_name, datetime and temp_dry_avg_2m
    station: Target station

    Output
    ------
    series: Hourly target series
    exog_df: Hourly neighbour observations
    df_complete: series joined with exog_df
    """
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

    df_complete = series.to_frame(name=station).join(exog_df)

    return series, exog_df, df_complete


def run_all_experiments(training_weeks, repeats, calibration_days, levels, max_workers):
    """
    Run every dataset, station and onset, and write the results to output/interval_coverage/.

    Input
    -----
    training_weeks: Weeks of training data per onset
    repeats: Failure onsets per station
    calibration_days: Holdout lengths in days
    levels: Nominal coverage levels
    max_workers: Parallel worker processes (None lets the pool decide)
    """
    training_days = training_weeks * 7
    rows = []

    for dataset_name, dataset_info in DATASETS.items():
        df = pd.read_csv(dataset_info["file"])

        for station in dataset_info["stations"]:
            series, exog_df, df_complete = prepare_station_data(df, station)

            tasks = [
                (dataset_name, i, station, series, exog_df, df_complete, training_days,
                 calibration_days, levels)
                for i in range(repeats)
            ]

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(run_single_experiment, task) for task in tasks]

                for future in as_completed(futures):
                    rows.extend(future.result())

    write_results_csv(
        f"output/interval_coverage/interval_coverage_{training_weeks}_weeks.csv",
        [
            "Dataset",
            "Station",
            "Repeat",
            "Train_start",
            "Forecast_start",
            "Calibration_days",
            "Method",
            "Level",
            "Horizon",
            "N_total",
            "N_inside",
            "Coverage",
            "Mean_width",
            "Mean_interval_score"
        ],
        rows
    )

    print(f"Wrote {len(rows)} rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the empirical coverage of the two-stage model's prediction intervals "
                    "as a function of outage duration, across all datasets, stations and failure "
                    "onsets (P1.5)."
    )
    parser.add_argument("--training_weeks", type=int, default=8,
                        help="Number of weeks to use for training (default: 8).")
    parser.add_argument("--repeats", type=int, default=NUMBER_OF_REPEATS,
                        help=f"Number of failure onsets per station (default: {NUMBER_OF_REPEATS}). "
                             "Lower it for a quick trial run.")
    parser.add_argument("--calibration_days", type=int, nargs="+",
                        default=DEFAULT_CALIBRATION_DAYS,
                        help=f"Holdout lengths in days (default: {DEFAULT_CALIBRATION_DAYS}).")
    parser.add_argument("--levels", type=float, nargs="+", default=DEFAULT_LEVELS,
                        help=f"Nominal coverage levels to evaluate (default: {DEFAULT_LEVELS}).")
    parser.add_argument("--max_workers", type=int, default=None,
                        help="Number of parallel worker processes (default: let the pool decide).")
    args = parser.parse_args()

    run_all_experiments(args.training_weeks, args.repeats, args.calibration_days, args.levels,
                        args.max_workers)
