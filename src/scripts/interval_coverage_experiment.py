import argparse
import zlib
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from src.models.regsarima.train import RESIDUAL_ORDER, RESIDUAL_SEASONAL_ORDER
from src.utils.datasets import DATASETS, load_station_data
from src.utils.experiment_utils import write_results_csv
from src.evaluation.interval_coverage import conformal_bounds, empirical_bounds, summarise_interval
from src.models.regsarima.train import train_regression_sarima_errors
from src.models.regsarima.execute_forecast import forecast_two_stage
from src.utils.generate_forecast_start import generate_forecast_start

# Must match model_comparison_experiment.py to keep the onsets paired
SEED = 42

# Forecasting horizons in hours (4h, 12h, 1D, 2D, 4D, 7D, 14D, 21D, 30D)
HORIZONS = [4, 12, 24, 48, 96, 168, 336, 504, 720]

NUMBER_OF_REPEATS = 10

# Lengths of the held-out validation episode. The total history stays at the training weeks, so a longer episode
# leaves the calibration fit less data
CALIBRATION_DAYS = [3, 7, 14]

# Nominal level of the prediction intervals
LEVEL = 0.95

# Both bands come from the same calibration errors, so comparing them costs no extra fits
BAND_METHODS = {"quantile": empirical_bounds, "conformal": conformal_bounds}


def predict(df, exog_cols, model, index):
    """RegSARIMA's reconstruction over index, with short gaps in the neighbour readings interpolated."""
    predictions, _ = forecast_two_stage(model, df.loc[index, exog_cols].interpolate(limit_direction="both"))
    return predictions


def run_intervals(df_complete, target_station, exog_cols, train_start, forecast_start, test_index):
    """
    Interval bounds over the reconstruction window, for every calibration length and band method. The reconstruction forecast is fitted once and shared, so the cost is one test fit plus
    one calibration fit per calibration length.

    Input
    -----
    df_complete: Wide DataFrame, one column per station, hourly
    target_station: Target station column
    exog_cols: Neighbour station columns used as regressors
    train_start, forecast_start: Training window boundaries
    test_index: Reconstruction window

    Output
    ------
    bounds: Dict mapping (days, method) to a (lower, upper) tuple
    """
    # Fitted on the full history and forecast once; only the band depends on the holdout
    test_model = train_regression_sarima_errors(
        df_complete[train_start:forecast_start], target_station, exog_cols,
        RESIDUAL_ORDER, RESIDUAL_SEASONAL_ORDER, max_iter=1000
    )
    test_predictions = predict(df_complete, exog_cols, test_model, test_index)

    bounds = {}

    for days in CALIBRATION_DAYS:
        calibration_start = forecast_start - pd.Timedelta(days=days)

        # Starts one step after the boundary, as in the model's execute_forecast module
        calibration_index = df_complete.loc[calibration_start:forecast_start].index[1:]

        # Fit with the holdout carved out, then reconstruct it to measure real forecast errors
        calibration_model = train_regression_sarima_errors(
            df_complete[train_start:calibration_start], target_station, exog_cols,
            RESIDUAL_ORDER, RESIDUAL_SEASONAL_ORDER, max_iter=1000
        )
        calibration_predictions = predict(df_complete, exog_cols, calibration_model, calibration_index)
        calibration_errors = (
            df_complete.loc[calibration_index, target_station] - calibration_predictions
        )

        for method, build_bounds in BAND_METHODS.items():
            bounds[(days, method)] = build_bounds(test_predictions, calibration_errors, LEVEL)

    return bounds


def run_single_experiment(task):
    """
    Coverage rows for one (dataset, station, onset).

    Input
    -----
    task: Tuple of (dataset_name, repeat_id, station, series, exog_df, df_complete, training_days)

    Output
    ------
    rows: One row per (calibration length, method, horizon)
    """
    dataset_name, repeat_id, station, series, exog_df, df_complete, training_days = task

    # Same onset as the accuracy experiment, since SEED, its per-station offset and the history length match
    station_seed = SEED + zlib.crc32(station.encode())
    forecast_start = generate_forecast_start(
        series=series, seed=station_seed, repeat_id=repeat_id,
        max_history_days=training_days, max_horizon=max(HORIZONS)
    )

    train_start = forecast_start - pd.Timedelta(days=training_days)

    # Starts one step after the onset, as in the model's execute_forecast module
    test_index = df_complete.loc[forecast_start:].index[1:max(HORIZONS) + 1]

    print(f"{dataset_name} {station} onset {forecast_start} (repeat {repeat_id})")

    bounds = run_intervals(
        df_complete, station, exog_df.columns.tolist(), train_start, forecast_start, test_index
    )

    actual = df_complete.loc[test_index, station]

    rows = []
    for (days, method), (lower, upper) in bounds.items():
        for horizon in HORIZONS:
            # Cumulative over hours 1..horizon, matching how MAE is reported per horizon
            window = test_index[:horizon]
            summary = summarise_interval(lower.loc[window], upper.loc[window], actual.loc[window])

            rows.append([
                dataset_name,
                station,
                repeat_id,
                train_start,
                forecast_start,
                days,
                method,
                horizon,
                summary["n_total"],
                summary["n_inside"],
                summary["coverage"],
                summary["mean_width"]
            ])

    return rows


def run_all_experiments(training_weeks, repeats, max_workers):
    """
    Run every dataset, station and onset, and write the results to output/interval_coverage/.

    Input
    -----
    training_weeks: Weeks of training data per onset
    repeats: Failure onsets per station
    max_workers: Parallel worker processes (None lets the pool decide)
    """
    training_days = training_weeks * 7
    rows = []

    for dataset_name, dataset_info in DATASETS.items():
        df = pd.read_csv(dataset_info["file"])

        for station in dataset_info["stations"]:
            df_complete = load_station_data(df, station)
            series = df_complete[station]
            exog_df = df_complete.drop(columns=station)

            tasks = [
                (dataset_name, i, station, series, exog_df, df_complete, training_days)
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
            "Horizon",
            "N_total",
            "N_inside",
            "Coverage",
            "Mean_width"
        ],
        rows
    )

    print(f"Wrote {len(rows)} rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the empirical coverage of the two-stage model's prediction intervals "
                    "as a function of outage duration, across all datasets, stations and failure "
                    "onsets."
    )
    parser.add_argument("--training_weeks", type=int, default=8,
                        help="Number of weeks to use for training (default: 8).")
    parser.add_argument("--repeats", type=int, default=NUMBER_OF_REPEATS,
                        help=f"Number of failure onsets per station (default: {NUMBER_OF_REPEATS}). "
                             "Lower it for a quick trial run.")
    parser.add_argument("--max_workers", type=int, default=None,
                        help="Number of parallel worker processes (default: let the pool decide).")
    args = parser.parse_args()

    run_all_experiments(args.training_weeks, args.repeats, args.max_workers)
