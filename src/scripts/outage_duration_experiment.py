"""
Experiment: does the exogenous-station beta extracted from ARIMAX forecast worse than the
OLS beta as the "outage duration" (forecast horizon, i.e. how long the target station's own
recent history has stopped being informative) grows -- and does the full ARIMAX model's
forecast converge toward its own (worse) beta-only floor as that happens, the way the
two-stage model converges toward the (better) OLS floor?

For each station/repeat we fit OLS, ARIMAX (standard (2,0,0)(1,0,1,24) order), and the
two-stage model once, then forecast each ONCE over the longest horizon and slice that single
forecast for every shorter horizon (instead of re-forecasting per horizon). Both the
cumulative error (averaged over hours 1..h) and the pointwise error (at hour h alone) are
recorded, since they tell a different story: cumulative is diluted by the accurate short
leads and is the practically relevant "average error over an outage of length h", while
pointwise isolates the error at exactly h hours out, which is what actually shows the AR
contribution decaying to zero.

- ols_beta:      OLS regression alone: beta_ols'x_t + intercept
- arimax_beta:   Only ARIMAX's exogenous coefficients, no AR/seasonal contribution: beta_arimax'x_t
- arimax_full:   The actual fitted ARIMAX model's forecast (AR/seasonal included)
- twostage_full: The actual fitted two-stage model's forecast (residual SARIMA included)

python -m src.scripts.outage_duration_experiment
"""
import csv
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from src.utils.generate_forecast_start import generate_forecast_start
from src.models.arima.train import train_sarima_model
from src.models.linear_regression.train import train_neighbour_regression
from src.models.regression_sarima_errors.train import train_regression_sarima_errors
from src.models.arima.repeat_forecast import _run_single_forecast as run_arimax
from src.models.regression_sarima_errors.execute_forecast import run_single_forecast as run_regression_sarima_errors
from src.scripts.models_comparison_experiment import DATASETS as ALL_DATASETS

# Synthetic is left out for now: ARIMAX fitting is much slower on it, and KMI + Turku already
# give two independent real-world station networks
DATASETS = {name: info for name, info in ALL_DATASETS.items() if name != "SYNTHETIC"}

SEED = 42
NUMBER_OF_REPEATS = 20
TRAIN_WEEKS = 8
HORIZONS = [4, 12, 24, 48, 96, 168, 336, 504, 720]
MAX_HORIZON = max(HORIZONS)

ARIMAX_ORDER = (2, 0, 0)
ARIMAX_SEASONAL_ORDER = (1, 0, 1, 24)
RESIDUAL_SARIMA_ORDER = (3, 0, 0)
RESIDUAL_SARIMA_SEASONAL_ORDER = (1, 0, 0, 24)


def run_single_experiment(task):
    """
    Fit OLS, ARIMAX, and the two-stage model once on one station/repeat's training window,
    forecast each once over MAX_HORIZON, and slice that forecast for every shorter horizon.

    Input
    -----
    task: (dataset_name, station, repeat_id, series, exog_df, df_complete)

    Output
    ------
    A list of one row (dict) per horizon, with both the cumulative and the pointwise MAE.
    """
    dataset_name, station, repeat_id, series, exog_df, df_complete = task

    forecast_start = generate_forecast_start(
        series=series, seed=SEED, repeat_id=repeat_id,
        max_history_days=TRAIN_WEEKS * 7, max_horizon=MAX_HORIZON
    )
    train_start = forecast_start - pd.Timedelta(weeks=TRAIN_WEEKS)
    train_end = forecast_start
    test_end = train_end + pd.Timedelta(hours=MAX_HORIZON)

    series_train = series[train_start:train_end]
    exog_train = exog_df[train_start:train_end]
    exog_cols = exog_train.columns.tolist()

    print(f"{dataset_name} / {station} / repeat {repeat_id}: {train_start} to {train_end}")

    ols_model = train_neighbour_regression(df_complete[train_start:train_end], station, exog_cols)
    ols_coefs = pd.Series(ols_model.coef_, index=exog_cols)

    arimax_result = train_sarima_model(
        series=series_train, exog_df=exog_train,
        arima_order=ARIMAX_ORDER, seasonal_order=ARIMAX_SEASONAL_ORDER, max_iter=1000
    )
    arimax_coefs = arimax_result.params[exog_cols]

    twostage_model = train_regression_sarima_errors(
        df_complete[train_start:train_end], station, exog_cols,
        RESIDUAL_SARIMA_ORDER, RESIDUAL_SARIMA_SEASONAL_ORDER
    )

    # Same known-neighbour-observations assumption every model in this codebase makes,
    # over the whole MAX_HORIZON window at once
    test_index = df_complete.loc[train_end:test_end].index[1:]
    X_test = df_complete.loc[test_index, exog_cols].interpolate(limit_direction="both")
    y_actual = df_complete.loc[test_index, station]

    ols_beta_series = X_test @ ols_coefs + ols_model.intercept_
    arimax_beta_series = X_test @ arimax_coefs

    arimax_full_result = run_arimax(
        repeat_id, series=series, exog_df=exog_df, model=arimax_result,
        start_date=train_start, end_date=test_end, hours_to_forecast=MAX_HORIZON,
        arima_order=ARIMAX_ORDER, seasonal_order=ARIMAX_SEASONAL_ORDER,
        confidence_score=False, use_LASSO_selection=False, max_iter=1000, plot=False
    )
    arimax_full_series = arimax_full_result[-1]

    twostage_full_result = run_regression_sarima_errors(
        df=df_complete, target_station=station, model=twostage_model, exog_cols=exog_cols,
        start=train_start, train_end=train_end, test_end=test_end, mode="forecast"
    )
    twostage_full_series = twostage_full_result[-1]

    forecasts = {
        "ols_beta": ols_beta_series,
        "arimax_beta": arimax_beta_series,
        "arimax_full": arimax_full_series,
        "twostage_full": twostage_full_series,
    }
    errors = {name: (y_actual - series).abs() for name, series in forecasts.items()}

    rows = []
    for horizon in HORIZONS:
        row = {"dataset": dataset_name, "station": station, "repeat_id": repeat_id, "horizon": horizon}

        for name, error_series in errors.items():
            row[f"{name}_cumulative_mae"] = error_series.iloc[:horizon].mean()
            row[f"{name}_pointwise_mae"] = error_series.iloc[horizon - 1]

        rows.append(row)

    return rows


def run_all_experiments():
    """Run the outage-duration experiment across every dataset, station, and repeat, saving to CSV."""
    output_file = "output/outage_duration_experiment.csv"
    total_tasks = sum(len(info["stations"]) for info in DATASETS.values()) * NUMBER_OF_REPEATS

    writer = None
    rows_written = 0

    with open(output_file, "w", newline="") as f:
        for dataset_name, dataset_info in DATASETS.items():
            df = pd.read_csv(dataset_info["file"])

            for station in dataset_info["stations"]:
                station_data = df[df["station_name"] == station]
                series = pd.Series(
                    station_data["temp_dry_avg_2m"].values,
                    index=pd.to_datetime(station_data["datetime"])
                ).asfreq("1h").dropna()

                other_stations = [s for s in df["station_name"].unique() if s != station]
                exog_data = df[df["station_name"].isin(other_stations)]
                exog_pivot = exog_data.pivot_table(index="datetime", columns="station_name", values="temp_dry_avg_2m")
                exog_pivot.index = pd.to_datetime(exog_pivot.index)
                exog_df = exog_pivot.reindex(series.index)

                series = series.resample("1h").mean().interpolate(limit_direction="both")
                exog_df = exog_df.resample("1h").mean().interpolate(limit_direction="both")

                df_complete = series.to_frame(name=station).join(exog_df)

                # Submit only this station's repeats at a time, so at most MAX_WORKERS copies
                # of its DataFrames are in flight instead of every station's at once
                tasks = [
                    (dataset_name, station, repeat_id, series, exog_df, df_complete)
                    for repeat_id in range(NUMBER_OF_REPEATS)
                ]

                # Sequential (max_workers=1): this machine only has a few GB of RAM actually free
                # at the moment, and running one SARIMAX fit at a time keeps peak memory minimal
                # even though it is slower than parallel workers
                with ProcessPoolExecutor(max_workers=1, max_tasks_per_child=1) as executor:
                    futures = [executor.submit(run_single_experiment, task) for task in tasks]

                    for future in as_completed(futures):
                        for row in future.result():
                            if writer is None:
                                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                                writer.writeheader()
                            writer.writerow(row)
                            f.flush()
                        rows_written += 1
                        print(f"[{rows_written}/{total_tasks}] station/repeat done")


if __name__ == "__main__":
    run_all_experiments()
