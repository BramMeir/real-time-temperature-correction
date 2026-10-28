import argparse
import os
import numpy as np
import pandas as pd
from src.models.regsarima.train import RESIDUAL_ORDER, RESIDUAL_SEASONAL_ORDER
from src.utils.datasets import DATASETS, load_station_data
from src.utils.experiment_utils import get_worker_df_complete, run_tasks_for_station
from src.models.regsarima.confidence_score import two_stage_forecast_with_confidence_score

OUTPUT_DIR = "output/confidence_analysis_regression_sarima_errors"

# Every period holds WEEKS of training data, the last CALIBRATION_DAYS of which are held out for the score, followed
# by a reconstruction of HOURS_TO_FORECAST
WEEKS = 8
CALIBRATION_DAYS = 14
HOURS_TO_FORECAST = 168

NUMBER_OF_REPEATS = 20
SEED = 47


def run_period(task):
    """
    Score one period (station, start, end) on the station's data of the worker process, returning a one-row list
    with its MAE, MSE, reliability score and interval width.
    """
    station, start, end = task
    df_complete = get_worker_df_complete()
    result = two_stage_forecast_with_confidence_score(
        df_complete[(df_complete.index >= start) & (df_complete.index <= end)], station,
        exog_cols=list(df_complete.columns[1:]), hours_to_forecast=HOURS_TO_FORECAST, arima_order=RESIDUAL_ORDER,
        seasonal_order=RESIDUAL_SEASONAL_ORDER, max_iter=1000, calibration_days=CALIBRATION_DAYS
    )
    return [(result["mae"], result["mse"], result["confidence_score"], result["interval_width"])]


def run_station(df_complete, station, max_workers):
    """
    Score NUMBER_OF_REPEATS random periods of one station and average the realised error and the reliability
    indicators over them.

    Input
    -----
    df_complete: DataFrame with the target station and all neighbouring stations as columns
    station: Name of the target station
    max_workers: Number of worker processes

    Output
    ------
    Dictionary with the mean MAE, MSE, reliability score and interval width of the station
    """
    period = pd.Timedelta(weeks=WEEKS) + pd.Timedelta(hours=HOURS_TO_FORECAST)
    possible_starts = df_complete.index[df_complete.index <= df_complete.index.max() - period]

    np.random.seed(SEED)
    starts = np.random.choice(possible_starts, size=NUMBER_OF_REPEATS, replace=False)

    tasks = [(station, start, start + period) for start in starts]
    results = np.array(run_tasks_for_station(df_complete, tasks, run_period, (), max_workers=max_workers))

    mae, mse, confidence, interval_size = results.mean(axis=0)

    return {"target_station": station, "mae_mean": mae, "mse_mean": mse, "confidence_mean": confidence,
            "interval_size_mean": interval_size}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reliability score and interval width of RegSARIMA against its "
                                                 "realised error, for every station of the synthetic network.")
    parser.add_argument("--max_workers", type=int, default=10, help="Number of worker processes (default: 10)")
    args = parser.parse_args()

    df = pd.read_csv(DATASETS["SYNTHETIC"]["file"])
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for station_index, station in enumerate(sorted(df["station_name"].unique())):
        result = run_station(load_station_data(df, station), station, args.max_workers)
        pd.DataFrame([result]).to_csv(f"{OUTPUT_DIR}/confidence_station_{station_index}.csv", index=False)
        print(f"{station}: MAE {result['mae_mean']:.3f}, reliability score {result['confidence_mean']:.3f}")
