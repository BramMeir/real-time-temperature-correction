"""
Script: heatwave_experiment.py

How well the two-stage model reconstructs a failing station during a heat wave, and whether it underestimates the heat.

The heat waves are the relative P90 events of detect_heatwaves.py, detected on the network median. Every station of
the network is reconstructed around every event, with the outage starting LEADS days before the first heat wave day:
a positive lead is an outage that is already running when the heat arrives, a negative one an outage that starts while
the heat wave is under way. The model is trained on the TRAINING_WEEKS before the onset, as in the main comparison, and
scored on the heat wave days plus RECOVERY_DAYS after them.

Each event also gets up to NUMBER_OF_CONTROLS control windows: ordinary days within CONTROL_SEARCH_DAYS of the event in
the same year, of the same length, at least CONTROL_MARGIN_DAYS away from any heat wave and not overlapping each other
in their scored days. They are reconstructed with the
same leads, so a heat wave window and its controls differ only in the weather, and the difference between their errors
is the effect of the heat itself. The controls are drawn once per event and shared by all stations of the network.

Writes the hourly reconstructions and a table of the windows to output/heatwave_experiment/. The metrics are computed
from these afterwards.

Run:
    python -m src.scripts.heatwave_experiment
"""
import argparse
import zlib
import numpy as np
import pandas as pd
from src.evaluation.experiment_utils import get_worker_df_complete, run_tasks_for_station, write_results_csv
from src.models.regression_sarima_errors.train import train_regression_sarima_errors
from src.models.regression_sarima_errors.execute_forecast import run_single_forecast
from src.scripts.detect_heatwaves import DATASETS, load_hourly, daily_extremes, relative_events

# The RMI record holds a single 3-day P90 event, too little to draw conclusions from
EXPERIMENT_DATASETS = ["TURKU", "SYNTHETIC"]

PERCENTILE = 90

# Days the outage has been running when the heat wave starts; -2 is an outage starting on its third day
LEADS = [-2, 0, 1, 3, 7, 14]

TRAINING_WEEKS = 8

# Days after the heat wave that are scored too, to see how the error decays once the heat breaks
RECOVERY_DAYS = 2

NUMBER_OF_CONTROLS = 5
CONTROL_SEARCH_DAYS = 42
CONTROL_MARGIN_DAYS = 3

SEED = 42

# Residual SARIMA orders as used elsewhere for this model
RESIDUAL_ORDER, RESIDUAL_SEASONAL_ORDER = (3, 0, 0), (1, 0, 0, 24)

OUTPUT_DIR = "output/heatwave_experiment"


def draw_controls(event_start, n_days, event_days, first_hour, last_day, rng):
    """
    Draw the start days of the control windows of one heat wave.

    Input
    -----
    event_start: First day of the heat wave
    n_days: Length of the heat wave in days
    event_days: DatetimeIndex with every heat wave day of the dataset
    first_hour: First hour of the record, which the training window of the longest lead must not precede
    last_day: Last complete day of the record, which the scored days must not pass
    rng: Random generator

    Output
    ------
    starts: Sorted list of at most NUMBER_OF_CONTROLS start days
    """
    candidates = []

    for offset in range(-CONTROL_SEARCH_DAYS, CONTROL_SEARCH_DAYS + 1):
        start = event_start + pd.Timedelta(days=offset)
        last_scored = start + pd.Timedelta(days=n_days - 1 + RECOVERY_DAYS)
        earliest_train = start - pd.Timedelta(days=max(LEADS), weeks=TRAINING_WEEKS)

        guard = pd.date_range(
            start - pd.Timedelta(days=CONTROL_MARGIN_DAYS),
            last_scored + pd.Timedelta(days=CONTROL_MARGIN_DAYS)
        )

        if earliest_train >= first_hour and last_scored <= last_day and not guard.isin(event_days).any():
            candidates.append(start)

    # Take candidates in random order, skipping those whose scored days overlap a control already chosen, so the
    # controls are distinct stretches of weather rather than near-copies of each other. Around the longest events,
    # in summers with several heat waves, fewer than NUMBER_OF_CONTROLS fit (Turku 2014-07-21 and 2018-07-23), and
    # those events are compared with the controls that do
    span = pd.Timedelta(days=n_days - 1 + RECOVERY_DAYS)
    chosen = []

    for i in rng.permutation(len(candidates)):
        if all(abs(candidates[i] - start) > span for start in chosen):
            chosen.append(candidates[i])

    return sorted(chosen[:NUMBER_OF_CONTROLS])


def build_windows(dataset, hourly):
    """
    The heat wave windows of one dataset, each followed by its controls.

    Input
    -----
    dataset: Dataset name
    hourly: DataFrame indexed by hour, one column per station

    Output
    ------
    windows: List of (event_id, control, first day, last day) tuples; control is 0 for the heat wave itself and
      1..NUMBER_OF_CONTROLS for its controls
    """
    tmax, _ = daily_extremes(hourly)
    events, _ = relative_events(tmax.median(axis=1), PERCENTILE)

    event_days = pd.DatetimeIndex(np.concatenate([pd.date_range(start, end) for start, end in events]))

    windows = []

    for start, end in events:
        event_id = f"{dataset}_{start.date()}"
        n_days = (end - start).days + 1

        # Seeded per event, so adding an event to the dataset leaves the controls of the others unchanged
        rng = np.random.default_rng([SEED, zlib.crc32(event_id.encode())])
        starts = [start] + draw_controls(start, n_days, event_days, hourly.index[0], tmax.index[-1], rng)

        for control, window_start in enumerate(starts):
            windows.append((event_id, control, window_start, window_start + pd.Timedelta(days=n_days - 1)))

    return windows


def run_window(task):
    """
    Reconstruct one station over one window at one lead.

    Input
    -----
    task: Tuple of (dataset, station, event_id, control, window_start, window_end, lead)

    Output
    ------
    result: One-element list with (window_row, hourly), where hourly is a DataFrame with the scored hours
    """
    dataset, station, event_id, control, window_start, window_end, lead = task
    df = get_worker_df_complete()
    exog_cols = [col for col in df.columns if col != station]

    onset = window_start - pd.Timedelta(days=lead)
    train_start = onset - pd.Timedelta(weeks=TRAINING_WEEKS)
    test_end = window_end + pd.Timedelta(days=RECOVERY_DAYS, hours=23)

    model = train_regression_sarima_errors(
        df[train_start:onset], station, exog_cols, RESIDUAL_ORDER, RESIDUAL_SEASONAL_ORDER, max_iter=1000
    )

    *_, predictions = run_single_forecast(
        df=df, target_station=station, model=model, exog_cols=exog_cols,
        start=train_start, train_end=onset, test_end=test_end
    )

    # From the first heat wave hour, or from the first hour after the onset if the outage starts during the heat wave
    scored = predictions[window_start:]

    hourly = pd.DataFrame({
        "Dataset": dataset,
        "Station": station,
        "Event_id": event_id,
        "Control": control,
        "Lead_days": lead,
        "Datetime": scored.index,
        "Observed": df.loc[scored.index, station].values,
        "Predicted": scored.values
    })

    regression, _ = model

    window_row = [
        dataset, station, event_id, control, lead, window_start.date(), window_end.date(), train_start, onset,
        regression.coef_.sum()
    ]

    return [(window_row, hourly)]


def main(datasets, max_workers):
    window_rows, hourly_frames = [], []

    for dataset in datasets:
        hourly = load_hourly(DATASETS[dataset])
        windows = build_windows(dataset, hourly)

        tasks = [
            (dataset, station, event_id, control, window_start, window_end, lead)
            for station in hourly.columns
            for event_id, control, window_start, window_end in windows
            for lead in LEADS
        ]

        print(f"{dataset}: {len(windows)} windows x {hourly.shape[1]} stations x {len(LEADS)} leads = {len(tasks)}")

        for window_row, frame in run_tasks_for_station(hourly, tasks, run_window, (), max_workers):
            window_rows.append(window_row)
            hourly_frames.append(frame)

    write_results_csv(
        f"{OUTPUT_DIR}/windows_{TRAINING_WEEKS}_weeks.csv",
        [
            "Dataset", "Station", "Event_id", "Control", "Lead_days", "Window_start", "Window_end", "Train_start",
            "Onset", "Sum_beta"
        ],
        window_rows
    )

    pd.concat(hourly_frames).to_csv(
        f"{OUTPUT_DIR}/RegressionSARIMAErrors_{TRAINING_WEEKS}_weeks.csv.gz", index=False, compression="gzip"
    )

    print(f"Wrote {len(window_rows)} windows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reconstruct every station around the heat waves and their control windows with the two-stage "
                    "model."
    )
    parser.add_argument("--datasets", nargs="+", default=EXPERIMENT_DATASETS, choices=EXPERIMENT_DATASETS,
                        help=f"Datasets to run on (default: {EXPERIMENT_DATASETS}).")
    parser.add_argument("--max_workers", type=int, default=None,
                        help="Number of parallel worker processes (default: let the pool decide).")
    args = parser.parse_args()

    main(args.datasets, args.max_workers)
