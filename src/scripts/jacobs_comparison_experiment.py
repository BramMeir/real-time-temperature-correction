"""
Script: jacobs_comparison_experiment.py

Comparison of RegSARIMA with the ERA5 debiasing of Jacobs et al. (2024) on the same simulated gaps in
the TURCLIM network.

Gaps follow their technique evaluation: per station and gap length, 1000 single gaps whose start is
drawn uniformly over the record, 60 days away from both ends. Each gap is filled by
- their techniques, with their selection parameters (60-day learning period, time variation 1 h):
  linear interpolation (LI), original ERA5, and the LR and MB debiasing with the learning period
  placed symmetrically around the gap, plus MB with the two halves of the learning period weighted
  separately (positioning 'separate'). Their algorithm combines LI and the two MB variants by gap
  length, which is left to the analysis, as their code and paper place the switch to 'separate' at
  15 h and 15 days respectively;
- RegSARIMA in real time, exactly as in the model comparison: trained on the eight weeks before the
  gap and forecast from its start;
- RegSARIMA retrospectively, trained on four weeks on either side of the gap with the residual
  inside it estimated by the Kalman smoother (src/models/regression_sarima_errors/retrospective.py);
- the regression of both RegSARIMA variants on its own, to separate the contribution of the
  residual model.

The MSE of every gap and method is written to one row, so that the analysis can compute both the
RMSE averaged over gaps, as in the paper, and their error E, the MSE averaged over gaps.

Run:
    python -m src.scripts.jacobs_comparison_experiment --repetitions 20
    python -m src.scripts.jacobs_comparison_experiment --stations Betel --max_workers 16
"""
import argparse
import csv
import os
import zlib
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from src.evaluation.comparison_amber.evaluation_gf_techniques import makegap
from src.evaluation.comparison_amber.gf_techniques import (
    GF_linint,
    GF_fillmodel,
    GF_debmodelReg,
    GF_debmodelMeanbias,
)
from src.evaluation.comparison_amber.turku_era5 import load_turku_with_era5
from src.models.regression_sarima_errors.execute_forecast import run_single_forecast
from src.models.regression_sarima_errors.retrospective import fill_gap_retrospective
from src.models.regression_sarima_errors.train import train_regression_sarima_errors

OUTPUT_DIR = "output/jacobs_comparison"

STATIONS = ["Betel", "Kurala", "Puutori", "Tuorla", "Virastotalo", "Ylijoki"]

SEED = 42

# Gap lengths in hours: the outage durations of the model comparison plus the short gaps of
# Jacobs et al., around their switch from LI to ERA5 debiasing at 5 h
GAP_LENGTHS = [1, 3, 4, 5, 7, 12, 24, 48, 96, 168, 336, 504, 720]

REPETITIONS = 1000

# Gaps stay this far from both ends of the record, as in their evaluation (their seasonal span). It
# also leaves room for the eight weeks of training before the gap and the four weeks after it.
MARGIN_DAYS = 60

# Their selection parameters
SEASONAL_SPAN = 60
TIME_VARIATION = 1

# As in the model comparison
TRAINING_DAYS = 56
ARIMA_ORDER = (3, 0, 0)
SEASONAL_ORDER = (1, 0, 0, 24)

# Gaps per task, small enough to spread a station over many workers
CHUNK_SIZE = 25

# Set by _init_worker, once per worker process
_df = None


def _init_worker():
    global _df
    _df = load_turku_with_era5()


def draw_gap_starts(n_hours, station, gap_length, repetitions):
    """Positions of the first missing hour of every gap, uniform within the margins."""
    rng = np.random.default_rng(SEED + zlib.crc32(station.encode()) + gap_length)
    margin = MARGIN_DAYS * 24
    return rng.integers(margin, n_hours - margin - gap_length, size=repetitions)


def fill_one_gap(df, station, gap_start, gap_length):
    """Fill one gap with every method and return a dict of method name to the series over the gap."""
    gap_end = gap_start + pd.Timedelta(hours=gap_length - 1)
    era5 = station + "_ERA5"
    exog_cols = [s for s in STATIONS if s != station]

    # Jacobs et al., on a slice that holds their learning period on both sides of the gap
    span = pd.Timedelta(days=SEASONAL_SPAN)
    gapped = makegap(
        df.loc[gap_start - span:gap_end + span, [station, era5]], station, gap_length,
        gap_start.strftime("%Y-%m-%d %H:%M:%S")
    )
    name_gapped = station + "_gap"
    fills = {
        "LI": GF_linint(gapped, name_gapped)["P_linint"],
        "ERA5": GF_fillmodel(gapped, name_gapped, era5)["P_fillmodel"],
        "LR": GF_debmodelReg(gapped, name_gapped, era5, SEASONAL_SPAN, TIME_VARIATION, "both", 1, False)["P_debmodelReg"],
        "MB_both": GF_debmodelMeanbias(gapped, name_gapped, era5, SEASONAL_SPAN, TIME_VARIATION, "both", False)["P_debmodelMeanbias"],
        "MB_separate": GF_debmodelMeanbias(gapped, name_gapped, era5, SEASONAL_SPAN, TIME_VARIATION, "separate", False)["P_debmodelMeanbias"],
    }

    # RegSARIMA in real time, trained and forecast as in models_comparison_experiment.py
    train_end = gap_start - pd.Timedelta(hours=1)
    train_start = train_end - pd.Timedelta(days=TRAINING_DAYS)
    model = train_regression_sarima_errors(
        df.loc[train_start:train_end], station, exog_cols, ARIMA_ORDER, SEASONAL_ORDER, max_iter=1000
    )
    fills["RegSARIMA_realtime"] = run_single_forecast(
        df, station, model=model, exog_cols=exog_cols, start=train_start, train_end=train_end, test_end=gap_end
    )[3]
    fills["Regression_realtime"] = pd.Series(
        model[0].predict(df.loc[gap_start:gap_end, exog_cols]), index=df.loc[gap_start:gap_end].index
    )

    # RegSARIMA retrospectively
    fills["RegSARIMA_retrospective"], fills["Regression_retrospective"] = fill_gap_retrospective(
        df, station, exog_cols, gap_start, gap_end, arima_order=ARIMA_ORDER, seasonal_order=SEASONAL_ORDER
    )

    return {method: fill.loc[gap_start:gap_end] for method, fill in fills.items()}


def run_chunk(station, gap_length, gap_ids, positions):
    """Fill a chunk of gaps and return one row per gap and method with the MSE over the gap."""
    rows = []
    for gap_id, position in zip(gap_ids, positions):
        gap_start = _df.index[position]
        gap_end = gap_start + pd.Timedelta(hours=gap_length - 1)
        truth = _df.loc[gap_start:gap_end, station].values

        for method, fill in fill_one_gap(_df, station, gap_start, gap_length).items():
            rows.append([station, gap_length, gap_id, gap_start, method, float(np.mean((fill.values - truth) ** 2))])
    return rows


def run_all(stations, gap_lengths, repetitions, max_workers, output_path):
    """Run every station and gap length in a worker pool, appending the rows to output_path as chunks finish."""
    n_hours = len(load_turku_with_era5())

    tasks = []
    for station in stations:
        for gap_length in gap_lengths:
            positions = draw_gap_starts(n_hours, station, gap_length, repetitions)
            for first in range(0, repetitions, CHUNK_SIZE):
                ids = list(range(first, min(first + CHUNK_SIZE, repetitions)))
                tasks.append((station, gap_length, ids, positions[ids]))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["station", "gap_length", "gap_id", "gap_start", "method", "mse"])

        with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_worker) as executor:
            futures = [executor.submit(run_chunk, *task) for task in tasks]
            for done, future in enumerate(as_completed(futures), start=1):
                writer.writerows(future.result())
                f.flush()
                print(f"Finished {done}/{len(tasks)} chunks", flush=True)

    print(f"Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare RegSARIMA with the ERA5 debiasing of Jacobs et al. (2024).")
    parser.add_argument("--stations", nargs="+", default=STATIONS, help="TURCLIM target stations")
    parser.add_argument("--gap_lengths", type=int, nargs="+", default=GAP_LENGTHS, help="Gap lengths in hours")
    parser.add_argument("--repetitions", type=int, default=REPETITIONS, help="Number of random gaps per station and gap length")
    parser.add_argument("--max_workers", type=int, default=None, help="Parallel worker processes (default lets the pool decide)")
    args = parser.parse_args()

    label = args.stations[0] if len(args.stations) == 1 else "all"
    output_path = f"{OUTPUT_DIR}/jacobs_comparison_{label}_rep{args.repetitions}.csv"
    run_all(args.stations, args.gap_lengths, args.repetitions, args.max_workers, output_path)
