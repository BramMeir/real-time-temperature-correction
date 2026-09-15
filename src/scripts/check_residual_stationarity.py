"""
Script to check whether the two-stage model's assumption that the stage-one regression
residuals are already stationary and mean zero actually holds. The residual SARIMA grid
search (determine_residual_sarima_order.py) restricts its candidate orders to d=D=0 on
that assumption; this script verifies it with the Augmented Dickey-Fuller test, pooled
over the same datasets, stations and training periods.

python -m src.scripts.check_residual_stationarity
"""
import os
import csv
import argparse
import statistics
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from src.utils.generate_forecast_start import generate_forecast_start
from src.models.regression_sarima_errors.train import stage_one_residuals
from src.scripts.determine_residual_sarima_order import DATASETS, load_station_series, SEED, HORIZONS, MAX_HISTORY_DAYS

# Number of training periods to evaluate per station
NUMBER_OF_REPEATS = 10


def test_stationarity(residuals):
    """
    Run the Augmented Dickey-Fuller test on a residual series.

    Input
    -----
    residuals: Pandas Series with the stage-one regression residuals

    Output
    ------
    Returns a dict with the residual mean, and the ADF statistic/p-value.
    """
    adf_statistic, adf_pvalue, *_ = adfuller(residuals, autolag="AIC")

    return {
        "residual_mean": residuals.mean(),
        "adf_statistic": adf_statistic,
        "adf_pvalue": adf_pvalue,
    }


def summarise(rows):
    """
    Summarise the stationarity tests over all evaluated windows.

    Input
    -----
    rows: List of per-window result dictionaries, as produced by run_stationarity_check

    Output
    ------
    Prints the share of windows where the ADF test rejects a unit root (p < 0.05), and
    the mean absolute residual mean, both overall and grouped by dataset.
    """
    def report(label, group):
        n = len(group)
        adf_stationary = sum(1 for r in group if r["adf_pvalue"] < 0.05)
        mean_abs_mean = statistics.fmean(abs(r["residual_mean"]) for r in group)
        print(f"{label:<12}{n:>8}{adf_stationary:>8}/{n:<8}{mean_abs_mean:>16.4f}")

    print(f"\n{'':<12}{'windows':>8}{'ADF ok':>16}{'mean |resid mean|':>18}")
    for dataset_name in sorted(set(r["dataset"] for r in rows)):
        report(dataset_name, [r for r in rows if r["dataset"] == dataset_name])
    report("overall", rows)


def run_stationarity_check(training_weeks, repeats, output_file, datasets=None):
    """
    Run the ADF and KPSS tests on the stage-one residuals over all datasets, stations and
    training periods, and save the per-window results to a CSV file.

    Input
    -----
    training_weeks: Number of weeks of training data to compute the residuals on
    repeats: Number of training periods to evaluate per station
    output_file: Path of the CSV file to write the per-window results to
    datasets: List of dataset names to evaluate (default is None, which uses all of them)
    """
    rows = []

    for dataset_name in (datasets or DATASETS):
        dataset_info = DATASETS[dataset_name]
        df = pd.read_csv(dataset_info["file"])

        for station in dataset_info["stations"]:
            series, df_complete, exog_cols = load_station_series(df, station)

            for repeat_id in range(repeats):
                forecast_start = generate_forecast_start(
                    series=series,
                    seed=SEED,
                    repeat_id=repeat_id,
                    max_history_days=max(MAX_HISTORY_DAYS, training_weeks * 7),
                    max_horizon=max(HORIZONS)
                )

                train_start = forecast_start - pd.Timedelta(days=training_weeks * 7)
                train_end = forecast_start

                _, residuals = stage_one_residuals(
                    df_complete[train_start:train_end], station, exog_cols
                )

                result = test_stationarity(residuals)

                print(f"{dataset_name} {station} repeat {repeat_id} "
                      f"({train_start.date()} to {train_end.date()}): "
                      f"ADF p={result['adf_pvalue']:.4f}, mean={result['residual_mean']:.4f}")

                rows.append({
                    "dataset": dataset_name,
                    "station": station,
                    "repeat_id": repeat_id,
                    "train_start": train_start,
                    "train_end": train_end,
                    **result
                })

    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summarise(rows)
    print(f"\nPer-window results written to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check whether the two-stage model's stage-one regression "
                                                 "residuals are stationary and mean zero, as assumed by the "
                                                 "residual SARIMA grid search, using the Augmented Dickey-Fuller "
                                                 "test pooled over datasets, stations and training periods.")
    parser.add_argument("--training_weeks", type=int, default=8,
                        help="Number of weeks to use for training (default: 8, as in the comparison experiment)")
    parser.add_argument("--repeats", type=int, default=NUMBER_OF_REPEATS,
                        help=f"Number of training periods per station (default: {NUMBER_OF_REPEATS})")
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS),
                        help="Datasets to evaluate (default: all of them)")
    args = parser.parse_args()

    os.makedirs("output/residual_stationarity", exist_ok=True)

    run_stationarity_check(
        training_weeks=args.training_weeks,
        repeats=args.repeats,
        output_file=f"output/residual_stationarity/results_{args.training_weeks}_weeks.csv",
        datasets=args.datasets
    )
