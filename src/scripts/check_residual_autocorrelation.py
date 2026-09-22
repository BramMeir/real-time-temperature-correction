"""
Script to check whether the stage-one regression residuals still carry autocorrelation, which is what
motivates fitting a SARIMA on them instead of stopping at the neighbour regression. Runs over the same
datasets, stations and training windows as the other residual diagnostics, and repeats every test on
the stage-two residuals (the SARIMA innovations), which should be close to white noise.

The Ljung-Box test is the one to read: it tests jointly whether the autocorrelations up to a lag are
all zero. The Durbin-Watson statistic is reported next to it because it is the familiar regression
diagnostic, but it only looks at lag 1 (2 means no lag-1 autocorrelation, below 2 positive), so it
says nothing about the daily cycle the seasonal part is there for.

python -m src.scripts.check_residual_autocorrelation
"""
import os
import csv
import argparse
import statistics
import pandas as pd
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf
from src.utils.generate_forecast_start import generate_forecast_start
from src.models.regression_sarima_errors.train import stage_one_residuals, fit_residual_sarima
from src.scripts.determine_residual_sarima_order import DATASETS, load_station_series, SEED, HORIZONS, MAX_HISTORY_DAYS

# Number of training periods to evaluate per station
NUMBER_OF_REPEATS = 10

# Lags the Ljung-Box test is evaluated at: the first lag, one day and two days
LJUNG_BOX_LAGS = [1, 24, 48]

# Order of the residual SARIMA, the one the two-stage model forecasts with
ARIMA_ORDER = (2, 0, 0)
SEASONAL_ORDER = (1, 0, 1, 24)

# Free ARMA parameters of that SARIMA, which Ljung-Box has to discount on the stage-two residuals
MODEL_DF = ARIMA_ORDER[0] + ARIMA_ORDER[2] + SEASONAL_ORDER[0] + SEASONAL_ORDER[2]

# The repeat whose residual series is written out for the ACF/PACF figure
PLOT_REPEAT_ID = 0


def test_autocorrelation(residuals, model_df=0):
    """
    Test a residual series for autocorrelation.

    Input
    -----
    residuals: Pandas Series with the residuals
    model_df: Number of fitted ARMA parameters to discount in the Ljung-Box degrees of freedom

    Output
    ------
    Returns a dict with the Durbin-Watson statistic, the ACF at lag 1 and 24, and the Ljung-Box
    p-value at every lag in LJUNG_BOX_LAGS that exceeds model_df.
    """
    # A lag that does not exceed the number of fitted parameters leaves the test without degrees of
    # freedom, so the lag-1 test is dropped for the stage-two residuals
    lags = [lag for lag in LJUNG_BOX_LAGS if lag > model_df]

    ljung_box = acorr_ljungbox(residuals, lags=lags, model_df=model_df)
    autocorrelations = acf(residuals, nlags=max(LJUNG_BOX_LAGS), fft=True)

    return {
        "durbin_watson": durbin_watson(residuals),
        "acf_lag_1": autocorrelations[1],
        "acf_lag_24": autocorrelations[24],
        **{f"ljung_box_p_lag_{lag}": ljung_box.loc[lag, "lb_pvalue"] for lag in lags}
    }


def summarise(rows):
    """
    Summarise the autocorrelation tests over all evaluated windows.

    Input
    -----
    rows: List of per-window result dictionaries, as produced by run_autocorrelation_check

    Output
    ------
    Prints, per stage and per dataset, the share of windows where Ljung-Box rejects white noise at
    lag 24 (p < 0.05), and the mean Durbin-Watson statistic and lag-1/lag-24 autocorrelation.
    """
    def report(label, stage, group):
        n = len(group)
        rejected = sum(1 for r in group if r[f"{stage}_ljung_box_p_lag_24"] < 0.05)
        print(f"{label:<12}{n:>8}{rejected:>8}/{n:<8}"
              f"{statistics.fmean(r[f'{stage}_durbin_watson'] for r in group):>10.3f}"
              f"{statistics.fmean(r[f'{stage}_acf_lag_1'] for r in group):>12.3f}"
              f"{statistics.fmean(r[f'{stage}_acf_lag_24'] for r in group):>12.3f}")

    for stage, title in [("stage_one", "Stage-one residuals (regression only)"),
                         ("stage_two", "Stage-two residuals (SARIMA innovations)")]:
        print(f"\n{title}")
        print(f"{'':<12}{'windows':>8}{'LB rejects':>16}{'DW':>10}{'ACF(1)':>12}{'ACF(24)':>12}")
        for dataset_name in sorted(set(r["dataset"] for r in rows)):
            report(dataset_name, stage, [r for r in rows if r["dataset"] == dataset_name])
        report("overall", stage, rows)


def run_autocorrelation_check(training_weeks, repeats, output_dir, datasets=None):
    """
    Test the stage-one and stage-two residuals for autocorrelation over all datasets, stations and
    training periods, and save the per-window results to a CSV file.

    Input
    -----
    training_weeks: Number of weeks of training data to compute the residuals on
    repeats: Number of training periods to evaluate per station
    output_dir: Directory to write the per-window results and the plotted residual series to
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
                residual_model = fit_residual_sarima(residuals, ARIMA_ORDER, SEASONAL_ORDER)
                innovations = residual_model.resid

                stage_one = test_autocorrelation(residuals)
                stage_two = test_autocorrelation(innovations, model_df=MODEL_DF)

                print(f"{dataset_name} {station} repeat {repeat_id} "
                      f"({train_start.date()} to {train_end.date()}): "
                      f"stage one DW={stage_one['durbin_watson']:.3f}, "
                      f"LB(24) p={stage_one['ljung_box_p_lag_24']:.4f} | "
                      f"stage two DW={stage_two['durbin_watson']:.3f}, "
                      f"LB(24) p={stage_two['ljung_box_p_lag_24']:.4f}")

                rows.append({
                    "dataset": dataset_name,
                    "station": station,
                    "repeat_id": repeat_id,
                    "train_start": train_start,
                    "train_end": train_end,
                    **{f"stage_one_{key}": value for key, value in stage_one.items()},
                    **{f"stage_two_{key}": value for key, value in stage_two.items()}
                })

                # Keep one window's residuals per dataset, so the ACF/PACF figure shows exactly the
                # series these tests are run on instead of fitting its own
                if station == dataset_info["stations"][0] and repeat_id == PLOT_REPEAT_ID:
                    pd.DataFrame({"stage_one": residuals, "stage_two": innovations}).to_csv(
                        os.path.join(output_dir, f"residuals_{dataset_name}_{station}.csv"),
                        index_label="datetime"
                    )

    output_file = os.path.join(output_dir, f"results_{training_weeks}_weeks.csv")
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summarise(rows)
    print(f"\nPer-window results written to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check whether the stage-one regression residuals still carry "
                                                 "autocorrelation, and whether the stage-two SARIMA removes it, "
                                                 "using the Ljung-Box test and the Durbin-Watson statistic pooled "
                                                 "over datasets, stations and training periods.")
    parser.add_argument("--training_weeks", type=int, default=8,
                        help="Number of weeks to use for training (default: 8, as in the comparison experiment)")
    parser.add_argument("--repeats", type=int, default=NUMBER_OF_REPEATS,
                        help=f"Number of training periods per station (default: {NUMBER_OF_REPEATS})")
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS),
                        help="Datasets to evaluate (default: all of them)")
    args = parser.parse_args()

    output_dir = "output/residual_autocorrelation"
    os.makedirs(output_dir, exist_ok=True)

    run_autocorrelation_check(
        training_weeks=args.training_weeks,
        repeats=args.repeats,
        output_dir=output_dir,
        datasets=args.datasets
    )
