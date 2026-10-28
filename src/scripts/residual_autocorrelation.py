import argparse
import pandas as pd
from statsmodels.tsa.stattools import acf
from src.utils.datasets import DATASETS, load_station_data
from src.utils.generate_forecast_start import generate_forecast_start
from src.utils.experiment_utils import write_results_csv
from src.models.regsarima.train import stage_one_residuals, fit_residual_sarima
from src.scripts.residual_sarima_order import SEED, HORIZONS, MAX_HISTORY_DAYS

# Number of training periods to evaluate per station
NUMBER_OF_REPEATS = 10

# Order of the residual SARIMA, the one the two-stage model forecasts with
ARIMA_ORDER = (3, 0, 0)
SEASONAL_ORDER = (1, 0, 0, 24)


def run_autocorrelation_check(training_weeks, repeats, output_file):
    """
    Compute the lag-1 autocorrelation of the stage-one residuals and of the stage-two residuals (the SARIMA
    innovations) over all datasets, stations and training periods, and save the per-window results to a CSV file.

    Input
    -----
    training_weeks: Number of weeks of training data to compute the residuals on
    repeats: Number of training periods to evaluate per station
    output_file: Path of the CSV file to write the per-window results to
    """
    rows = []

    for dataset_name, dataset_info in DATASETS.items():
        df = pd.read_csv(dataset_info["file"])

        for station in dataset_info["stations"]:
            df_complete = load_station_data(df, station)
            series = df_complete[station]
            exog_cols = list(df_complete.columns[1:])

            for repeat_id in range(repeats):
                forecast_start = generate_forecast_start(
                    series=series,
                    seed=SEED,
                    repeat_id=repeat_id,
                    max_history_days=max(MAX_HISTORY_DAYS, training_weeks * 7),
                    max_horizon=max(HORIZONS)
                )

                train_start = forecast_start - pd.Timedelta(days=training_weeks * 7)

                _, residuals = stage_one_residuals(df_complete[train_start:forecast_start], station, exog_cols)
                innovations = fit_residual_sarima(residuals, ARIMA_ORDER, SEASONAL_ORDER).resid

                rows.append([dataset_name, station, repeat_id, train_start, forecast_start,
                             acf(residuals, nlags=1, fft=True)[1], acf(innovations, nlags=1, fft=True)[1]])

    write_results_csv(output_file, ["dataset", "station", "repeat_id", "train_start", "train_end",
                                    "stage_one_acf_lag_1", "stage_two_acf_lag_1"], rows)

    results = pd.DataFrame(rows, columns=["dataset", "station", "repeat_id", "train_start", "train_end",
                                          "stage_one", "stage_two"])
    print(f"Lag-1 autocorrelation over {len(results)} windows: stage one mean {results['stage_one'].mean():.2f}, "
          f"stage two largest |value| {results['stage_two'].abs().max():.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lag-1 autocorrelation of the stage-one regression residuals and of "
                                                 "the stage-two SARIMA innovations, over datasets, stations and "
                                                 "training periods.")
    parser.add_argument("--training_weeks", type=int, default=8,
                        help="Number of weeks to use for training (default: 8, as in the comparison experiment)")
    parser.add_argument("--repeats", type=int, default=NUMBER_OF_REPEATS,
                        help=f"Number of training periods per station (default: {NUMBER_OF_REPEATS})")
    args = parser.parse_args()

    run_autocorrelation_check(
        training_weeks=args.training_weeks,
        repeats=args.repeats,
        output_file=f"output/residual_autocorrelation/results_{args.training_weeks}_weeks.csv"
    )
