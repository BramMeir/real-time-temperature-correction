import argparse
import pandas as pd
from src.utils.datasets import DATASETS, load_station_data
from src.utils.generate_forecast_start import generate_forecast_start
from src.utils.experiment_utils import write_results_csv
from src.models.regsarima.train import stage_one_residuals
from src.models.regsarima.grid_search import build_candidate_grid, residual_sarima_grid_search

# Seed that is used to define the training periods (for reproducibility)
SEED = 42

# Forecasting horizons in hours (4h, 12h, 1D, 2D, 4D, 7D, 14D, 21D, 30D)
HORIZONS = [4, 12, 24, 48, 96, 168, 336, 504, 720]

# Number of training periods to evaluate per station, fewer than in the comparison experiment
# because every window is fitted once per candidate order
NUMBER_OF_REPEATS = 3

# History that generate_forecast_start has to keep available, matching the comparison experiment so
# the order is selected on the same training windows the model is evaluated on
MAX_HISTORY_DAYS = 8 * 7


def run_grid_search(training_weeks, repeats, output_file):
    """
    Run the grid search on the stage-one residuals over all datasets, stations and training periods, save the AIC
    of every candidate per window to a CSV file and print the candidates ranked by their mean AIC rank.

    Input
    -----
    training_weeks: Number of weeks of training data to fit the residual models on
    repeats: Number of training periods to evaluate per station
    output_file: Path of the CSV file to write the per-window results to
    """
    candidates = build_candidate_grid()
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

                # Residuals from the training window only, so the order is never selected on data the model is
                # evaluated on
                _, residuals = stage_one_residuals(df_complete[train_start:forecast_start], station, exog_cols)

                print(f"{dataset_name} {station} repeat {repeat_id} ({train_start.date()} to {forecast_start.date()})")

                _, _, window_rows = residual_sarima_grid_search(residuals, candidates=candidates, verbose=False)

                rows += [[dataset_name, station, repeat_id, train_start, forecast_start,
                          row["order"], row["seasonal_order"], row["aic"]] for row in window_rows]

    columns = ["dataset", "station", "repeat_id", "train_start", "train_end", "order", "seasonal_order", "aic"]
    write_results_csv(output_file, columns, rows)

    # Every window gets an equal vote: the candidates are ranked by AIC within each window, and the order is the one
    # with the best mean rank. The mean AIC itself would let the few windows that separate the candidates most decide
    results = pd.DataFrame(rows, columns=columns).astype({"order": str, "seasonal_order": str})
    results["rank"] = results.groupby(["dataset", "station", "repeat_id"])["aic"].rank()
    ranking = results.groupby(["order", "seasonal_order"])["rank"].mean().sort_values()
    print(f"\nCandidates ranked by their mean AIC rank over {len(rows) // len(candidates)} windows:")
    print(ranking.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select the orders of the stage-two SARIMA of the two-stage model "
                                                 "by a grid search on the stage-one residuals, pooled over datasets, "
                                                 "stations and training periods.")
    parser.add_argument("--training_weeks", type=int, default=8,
                        help="Number of weeks to use for training (default: 8, as in the comparison experiment)")
    parser.add_argument("--repeats", type=int, default=NUMBER_OF_REPEATS,
                        help=f"Number of training periods per station (default: {NUMBER_OF_REPEATS})")
    args = parser.parse_args()

    run_grid_search(
        training_weeks=args.training_weeks,
        repeats=args.repeats,
        output_file=f"output/residual_sarima_order_grid_search/results_{args.training_weeks}_weeks.csv"
    )
