"""
Main script to determine the SARIMA order for stage two of the two-stage regression model, by running
a grid search on the stage-one residuals rather than on the raw temperature series. Runs over all
datasets, stations and training periods, and saves the results in a CSV file for further analysis.

python -m src.scripts.determine_residual_sarima_order
"""
import os
import csv
import argparse
import statistics
import pandas as pd
from src.utils.generate_forecast_start import generate_forecast_start
from src.models.regression_sarima_errors.train import stage_one_residuals
from src.models.regression_sarima_errors.grid_search import build_candidate_grid, residual_sarima_grid_search

# Metadata about the used datasets and stations to evaluate on
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


def load_station_series(df, station):
    """
    Build the hourly target series and neighbour DataFrame for one station, using the same
    preprocessing as the comparison experiment.

    Input
    -----
    df: DataFrame with the preprocessed dataset in long format
    station: Name of the target station

    Output
    ------
    series: Target time series with an hourly datetime index
    df_complete: DataFrame with the target station and all neighbouring stations as columns
    exog_cols: List of the neighbouring station column names
    """
    # Select target station
    station_data = df[df["station_name"] == station]

    # Create target time series with a datetime index
    series = pd.Series(
        station_data["temp_dry_avg_2m"].values,
        index=pd.to_datetime(station_data["datetime"])
    ).asfreq("1h").dropna()

    # Create exogenous DataFrame
    other_stations = [s for s in df["station_name"].unique() if s != station]

    exog_data = df[df["station_name"].isin(other_stations)]

    # Pivot to get each station as a separate column
    exog_pivot = exog_data.pivot_table(
        index="datetime", columns="station_name", values="temp_dry_avg_2m"
    )

    # Create datetime index
    exog_pivot.index = pd.to_datetime(exog_pivot.index)

    # Remove the missing timestamps and align with the main series
    exog_df = exog_pivot.reindex(series.index)

    # Resample data by taking the mean
    series = series.resample("1h").mean().interpolate(limit_direction="both")
    exog_df = exog_df.resample("1h").mean().interpolate(limit_direction="both")

    df_complete = series.to_frame(name=station).join(exog_df)

    return series, df_complete, exog_df.columns.tolist()


def summarise(rows, criterion):
    """
    Rank the candidate orders over all evaluated windows.

    Reports the mean criterion, which weighs a window by how far apart it drives the candidates, the
    mean rank, which gives every window an equal vote, and the number of windows won, which shows
    whether the criterion separates the candidates at all.

    Input
    -----
    rows: List of result dictionaries, one per (window, candidate)
    criterion: Information criterion that was minimised, either "aic" or "bic"

    Output
    ------
    Returns a list of summary dictionaries, sorted by the mean of the criterion.
    """
    # Group the results per window, so the candidates can be ranked within each window
    windows = {}
    for row in rows:
        windows.setdefault((row["dataset"], row["station"], row["repeat_id"]), []).append(row)

    ranks = {}
    wins = {}
    for window_rows in windows.values():
        for rank, row in enumerate(sorted(window_rows, key=lambda r: r[criterion]), 1):
            candidate = (row["order"], row["seasonal_order"])
            ranks.setdefault(candidate, []).append(rank)
            if rank == 1:
                wins[candidate] = wins.get(candidate, 0) + 1

    values = {}
    for row in rows:
        values.setdefault((row["order"], row["seasonal_order"]), []).append(row[criterion])

    summary = [
        {
            "order": candidate[0],
            "seasonal_order": candidate[1],
            "mean_criterion": statistics.fmean(values[candidate]),
            "mean_rank": statistics.fmean(ranks[candidate]),
            "windows_won": wins.get(candidate, 0)
        }
        for candidate in values
    ]

    return sorted(summary, key=lambda entry: entry["mean_criterion"])


def run_grid_search(training_weeks, repeats, criterion, seasonal_period, output_file, datasets=None):
    """
    Run the grid search on the stage-one residuals over all datasets, stations and training periods,
    and save the results to a CSV file.

    Input
    -----
    training_weeks: Number of weeks of training data to fit the residual models on
    repeats: Number of training periods to evaluate per station
    criterion: Information criterion to minimise, either "aic" or "bic"
    seasonal_period: Seasonal period of the candidate orders
    output_file: Path of the CSV file to write the per-window results to
    datasets: List of dataset names to evaluate (default is None, which uses all of them)
    """
    candidates = build_candidate_grid(S=seasonal_period)

    print(f"Evaluating {len(candidates)} candidate orders on the stage-one residuals, "
          f"minimising {criterion.upper()}\n")

    rows = []

    # Loop through all combinations of dataset, station and training period
    for dataset_name in (datasets or DATASETS):
        dataset_info = DATASETS[dataset_name]
        # Read the preprocessed data
        df = pd.read_csv(dataset_info["file"])

        for station in dataset_info["stations"]:
            series, df_complete, exog_cols = load_station_series(df, station)

            for repeat_id in range(repeats):
                # Generate random training period (start and end date) for the given dataset and station
                forecast_start = generate_forecast_start(
                    series=series,
                    seed=SEED,
                    repeat_id=repeat_id,
                    max_history_days=max(MAX_HISTORY_DAYS, training_weeks * 7),
                    max_horizon=max(HORIZONS)
                )

                train_start = forecast_start - pd.Timedelta(days=training_weeks * 7)
                train_end = forecast_start

                # Residuals from the training window only, so the order is never selected on data
                # the model is evaluated on
                _, residuals = stage_one_residuals(
                    df_complete[train_start:train_end], station, exog_cols
                )

                print(f"{dataset_name} {station} repeat {repeat_id} "
                      f"({train_start.date()} to {train_end.date()}, {len(residuals)} hours)")

                _, _, window_rows = residual_sarima_grid_search(
                    residuals, candidates=candidates, criterion=criterion, verbose=False
                )

                for row in window_rows:
                    rows.append({
                        "dataset": dataset_name,
                        "station": station,
                        "repeat_id": repeat_id,
                        "train_start": train_start,
                        "train_end": train_end,
                        **row
                    })

    # Save all results to a CSV file for later analysis
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Report the ranking over all windows, which is what the order should be selected on
    summary = summarise(rows, criterion)
    n_windows = len(rows) // len(candidates)

    print(f"\nCandidates ranked over {n_windows} windows, by mean {criterion.upper()}")
    header = f"{'order':<24}{'mean ' + criterion.upper():>14}{'mean rank':>12}{'windows won':>14}"
    print(header)
    for entry in summary:
        label = f"{entry['order']}x{entry['seasonal_order']}"
        print(f"{label:<24}{entry['mean_criterion']:14.2f}{entry['mean_rank']:12.1f}"
              f"{entry['windows_won']:14d}")

    best = summary[0]
    print(f"\nBest by mean {criterion.upper()}: "
          f"arima_order={best['order']}, seasonal_order={best['seasonal_order']}")
    print(f"Per-window results written to {output_file}")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Determine the SARIMA order for stage two of the two-stage "
                                                 "regression model by running a grid search on the stage-one "
                                                 "residuals, pooled over datasets, stations and training "
                                                 "periods. The results are saved to a CSV file.")
    parser.add_argument("--training_weeks", type=int, default=8,
                        help="Number of weeks to use for training (default: 8, as in the comparison experiment)")
    parser.add_argument("--repeats", type=int, default=NUMBER_OF_REPEATS,
                        help=f"Number of training periods per station (default: {NUMBER_OF_REPEATS})")
    parser.add_argument("--criterion", choices=["aic", "bic"], default="aic",
                        help="Information criterion to minimise (default: aic)")
    parser.add_argument("--seasonal_period", type=int, default=24,
                        help="Seasonal period of the candidate orders (default: 24)")
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS),
                        help="Datasets to evaluate (default: all of them)")
    args = parser.parse_args()

    # Make sure the output directory exists
    os.makedirs("output/residual_sarima_order", exist_ok=True)

    run_grid_search(
        training_weeks=args.training_weeks,
        repeats=args.repeats,
        criterion=args.criterion,
        seasonal_period=args.seasonal_period,
        output_file=f"output/residual_sarima_order/results_{args.training_weeks}_weeks_{args.criterion}.csv",
        datasets=args.datasets
    )
