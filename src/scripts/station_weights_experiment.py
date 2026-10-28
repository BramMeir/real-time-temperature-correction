import os
import argparse
import pandas as pd
from src.utils.datasets import DATASETS, load_station_data
from src.utils.generate_forecast_start import generate_forecast_start
from src.utils.load_station_coordinates import load_station_coordinates
from src.models.idw.train import _haversine_distance
from src.models.neighbour_regression.train import train_neighbour_regression
from src.scripts.residual_sarima_order import SEED, HORIZONS, MAX_HISTORY_DAYS


OUTPUT_DIR = "output/stage_one_weights"

# Number of training periods per target
NUMBER_OF_REPEATS = 50

# Length of every training period, as in the comparison experiment
TRAINING_WEEKS = 8


def fit_weights(df_complete, series, target, exog_cols, repeats):
    """
    Fit the neighbour regression of one target over several training periods and return its coefficients.

    Input
    -----
    df_complete: DataFrame with the target station and all neighbouring stations as columns
    series: Target time series with an hourly datetime index, used to draw the training periods
    target: Name of the target station
    exog_cols: List of the neighbouring station column names
    repeats: Number of training periods

    Output
    ------
    Returns a DataFrame with one row per training period and neighbour, with the standardised coefficient.
    """
    rows = []
    for repeat_id in range(repeats):
        # Training periods drawn as in the residual diagnostics (residual_autocorrelation.py)
        train_end = generate_forecast_start(
            series=series,
            seed=SEED,
            repeat_id=repeat_id,
            max_history_days=max(MAX_HISTORY_DAYS, TRAINING_WEEKS * 7),
            max_horizon=max(HORIZONS)
        )
        train_start = train_end - pd.Timedelta(weeks=TRAINING_WEEKS)
        train = df_complete[train_start:train_end]

        regression = train_neighbour_regression(train, target, exog_cols)

        # Standardise on the rows the regression was fitted on
        data = train[[target] + exog_cols].dropna()
        scale = data[exog_cols].std() / data[target].std()

        for neighbour, coef in zip(exog_cols, regression.coef_):
            rows.append({"target": target, "neighbour": neighbour, "std_coef": coef * scale[neighbour]})

    return pd.DataFrame(rows)


def summarise_weights(fits, coords):
    """
    Average the coefficients of every target and neighbour over the training periods and rank the neighbours.

    Input
    -----
    fits: DataFrame with one row per target, training period and neighbour, as returned by fit_weights
    coords: Dict mapping station_name to a (latitude, longitude) tuple

    Output
    ------
    Returns a DataFrame with one row per target and neighbour: the weight (the mean absolute standardised
    coefficient), the distance between them, and the rank of the neighbour by weight and by distance (1 = highest /
    nearest).
    """
    weights = (fits.assign(weight=fits["std_coef"].abs())
               .groupby(["target", "neighbour"], as_index=False)["weight"].mean())

    weights["distance_km"] = [
        _haversine_distance(*coords[target], *coords[neighbour])
        for target, neighbour in zip(weights["target"], weights["neighbour"])
    ]
    weights["weight_rank"] = weights.groupby("target")["weight"].rank(ascending=False, method="first").astype(int)
    weights["distance_rank"] = weights.groupby("target")["distance_km"].rank(method="first").astype(int)

    return weights.sort_values(["target", "weight_rank"]).reset_index(drop=True)


def run_weight_analysis(repeats):
    """
    Fit the first stage for every target of the synthetic network over the training periods and save the averaged
    weights to OUTPUT_DIR.

    Input
    -----
    repeats: Number of training periods per target
    """
    df = pd.read_csv(DATASETS["SYNTHETIC"]["file"])
    coords = load_station_coordinates("SYNTHETIC")

    fits = []
    for target in df["station_name"].unique():
        df_complete = load_station_data(df, target)
        series = df_complete[target]
        exog_cols = list(df_complete.columns[1:])
        fits.append(fit_weights(df_complete, series, target, exog_cols, repeats))
        print(f"{target}: fitted on {repeats} training periods, {len(exog_cols)} neighbours", flush=True)

    fits = pd.concat(fits, ignore_index=True)
    weights = summarise_weights(fits, coords)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    weights.to_csv(os.path.join(OUTPUT_DIR, "weights.csv"), index=False)

    print("\nFour highest-weighted neighbours per target (distance rank in brackets):")
    for target, group in weights[weights["weight_rank"] <= 4].groupby("target"):
        top = ", ".join(f"{row.neighbour} ({row.distance_rank})" for row in group.itertuples())
        print(f"{target}: {top}")
    print(f"\nWeights written to {OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Average the coefficients of the neighbour regression (the first "
                                                 "stage of the two-stage model) per target of the synthetic "
                                                 "network over many training periods.")
    parser.add_argument("--repeats", type=int, default=NUMBER_OF_REPEATS,
                        help=f"Number of training periods per target (default: {NUMBER_OF_REPEATS})")
    args = parser.parse_args()

    run_weight_analysis(repeats=args.repeats)
