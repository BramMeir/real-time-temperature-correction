"""
Script to determine which stations the first stage of the two-stage model (the neighbour regression) relies on,
per target of the synthetic network. For every target, the regression is fitted on the full network over
NUMBER_OF_REPEATS training periods, and the coefficient of every neighbour is averaged over those periods.
The result feeds the station weight maps (src/plotting/plot_stage_one_weight_maps.py).

The neighbours are strongly collinear, so a single coefficient varies considerably from one training period to
the next and is often negative. A neighbour is therefore ranked by its mean absolute coefficient over many
periods rather than by one fit. The coefficients are standardised afterwards (coef * std(x) / std(y) over the
training period) without refitting, so the weights of stations with a different natural variance are
comparable. The raw coefficients are kept next to them.

python -m src.scripts.determine_stage_one_weights [--targets Vielsalm Station_Eeklo]
"""
import os
import argparse
import pandas as pd
from src.utils.generate_forecast_start import generate_forecast_start
from src.utils.load_station_coordinates import load_station_coordinates
from src.models.idw.train import _haversine_distance
from src.models.linear_regression.train import train_neighbour_regression
from src.scripts.determine_residual_sarima_order import load_station_series, SEED, HORIZONS, MAX_HISTORY_DAYS

DATA_FILE = "data/Synthetic/temperature_data.csv"

OUTPUT_DIR = "output/stage_one_weights"

# Number of training periods per target, as in the coefficient analysis of the thesis
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
    Returns a DataFrame with one row per training period and neighbour, with the raw and the standardised
    coefficient.
    """
    rows = []
    for repeat_id in range(repeats):
        # Training periods drawn as in the residual diagnostics (check_residual_autocorrelation.py)
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
            rows.append({
                "target": target,
                "neighbour": neighbour,
                "repeat_id": repeat_id,
                "train_start": train_start,
                "train_end": train_end,
                "raw_coef": coef,
                "std_coef": coef * scale[neighbour],
            })

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
    Returns a DataFrame with one row per target and neighbour: the distance between them, the mean absolute
    standardised and raw coefficient, the mean signed standardised coefficient, the share of periods with a
    negative coefficient, and the rank of the neighbour by weight and by distance (1 = highest / nearest).
    """
    fits = fits.assign(abs_std_coef=fits["std_coef"].abs(), abs_raw_coef=fits["raw_coef"].abs(),
                       negative=fits["std_coef"] < 0)
    weights = fits.groupby(["target", "neighbour"]).agg(
        weight=("abs_std_coef", "mean"),
        weight_raw=("abs_raw_coef", "mean"),
        mean_std_coef=("std_coef", "mean"),
        share_negative=("negative", "mean"),
        periods=("repeat_id", "nunique"),
    ).reset_index()

    weights["distance_km"] = [
        _haversine_distance(*coords[target], *coords[neighbour])
        for target, neighbour in zip(weights["target"], weights["neighbour"])
    ]
    weights["weight_rank"] = weights.groupby("target")["weight"].rank(ascending=False, method="first").astype(int)
    weights["distance_rank"] = weights.groupby("target")["distance_km"].rank(method="first").astype(int)

    return weights.sort_values(["target", "weight_rank"]).reset_index(drop=True)


def run_weight_analysis(targets, repeats):
    """
    Fit the first stage for every target over the training periods and save the per-period coefficients and
    the averaged weights to OUTPUT_DIR.

    Input
    -----
    targets: List of target station names (None uses every station of the network)
    repeats: Number of training periods per target
    """
    df = pd.read_csv(DATA_FILE)
    coords = load_station_coordinates("SYNTHETIC")

    fits = []
    for target in targets or list(df["station_name"].unique()):
        series, df_complete, exog_cols = load_station_series(df, target)
        fits.append(fit_weights(df_complete, series, target, exog_cols, repeats))
        print(f"{target}: fitted on {repeats} training periods, {len(exog_cols)} neighbours", flush=True)

    fits = pd.concat(fits, ignore_index=True)
    weights = summarise_weights(fits, coords)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fits.to_csv(os.path.join(OUTPUT_DIR, "coefficients_per_period.csv"), index=False)
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
    parser.add_argument("--targets", nargs="+", default=None,
                        help="Target stations to fit (default: every station of the synthetic network)")
    parser.add_argument("--repeats", type=int, default=NUMBER_OF_REPEATS,
                        help=f"Number of training periods per target (default: {NUMBER_OF_REPEATS})")
    args = parser.parse_args()

    run_weight_analysis(targets=args.targets, repeats=args.repeats)
