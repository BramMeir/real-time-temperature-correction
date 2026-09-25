"""
Distribution of the per window RMSE of every model in the models comparison
(src/scripts/models_comparison_experiment.py). The mean RMSE per horizon hides how the error is spread
over the windows, so this prints the median, the upper percentiles and the worst window per model, and
lists the worst windows themselves. Only reads the results, nothing is written.
"""
import glob
import os

import numpy as np
import pandas as pd

RESULTS = "output/models_comparison/*_8_weeks.csv"
HORIZONS = [4, 24, 168, 720]
# Models that were run but are not part of the paper's comparison
EXCLUDED_MODELS = ["XGBoost"]
NUMBER_OF_WORST = 10


def load_results(pattern=RESULTS):
    """
    Read the comparison results of all models and add the RMSE per window.

    Input
    -----
    pattern: Glob pattern of the result files (default is RESULTS)

    Output
    ------
    Returns a DataFrame with one row per model, window and horizon.
    """
    df = pd.concat(pd.read_csv(path) for path in glob.glob(pattern))
    df = df[~df["Model"].isin(EXCLUDED_MODELS)].copy()
    df["RMSE"] = np.sqrt(df["MSE"])

    return df


def summarize(df, horizons=HORIZONS):
    """
    Summary of the per window RMSE per model and horizon.

    Input
    -----
    df: DataFrame as returned by load_results
    horizons: Horizons to summarize (default is HORIZONS)

    Output
    ------
    Returns a DataFrame with the mean, median, 90th and 99th percentile and maximum per model and
    horizon, and the number of windows whose RMSE exceeds 5 degrees.
    """
    selected = df[df["Horizon"].isin(horizons)]
    grouped = selected.groupby(["Horizon", "Model"])["RMSE"]

    summary = grouped.agg(
        Mean="mean",
        Median="median",
        P90=lambda values: values.quantile(0.9),
        P99=lambda values: values.quantile(0.99),
        Max="max",
        Above_5=lambda values: int((values > 5).sum()),
        Windows="count",
    )

    return summary.sort_values(["Horizon", "Mean"])


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 200)

    results = load_results()
    print(summarize(results).round(3).to_string())

    for horizon in [max(HORIZONS)]:
        worst = results[results["Horizon"] == horizon].nlargest(NUMBER_OF_WORST, "RMSE")
        print(f"\nWorst {NUMBER_OF_WORST} windows over {horizon} hours")
        print(worst[["Model", "Dataset", "Station", "Train_start", "MAE", "RMSE"]].round(3).to_string(index=False))

    print(f"\nRead from {os.path.dirname(RESULTS)}")
