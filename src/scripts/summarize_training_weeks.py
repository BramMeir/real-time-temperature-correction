"""
Accuracy of every model that learns from its training window, for windows of 1 to 26 weeks, from the runs of
hpc/submit_training_weeks.sh (src/scripts/models_comparison_experiment.py with fixed forecast starts). All windows
reconstruct the same outages, so a difference between two windows comes from the window alone.

Prints the mean RMSE per model and window over all windows and outage durations, and the same relative to the
eight weeks of the benchmark, and writes both to OUTPUT_DIR. Only reads the results otherwise.

python -m src.scripts.summarize_training_weeks
"""
import glob
import os

import numpy as np
import pandas as pd

RESULTS = "output/training_weeks_comparison/*_weeks.csv"
OUTPUT_DIR = "output/training_weeks_comparison/summary"
REFERENCE_WEEKS = 8

# Outage durations in hours that are summarised separately, besides the mean over all of them
SHORT_HORIZONS = [1, 4, 12, 24]
LONG_HORIZONS = [168, 336, 504, 720]


def load_results(pattern=RESULTS):
    """
    Read the results of all models and windows and add the RMSE per window.

    Input
    -----
    pattern: Glob pattern of the result files (default is RESULTS)

    Output
    ------
    Returns a DataFrame with one row per model, training window, forecast start and horizon.
    """
    frames = []
    for path in glob.glob(pattern):
        df = pd.read_csv(path)
        # The window length is only in the file name, <model>_<weeks>_weeks.csv
        df["Weeks"] = int(os.path.basename(path).rsplit("_", 2)[1])
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    df["RMSE"] = np.sqrt(df["MSE"])

    return df


def check_shared_starts(df):
    """
    Raise if two windows of a model were run on different forecast starts, which would make them incomparable.

    Input
    -----
    df: DataFrame as returned by load_results
    """
    starts = df.groupby(["Model", "Weeks"])["Train_end"].apply(lambda s: tuple(sorted(s.unique())))
    for model, per_window in starts.groupby(level="Model"):
        if per_window.nunique() != 1:
            raise ValueError(f"{model} was run on different forecast starts for different windows")


def rmse_table(df, horizons=None):
    """
    Mean RMSE per model and training window.

    Input
    -----
    df: DataFrame as returned by load_results
    horizons: Horizons to average over (default is None, which uses all of them)

    Output
    ------
    Returns a DataFrame with one row per model and one column per window.
    """
    if horizons is not None:
        df = df[df["Horizon"].isin(horizons)]

    table = df.pivot_table(index="Model", columns="Weeks", values="RMSE", aggfunc="mean")

    return table.sort_values(REFERENCE_WEEKS)


def relative_table(table):
    """Change of the mean RMSE relative to the reference window, in percent."""
    return 100 * (table.div(table[REFERENCE_WEEKS], axis=0) - 1)


if __name__ == "__main__":
    pd.set_option("display.width", 200)

    results = load_results()
    check_shared_starts(results)

    windows = results.groupby(["Model", "Weeks"]).apply(lambda g: g[["Dataset", "Station", "Train_end"]]
                                                        .drop_duplicates().shape[0])
    print(f"Forecast starts per model and window: {sorted(windows.unique())}\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for name, horizons in [("all", None), ("short", SHORT_HORIZONS), ("long", LONG_HORIZONS)]:
        table = rmse_table(results, horizons)
        relative = relative_table(table)

        label = "all outage durations" if horizons is None else f"outages of {horizons} hours"
        print(f"Mean RMSE (degrees C), {label}")
        print(table.round(3).to_string())
        print(f"\nChange relative to {REFERENCE_WEEKS} weeks (%), {label}")
        print(relative.round(1).to_string(), "\n")

        table.to_csv(os.path.join(OUTPUT_DIR, f"rmse_{name}.csv"))
        relative.to_csv(os.path.join(OUTPUT_DIR, f"rmse_relative_{name}.csv"))

    print(f"Tables written to {OUTPUT_DIR}")
