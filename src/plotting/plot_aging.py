import argparse
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from src.utils.datasets import DATASETS as COMPARISON_DATASETS

# Define constants for file paths and dataset labels
INPUT_TEMPLATE = "output/aging_model_regression_sarima_errors/results_{}_100.csv"
DATASETS = ["RMI", "SYNTHETIC", "TURKU"]
DATASET_LABELS = {
    "RMI": "RMI",
    "SYNTHETIC": "Synthetic",
    "TURKU": "TURCLIM"
}
OUTPUT_DIR = "plots/aging_model_regression_sarima_errors/"

# Error metric to report, set from --metric. RMSE is derived per row as sqrt(MSE), so the
# reported value is the mean of the per-forecast RMSEs, as in plot_model_comparison.py.
METRIC = "RMSE"

# An episode is one onset of one target, with its reconstructions at every age gap
EPISODE = ["dataset", "station", "train_end"]

NUMBER_OF_RESAMPLES = 10_000
SEED = 42


def _plot_path(name):
    """Keep the MAE filenames unchanged, suffix any other metric so it doesn't overwrite them."""
    suffix = "" if METRIC == "MAE" else f"_{METRIC.lower()}"
    return os.path.join(OUTPUT_DIR, f"{name}{suffix}.pdf")


def load_data():
    """
    Load the results from the CSV files for each dataset and combine them into a single DataFrame.

    Output:
    -------
    A DataFrame containing the combined results from all datasets.
    """
    dfs = []

    for dataset in DATASETS:
        df = pd.read_csv(INPUT_TEMPLATE.format(dataset))
        # The stored run of the paper also holds three synthetic targets that the experiment no longer uses
        df = df[df["station"].isin(COMPARISON_DATASETS[dataset]["stations"])]

        # Ensure correct data types
        df["age_gap"] = df["age_gap"].astype(int)
        df["mae"] = df["mae"].astype(float)
        df["rmse"] = np.sqrt(df["mse"].astype(float))
        df["mae_fresh"] = df["mae_fresh"].astype(float)
        df["rmse_fresh"] = np.sqrt(df["mse_fresh"].astype(float))

        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def extra_error(df, rng):
    """
    Extra error (METRIC) of the aged model relative to a fresh model trained right before the same
    reconstruction, (mean aged / mean fresh - 1) * 100 %, per age gap of one dataset. Both models
    reconstruct the same windows, so the season of a window weighs on both alike and drops out of the
    ratio. The 95 % interval is a percentile bootstrap over episodes, keeping the windows of an episode
    together since they share one aged model.

    Input
    -----
    df: DataFrame with the rows of one dataset, as returned by load_data
    rng: Random generator

    Output
    ------
    extra: DataFrame indexed by age gap with the columns extra, lower and upper, in %
    """
    column = METRIC.lower()

    # Summed per episode rather than pivoted, because a target can draw the same onset in two repeats
    # (three times for RMI). Such an episode then counts twice, as in the plain means
    aged = df.pivot_table(index=EPISODE, columns="age_gap", values=column, aggfunc="sum")
    fresh = df.pivot_table(index=EPISODE, columns="age_gap", values=f"{column}_fresh", aggfunc="sum")

    # The ratio of the means over the windows equals the ratio of the sums, as both count the same windows
    estimate = (aged.sum() / fresh.sum() - 1) * 100

    # Every resample draws as many episodes as there are, with replacement, and recomputes the extra error
    resampled = []
    for _ in range(NUMBER_OF_RESAMPLES):
        drawn = rng.integers(len(aged), size=len(aged))
        resampled.append((aged.values[drawn].sum(axis=0) / fresh.values[drawn].sum(axis=0) - 1) * 100)
    lower, upper = np.percentile(resampled, [2.5, 97.5], axis=0)

    return pd.DataFrame({"extra": estimate, "lower": lower, "upper": upper}, index=aged.columns)


def plot_datasets(df):
    """
    Plot the extra error (METRIC) of the aged model relative to a freshly trained one against the time
    since its training, for each dataset, with 95 % bootstrap intervals.

    Input:
    ------
    df: A DataFrame containing the aged and fresh error values per dataset, episode, and age gap.
    """
    rng = np.random.default_rng(SEED)

    plt.figure(figsize=(9, 5))
    plt.axhline(0, color="black", linewidth=0.8)

    for dataset in df["dataset"].unique():
        extra = extra_error(df[df["dataset"] == dataset], rng)

        plt.errorbar(
            extra.index,
            extra["extra"],
            yerr=[extra["extra"] - extra["lower"], extra["upper"] - extra["extra"]],
            capsize=4,
            marker="o",
            label=DATASET_LABELS.get(dataset, dataset)
        )

    plt.xlabel("Days since last training")
    plt.ylabel(f"Extra {METRIC} relative to a freshly trained model (%)")
    plt.legend(title="Datasets", frameon=True, loc="best")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(_plot_path("aging_datasets"))
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot the extra RMSE of an aged RegSARIMA relative to a freshly trained"
                                                 " one against the time since its training, per network.")
    parser.add_argument("--metric", choices=["MAE", "RMSE"], default="RMSE")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS,
                        help="The networks to plot (default: all three).")
    args = parser.parse_args()
    METRIC = args.metric
    DATASETS = args.datasets

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Larger axis labels, tick labels and legend than matplotlib's 10 pt, so they stay readable once LaTeX scales the
    # 9-inch figure to the text width. Arial is one of the fonts Elsevier recommends for artwork, and it is set 1 pt
    # larger than matplotlib's default DejaVu Sans would need, since its lowercase letters are smaller
    plt.rcParams.update({"font.family": "Arial", "font.size": 11, "axes.labelsize": 14, "xtick.labelsize": 13,
                         "ytick.labelsize": 13, "legend.fontsize": 12, "legend.title_fontsize": 12})
    # Saved as vector PDF, which Elsevier prefers for charts, with the fonts embedded as TrueType instead of
    # matplotlib's default Type 3
    plt.rcParams["pdf.fonttype"] = 42

    df = load_data()
    plot_datasets(df)
