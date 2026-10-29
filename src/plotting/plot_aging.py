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

        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def add_confidence_interval(grouped_df, std_col, count_col):
    """
    Adds confidence interval columns to the given grouped DataFrame.

    Input:
    ------
    - grouped_df: DataFrame that has been grouped and aggregated, containing mean, std, and count columns.
    - std_col: Name of the column containing the standard deviation values.
    - count_col: Name of the column containing the count of samples for each group.

    Output:
    -------
    - grouped_df: The same DataFrame with two new columns added: 'ci_lower' and 'ci_upper',
                  representing the lower and upper bounds of the 95% confidence interval for the mean.
    """
    grouped_df["ci95"] = 1.96 * (grouped_df[std_col] / np.sqrt(grouped_df[count_col]))
    return grouped_df


def plot_datasets(df):
    """
    Plot the error (METRIC) degradation over time for each dataset, showing the mean and confidence intervals.

    Input:
    ------
    df: A DataFrame containing the error values per dataset, station, and age gap.
    """
    column = METRIC.lower()
    grouped = df.groupby(["dataset", "age_gap"]).agg(
        error_mean=(column, "mean"),
        error_std=(column, "std"),
        count=(column, "count")
    ).reset_index()

    # Add confidence intervals to the grouped DataFrame
    grouped = add_confidence_interval(grouped, "error_std", "count")

    plt.figure(figsize=(9, 5))

    for dataset in grouped["dataset"].unique():
        sub = grouped[grouped["dataset"] == dataset]

        plt.errorbar(
            sub["age_gap"],
            sub["error_mean"],
            yerr=sub["ci95"],
            capsize=4,
            marker="o",
            label=DATASET_LABELS.get(dataset, dataset)
        )

    plt.xlabel("Days since last training")
    plt.ylabel(f"Mean {METRIC} (°C)")
    plt.legend(title="Datasets", frameon=True, loc="best")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(_plot_path("aging_datasets"))
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot the RMSE of RegSARIMA against the time since its training, per network.")
    parser.add_argument("--metric", choices=["MAE", "RMSE"], default="RMSE")
    METRIC = parser.parse_args().metric

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
