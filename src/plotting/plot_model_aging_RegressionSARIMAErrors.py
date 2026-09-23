"""
Script: plot_model_aging_RegressionSARIMAErrors.py
Description: Plots to analyze the effect of model aging on the two-stage regression with SARIMA
errors model's performance. Mirrors plot_model_aging.py, which ran this analysis for ARIMAX.

Usage: python -m src.plotting.plot_model_aging_RegressionSARIMAErrors [--metric MAE|RMSE]
"""
import argparse
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

# Define constants for file paths and dataset labels
INPUT_TEMPLATE = "output/aging_model_regression_sarima_errors/results_{}_100.csv"
DATASETS = ["KMI", "SYNTHETIC", "TURKU"]
DATASET_LABELS = {
    "KMI": "RMI",
    "SYNTHETIC": "Synthetic",
    "TURKU": "TURCLIM"
}
OUTPUT_DIR = "plots/aging_model_regression_sarima_errors/"

# Error metric to report, set from --metric. RMSE is derived per row as sqrt(MSE), so the
# reported value is the mean of the per-forecast RMSEs, as in plot_model_comparisons.py.
METRIC = "MAE"


def _plot_path(name):
    """Keep the MAE filenames unchanged, suffix any other metric so it doesn't overwrite them."""
    suffix = "" if METRIC == "MAE" else f"_{METRIC.lower()}"
    return os.path.join(OUTPUT_DIR, f"{name}{suffix}.png")


def load_data():
    """
    Load the results from the CSV files for each dataset and combine them into a single DataFrame.

    Output:
    -------
    A DataFrame containing the combined results from all datasets.
    """
    dfs = []

    for dataset in DATASETS:
        file_path = INPUT_TEMPLATE.format(dataset)

        if not os.path.exists(file_path):
            print(f"Warning: file not found for {dataset}: {file_path}")
            continue

        df = pd.read_csv(file_path)

        # Ensure correct data types
        df["age_gap"] = df["age_gap"].astype(int)
        df["mae"] = df["mae"].astype(float)
        df["rmse"] = np.sqrt(df["mse"].astype(float))

        dfs.append(df)

    if not dfs:
        raise ValueError("No data files could be loaded.")

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
    plt.savefig(_plot_path("aging_datasets"), dpi=300)
    plt.close()


def plot_relative(df):
    """
    Plot the relative degradation of the error (METRIC) over time for each dataset, showing how the performance degrades relative
    to the starting point.

    Input:
    ------
    df: A DataFrame containing the error values per dataset, station, and age gap, with columns 'dataset', 'station', 'age_gap', and 'mae'/'rmse'.
    """
    column = METRIC.lower()

    # Compute baseline per dataset + station
    baseline = df[df["age_gap"] == 0].groupby(
        ["dataset", "station"]
    )[column].mean().reset_index()

    baseline = baseline.rename(columns={column: "error_baseline"})

    df = df.merge(baseline, on=["dataset", "station"])
    df["rel_error"] = df[column] / df["error_baseline"]

    # Aggregate
    grouped = df.groupby(["dataset", "age_gap"]).agg(
        rel_mean=("rel_error", "mean"),
        rel_std=("rel_error", "std"),
        count=("rel_error", "count")
    ).reset_index()

    # Add confidence intervals (same as elsewhere)
    grouped = add_confidence_interval(grouped, "rel_std", "count")

    plt.figure(figsize=(9, 5))

    for dataset in grouped["dataset"].unique():
        sub = grouped[grouped["dataset"] == dataset]

        plt.errorbar(
            sub["age_gap"],
            sub["rel_mean"],
            yerr=sub["ci95"],
            capsize=3,
            marker="o",
            linewidth=1.5,
            label=DATASET_LABELS.get(dataset, dataset)
        )

    # Subtle reference line at 1.0 to indicate no degradation
    plt.axhline(
        y=1.0,
        linestyle="--",
        linewidth=1.0,
        color="black",
        alpha=0.7
    )

    plt.xlabel("Days since last training")
    plt.ylabel(f"Relative {METRIC} (vs. day 0)")
    plt.legend(title="Datasets", frameon=True, loc="best")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(_plot_path("aging_relative"), dpi=300)
    plt.close()


def plot_overall(df):
    """
    Plot the overall error (METRIC) degradation over time, averaging across all datasets and stations.

    Input:
    ------
    df: A DataFrame containing the error values per dataset, station, and age gap.
    """
    column = METRIC.lower()
    grouped = df.groupby("age_gap").agg(
        error_mean=(column, "mean"),
        error_std=(column, "std"),
        count=(column, "count")
    ).reset_index()

    # Add confidence intervals to the grouped DataFrame
    grouped = add_confidence_interval(grouped, "error_std", "count")

    plt.figure(figsize=(9, 5))

    plt.errorbar(
        grouped["age_gap"],
        grouped["error_mean"],
        yerr=grouped["ci95"],
        capsize=4,
        marker="o",
    )

    plt.xlabel("Days since last training")
    plt.ylabel(f"Mean {METRIC} (°C)")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(_plot_path("aging_overall"), dpi=300)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric", choices=["MAE", "RMSE"], default="MAE")
    METRIC = parser.parse_args().metric

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_data()
    plot_datasets(df)
    plot_relative(df)
    plot_overall(df)
