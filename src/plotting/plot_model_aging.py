"""
Script: plot_model_aging.py
Description: Plots to analyze the effect of model aging on ARIMAX performance.

Usage: python -m src.plotting.plot_model_aging
"""
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

# Define constants for file paths and dataset labels
INPUT_TEMPLATE = "output/aging_model_fixed_gaps/results_{}_100.csv"
DATASETS = ["KMI", "SYNTHETIC", "TURKU"]
DATASET_LABELS = {
    "KMI": "RMI",
    "SYNTHETIC": "Synthetic",
    "TURKU": "TURCLIM"
}
OUTPUT_DIR = "plots/aging_model/"


def load_data():
    """
    Load the MAE results from the CSV files for each dataset and combine them into a single DataFrame.

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
    Plot the MAE degradation over time for each dataset, showing the mean and confidence intervals.

    Input:
    ------
    df: A DataFrame containing the MAE values per dataset, station, and age gap.
    """
    grouped = df.groupby(["dataset", "age_gap"]).agg(
        mae_mean=("mae", "mean"),
        mae_std=("mae", "std"),
        count=("mae", "count")
    ).reset_index()

    # Add confidence intervals to the grouped DataFrame
    grouped = add_confidence_interval(grouped, "mae_std", "count")

    plt.figure(figsize=(9, 5))

    for dataset in grouped["dataset"].unique():
        sub = grouped[grouped["dataset"] == dataset]

        plt.errorbar(
            sub["age_gap"],
            sub["mae_mean"],
            yerr=sub["ci95"],
            capsize=4,
            marker="o",
            label=DATASET_LABELS.get(dataset, dataset)
        )

    plt.xlabel("Days since last training")
    plt.ylabel("Mean MAE (°C)")
    plt.legend(title="Datasets", frameon=True, loc="best")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/aging_datasets.png", dpi=300)
    plt.close()


def plot_relative(df):
    """
    Plot the relative degradation of MAE over time for each dataset, showing how the performance degrades relative
    to the starting point.

    Input:
    ------
    df: A DataFrame containing the MAE values per dataset, station, and age gap, with columns 'dataset', 'station', 'age_gap', and 'mae'.
    """
    # Compute baseline per dataset + station
    baseline = df[df["age_gap"] == 0].groupby(
        ["dataset", "station"]
    )["mae"].mean().reset_index()

    baseline = baseline.rename(columns={"mae": "mae_baseline"})

    df = df.merge(baseline, on=["dataset", "station"])
    df["rel_mae"] = df["mae"] / df["mae_baseline"]

    # Aggregate
    grouped = df.groupby(["dataset", "age_gap"]).agg(
        rel_mean=("rel_mae", "mean"),
        rel_std=("rel_mae", "std"),
        count=("rel_mae", "count")
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
    plt.ylabel("Relative MAE (vs. day 0)")
    plt.legend(title="Datasets", frameon=True, loc="best")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/aging_relative.png", dpi=300)
    plt.close()


def plot_overall(df):
    """
    Plot the overall MAE degradation over time, averaging across all datasets and stations.

    Input:
    ------
    df: A DataFrame containing the MAE values per dataset, station, and age gap.
    """
    grouped = df.groupby("age_gap").agg(
        mae_mean=("mae", "mean"),
        mae_std=("mae", "std"),
        count=("mae", "count")
    ).reset_index()

    # Add confidence intervals to the grouped DataFrame
    grouped = add_confidence_interval(grouped, "mae_std", "count")

    plt.figure(figsize=(9, 5))

    plt.errorbar(
        grouped["age_gap"],
        grouped["mae_mean"],
        yerr=grouped["ci95"],
        capsize=4,
        marker="o",
    )

    plt.xlabel("Days since last training")
    plt.ylabel("Mean MAE (°C)")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/aging_overall.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_data()
    plot_datasets(df)
    plot_relative(df)
    plot_overall(df)
