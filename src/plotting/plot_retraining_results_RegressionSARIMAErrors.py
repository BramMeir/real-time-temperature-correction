"""
Script: plot_retraining_results_RegressionSARIMAErrors.py
Description: Plot the retraining-strategy results for the two-stage model. Mirrors
             plot_retraining_results.py, which ran this analysis for ARIMAX.

python -m src.plotting.plot_retraining_results_RegressionSARIMAErrors [--metric MAE|RMSE]
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

INPUT_PATH = "output/retraining_results_regression_sarima_errors/results_*.csv"
OUTPUT_DIR = "plots/retraining_results_regression_sarima_errors"
MODEL_LABELS = {
    "no_retrain": "No retraining",
    "lr_only": "Regression only",
    "sarima_only": "Residual SARIMA only",
    "both": "Both retrained"
}

# Error metric to report, set from --metric. RMSE is derived per row as sqrt(MSE), so the
# reported value is the mean of the per-forecast RMSEs, as in plot_model_comparisons.py.
METRIC = "MAE"


def _plot_path(name):
    """Keep the MAE filenames unchanged, suffix any other metric so it doesn't overwrite them."""
    suffix = "" if METRIC == "MAE" else f"_{METRIC.lower()}"
    return f"{OUTPUT_DIR}/{name}{suffix}.png"


def load_data():
    """Load all CSV files matching INPUT_PATH and concatenate them into a single DataFrame."""
    files = glob.glob(INPUT_PATH)
    df = pd.concat([pd.read_csv(f) for f in files])

    df["current_time"] = pd.to_datetime(df["current_time"])

    return df


def add_relative_time(df):
    """Add a 'days_since_start' column, so strategies can be compared regardless of absolute dates."""
    df["t0"] = df.groupby(
        ["dataset", "station", "repeat_id"]
    )["current_time"].transform("min")

    df["days_since_start"] = (
        (df["current_time"] - df["t0"]).dt.total_seconds() / (3600 * 24)
    )

    return df


def plot_per_dataset(df_avg):
    """Plot the average error (METRIC) over time per dataset and model type, showing the effect of each strategy."""
    datasets = df_avg["dataset"].unique()

    for dataset in datasets:
        plt.figure(figsize=(10, 6))

        subset_avg = df_avg[df_avg["dataset"] == dataset]

        # Loop through each model type (except "no_retrain") and plot its performance over time
        for model_type in MODEL_LABELS.keys():
            if model_type == "no_retrain":
                continue

            model_data = subset_avg[subset_avg["model_type"] == model_type]

            plt.plot(
                model_data["days_since_start"],
                model_data[METRIC.lower()],
                label=MODEL_LABELS.get(model_type, model_type),
                alpha=0.8,
                linestyle="-" if model_type == "both" else "--"
            )

            # Add an indicator for the retraining points (where days_since_start is a multiple of the
            # retraining interval)
            events = subset_avg[
                (subset_avg["model_type"] == model_type) & (subset_avg["retrain_event"] == 1)
            ]

            plt.scatter(
                events["days_since_start"],
                events[METRIC.lower()],
                marker="x",
                s=50,
                alpha=0.7
            )

        # Plot the "no_retrain" strategy separately to ensure it's visible and serves as a clear
        # baseline
        no_retrain_data = subset_avg[subset_avg["model_type"] == "no_retrain"]

        plt.plot(
            no_retrain_data["days_since_start"],
            no_retrain_data[METRIC.lower()],
            label=MODEL_LABELS["no_retrain"],
            color="black",
            linewidth=2,
            linestyle="--",
            zorder=10
        )

        plt.xlabel("Days since start of simulation")
        plt.ylabel(f"Mean {METRIC} (°C)")
        plt.legend()
        plt.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(_plot_path(f"retraining_strategies_{dataset}"), dpi=300)
        plt.close()


def plot_training_event_cost(df_avg):
    """Dot plot of training duration per retraining event, for each dataset and model type."""
    datasets = df_avg["dataset"].unique()

    for dataset in datasets:
        plt.figure(figsize=(10, 6))

        subset = df_avg[
            (df_avg["dataset"] == dataset) & (df_avg["retrain_event"] == 1)
        ]

        for model_type in MODEL_LABELS.keys():
            if model_type == "no_retrain":
                continue

            model_data = subset[subset["model_type"] == model_type]

            plt.scatter(
                model_data["days_since_start"],
                model_data["retraining_duration"],
                label=MODEL_LABELS[model_type],
                s=60,
                alpha=0.8
            )

        plt.xlabel("Days since start of simulation")
        plt.ylabel("Training time per retraining event (s)")
        plt.legend(loc="upper left", frameon=True)
        plt.ylim(top=subset["retraining_duration"].max() + 10)
        plt.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(_plot_path(f"training_event_cost_{dataset}"), dpi=300)
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric", choices=["MAE", "RMSE"], default="MAE")
    METRIC = parser.parse_args().metric

    # Make sure the output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_data()
    df = add_relative_time(df)
    df["rmse"] = df["mse"] ** 0.5

    # Aggregate to get the average error per dataset, model_type, and days_since_start
    df_avg = df.groupby(
        ["dataset", "model_type", "days_since_start"]
    ).agg({
        "mae": "mean",
        "rmse": "mean",
        "retraining_duration": "mean",
        "retrain_event": "max"  # To know if a retraining event occurred at that time point
    }).reset_index()

    plot_per_dataset(df_avg)
    # The training cost does not depend on the metric, so only the MAE run draws it
    if METRIC == "MAE":
        plot_training_event_cost(df_avg)
