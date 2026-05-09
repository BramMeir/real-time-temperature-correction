"""
Script: plot_retraining_results.py
Description: Plot the results of the retraining experiments, showing model aging and the effect of different
             retraining strategies.

Example usage:
python -m src.plotting.plot_retraining_results
"""
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

INPUT_PATH = "output/retraining_results/results_*.csv"
OUTPUT_DIR = "plots/retraining_results"
DATASET_LABELS = {
    "KMI": "KMI",
    "SYNTHETIC": "Synthetisch",
    "TURKU": "Turku"
}
MODEL_LABELS = {
    "no_retrain": "Geen hertraining",
    "inc_3d": "Incrementele hertraining (3 dagen)",
    "inc_15d": "Incrementele hertraining (15 dagen)",
    "full_3d": "Volledige hertraining (3 dagen)",
    "full_15d": "Volledige hertraining (15 dagen)"
}


def load_data():
    """
    Load all CSV files matching the input pattern and concatenate them into a single DataFrame.

    Output:
    df: A DataFrame containing all the results from the retraining experiments.
    """
    files = glob.glob(INPUT_PATH)
    df = pd.concat([pd.read_csv(f) for f in files])

    df["current_time"] = pd.to_datetime(df["current_time"])

    return df


def add_relative_time(df):
    """
    Adds a column 'days_since_start' to the DataFrame, which represents the number of days since the start of the simulation.
    This is necessary to compare the performance of strategies over time, regardless of the absolute dates of the simulations.

    Input:
    ------
    df: A DataFrame containing the retraining results.

    Output:
    -------
    df: A DataFrame with the added 'days_since_start' column.
    """
    df["t0"] = df.groupby(
        ["dataset", "station", "repeat_id"]
    )["current_time"].transform("min")

    df["days_since_start"] = (
        (df["current_time"] - df["t0"]).dt.total_seconds() / (3600 * 24)
    )

    return df


def plot_per_dataset(df_avg):
    """
    Plot the average MAE over time for each dataset and model type, showing how the performance degrades (model aging) and
    how retraining strategies affect it.

    Input:
    ------
    df_avg: A DataFrame containing the average MAE per dataset, model type, and days since start.
    """
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
                model_data["mae"],
                label=MODEL_LABELS.get(model_type, model_type),
                alpha=0.8,
                linestyle="-" if "full" in model_type else "--"
            )

            # Add an indicator for the retraining points (where days_since_start is a multiple of the retraining interval)
            events = subset_avg[
                (subset_avg["model_type"] == model_type) & (subset_avg["retrain_event"] == 1)
            ]

            plt.scatter(
                events["days_since_start"],
                events["mae"],
                marker="x",
                s=50,
                alpha=0.7
            )

        # Plot the "no_retrain" strategy separately to ensure it's visible and serves as a clear baseline
        no_retrain_data = subset_avg[subset_avg["model_type"] == "no_retrain"]

        plt.plot(
            no_retrain_data["days_since_start"],
            no_retrain_data["mae"],
            label=MODEL_LABELS["no_retrain"],
            color="black",
            linewidth=2,
            linestyle="--",
            zorder=10
        )

        plt.xlabel("Dagen sinds start van simulatie")
        plt.ylabel("Gemiddelde MAE (°C)")
        plt.legend()
        plt.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/retraining_strategies_{dataset}.png", dpi=300)
        plt.close()


def plot_training_event_cost(df_avg):
    """
    Dot plot showing the average training duration for each retraining event, for each dataset and model type.

    Input:
    ------
    df_avg: A DataFrame containing the average MAE and retraining duration per dataset, model type, and days since start.
    """
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

        plt.xlabel("Dagen sinds start van simulatie")
        plt.ylabel("Trainingstijd per hertraining (s)")
        plt.legend(loc="upper left", frameon=True)
        plt.ylim(top=subset["retraining_duration"].max() + 10)
        plt.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/training_event_cost_{dataset}.png", dpi=300)
        plt.close()


if __name__ == "__main__":
    # Make sure the output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_data()
    df = add_relative_time(df)

    # Aggregate to get average MAE per dataset, model_type, and days_since_start
    df_avg = df.groupby(
        ["dataset", "model_type", "days_since_start"]
    ).agg({
        "mae": "mean",
        "retraining_duration": "mean",
        "retrain_event": "max"  # To know if a retraining event occurred at that time point
    }).reset_index()

    plot_per_dataset(df_avg)
    plot_training_event_cost(df_avg)
