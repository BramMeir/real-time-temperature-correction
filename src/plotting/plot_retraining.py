import argparse
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
from src.utils.datasets import DATASETS

INPUT_PATH = "output/retraining_results_regression_sarima_errors/results_*.csv"
OUTPUT_DIR = "plots/retraining_results_regression_sarima_errors"
MODEL_LABELS = {
    "no_retrain": "No retraining",
    "lr_only": "Regression only",
    "sarima_only": "Residual SARIMA only",
    "both": "Both retrained"
}

# Line style per strategy. The colours are from the Okabe-Ito palette and stay distinguishable with colour-vision
# deficiencies; the strategies that nearly coincide (regression only and both, residual SARIMA only and none) get
# strongly contrasting colours. Each partial strategy is dashed and drawn on top of the full strategy it follows, so
# both stay visible where they overlap
STRATEGY_STYLES = {
    "lr_only": {"color": "#E69F00", "linestyle": "--", "linewidth": 2, "zorder": 3},
    "sarima_only": {"color": "#009E73", "linestyle": "--", "linewidth": 2, "zorder": 3},
    "both": {"color": "#0072B2", "linestyle": "-", "linewidth": 2, "zorder": 2},
    "no_retrain": {"color": "black", "linestyle": "-", "linewidth": 2, "zorder": 2},
}

# Error metric to report, set from --metric. RMSE is derived per row as sqrt(MSE), so the
# reported value is the mean of the per-forecast RMSEs, as in plot_model_comparison.py.
METRIC = "RMSE"


def _plot_path(name):
    """Keep the MAE filenames unchanged, suffix any other metric so it doesn't overwrite them."""
    suffix = "" if METRIC == "MAE" else f"_{METRIC.lower()}"
    return f"{OUTPUT_DIR}/{name}{suffix}.pdf"


def load_data():
    """Load all CSV files matching INPUT_PATH and concatenate them into a single DataFrame."""
    files = glob.glob(INPUT_PATH)
    df = pd.concat([pd.read_csv(f) for f in files])

    # The stored run of the paper also holds three synthetic targets that the experiment no longer uses
    targets = {(dataset, station) for dataset, info in DATASETS.items() for station in info["stations"]}
    df = df[[target in targets for target in zip(df["dataset"], df["station"])]]

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
    """
    Plot the average error (METRIC) over time per dataset and model type, and below it how much each retraining
    strategy lowers that error compared with the model that is never retrained.
    """
    datasets = df_avg["dataset"].unique()
    metric = METRIC.lower()

    for dataset in datasets:
        fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={"height_ratios": [3, 2]})

        subset_avg = df_avg[df_avg["dataset"] == dataset]
        never_retrained = subset_avg[subset_avg["model_type"] == "no_retrain"].set_index("days_since_start")[metric]

        # The reference without retraining first in the legend, then the partial strategies and both stages
        for model_type in ["no_retrain", "lr_only", "sarima_only", "both"]:
            model_data = subset_avg[subset_avg["model_type"] == model_type].set_index("days_since_start")[metric]
            style = STRATEGY_STYLES[model_type]

            top.plot(model_data.index, model_data.values, label=MODEL_LABELS[model_type], **style)

            if model_type != "no_retrain":
                reduction = never_retrained - model_data
                bottom.plot(reduction.index, reduction.values, **style)

        bottom.axhline(0, color="black", linewidth=1)

        top.set_ylabel(f"Mean {METRIC} (°C)")
        bottom.set_ylabel(f"{METRIC} reduction (°C)")
        bottom.set_xlabel("Days since start of simulation")
        top.legend(fontsize=13, ncol=2)

        for ax in (top, bottom):
            ax.grid(alpha=0.3)

        fig.tight_layout()
        fig.savefig(_plot_path(f"retraining_strategies_{dataset}"))
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot the RMSE of RegSARIMA under every retraining strategy, per network.")
    parser.add_argument("--metric", choices=["MAE", "RMSE"], default="RMSE")
    METRIC = parser.parse_args().metric

    # Make sure the output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Larger axis labels and tick labels than matplotlib's 10 pt, so they stay readable once LaTeX scales the 10-inch
    # figure to the text width. Arial is one of the fonts Elsevier recommends for artwork, and it is set 1 pt larger
    # than matplotlib's default DejaVu Sans would need, since its lowercase letters are smaller
    plt.rcParams.update({"font.family": "Arial", "font.size": 11, "axes.labelsize": 15, "xtick.labelsize": 14,
                         "ytick.labelsize": 14})
    # Saved as vector PDF, which Elsevier prefers for charts, with the fonts embedded as TrueType instead of
    # matplotlib's default Type 3
    plt.rcParams["pdf.fonttype"] = 42

    df = load_data()
    df = add_relative_time(df)
    df["rmse"] = df["mse"] ** 0.5

    # Aggregate to get the average error per dataset, model_type, and days_since_start
    df_avg = df.groupby(
        ["dataset", "model_type", "days_since_start"]
    ).agg({
        "mae": "mean",
        "rmse": "mean",
        "retraining_duration": "mean"
    }).reset_index()

    plot_per_dataset(df_avg)
