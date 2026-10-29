import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

RESULTS_GLOB = "output/confidence_analysis_regression_sarima_errors/confidence_station_*.csv"
PLOTS_DIR = "plots/confidence_analysis_regression_sarima_errors"


def scatter_panel(ax, x, y, title, xlabel, correlation, text_x):
    """Scatter one indicator against the realised MAE, with its least-squares line and correlation."""
    ax.scatter(x, y, alpha=0.6)

    x_sorted = np.sort(x)
    m, b = np.polyfit(x, y, 1)
    ax.plot(x_sorted, m * x_sorted + b)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.text(text_x, 0.95, f"r = {correlation:.2f}", transform=ax.transAxes, verticalalignment='top', fontsize=15)

    # A few ticks per axis are enough to read the trend
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.grid(alpha=0.2)


def plot_reliability(df, output_path):
    """
    Plot the reliability score and the interval width of every station against its realised MAE, side by side.

    Input
    -----
    df: One row per station, with mae_mean, confidence_mean and interval_size_mean
    output_path: Path of the PDF to write
    """
    corr_conf = df["confidence_mean"].corr(df["mae_mean"])
    corr_interval = df["interval_size_mean"].corr(df["mae_mean"])

    print(f"Correlation (confidence vs MAE): {corr_conf:.3f}")
    print(f"Correlation (interval size vs MAE): {corr_interval:.3f}")

    _, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    scatter_panel(axes[0], df["confidence_mean"], df["mae_mean"], "Error-based score",
                  "Reliability score (1 / (1 + MAE))", corr_conf, 0.825)
    axes[0].set_ylabel("Mean MAE (°C)")
    scatter_panel(axes[1], df["interval_size_mean"], df["mae_mean"], "Prediction-interval width",
                  "95 % interval width (°C)", corr_interval, 0.05)

    plt.tight_layout()
    plt.savefig(output_path)


if __name__ == "__main__":
    # Larger axis labels, tick labels and panel titles than matplotlib's defaults, so they stay readable once LaTeX
    # scales the 12-inch figure to the text width. Arial is one of the fonts Elsevier recommends for artwork, and it is
    # set 1 pt larger than matplotlib's default DejaVu Sans would need, since its lowercase letters are smaller
    plt.rcParams.update({"font.family": "Arial", "font.size": 11, "axes.labelsize": 17, "axes.titlesize": 16,
                         "xtick.labelsize": 15, "ytick.labelsize": 15})
    # Saved as vector PDF, which Elsevier prefers for charts, with the fonts embedded as TrueType instead of
    # matplotlib's default Type 3
    plt.rcParams["pdf.fonttype"] = 42

    df = pd.concat([pd.read_csv(f) for f in glob.glob(RESULTS_GLOB)], ignore_index=True)

    # Casino_Oostende's realised error is not representative: its nearest neighbours are too far/dissimilar
    # for a meaningful spatial correction, which distorts the scale of the plot without adding information
    df = df[df["target_station"] != "Casino_Oostende"]

    os.makedirs(PLOTS_DIR, exist_ok=True)
    plot_reliability(df, f"{PLOTS_DIR}/confidence_combined_corr.pdf")
