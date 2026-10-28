"""
Group of functions to plot the impact of the training window on the two-stage regression with SARIMA
errors model, based on the joltik runs of hpc/submit_training_weeks_regsarima_joltik.sh, whose training times
were measured with the allocation of the model comparison.
Mirrors plot_training_time_impact_ARIMAX.py so the two models can be compared on the same kind of plots.
The plots are saved in the "plots" directory for further analysis and presentation.

python -m src.plotting.plot_training_history [--metric MAE|RMSE]
"""
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import glob

# Error metric to report, set from --metric. RMSE is derived per row as sqrt(MSE), so the
# reported value is the mean of the per-forecast RMSEs, as in plot_model_comparison.py.
METRIC = "RMSE"


def _file_name(name):
    """Keep the MAE filenames unchanged, suffix any other metric so it doesn't overwrite them."""
    suffix = "" if METRIC == "MAE" else f"_{METRIC.lower()}"
    return f"{name}{suffix}.pdf"


def load_training_weeks_results(results_dir="output/training_weeks_regression_sarima_errors_joltik"):
    """
    Load all RegressionSARIMAErrors_*_weeks.csv files of hpc/submit_training_weeks_regsarima_joltik.sh into one
    dataframe, with the columns renamed to the ones the plot functions use.

    Resulting columns:
    dataset, station, training_weeks,
    train_start, train_end,
    horizon, mae, mse, training_duration

    Input:
    ------
    results_dir: directory where the results csv files are stored

    Output:
    ------
    combined_df: DataFrame containing all the results from the csv files
    """
    pattern = os.path.join(results_dir, "RegressionSARIMAErrors_*_weeks.csv")
    files = sorted(glob.glob(pattern))

    dfs = []

    for f in files:
        df = pd.read_csv(f).rename(columns={
            "Dataset": "dataset", "Station": "station", "Train_start": "train_start", "Train_end": "train_end",
            "Horizon": "horizon", "MAE": "mae", "MSE": "mse", "Train_duration_seconds": "training_duration",
        })

        # The window length is only in the file name, RegressionSARIMAErrors_<weeks>_weeks.csv
        df["training_weeks"] = int(os.path.basename(f).rsplit("_", 2)[1])

        df["train_start"] = pd.to_datetime(df["train_start"])
        df["train_end"] = pd.to_datetime(df["train_end"])

        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)

    # Ensure correct dtypes
    combined_df["training_weeks"] = combined_df["training_weeks"].astype(int)
    combined_df["horizon"] = combined_df["horizon"].astype(int)
    combined_df["rmse"] = combined_df["mse"] ** 0.5

    return combined_df


def plot_performance_vs_training_time(df, output_dir):
    """
    Scatter plot of average error (METRIC) vs average training time, per number of training weeks.
    This helps to visualize the trade-off between training time and performance.

    Input:
    ------
    df: DataFrame containing the results (columns: dataset,station,training_weeks,train_start,train_end,horizon,mae,mse,training_duration)
    output_dir: directory where the plot will be saved
    """

    agg = (
        df.groupby("training_weeks")
        .agg(
            error_mean=(METRIC.lower(), "mean"),
            duration_mean=("training_duration", "mean"),
        )
        .reset_index()
    )

    plt.figure(figsize=(9, 5))

    plt.scatter(
        agg["duration_mean"],
        agg["error_mean"],
        s=60,
        color="C0",
        zorder=3
    )

    # Connect points to show evolution as training weeks increase
    agg_sorted = agg.sort_values("training_weeks")

    plt.plot(
        agg_sorted["duration_mean"],
        agg_sorted["error_mean"],
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        color="C0",
        zorder=2,
    )

    # Place the labels compactly inside the plot
    for _, row in agg.iterrows():
        plt.annotate(
            f'{int(row["training_weeks"])} {"weeks" if row["training_weeks"] != 1 else "week"}',  # short label
            (row["duration_mean"], row["error_mean"]),
            xytext=(3, 5),
            textcoords="offset points",
            fontsize=11,
            ha="left",
            va="bottom",
        )

    plt.xlabel("Mean training time (seconds)")
    plt.ylabel(f"Mean {METRIC} (°C)")

    # Subtle grid (zorder keeps the points and labels above the grid)
    plt.grid(True, linestyle="-", alpha=0.4, zorder=0)

    # Extra padding around the axes so the labels are not clipped
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    plt.xlim(left=x_min - 0.05 * (x_max - x_min), right=x_max + 0.05 * (x_max - x_min))
    plt.ylim(bottom=y_min - 0.05 * (y_max - y_min), top=y_max + 0.05 * (y_max - y_min))

    plt.tight_layout(pad=1.0)

    plt.savefig(os.path.join(output_dir, _file_name("performance_vs_training_time")))
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric", choices=["MAE", "RMSE"], default="RMSE")
    METRIC = parser.parse_args().metric

    output_dir = "plots/training_time_impact_regression_sarima_errors_joltik"
    os.makedirs(output_dir, exist_ok=True)
    # Larger axis labels and tick labels than matplotlib's 10 pt, so they stay readable once LaTeX scales the 9-inch
    # figure to the text width
    plt.rcParams.update({"axes.labelsize": 13, "xtick.labelsize": 12, "ytick.labelsize": 12})
    # Saved as vector PDF, which Elsevier prefers for charts, with the fonts embedded as TrueType instead of
    # matplotlib's default Type 3
    plt.rcParams["pdf.fonttype"] = 42

    results_dir = "output/training_weeks_regression_sarima_errors_joltik"
    df = load_training_weeks_results(results_dir)

    plot_performance_vs_training_time(df, output_dir)
