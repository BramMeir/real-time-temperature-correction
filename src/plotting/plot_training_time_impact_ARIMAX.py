"""
Group of functions to plot model comparisons based on the results of the experiments (evaluation/model_comparison_experiment.py).
Each function generates a specific plot to visualize different aspects of model performance, such as
overall performance, performance across forecast horizons, ...
The plots are saved in the "plots" directory for further analysis and presentation.

python -m src.plotting.plot_model_comparisons
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import glob


def load_training_weeks_results(results_dir="output/optimal_nr_training_weeks"):
    """
    Load all results_*_weeks.csv files into one dataframe.

    Expected columns:
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
    # Include both the seed42 and seed47 results, for comparison
    pattern = os.path.join(results_dir, "results_*_weeks.csv")
    files = sorted(glob.glob(pattern))

    dfs = []

    for f in files:
        df = pd.read_csv(f)

        # Infer training weeks from filename, because this is not included in the csv files directly
        if "training_weeks" not in df.columns:
            weeks = int(os.path.basename(f).split("_")[1])
            df["training_weeks"] = weeks

        # Parse timestamps (optional but clean)
        df["train_start"] = pd.to_datetime(df["train_start"])
        df["train_end"] = pd.to_datetime(df["train_end"])

        # Make sure training_duration is in seconds (it is stored as a timedelta in the csv)
        df["training_duration"] = pd.to_timedelta(df["training_duration"]).dt.total_seconds()

        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)

    # Ensure correct dtypes
    combined_df["training_weeks"] = combined_df["training_weeks"].astype(int)
    combined_df["horizon"] = combined_df["horizon"].astype(int)

    # Filter on synthetic dataset
    # combined_df = combined_df[combined_df["dataset"] == "TURKU"]

    return combined_df


def plot_mae_vs_training_weeks(df, output_dir):
    """
    Average MAE in function of the number of training weeks, with error bands for standard deviation.

    Input:
    ------
    df: DataFrame containing the results (columns: dataset,station,training_weeks,train_start,train_end,horizon,mae,mse,training_duration)
    output_dir: directory where the plot will be saved
    """

    summary = (
        df.groupby("training_weeks")["mae"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    # 95% confidence interval
    summary["ci95"] = 1.96 * (
        summary["std"] / (summary["count"] ** 0.5)
    )

    plt.figure(figsize=(9, 5))

    plt.errorbar(
        summary["training_weeks"],
        summary["mean"],
        yerr=summary["ci95"],
        marker="o",
        capsize=4,
    )

    # Zoom in on the relevant region
    ymin = summary["mean"].min() * 0.95
    ymax = summary["mean"].min() * 1.25

    plt.ylim(ymin, ymax)
    plt.xlim(left=0)

    plt.xlabel("Number of training weeks")
    plt.ylabel("Mean MAE (°C)")

    # Subtle grid
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "performance_vs_training_weeks.png"))
    plt.close()


def plot_training_time_vs_weeks(df, output_dir):
    """
    Average trainingtime in function of the number of training weeks, with error bands for 95% confidence interval.

    Input:
    ------
    df: DataFrame containing the results (columns: dataset,station,training_weeks,train_start,train_end,horizon,mae,mse,training_duration)
    output_dir: directory where the plot will be saved
    """

    agg = (
        df.groupby("training_weeks")["training_duration"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    # 95% confidence interval
    agg["ci95"] = 1.96 * (
        agg["std"] / (agg["count"] ** 0.5)
    )

    plt.figure(figsize=(9, 5))

    plt.errorbar(
        agg["training_weeks"],
        agg["mean"],
        yerr=agg["ci95"],
        marker="o",
        capsize=4,
    )

    plt.xlabel("Number of training weeks")
    plt.ylabel("Mean training time (seconds)")

    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, "training_time_vs_training_weeks.png"))
    plt.close()


def plot_performance_vs_training_time(df, output_dir):
    """
    Scatter plot of average MAE vs average training time, per number of training weeks.
    This helps to visualize the trade-off between training time and performance.

    Input:
    ------
    df: DataFrame containing the results (columns: dataset,station,training_weeks,train_start,train_end,horizon,mae,mse,training_duration)
    output_dir: directory where the plot will be saved
    """

    agg = (
        df.groupby("training_weeks")
        .agg(
            mae_mean=("mae", "mean"),
            duration_mean=("training_duration", "mean"),
        )
        .reset_index()
    )

    plt.figure(figsize=(9, 5))

    plt.scatter(
        agg["duration_mean"],
        agg["mae_mean"],
        s=60,
        color="C0",
        zorder=3
    )

    # Connect points to show evolution as training weeks increase
    agg_sorted = agg.sort_values("training_weeks")

    plt.plot(
        agg_sorted["duration_mean"],
        agg_sorted["mae_mean"],
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
            (row["duration_mean"], row["mae_mean"]),
            xytext=(3, 5),
            textcoords="offset points",
            fontsize=9,
            ha="left",
            va="bottom",
        )

    plt.xlabel("Mean training time (seconds)")
    plt.ylabel("Mean MAE (°C)")

    # Subtle grid (zorder keeps the points and labels above the grid)
    plt.grid(True, linestyle="-", alpha=0.4, zorder=0)

    # Extra padding around the axes so the labels are not clipped
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    plt.xlim(left=x_min - 0.05 * (x_max - x_min), right=x_max + 0.05 * (x_max - x_min))
    plt.ylim(bottom=y_min - 0.05 * (y_max - y_min), top=y_max + 0.05 * (y_max - y_min))

    plt.tight_layout(pad=1.0)

    plt.savefig(os.path.join(output_dir, "performance_vs_training_time.png"))
    plt.close()


def plot_mae_vs_weeks_per_horizon(df, output_dir):
    """
    MAE in function of the number of training weeks, per forecast horizon. This helps to understand
    the impact of training time on performance for different forecast horizons.

    Input:
    ------
    df: DataFrame containing the results (columns: dataset,station,training_weeks,train_start,train_end,horizon,mae,mse,training_duration)
    output_dir: directory where the plot will be saved
    """

    plt.figure(figsize=(9, 5))

    horizons = sorted(df["horizon"].unique())

    for i, horizon in enumerate(horizons):
        group = df[df["horizon"] == horizon]

        agg = (
            group.groupby("training_weeks")["mae"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )

        plt.errorbar(
            agg["training_weeks"],
            agg["mean"],
            marker="o",
            capsize=3,
            label=f"{horizon} h",
        )

    plt.xlabel("Number of training weeks")
    plt.ylabel("Mean MAE (°C)")

    plt.grid(True, linestyle="-", alpha=0.3)

    plt.legend(
        title="Outage duration",
        ncol=2,
    )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "performance_vs_training_weeks_per_horizon.png"))
    plt.close()


if __name__ == "__main__":
    output_dir = "plots/training_time_impact"
    os.makedirs(output_dir, exist_ok=True)

    results_dir = "output/optimal_nr_training_weeks_doduo"
    df = load_training_weeks_results(results_dir)

    # Make the plots
    plot_mae_vs_training_weeks(df, output_dir)
    plot_training_time_vs_weeks(df, output_dir)
    plot_performance_vs_training_time(df, output_dir)
    plot_mae_vs_weeks_per_horizon(df, output_dir)
