"""
Plot the comparison between the two-stage regression with SARIMA errors model trained on all
neighbouring stations and the same model trained on the LASSO-selected subset, based on the results
of determine_LASSO_performance_RegressionSARIMAErrors.py. Mirrors plot_LASSO_selection_improvement.py.
The plots are saved in the "plots" directory for further analysis and presentation.

python -m src.plotting.plot_LASSO_performance_RegressionSARIMAErrors [--metric MAE|RMSE]
"""
import argparse
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Error metric to report, set from --metric. RMSE is derived per row as sqrt(MSE), so the
# reported value is the mean of the per-forecast RMSEs, as in plot_model_comparisons.py.
METRIC = "MAE"


def load_results(results_dir="output/LASSO_performance_regression_sarima_errors"):
    """
    Load all results_*_weeks.csv files into one dataframe.

    Input
    -----
    results_dir: directory where the results csv files are stored

    Output
    ------
    df: DataFrame containing all the results from the csv files
    """
    files = sorted(Path(results_dir).glob("results_*_weeks.csv"))

    if not files:
        raise RuntimeError(f"No result files found in {results_dir}.")

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["rmse"] = df["mse"] ** 0.5

    return df


def plot_improvement_per_station(df, output_dir):
    """
    Bar plot of the error (METRIC) difference (all stations minus LASSO) per target station, averaged over
    repeats and horizons. A positive bar means LASSO improved on using all stations.

    Input
    -----
    df: DataFrame containing the results
    output_dir: directory where the plot will be saved
    """
    per_station = (
        df.pivot_table(index=["dataset", "station"], columns="arm", values=METRIC.lower(), aggfunc="mean")
        .reset_index()
    )

    per_station["difference"] = per_station["all"] - per_station["lasso"]
    per_station = per_station.sort_values("difference", ascending=False)

    colors = per_station["difference"].apply(lambda x: "tab:green" if x > 0 else "tab:red")

    plt.figure(figsize=(10, 6))

    plt.bar(
        per_station["station"],
        per_station["difference"],
        color=colors,
        alpha=0.85
    )

    baseline_error = per_station["all"].mean()
    lasso_error = per_station["lasso"].mean()
    mean_improvement = per_station["difference"].mean()

    plt.text(
        0.99, 0.98,
        (
            f"Mean {METRIC} (all stations): {baseline_error:.3f} °C\n"
            f"Mean {METRIC} (LASSO selection): {lasso_error:.3f} °C\n"
            f"Mean improvement: {mean_improvement:.3f} °C"
        ),
        transform=plt.gca().transAxes,
        fontsize=11,
        horizontalalignment="right",
        verticalalignment="top",
        bbox=dict(boxstyle="round", alpha=0.5, facecolor="white", edgecolor="gray")
    )

    plt.axhline(0, linestyle="--")

    plt.ylabel(f"{METRIC} improvement over using all stations (°C)")
    plt.xlabel("Target stations (ranked by improvement)")
    plt.xticks([])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{METRIC.lower()}_improvement_per_station.png"))
    plt.close()


def plot_error_vs_horizon(df, output_dir):
    """
    Mean error (METRIC) per horizon, one line per arm, to see whether LASSO's effect (if any) depends on lead time.

    Input
    -----
    df: DataFrame containing the results
    output_dir: directory where the plot will be saved
    """
    plt.figure(figsize=(9, 5))

    for arm, group in df.groupby("arm"):
        agg = group.groupby("horizon")[METRIC.lower()].mean().reset_index()
        plt.plot(agg["horizon"], agg[METRIC.lower()], marker="o", label=arm)

    plt.xlabel("Forecast horizon (hours)")
    plt.ylabel(f"Mean {METRIC} (°C)")
    plt.legend(title="Stations used")
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{METRIC.lower()}_vs_horizon_per_arm.png"))
    plt.close()


def plot_training_duration_vs_nr_stations(df, output_dir):
    """
    Scatter plot of training duration against the number of stations used, pooling both arms. The
    LASSO arm naturally spans a range of station counts (it varies per station and repeat), which
    makes it possible to see directly whether fewer stations actually train faster. Mirrors the
    ARIMAX training_time_versus_nr_exog_stations.png plot, but for the two-stage model.

    Input
    -----
    df: DataFrame containing the results
    output_dir: directory where the plot will be saved

    Output
    ------
    correlation: Pearson correlation between nr_stations and training_duration, for use in the
        accompanying explanatory note
    """
    # One row per model fit: training_duration and nr_stations repeat across horizons otherwise
    per_fit = df.drop_duplicates(subset=["dataset", "station", "repeat_id", "arm"])

    correlation = per_fit["nr_stations"].corr(per_fit["training_duration"])

    plt.figure(figsize=(9, 5))

    plt.scatter(
        per_fit["nr_stations"],
        per_fit["training_duration"],
        alpha=0.4,
        s=20
    )

    plt.text(
        0.99, 0.98,
        f"Correlation: {correlation:.2f}",
        transform=plt.gca().transAxes,
        fontsize=11,
        horizontalalignment="right",
        verticalalignment="top",
        bbox=dict(boxstyle="round", alpha=0.5, facecolor="white", edgecolor="gray")
    )

    plt.xlabel("Number of exogenous stations used")
    plt.ylabel("Training duration (s)")
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_duration_vs_nr_stations.png"))
    plt.close()

    return correlation


def generate_lasso_properties_table(df, duration_vs_stations_correlation):
    """
    Generate a LaTeX table with LASSO selection properties, in the same compact style as
    plot_LASSO_selection_improvement.py's generate_lasso_properties_table.

    Input
    -----
    df: DataFrame containing the results
    """
    lasso = df[df["arm"] == "lasso"]
    all_stations = df[df["arm"] == "all"]

    total_stations = df.groupby(["dataset", "station"])["nr_stations"].max().mean()

    mean_nr_stations = lasso["nr_stations"].mean()
    median_nr_stations = lasso["nr_stations"].median()
    minimum_nr_stations = lasso["nr_stations"].min()
    maximum_nr_stations = lasso["nr_stations"].max()

    mean_selection_time = lasso["lasso_selection_duration"].mean()
    mean_training_time_all = all_stations["training_duration"].mean()
    mean_training_time_lasso = lasso["training_duration"].mean()

    mean_speedup = mean_training_time_all / (mean_selection_time + mean_training_time_lasso)

    latex_code = f"""
    \\begin{{tabular*}}{{0.8\\textwidth}}{{l@{{\\extracolsep{{\\fill}}}}c}}
    \\toprule
    \\multicolumn{{2}}{{l}}{{\\textbf{{LASSO selection properties}}}} \\\\
    \\midrule
    Total number of available stations & {total_stations:.0f} \\\\
    Mean number of selected stations & {mean_nr_stations:.1f} \\\\
    Range of selected stations & {minimum_nr_stations:.0f} - {maximum_nr_stations:.0f} (median: {median_nr_stations:.0f}) \\\\
    \\addlinespace[2pt]
    LASSO selection time (s) & {mean_selection_time:.2f} \\\\
    Training time (s) & {mean_training_time_lasso:.2f} \\\\
    Mean speed-up vs. all stations & $\\times${mean_speedup:.1f} \\\\
    \\bottomrule
    \\end{{tabular*}}
    """

    print("\n=== LASSO Properties Table ===\n")
    print(latex_code)

    print(
        f"Unlike ARIMAX, where every exogenous station adds a coefficient jointly optimised inside\n"
        f"the SARIMAX likelihood, this model's expensive step (the residual SARIMA, order (3,0,0)x\n"
        f"(1,0,0,24)) always has the same 5 parameters regardless of how many stations feed stage one.\n"
        f"Only the OLS regression scales with station count, and OLS is fast enough that it barely\n"
        f"registers: correlation between number of stations and training duration is "
        f"{duration_vs_stations_correlation:.2f} (near zero), so trimming stations can't meaningfully\n"
        f"speed up training here, unlike for ARIMAX.\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric", choices=["MAE", "RMSE"], default="MAE")
    METRIC = parser.parse_args().metric

    output_dir = "plots/LASSO_regression_sarima_errors"
    os.makedirs(output_dir, exist_ok=True)

    df = load_results()

    plot_improvement_per_station(df, output_dir)
    plot_error_vs_horizon(df, output_dir)

    # Training duration and selection properties do not depend on the metric, so only the MAE run
    # draws and prints them
    if METRIC == "MAE":
        duration_vs_stations_correlation = plot_training_duration_vs_nr_stations(df, output_dir)
        generate_lasso_properties_table(df, duration_vs_stations_correlation)
