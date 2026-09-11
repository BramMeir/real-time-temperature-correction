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


def plot_overall_model_performance(df):
    """
    1. Plot overall MAE per model with error bars representing standard deviation across stations/datasets/horizons.

    Input:
    ------
    df: DataFrame containing all the results (columns: dataset,station,model,train_start,train_end,horizon,mae,mse)
    """
    summary_mae = df.groupby("Model")["MAE"].agg(["mean", "std"]).reset_index()

    plt.figure(figsize=(9, 5))
    plt.bar(
        summary_mae["Model"].map(_display_name),
        summary_mae["mean"],
        yerr=summary_mae["std"],
        capsize=5,
    )
    plt.xlabel("Model")
    plt.ylabel("Mean MAE (°C)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig("plots/overall_model_performance.png")
    plt.close()


# Model keys as written in the result csv files -> the names used in the paper
LATEX_MODEL_NAMES = {
    "RegressionSARIMAErrors": "Regression + SARIMA errors",
    "LinearRegression": "Neighbour regression",
    "IDW": "IDW (concurrent)",
    "Climatology": "Hourly climatology",
}


def _display_name(model):
    """Model key as written in the result csv files -> the name used in the paper."""
    return LATEX_MODEL_NAMES.get(model, model)


def _latex_cell(mean, std, decimals, is_best):
    """Format 'mean ± std' as a math-mode LaTeX cell, bold if it is the best value."""
    value = f"{mean:.{decimals}f} \\pm {std:.{decimals}f}"
    return f"$\\mathbf{{{value}}}$" if is_best else f"${value}$"


def generate_overall_results_table(df):
    """
    1. Generate a summary table with MAE, MSE, training and prediction time (mean ± std)
    and print it as a booktabs LaTeX table in the style of the paper (tab:overall).

    Input:
    ------
    df: DataFrame containing all the results (columns: dataset,station,model,train_start,train_end,horizon,mae,mse)
    """
    summary = (
        df.groupby("Model")
        .agg(
            mae_mean=("MAE", "mean"),
            mae_std=("MAE", "std"),
            mse_mean=("MSE", "mean"),
            mse_std=("MSE", "std"),
            train_time_mean=("Train_duration_seconds", "mean"),
            train_time_std=("Train_duration_seconds", "std"),
            forecast_time_mean=("Forecast_duration_seconds", "mean"),
            forecast_time_std=("Forecast_duration_seconds", "std")
        )
        .sort_values(by="mae_mean")
        .reset_index()
    )

    # (mean column, std column, decimals) per metric column, in table order
    metrics = [
        ("mae_mean", "mae_std", 3),
        ("mse_mean", "mse_std", 3),
        ("train_time_mean", "train_time_std", 2),
        ("forecast_time_mean", "forecast_time_std", 2),
    ]

    header = [
        "\\textbf{Model}",
        "\\textbf{MAE (\\textcelsius)}",
        "\\textbf{MSE (${}^{\\circ}$C$^2$)}",
        "\\textbf{Train (s)}",
        "\\textbf{Predict (s)}",
    ]

    rows = []
    for _, r in summary.iterrows():
        row = [LATEX_MODEL_NAMES.get(r["Model"], r["Model"])]
        for mean_col, std_col, decimals in metrics:
            is_best = r[mean_col] == summary[mean_col].min()
            row.append(_latex_cell(r[mean_col], r[std_col], decimals, is_best))
        rows.append(row)

    # Pad every column so the ampersands line up in the .tex source
    widths = [max(len(row[i]) for row in [header] + rows) for i in range(len(header))]

    def format_row(cells):
        padded = [cell.ljust(width) for cell, width in zip(cells, widths)]
        return "    " + " & ".join(padded).rstrip() + " \\\\"

    caption = (
        "Mean error metrics across all stations, datasets, and horizons "
        "($\\pm$ one standard deviation over experiments). "
        "Training and prediction times measured on the UGent HPC."
    )

    lines = [
        "\\begin{table}[htb]",
        "  \\centering",
        f"  \\caption{{{caption}}}",
        "  \\label{tab:overall}",
        "  \\begin{tabular}{lrrrr}",
        "    \\toprule",
        format_row(header),
        "    \\midrule",
        *[format_row(row) for row in rows],
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
    ]

    print("\n=== Overall Model Performance ===")
    print(summary[["Model"] + [m[0] for m in metrics]].to_string(index=False))

    print("\nLaTeX code for the overall results table:")
    print("\n".join(lines))


def plot_performance_vs_horizon(df):
    """
    2. Plot MAE vs forecast horizon for each model.

    Input:
    ------
    df: DataFrame containing all the results (columns: Dataset,Station,Model,Train_start,Train_end,Horizon,MAE,MSE)
    """
    # Compute statistics
    summary = (
        df.groupby(["Model", "Horizon"])["MAE"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    # 95% confidence interval
    summary["ci95"] = 1.96 * (summary["std"] / (summary["count"] ** 0.5))

    plt.figure(figsize=(8, 5))

    for model in summary["Model"].unique():
        model_data = summary[summary["Model"] == model]

        plt.errorbar(
            model_data["Horizon"],
            model_data["mean"],
            yerr=model_data["ci95"],
            marker="o",
            capsize=4,
            label=_display_name(model)
        )

    plt.xlabel("Outage duration (days)")
    plt.ylabel("Mean MAE (°C)")

    # Set both axis limits to start at 0 for better visualization
    plt.xlim(left=0)
    plt.ylim(bottom=0)

    plt.legend()
    plt.savefig("plots/performance_vs_horizon.png")
    plt.close()


def plot_performance_vs_horizon_with_zoom(df):
    """
    2. Zoomed version of MAE vs horizon without ARIMA to highlight difference between the other models.

    Input:
    ------
    df: DataFrame containing all the results (columns: Dataset,Station,Model,Train_start,Train_end,Horizon,MAE,MSE)
    """
    summary = (
        df.groupby(["Model", "Horizon"])["MAE"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    summary["ci95"] = 1.96 * (summary["std"] / (summary["count"] ** 0.5))

    _, (ax_top, ax_zoom) = plt.subplots(
        2, 1, sharex=True, figsize=(10, 8),
        gridspec_kw={"height_ratios": [2, 1]}
    )

    # Define zoom range
    zoom_min = 0.2
    zoom_max = 1.3

    for model in summary["Model"].unique():
        model_data = summary[summary["Model"] == model]

        ax_top.errorbar(
            model_data["Horizon"],
            model_data["mean"],
            yerr=model_data["ci95"],
            marker="o",
            capsize=4,
            label=_display_name(model)
        )

        ax_zoom.errorbar(
            model_data["Horizon"],
            model_data["mean"],
            yerr=model_data["ci95"],
            marker="o",
            capsize=4
        )

    # Top plot
    ax_top.set_ylabel("Mean MAE (°C)")
    ax_top.set_xlim(left=0)

    # Show zoom region
    ax_top.axhline(zoom_min, linestyle="--", color="gray")
    ax_top.axhline(zoom_max, linestyle="--", color="gray")

    # Zoom plot
    ax_zoom.set_ylim(zoom_min, zoom_max)
    ax_zoom.set_ylabel("Mean MAE (°C)")
    ax_zoom.set_xlabel("Outage duration (days)")

    # Two columns keep the (now larger) legend from covering the baseline curves
    ax_top.legend(loc="upper left", ncol=2, fontsize=9)

    plt.tight_layout()
    plt.savefig("plots/performance_vs_horizon_zoomed.png")
    plt.close()


def plot_dataset_comparison(df):
    """
    3. Plot MAE per dataset for each model.

    Input:
    ------
    df: DataFrame containing all the results (columns: Dataset,Station,Model,Train_start,Train_end,Horizon,MAE,MSE)
    """
    summary = df.groupby(["Dataset", "Model"])["MAE"].mean().reset_index()

    pivot = summary.pivot(index="Dataset", columns="Model", values="MAE")
    pivot = pivot.rename(columns=_display_name)

    _, ax = plt.subplots(figsize=(8, 5))

    pivot.plot(kind="bar", ax=ax, width=0.9)

    ax.set_ylabel("Mean MAE (°C)")
    # ax.set_title("Model performance per dataset")

    # Add value labels on top of bars
    for container in ax.containers:
        ax.bar_label(
            container,
            fmt="%.2f",
            padding=3,
            fontsize=8
        )

    # Remove the unncessary x-axis label ("dataset")
    ax.set_xlabel(None)

    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("plots/dataset_comparison.png")
    plt.close()


def plot_station_variability(df):
    """
    4. Plot MAE per station for each model.

    Input:
    ------
    df: DataFrame containing all the results (columns: Dataset,Station,Model,Train_start,Train_end,Horizon,MAE,MSE)
    """
    df["station_short"] = df["Station"].apply(lambda x: x[:12] + "…" if len(x) > 12 else x)

    summary = df.groupby(["station_short", "Model"])["MAE"].mean().reset_index()
    pivot = summary.pivot(index="station_short", columns="Model", values="MAE")
    pivot = pivot.rename(columns=_display_name)

    _, ax = plt.subplots(figsize=(8, 5))

    pivot.plot(kind="bar", ax=ax)

    ax.set_ylabel("Mean MAE (°C)")
    # ax.set_title("Model performance per station")

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("plots/station_variability.png")
    plt.close()


def plot_urban_vs_rural_comparison(df):
    """
    5.
    Compare model performance between:
    - rural-only dataset (KMI)
    - mixed urban/rural datasets (Turku + Synthetic)

    Input:
    ------
    df: DataFrame containing all the results (columns: Dataset,Station,Model,Train_start,Train_end,Horizon,MAE,MSE)
    """
    df = df.copy()

    # Label dataset type
    df["DatasetType"] = df["Dataset"].apply(
        lambda x: "Rural-only" if x == "KMI" else "Mixed urban/rural"
    )

    # Aggregate
    summary = (
        df.groupby(["Model", "DatasetType"])["MAE"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    summary["ci95"] = 1.96 * (summary["std"] / (summary["count"] ** 0.5))

    models = summary["Model"].unique()

    rural = summary[summary["DatasetType"] == "Rural-only"]
    mixed = summary[summary["DatasetType"] == "Mixed urban/rural"]

    x = range(len(models))
    width = 0.35

    plt.figure(figsize=(9, 5))

    plt.bar(
        [i - width / 2 for i in x],
        rural["mean"],
        width,
        yerr=rural["ci95"],
        capsize=4,
        label="Rural stations only"
    )

    plt.bar(
        [i + width / 2 for i in x],
        mixed["mean"],
        width,
        yerr=mixed["ci95"],
        capsize=4,
        label="Mixed rural/urban"
    )

    plt.xticks(
        x,
        [_display_name(m) for m in models],
        rotation=30,
        ha="right",
    )
    plt.ylabel("Mean MAE (°C)")

    plt.legend()

    plt.tight_layout()
    plt.savefig("plots/rural_vs_mixed_performance.png")
    plt.close()


if __name__ == "__main__":
    output_dir = "output/models_comparison"

    # Load all the csv files generated by the model comparison experiments
    csv_files = [
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.endswith("8_weeks.csv")
    ]

    # Read and concatenate all the csv files into a single DataFrame
    dfs = [pd.read_csv(f) for f in csv_files]

    df = pd.concat(dfs, ignore_index=True)

    # Convert horizon to days for better readability
    df["Horizon"] = df["Horizon"] / 24

    # Create plot directory if it does not exist
    os.makedirs("plots", exist_ok=True)

    plot_overall_model_performance(df)
    plot_performance_vs_horizon(df)
    plot_performance_vs_horizon_with_zoom(df)
    plot_dataset_comparison(df)
    plot_station_variability(df)
    plot_urban_vs_rural_comparison(df)

    generate_overall_results_table(df)
