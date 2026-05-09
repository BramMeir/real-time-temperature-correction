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

    plt.figure()
    plt.bar(summary_mae["Model"], summary_mae["mean"], yerr=summary_mae["std"], capsize=5)
    plt.xlabel("Model")
    plt.ylabel("Gemiddelde MAE (°C)")
    plt.savefig("plots/overall_model_performance.png")
    plt.close()


def generate_overall_results_table(df):
    """
    1. Generate a summary table with
        MAE = mean ± std
        MSE = mean ± std
    and export it to LaTeX.
    Input:
    ------
    df: DataFrame containing all the results (columns: dataset,station,model,train_start,train_end,horizon,mae,mse)
    """
    # Compute statistics
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
        .reset_index()
    )

    # Determine best values for highlighting
    best_mae = summary["mae_mean"].min()
    best_mse = summary["mse_mean"].min()
    best_train_time = summary["train_time_mean"].min()
    best_forecast_time = summary["forecast_time_mean"].min()

    # Helper function to format values and highlight best ones
    def format_value(mean, std, best):
        value = f"{mean:.3f} ± {std:.3f}"
        return f"\\textbf{{{value}}}" if mean == best else value
    
    def format_time(mean, std, best):
        formatted = f"{mean:.2f} ± {std:.2f} s"
        return f"\\textbf{{{formatted}}}" if mean == best else formatted

    # Sort by MAE
    summary = summary.sort_values(by="mae_mean").reset_index(drop=True)

    # Format as "mean ± std" and highlight best values
    summary["MAE"] = summary.apply(
        lambda r: format_value(r.mae_mean, r.mae_std, best_mae), axis=1
    )
    summary["MSE"] = summary.apply(
        lambda r: format_value(r.mse_mean, r.mse_std, best_mse), axis=1
    )

    summary["Training Time"] = summary.apply(
        lambda r: format_time(r.train_time_mean, r.train_time_std, best_train_time), axis=1
    )

    summary["Forecasting Time"] = summary.apply(
        lambda r: format_time(r.forecast_time_mean, r.forecast_time_std, best_forecast_time), axis=1
    )

    # Keep only formatted columns
    summary_table = summary[["Model", "MAE", "MSE", "Training Time", "Forecasting Time"]]

    print("\n=== Overall Model Performance ===")
    print(summary_table)

    # Export LaTeX table
    latex_code = summary_table.to_latex(
        index=False,
        escape=False,   # important for ± symbol
        column_format="lccccc"
    )

    print("\nLaTeX code for the overall results table:")
    print(latex_code)


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
            label=model
        )

    plt.xlabel("Voorspellingshorizon (dagen)")
    plt.ylabel("Gemiddelde MAE (°C)")

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
            label=model
        )

        ax_zoom.errorbar(
            model_data["Horizon"],
            model_data["mean"],
            yerr=model_data["ci95"],
            marker="o",
            capsize=4
        )

    # Top plot
    ax_top.set_ylabel("Gemiddelde MAE (°C)")
    ax_top.set_xlim(left=0)

    # Show zoom region
    ax_top.axhline(zoom_min, linestyle="--", color="gray")
    ax_top.axhline(zoom_max, linestyle="--", color="gray")

    # Zoom plot
    ax_zoom.set_ylim(zoom_min, zoom_max)
    ax_zoom.set_ylabel("Gemiddelde MAE (°C)")
    ax_zoom.set_xlabel("Voorspellingshorizon (dagen)")

    ax_top.legend(loc="upper left")

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

    _, ax = plt.subplots(figsize=(8, 5))

    pivot.plot(kind="bar", ax=ax, width=0.9)

    ax.set_ylabel("Gemiddelde MAE (°C)")
    # ax.set_title("Modelprestaties per dataset")

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

    _, ax = plt.subplots(figsize=(8, 5))

    pivot.plot(kind="bar", ax=ax)

    ax.set_ylabel("Gemiddelde MAE (°C)")
    # ax.set_title("Modelprestaties per station")

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
        label="Enkel landelijke stations"
    )

    plt.bar(
        [i + width / 2 for i in x],
        mixed["mean"],
        width,
        yerr=mixed["ci95"],
        capsize=4,
        label="Gemengd landelijk/stedelijk"
    )

    plt.xticks(x, models, rotation=0)
    plt.ylabel("Gemiddelde MAE (°C)")

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
