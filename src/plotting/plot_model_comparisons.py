"""
Group of functions to plot model comparisons based on the results of the experiments (evaluation/model_comparison_experiment.py).
Each function generates a specific plot to visualize different aspects of model performance, such as
overall performance, performance across forecast horizons, ...
The plots are saved in the "plots" directory for further analysis and presentation.

python -m src.plotting.plot_model_comparisons [--metric MAE|RMSE]
"""
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

# Error metric to report, set from --metric. RMSE is derived per row as sqrt(MSE), so the
# reported value is the mean of the per-forecast RMSEs, matching how the MAE is averaged.
METRIC = "MAE"

METRIC_CAPTIONS = {"MAE": "Mean absolute error", "RMSE": "Root mean squared error"}

# Zoom window in degrees for the zoomed horizon plot, per metric
ZOOM_RANGES = {"MAE": (0.2, 1.3), "RMSE": (0.3, 1.6)}


def _metric_label():
    return f"Mean {METRIC} (°C)"


def _plot_path(name):
    """Keep the MAE filenames unchanged, suffix any other metric so it doesn't overwrite them."""
    suffix = "" if METRIC == "MAE" else f"_{METRIC.lower()}"
    return f"plots/{name}{suffix}.png"


def plot_overall_model_performance(df):
    """
    1. Plot the overall error (METRIC) per model with error bars representing standard deviation across stations/datasets/horizons.

    Input:
    ------
    df: DataFrame containing all the results (columns: dataset,station,model,train_start,train_end,horizon,mae,mse)
    """
    summary_errors = df.groupby("Model")[METRIC].agg(["mean", "std"]).reset_index()

    plt.figure(figsize=(9, 5))
    plt.bar(
        summary_errors["Model"].map(_display_name),
        summary_errors["mean"],
        yerr=summary_errors["std"],
        capsize=5,
    )
    plt.xlabel("Model")
    plt.ylabel(_metric_label())
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(_plot_path("overall_model_performance"))
    plt.close()


# Model keys as written in the result csv files -> the names used in the paper
LATEX_MODEL_NAMES = {
    "RegressionSARIMAErrors": "RegSARIMA",
    "ARIMAX": "SARIMAX",
    "ARIMA": "SARIMA",
    "LinearRegression": "Neighbour regression",
    "Climatology": "Hourly climatology",
    "RF": "Random forest",
}

# Models that were run but are not part of the paper's comparison
EXCLUDED_MODELS = ["XGBoost"]

# One fixed colour per model: matplotlib's default cycle has only ten colours, so with more
# models two curves would share a colour
MODEL_COLOURS = {
    "RegressionSARIMAErrors": "black",
    "ARIMAX": "tab:blue",
    "LinearRegression": "tab:cyan",
    "IDW": "tab:olive",
    "TCN": "tab:red",
    "LSTM": "tab:purple",
    "Transformer": "tab:pink",
    "MLP": "tab:brown",
    "RF": "tab:orange",
    "ARIMA": "tab:green",
    "Persistence": "tab:gray",
    "Climatology": "goldenrod",
}


def _display_name(model):
    """Model key as written in the result csv files -> the name used in the paper."""
    return LATEX_MODEL_NAMES.get(model, model)


# Outage durations shown in the results table, as (hours, column label)
REPORTED_HORIZONS = [(4, "4\\,h"), (24, "24\\,h"), (168, "7\\,d"), (720, "30\\,d")]

# Models grouped by the information they are given, in table order
MODEL_FAMILIES = [
    ("Temporal only", ["ARIMA", "Persistence", "Climatology"]),
    ("Spatial only", ["LinearRegression", "IDW"]),
    ("Spatial and temporal",
     ["RegressionSARIMAErrors", "ARIMAX", "TCN", "LSTM", "MLP", "Transformer", "RF"]),
]


def _latex_cell(value, decimals, is_best):
    """Format a value as a math-mode LaTeX cell, bold if it is the best of its column."""
    text = f"{value:.{decimals}f}"
    return f"$\\mathbf{{{text}}}$" if is_best else f"${text}$"


def _family_order(errors):
    """Group the models of the results into the families of MODEL_FAMILIES, most accurate first."""
    accuracy = errors[[hours for hours, _ in REPORTED_HORIZONS]].mean(axis=1).sort_values()
    listed = {m for _, members in MODEL_FAMILIES for m in members}
    families = MODEL_FAMILIES + [("Other", [m for m in accuracy.index if m not in listed])]
    grouped = [(family, [m for m in accuracy.index if m in set(members)]) for family, members in families]
    return [(family, members) for family, members in grouped if members]


def generate_overall_results_table(df):
    """
    1. Generate a summary table with the error (METRIC) per outage duration and the training and prediction
    time, and print it as a booktabs LaTeX table in the style of the paper (tab:overall).

    Averaging the error over all horizons weights the table by the horizon grid rather than by the
    model, so the error is reported per outage duration instead.

    Input:
    ------
    df: DataFrame containing all the results, with the horizon in hours
    (columns: Dataset,Station,Model,Train_start,Train_end,Horizon,MAE,MSE)
    """
    errors = df.pivot_table(index="Model", columns="Horizon", values=METRIC, aggfunc="mean")
    cost = df.groupby("Model")[["Train_duration_seconds", "Forecast_duration_seconds"]].mean()

    missing = [label for hours, label in REPORTED_HORIZONS if hours not in errors.columns]
    if missing:
        raise ValueError(f"Horizons missing from the results: {missing}")

    # (column of the source table, decimals, mark the best) per table column, in table order
    columns = (
        [(errors[hours], 3, True) for hours, _ in REPORTED_HORIZONS]
        + [(cost["Train_duration_seconds"], 2, False), (cost["Forecast_duration_seconds"], 2, False)]
    )

    header = (
        ["\\textbf{Model}"]
        + [f"\\textbf{{{label}}}" for _, label in REPORTED_HORIZONS]
        + ["\\textbf{Train}", "\\textbf{Predict}"]
    )

    # Either ("group", family name) or ("model", cells), in table order
    body = []
    for family, models in _family_order(errors):
        body.append(("group", family))
        for model in models:
            cells = ["\\quad " + _display_name(model)]
            for values, decimals, mark_best in columns:
                is_best = mark_best and values[model] == values.min()
                cells.append(_latex_cell(values[model], decimals, is_best))
            body.append(("model", cells))

    # Pad every column so the ampersands line up in the .tex source
    rows = [cells for kind, cells in body if kind == "model"]
    widths = [max(len(row[i]) for row in [header] + rows) for i in range(len(header))]

    def format_row(cells):
        padded = [cell.ljust(width) for cell, width in zip(cells, widths)]
        return "    " + " & ".join(padded).rstrip() + " \\\\"

    def format_group(family):
        return f"    \\multicolumn{{{len(header)}}}{{l}}{{\\emph{{{family}}}}} \\\\"

    caption = (
        f"{METRIC_CAPTIONS[METRIC]} (\\textcelsius) per outage duration, averaged over stations and "
        "failure onsets. Models are grouped by the information they are given --- temporal is the "
        "past observations at the target station, spatial the concurrent observations at the "
        "neighbouring stations --- and ordered by their mean error within each group. Training and "
        "prediction times in seconds, measured on the UGent HPC."
    )

    lines = [
        "\\begin{table}[htb]",
        "  \\centering",
        f"  \\caption{{{caption}}}",
        "  \\label{tab:overall}",
        "  \\small",
        "  \\setlength{\\tabcolsep}{3pt}",
        f"  \\begin{{tabular*}}{{\\linewidth}}{{@{{\\extracolsep{{\\fill}}}}l{'r' * (len(header) - 1)}@{{}}}}",
        "    \\toprule",
        format_row(header),
        "    \\midrule",
    ]
    for index, (kind, value) in enumerate(body):
        if kind == "group":
            if index:
                lines.append("    \\addlinespace")
            lines.append(format_group(value))
        else:
            lines.append(format_row(value))
    lines += [
        "    \\bottomrule",
        "  \\end{tabular*}",
        "\\end{table}",
    ]

    print(f"\n=== Overall Model Performance ({METRIC}) ===")
    print(errors[[hours for hours, _ in REPORTED_HORIZONS]].join(cost).to_string())

    print("\nLaTeX code for the overall results table:")
    print("\n".join(lines))


# Dataset keys as written in the result csv files -> the names used in the paper, in table order
LATEX_DATASET_NAMES = {"KMI": "RMI", "TURKU": "TURCLIM", "SYNTHETIC": "Synthetic"}

# Outage durations shown per dataset in the appendix table, as (hours, column label)
DATASET_HORIZONS = [(4, "4\\,h"), (720, "30\\,d")]


def generate_dataset_results_table(df):
    """
    Generate a table with the error (METRIC) per dataset over a short and a long outage, and print it as
    a booktabs LaTeX table in the style of the paper (tab:perdataset).

    The best model is marked per column, so it can be read whether the ranking holds on every network.
    Values are compared as printed, so models that tie at the reported precision are all marked.
    The models keep the order of the overall table.

    Input:
    ------
    df: DataFrame containing all the results, with the horizon in hours
    (columns: Dataset,Station,Model,Train_start,Train_end,Horizon,MAE,MSE)
    """
    decimals = 3
    datasets = [d for d in LATEX_DATASET_NAMES if d in set(df["Dataset"])]
    errors = df.pivot_table(index="Model", columns=["Dataset", "Horizon"], values=METRIC, aggfunc="mean")
    columns = [(dataset, hours) for dataset in datasets for hours, _ in DATASET_HORIZONS]

    # Same grouping and order as the overall table
    overall = df.pivot_table(index="Model", columns="Horizon", values=METRIC, aggfunc="mean")
    families = _family_order(overall)

    rounded = errors[columns].round(decimals)
    best = rounded == rounded.min()

    header = ["\\textbf{Model}"] + [f"\\textbf{{{label}}}" for _ in datasets for _, label in DATASET_HORIZONS]
    body = []
    for _, models in families:
        body.append(None)
        for model in models:
            cells = [_display_name(model)] + [
                _latex_cell(errors.loc[model, column], decimals, best.loc[model, column]) for column in columns
            ]
            body.append(cells)

    # Pad every column so the ampersands line up in the .tex source
    rows = [cells for cells in body if cells]
    widths = [max(len(row[i]) for row in [header] + rows) for i in range(len(header))]

    def format_row(cells):
        padded = [cell.ljust(width) for cell, width in zip(cells, widths)]
        return "    " + " & ".join(padded).rstrip() + " \\\\"

    span = len(DATASET_HORIZONS)
    groups = " & ".join(
        f"\\multicolumn{{{span}}}{{c}}{{\\textbf{{{LATEX_DATASET_NAMES[d]}}}}}" for d in datasets
    )
    rules = "".join(
        f"\\cmidrule({'l' if i == len(datasets) - 1 else 'lr'}){{{2 + i * span}-{1 + (i + 1) * span}}}"
        for i in range(len(datasets))
    )

    stations = df.groupby("Dataset")["Station"].nunique()
    windows = df.groupby(["Dataset", "Station"])["Train_start"].nunique()
    # Acronyms keep their capitals, other names are lower case mid-sentence, as in the paper
    names = [name if name.isupper() else name.lower() for name in map(LATEX_DATASET_NAMES.get, datasets)]
    counts = [f"{name}: {stations[d]}" for name, d in zip(names, datasets)]
    counts[0] += " stations"
    caption = (
        f"{METRIC} (\\textcelsius) per network over 4-hour and 30-day outages ({', '.join(counts)}; "
        f"{windows.max()} windows per station). Bold marks the lowest error per column."
    )

    lines = [
        "\\begin{table}[!htbp]",
        "  \\centering",
        f"  \\caption{{{caption}}}",
        "  \\label{tab:perdataset}",
        "  \\small",
        "  \\setlength{\\tabcolsep}{4pt}",
        f"  \\begin{{tabular*}}{{\\linewidth}}{{@{{\\extracolsep{{\\fill}}}}l{'r' * len(columns)}@{{}}}}",
        "    \\toprule",
        f"    & {groups} \\\\",
        f"    {rules}",
        format_row(header),
        "    \\midrule",
    ]
    for index, cells in enumerate(body):
        if cells is None:
            if index:
                lines.append("    \\addlinespace")
        else:
            lines.append(format_row(cells))
    lines += [
        "    \\bottomrule",
        "  \\end{tabular*}",
        "\\end{table}",
    ]

    print(f"\n=== Model Performance per Dataset ({METRIC}) ===")
    print(errors[columns].to_string())

    print("\nLaTeX code for the per network results table:")
    print("\n".join(lines))


def plot_performance_vs_horizon(df):
    """
    2. Plot the error (METRIC) vs forecast horizon for each model.

    Input:
    ------
    df: DataFrame containing all the results (columns: Dataset,Station,Model,Train_start,Train_end,Horizon,MAE,MSE)
    """
    # Compute statistics
    summary = (
        df.groupby(["Model", "Horizon"])[METRIC]
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
            color=MODEL_COLOURS[model],
            label=_display_name(model)
        )

    plt.xlabel("Outage duration (days)")
    plt.ylabel(_metric_label())

    # Set both axis limits to start at 0 for better visualization
    plt.xlim(left=0)
    plt.ylim(bottom=0)

    plt.legend()
    plt.savefig(_plot_path("performance_vs_horizon"))
    plt.close()


def plot_performance_vs_horizon_with_zoom(df):
    """
    2. Zoomed version of the error vs horizon without ARIMA to highlight difference between the other models.

    Input:
    ------
    df: DataFrame containing all the results (columns: Dataset,Station,Model,Train_start,Train_end,Horizon,MAE,MSE)
    """
    summary = (
        df.groupby(["Model", "Horizon"])[METRIC]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    summary["ci95"] = 1.96 * (summary["std"] / (summary["count"] ** 0.5))

    _, (ax_top, ax_zoom) = plt.subplots(
        2, 1, sharex=True, figsize=(10, 8),
        gridspec_kw={"height_ratios": [3, 2]}
    )

    # Define zoom range
    zoom_min, zoom_max = ZOOM_RANGES[METRIC]

    for model in summary["Model"].unique():
        model_data = summary[summary["Model"] == model]

        ax_top.errorbar(
            model_data["Horizon"],
            model_data["mean"],
            yerr=model_data["ci95"],
            marker="o",
            capsize=4,
            color=MODEL_COLOURS[model],
            label=_display_name(model)
        )

        ax_zoom.errorbar(
            model_data["Horizon"],
            model_data["mean"],
            yerr=model_data["ci95"],
            marker="o",
            capsize=4,
            color=MODEL_COLOURS[model]
        )

    # Top plot
    ax_top.set_ylabel(_metric_label())
    ax_top.set_xlim(left=0)

    # Show zoom region
    ax_top.axhline(zoom_min, linestyle="--", color="gray")
    ax_top.axhline(zoom_max, linestyle="--", color="gray")

    # Zoom plot
    ax_zoom.set_ylim(zoom_min, zoom_max)
    ax_zoom.set_ylabel(_metric_label())
    ax_zoom.set_xlabel("Outage duration (days)")

    # Two columns keep the (now larger) legend from covering the baseline curves
    ax_top.legend(loc="upper left", ncol=2, fontsize=9)

    plt.tight_layout()
    plt.savefig(_plot_path("performance_vs_horizon_zoomed"))
    plt.close()


def plot_dataset_comparison(df):
    """
    3. Plot the error (METRIC) per dataset for each model.

    Input:
    ------
    df: DataFrame containing all the results (columns: Dataset,Station,Model,Train_start,Train_end,Horizon,MAE,MSE)
    """
    summary = df.groupby(["Dataset", "Model"])[METRIC].mean().reset_index()

    pivot = summary.pivot(index="Dataset", columns="Model", values=METRIC)
    pivot = pivot.rename(columns=_display_name)

    _, ax = plt.subplots(figsize=(8, 5))

    pivot.plot(kind="bar", ax=ax, width=0.9)

    ax.set_ylabel(_metric_label())
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
    plt.savefig(_plot_path("dataset_comparison"))
    plt.close()


def plot_station_variability(df):
    """
    4. Plot the error (METRIC) per station for each model.

    Input:
    ------
    df: DataFrame containing all the results (columns: Dataset,Station,Model,Train_start,Train_end,Horizon,MAE,MSE)
    """
    df["station_short"] = df["Station"].apply(lambda x: x[:12] + "…" if len(x) > 12 else x)

    summary = df.groupby(["station_short", "Model"])[METRIC].mean().reset_index()
    pivot = summary.pivot(index="station_short", columns="Model", values=METRIC)
    pivot = pivot.rename(columns=_display_name)

    _, ax = plt.subplots(figsize=(8, 5))

    pivot.plot(kind="bar", ax=ax)

    ax.set_ylabel(_metric_label())
    # ax.set_title("Model performance per station")

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(_plot_path("station_variability"))
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
        df.groupby(["Model", "DatasetType"])[METRIC]
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
    plt.ylabel(_metric_label())

    plt.legend()

    plt.tight_layout()
    plt.savefig(_plot_path("rural_vs_mixed_performance"))
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric", choices=["MAE", "RMSE"], default="MAE")
    METRIC = parser.parse_args().metric

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
    df = df[~df["Model"].isin(EXCLUDED_MODELS)]

    # Per-forecast RMSE, averaged the same way the MAE is
    df["RMSE"] = df["MSE"] ** 0.5

    # Create plot directory if it does not exist
    os.makedirs("plots", exist_ok=True)

    # The results table reports per outage duration, so it needs the horizon in hours
    generate_overall_results_table(df)
    generate_dataset_results_table(df)

    # Convert horizon to days for better readability
    df["Horizon"] = df["Horizon"] / 24

    plot_overall_model_performance(df)
    plot_performance_vs_horizon(df)
    plot_performance_vs_horizon_with_zoom(df)
    plot_dataset_comparison(df)
    plot_station_variability(df)
    plot_urban_vs_rural_comparison(df)
