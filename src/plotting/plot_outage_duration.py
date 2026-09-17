"""
Plot the outage duration experiment (src/scripts/outage_duration_experiment.py): as the forecast
horizon grows, ARIMAX's and the two-stage model's full forecasts converge onto their own
exogenous-beta-only floor. Both the cumulative error (averaged over hours 1..h, the practically
relevant "average error over an outage of length h") and the pointwise error (at hour h alone,
which isolates the AR contribution decaying to zero) are plotted, since they tell a different
story about where the convergence happens. The plots are saved in the "plots/outage_duration"
directory.

python -m src.plotting.plot_outage_duration
"""
import os
import pandas as pd
import matplotlib.pyplot as plt

# Fixed colour per model family, matching src/plotting/plot_short_horizon_comparison.py
ARIMAX_COLOUR = "#eb6834"
TWOSTAGE_COLOUR = "#2a78d6"

OUTPUT_FILE = "output/outage_duration_experiment.csv"
PLOT_DIR = "plots/outage_duration"

# Datasets shown side by side in the faceted comparison plot
FACETED_DATASETS = ["KMI", "TURKU"]

# Which error kind to plot, and the resulting file name / title
KINDS = {
    "cumulative": {
        "suffix": "cumulative_mae",
        "file_name": "outage_duration_cumulative.png",
        "title": "Full-model forecasts converge onto their exogenous-beta floor (cumulative error)",
    },
    "pointwise": {
        "suffix": "pointwise_mae",
        "file_name": "outage_duration_pointwise.png",
        "title": "Full-model forecasts converge onto their exogenous-beta floor (pointwise error)",
    },
}


def load_results(output_file, dataset_filter=None):
    """
    Load the outage duration experiment results and average them over stations and repeats.

    Input
    -----
    output_file: Path to the CSV written by src/scripts/outage_duration_experiment.py
    dataset_filter: Restrict to this dataset's rows, or None to keep every dataset in the file

    Output
    ------
    Returns a DataFrame indexed by horizon with the mean of every cumulative and pointwise column.
    """
    df = pd.read_csv(output_file)

    if dataset_filter is not None:
        df = df[df["dataset"] == dataset_filter]

    value_columns = [col for col in df.columns if col.endswith("_mae")]

    return df.groupby("horizon")[value_columns].mean()


def plot_outage_duration(summary, plot_dir, kind):
    """
    Plot the full-model forecasts against their beta-only floors as the forecast horizon grows,
    for one error kind (cumulative or pointwise).

    Input
    -----
    summary: DataFrame indexed by horizon, as returned by load_results
    plot_dir: Directory to write the figure to
    kind: "cumulative" or "pointwise", picking which columns and plot to produce
    """
    spec = KINDS[kind]
    suffix = spec["suffix"]
    horizons = summary.index.tolist()

    plt.figure(figsize=(8, 5))

    plt.plot(horizons, summary[f"arimax_full_{suffix}"], marker="o", color=ARIMAX_COLOUR, label="ARIMAX (full model)")
    plt.plot(horizons, summary[f"arimax_beta_{suffix}"], linestyle="--", color=ARIMAX_COLOUR, label=r"ARIMAX $\beta$ only")
    plt.plot(horizons, summary[f"twostage_full_{suffix}"], marker="o", color=TWOSTAGE_COLOUR, label="Two-stage (full model)")
    plt.plot(horizons, summary[f"ols_beta_{suffix}"], linestyle="--", color=TWOSTAGE_COLOUR, label=r"OLS $\beta$ only")

    # The horizons are spaced geometrically, so a log axis keeps the short leads readable
    plt.xscale("log")
    plt.xticks(horizons, [str(h) for h in horizons])
    plt.xlabel("Outage duration / forecast horizon (hours)")
    plt.ylabel("Mean absolute error (°C)")
    plt.title(spec["title"])
    plt.grid(alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, spec["file_name"]), dpi=200)
    plt.close()

    print(f"\n=== Mean absolute error per horizon, {kind} (°C) ===")
    print(f"{'Horizon (h)':<14}{'OLS beta':>12}{'ARIMAX beta':>14}{'ARIMAX full':>14}{'Two-stage full':>16}")
    for horizon in horizons:
        row = summary.loc[horizon]
        print(f"{horizon:<14}{row[f'ols_beta_{suffix}']:>12.4f}{row[f'arimax_beta_{suffix}']:>14.4f}"
              f"{row[f'arimax_full_{suffix}']:>14.4f}{row[f'twostage_full_{suffix}']:>16.4f}")


def plot_outage_duration_faceted(summaries, plot_dir, kind):
    """
    Plot the same comparison side by side for several datasets, one subplot each, sharing a
    single legend, so the datasets can be read against one another directly.

    Input
    -----
    summaries: Dict of {dataset_name: summary DataFrame indexed by horizon}, as returned by
              load_results for each dataset
    plot_dir: Directory to write the figure to
    kind: "cumulative" or "pointwise", picking which columns and plot to produce
    """
    spec = KINDS[kind]
    suffix = spec["suffix"]

    fig, axes = plt.subplots(1, len(summaries), figsize=(6 * len(summaries), 5), sharey=False)

    for ax, (dataset_name, summary) in zip(axes, summaries.items()):
        horizons = summary.index.tolist()

        ax.plot(horizons, summary[f"arimax_full_{suffix}"], marker="o", color=ARIMAX_COLOUR, label="ARIMAX (full model)")
        ax.plot(horizons, summary[f"arimax_beta_{suffix}"], linestyle="--", color=ARIMAX_COLOUR, label=r"ARIMAX $\beta$ only")
        ax.plot(horizons, summary[f"twostage_full_{suffix}"], marker="o", color=TWOSTAGE_COLOUR, label="Two-stage (full model)")
        ax.plot(horizons, summary[f"ols_beta_{suffix}"], linestyle="--", color=TWOSTAGE_COLOUR, label=r"OLS $\beta$ only")

        ax.set_xscale("log")
        ax.set_xticks(horizons)
        ax.set_xticklabels([str(h) for h in horizons])
        ax.set_xlabel("Outage duration / forecast horizon (hours)")
        ax.set_title(dataset_name)
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Mean absolute error (°C)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.05))
    fig.suptitle(spec["title"] + ", by dataset", y=1.12)

    fig.tight_layout()
    file_name = spec["file_name"].replace(".png", "_by_dataset.png")
    fig.savefig(os.path.join(plot_dir, file_name), dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(PLOT_DIR, exist_ok=True)

    summary = load_results(OUTPUT_FILE)
    print(f"{len(summary)} horizons, all datasets combined")

    for kind in KINDS:
        plot_outage_duration(summary, PLOT_DIR, kind)

    faceted_summaries = {name: load_results(OUTPUT_FILE, name) for name in FACETED_DATASETS}

    for kind in KINDS:
        plot_outage_duration_faceted(faceted_summaries, PLOT_DIR, kind)
