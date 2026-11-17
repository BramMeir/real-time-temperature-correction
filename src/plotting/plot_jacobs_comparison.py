import argparse
import os
import matplotlib.pyplot as plt
import pandas as pd
from src.scripts.jacobs_comparison_experiment import OUTPUT_DIR as RESULTS_DIR, REPETITIONS
from src.plotting.table_jacobs_comparison import PANELS, METHODS, add_jacobs_algorithm, rmse_summary

PLOT_DIR = "plots/jacobs_comparison"
PLOT_NAME = "jacobs_comparison_gap_length.pdf"

# Their algorithm switches from linear interpolation to debiased ERA5 at 5 h, so between the 4 h and 5 h gaps
SWITCH_AT = 4.5

# Labelled gap lengths in hours; every other simulated length gets a small tick without a label
LABELLED_LENGTHS = [(1, "1 h"), (4, "4 h"), (24, "1 d"), (168, "7 d"), (720, "30 d")]

LINE_STYLES = {
    "Jacobs": {"label": "Jacobs et al. (2024)", "color": "#d95f02", "marker": "o"},
    "RegSARIMA_realtime": {"label": "RegSARIMA, real time", "color": "black", "marker": "s"},
    "RegSARIMA_retrospective": {"label": "RegSARIMA, retrospective", "color": "black", "marker": "s",
                                "linestyle": "--", "markerfacecolor": "white"},
}


def plot_gap_length(summary):
    """
    Mean RMSE against gap length on a log axis, one panel for the rural and one for the urban stations, with a line
    for their algorithm and for both RegSARIMA variants.

    Input
    -----
    summary: Output of rmse_summary, with columns panel, method, gap_length and rmse
    """
    lengths = sorted(summary["gap_length"].unique())
    fig, axes = plt.subplots(1, len(PANELS), figsize=(11, 4), sharey=True)

    for ax, panel in zip(axes, PANELS):
        for method in METHODS:
            rows = summary[(summary["panel"] == panel) & (summary["method"] == method)]
            ax.plot(rows["gap_length"], rows["rmse"], linewidth=2, markersize=5, **LINE_STYLES[method])

        ax.axvline(SWITCH_AT, color="0.7", linewidth=0.8, zorder=0)
        for x, text, align in [(SWITCH_AT / 1.06, "interpolation", "right"),
                               (SWITCH_AT * 1.06, "debiased ERA5", "left")]:
            ax.text(x, 0.02, text, transform=ax.get_xaxis_transform(), fontsize=11, color="0.4", ha=align)

        ax.set_xscale("log")
        ax.set_xticks([hours for hours, _ in LABELLED_LENGTHS], [label for _, label in LABELLED_LENGTHS])
        ax.minorticks_off()
        ax.set_xticks(lengths, [""] * len(lengths), minor=True)
        ax.tick_params(axis="x", which="minor", length=4)

        ax.set_xlabel("Gap length")
        ax.set_title(panel, fontsize=15)
        ax.set_ylim(0, 1.15)
        ax.grid(color="0.92", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Mean RMSE (°C)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(METHODS), frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, PLOT_NAME), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot the comparison of RegSARIMA with Jacobs et al. (2024) "
                                                 "against gap length.")
    parser.add_argument("--repetitions", type=int, default=REPETITIONS, help="Repetitions of the run to read")
    args = parser.parse_args()

    os.makedirs(PLOT_DIR, exist_ok=True)
    # Arial with TrueType fonts embedded in the PDF, as in the other figures, sized so the text prints as large as in
    # the reliability figure, the other two-panel figure, once LaTeX scales both to the text width
    plt.rcParams.update({"font.family": "Arial", "font.size": 11, "axes.labelsize": 15, "xtick.labelsize": 14,
                         "ytick.labelsize": 14, "legend.fontsize": 13})
    plt.rcParams["pdf.fonttype"] = 42

    stations = sorted(station for group in PANELS.values() for station in group)
    results = add_jacobs_algorithm(pd.concat(
        pd.read_csv(f"{RESULTS_DIR}/jacobs_comparison_{station}_rep{args.repetitions}.csv") for station in stations
    ))
    plot_gap_length(rmse_summary(results, PANELS))
