"""
Script: plot_jacobs_comparison.py

Mean RMSE against gap length for the gap-filling algorithm of Jacobs et al. (2024) and both RegSARIMA
variants (src/scripts/jacobs_comparison_experiment.py), in two figures: one panel per station for the
two stations of their Fig. 6, and one panel per station type, with the gaps of its three stations
pooled. The gap lengths are spaced evenly, as in their figure, with the 95 % confidence interval of
the mean as error bars.

Their algorithm is assembled from the techniques by gap length as in Table 1 of their paper: linear
interpolation below 5 h, mean-bias debiased ERA5 with both halves of the learning period weighted
together from 5 h, and weighted separately from 15 days. Only the first switch is marked, since the
second leaves the error practically unchanged.

Run:
    python -m src.plotting.plot_jacobs_comparison
"""
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.scripts.jacobs_comparison_experiment import GAP_LENGTHS, OUTPUT_DIR, REPETITIONS

PLOTS_DIR = "plots/jacobs_comparison"

# Panels of each figure, as {figure name: {panel title: stations pooled in the panel}}. The urban stations are
# those of the UHI evaluation of Jacobs et al., the other three lie outside the city centre
FIGURES = {
    "ylijoki_betel": {"(a) Ylijoki (rural)": ["Ylijoki"], "(b) Betel (urban)": ["Betel"]},
    "rural_urban": {"(a) Rural stations": ["Kurala", "Tuorla", "Ylijoki"],
                    "(b) Urban stations": ["Betel", "Puutori", "Virastotalo"]},
}

# Switches of their algorithm, in hours
LI_UNTIL = 5
SEPARATE_FROM = 15 * 24

# RegSARIMA in black as in the model comparison of the paper, the retrospective variant dashed with open markers
METHODS = {
    "Jacobs": {"label": "Jacobs et al. (2024)", "color": "#c2410c"},
    "RegSARIMA_realtime": {"label": "RegSARIMA, real time", "color": "black"},
    "RegSARIMA_retrospective": {"label": "RegSARIMA, retrospective", "color": "black", "linestyle": "--",
                                "markerfacecolor": "white"},
}


def gap_label(hours):
    """Gap length as a tick label: hours below a day, days from a day on."""
    return f"{hours} h" if hours < 24 else f"{hours // 24} d"


def add_jacobs_algorithm(results):
    """Add their algorithm as a method, taking for every gap the technique it selects for the gap length."""
    technique = np.select(
        [results["gap_length"] < LI_UNTIL, results["gap_length"] < SEPARATE_FROM], ["LI", "MB_both"], "MB_separate"
    )
    algorithm = results[results["method"] == technique].assign(method="Jacobs")
    return pd.concat([results, algorithm])


def rmse_summary(results, panels):
    """Mean of the per-gap RMSE with the half-width of its 95 % interval, per panel, method and gap length."""
    panel_of = {station: panel for panel, stations in panels.items() for station in stations}
    results = results[results["station"].isin(panel_of)]
    results = results.assign(rmse=np.sqrt(results["mse"]), panel=results["station"].map(panel_of))
    summary = results.groupby(["panel", "method", "gap_length"])["rmse"].agg(["mean", "std", "count"])
    summary["ci95"] = 1.96 * summary["std"] / np.sqrt(summary["count"])
    return summary.reset_index()


def plot(summary, output_path):
    """Draw the mean RMSE against gap length with its 95 % interval, one panel per panel title in summary."""
    panels = list(summary["panel"].unique())
    fig, axes = plt.subplots(1, len(panels), figsize=(13, 5), sharey=True)
    x = np.arange(len(GAP_LENGTHS))
    # Between the last gap length below 5 h and 5 h itself
    switch = GAP_LENGTHS.index(LI_UNTIL) - 0.5

    for ax, title in zip(axes, panels):
        for method, style in METHODS.items():
            rows = summary[(summary["panel"] == title) & (summary["method"] == method)].set_index("gap_length")
            rows = rows.loc[GAP_LENGTHS]
            ax.errorbar(x, rows["mean"], yerr=rows["ci95"], marker="o", capsize=3, **style)

        ax.axvline(switch, linestyle=":", color="gray")
        ax.text(switch + 0.12, 0.03, "interpolation → ERA5", color="gray", fontsize=10)

        ax.set_title(title, loc="left")
        ax.set_xticks(x, [gap_label(hours) for hours in GAP_LENGTHS])
        ax.set_xlabel("Gap length")
        ax.grid(alpha=0.3)

    axes[0].set_ylim(bottom=0)
    axes[0].set_ylabel("Mean RMSE (°C)")
    # In the empty band of the urban panel, between RegSARIMA and the ERA5 debiasing
    axes[1].legend(loc="center right", bbox_to_anchor=(1, 0.55))

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    for extension in ["png", "pdf"]:
        plt.savefig(f"{output_path}.{extension}", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure to {output_path}.png and .pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot the comparison of RegSARIMA with Jacobs et al. (2024).")
    parser.add_argument("--repetitions", type=int, default=REPETITIONS, help="Repetitions of the run to plot")
    args = parser.parse_args()

    # Label sizes and TrueType fonts in the PDF as in the other paper figures
    plt.rcParams.update({"axes.labelsize": 13, "axes.titlesize": 13, "xtick.labelsize": 11, "ytick.labelsize": 12,
                         "legend.fontsize": 11})
    plt.rcParams["pdf.fonttype"] = 42

    stations = sorted({station for panels in FIGURES.values() for group in panels.values() for station in group})
    results = add_jacobs_algorithm(pd.concat(
        pd.read_csv(f"{OUTPUT_DIR}/jacobs_comparison_{station}_rep{args.repetitions}.csv") for station in stations
    ))

    pd.set_option("display.width", 200)
    for name, panels in FIGURES.items():
        summary = rmse_summary(results, panels)
        print(summary[summary["method"].isin(list(METHODS))]
              .pivot_table(index="gap_length", columns=["panel", "method"], values="mean").round(3))
        plot(summary, f"{PLOTS_DIR}/jacobs_comparison_{name}")
