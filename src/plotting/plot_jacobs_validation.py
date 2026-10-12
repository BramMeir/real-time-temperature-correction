"""
Script: plot_jacobs_validation.py

Figure and table for the reproduction of Fig. 6 of Jacobs et al. (2024): the error E (mean MSE over
the gaps) of the five gap-filling techniques per gap length, from our rerun of their code next to
the values read off their figure.

Their values were read by eye from the published figure, to about 0.02 degC^2 where the close-up
panels show them and about 0.3 degC^2 where only the full-scale panels do (linear interpolation from
7 h on, and original ERA5 at Betel). Their gaps were unseeded, so our rerun draws different gaps,
and a difference between the two is expected to stay within the sampling error of both runs. The
table therefore lists z = (ours - theirs) / sqrt(SE_ours^2 + SE_theirs^2), with their standard error
scaled from ours by the number of gaps, since their error bars cannot be read for most points.

Run:
    python -m src.plotting.plot_jacobs_validation
"""
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.scripts.jacobs_validation_experiment import OUTPUT_DIR, REPETITIONS

RESULTS_FILE = f"{OUTPUT_DIR}/jacobs_techniques_rep{REPETITIONS}.csv"
PLOTS_DIR = "plots/jacobs_validation"

GAP_LENGTHS = [1, 3, 5, 7, 10, 20, 30, 336]

# Number of gaps per length behind their figure
PUBLISHED_REPETITIONS = 1000

# Labels and colours as in their Fig. 6, so the two figures can be compared side by side
TECHNIQUES = {
    "linint": ("LI", "tab:blue"),
    "fillmodel": ("Original ERA5", "tab:orange"),
    "debmodelReg": ("Debiased: LR", "tab:green"),
    "debmodelMeanbias": ("Debiased: MB", "tab:red"),
    "debmodelTvar": ("Debiased: WMB", "tab:purple"),
}

# E (degC^2) per gap length in GAP_LENGTHS, read off Fig. 6 of Jacobs et al. (2024)
PUBLISHED = {
    "Ylijoki": {
        "linint": [0.22, 0.54, 1.18, 2.33, 4.38, 15.2, 14.3, 29.0],
        "fillmodel": [1.31, 1.25, 1.33, 1.19, 1.26, 1.44, 1.28, 1.25],
        "debmodelReg": [1.10, 1.07, 1.15, 0.99, 1.15, 1.29, 1.13, 1.11],
        "debmodelMeanbias": [1.12, 1.05, 1.16, 0.99, 1.13, 1.28, 1.10, 1.09],
        "debmodelTvar": [1.23, 1.15, 1.19, 1.06, 1.21, 1.38, 1.16, 1.20],
    },
    "Betel": {
        "linint": [0.09, 0.32, 0.67, 1.23, 2.52, 8.49, 7.82, 20.5],
        "fillmodel": [3.15, 3.50, 3.03, 3.47, 3.22, 3.09, 3.15, 3.25],
        "debmodelReg": [1.00, 1.13, 1.12, 1.02, 1.06, 1.00, 1.06, 1.15],
        "debmodelMeanbias": [1.32, 1.47, 1.38, 1.40, 1.32, 1.28, 1.33, 1.38],
        "debmodelTvar": [1.25, 1.31, 1.23, 1.27, 1.14, 1.16, 1.18, 1.22],
    },
}


def comparison_table(results):
    """Join our E and standard error with the published values, one row per station, technique and length."""
    published = pd.DataFrame([
        [station, length, technique, value]
        for station, techniques in PUBLISHED.items()
        for technique, values in techniques.items()
        for length, value in zip(GAP_LENGTHS, values)
    ], columns=["station", "gap_length", "technique", "E_published"])

    table = results.merge(published, on=["station", "gap_length", "technique"])
    sterr_published = table["sterr"] * np.sqrt(table["n"] / PUBLISHED_REPETITIONS)
    table["z"] = (table["E"] - table["E_published"]) / np.sqrt(table["sterr"] ** 2 + sterr_published ** 2)
    return table


def plot(table, output_path):
    """Draw our E with 95 % intervals next to the published values, one panel per station."""
    stations = list(PUBLISHED)
    x = np.arange(len(GAP_LENGTHS))
    fig, axes = plt.subplots(1, len(stations), figsize=(12, 4.5), sharey=True)

    for ax, station in zip(axes, stations):
        for i, (technique, (label, colour)) in enumerate(TECHNIQUES.items()):
            rows = table[(table["station"] == station) & (table["technique"] == technique)].sort_values("gap_length")
            # Spread the techniques slightly along x so their error bars do not overlap
            offset = (i - (len(TECHNIQUES) - 1) / 2) * 0.06
            ax.errorbar(x + offset, rows["E"], yerr=1.96 * rows["sterr"], color=colour, marker="o",
                        markersize=5, linewidth=1.5, capsize=2, label=f"{label}, rerun")
            ax.plot(x + offset, rows["E_published"], color=colour, marker="o", markersize=8,
                    markerfacecolor="none", linestyle="none", label=f"{label}, Jacobs et al.")

        ax.set_yscale("log")
        ax.set_xticks(x, GAP_LENGTHS)
        ax.set_xlabel("Gap length (h)")
        ax.set_title(station)
        ax.grid(axis="y", which="both", color="0.9", linewidth=0.5)

    axes[0].set_ylabel("E, mean MSE over gaps (°C$^2$)")
    axes[-1].legend(fontsize=7, ncol=2, loc="upper left")
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare the rerun of Jacobs et al. (2024) with their Fig. 6.")
    parser.add_argument("--results_file", default=RESULTS_FILE, help="CSV written by jacobs_validation_experiment.py")
    args = parser.parse_args()

    table = comparison_table(pd.read_csv(args.results_file))

    pd.set_option("display.width", 200)
    print(table.pivot_table(index=["station", "technique"], columns="gap_length", values="z").round(1))
    print()
    print(table.pivot_table(index=["station", "technique"], columns="gap_length", values=["E", "E_published"]).round(2))

    plot(table, f"{PLOTS_DIR}/jacobs_fig6_reproduction.png")
