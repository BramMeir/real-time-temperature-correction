"""
Script: plot_interval_coverage.py

Figure for experiment P1.5: empirical coverage of the two-stage model's prediction intervals
against outage duration.

One line per dataset, a reference line at the nominal level, and a band for the spread across
stations - marginal coverage can sit on the nominal line while isolated stations are badly
overconfident.

A line on the reference line means the interval is honest, below it the model is overconfident, and
a line that sags as the outage lengthens means the interval fails to widen fast enough.

Run:
    python -m src.plotting.plot_interval_coverage
    python -m src.plotting.plot_interval_coverage --level 0.9
"""
import argparse
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_GLOB = "output/interval_coverage/interval_coverage_*.csv"
PLOTS_DIR = "plots/interval_coverage"

DATASET_LABELS = {
    "KMI": "RMI (rural)",
    "TURKU": "TURCLIM (urban)",
    "SYNTHETIC": "Synthetic"
}

# Horizons in hours, labelled in the units a reader thinks in
HORIZON_LABELS = {
    4: "4h", 12: "12h", 24: "1d", 48: "2d", 96: "4d",
    168: "7d", 336: "14d", 504: "21d", 720: "30d"
}


def load_results(level, method=None, calibration_days=None):
    """
    Load every coverage CSV and keep the rows for one nominal level, optionally for one band method
    and calibration length.

    Input
    -----
    level: Nominal coverage level to plot
    method: Band method to keep, or None for all
    calibration_days: Holdout length to keep, or None for all

    Output
    ------
    df: The matching rows
    """
    files = glob.glob(RESULTS_GLOB)
    if not files:
        raise FileNotFoundError(
            f"No results found at {RESULTS_GLOB}. Run src.scripts.interval_coverage_experiment first."
        )

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    selected = df[np.isclose(df["Level"], level)]
    if selected.empty:
        raise ValueError(f"Level {level} not in the results. Available: {sorted(df.Level.unique())}")

    if method is not None:
        selected = selected[selected["Method"] == method]
    if calibration_days is not None:
        selected = selected[selected["Calibration_days"] == calibration_days]

    if selected.empty:
        raise ValueError(f"No rows for method={method}, calibration_days={calibration_days}.")

    return selected


def pooled_coverage(df, group_cols):
    """
    Coverage as the ratio of the pooled hour counts, not the mean of the per-window ratios, so short
    and long windows are weighted correctly.

    Input
    -----
    df: DataFrame with N_inside and N_total
    group_cols: Columns to group by

    Output
    ------
    summary: Group columns plus a Coverage column
    """
    grouped = df.groupby(group_cols, as_index=False)[["N_inside", "N_total"]].sum()
    grouped["Coverage"] = grouped["N_inside"] / grouped["N_total"]

    return grouped


def plot_coverage_vs_horizon(df, level, band_quantiles=(0.1, 0.9), show_band=True):
    """
    Draw the coverage-against-outage-duration figure and save it to plots/interval_coverage/.

    Input
    -----
    df: Coverage rows for one nominal level
    level: The nominal level, for the reference line and the axis label
    band_quantiles: Quantiles of the across-station spread shown as a band
    show_band: Whether to draw the spread
    """
    datasets = [d for d in DATASET_LABELS if d in df["Dataset"].unique()]
    horizons = sorted(df["Horizon"].unique())
    colours = plt.cm.tab10(np.linspace(0, 1, 10))

    # Pooled over onsets and stations: the line. Pooled per station: the spread behind it.
    overall = pooled_coverage(df, ["Dataset", "Horizon"])
    per_station = pooled_coverage(df, ["Dataset", "Station", "Horizon"])

    _, ax = plt.subplots(figsize=(8, 5))

    # Track the lowest value drawn so the axis zooms on the range that carries the result
    lowest = level

    for i, dataset in enumerate(datasets):
        line = overall[overall["Dataset"] == dataset].sort_values("Horizon")
        if line.empty:
            continue

        ax.plot(
            line["Horizon"], line["Coverage"],
            marker="o", markersize=4, color=colours[i], label=DATASET_LABELS[dataset]
        )
        lowest = min(lowest, line["Coverage"].min())

        if show_band:
            spread = (
                per_station[per_station["Dataset"] == dataset]
                .groupby("Horizon")["Coverage"]
                .quantile(list(band_quantiles))
                .unstack()
                .sort_index()
            )

            ax.fill_between(
                spread.index, spread[band_quantiles[0]], spread[band_quantiles[1]],
                color=colours[i], alpha=0.15, linewidth=0
            )
            lowest = min(lowest, spread[band_quantiles[0]].min())

    ax.axhline(level, color="black", linestyle="--", linewidth=1, alpha=0.6)

    ax.set_xscale("log")
    ax.set_xticks(horizons)
    ax.set_xticklabels([HORIZON_LABELS.get(h, str(h)) for h in horizons])
    ax.set_xlabel("Outage duration")

    ax.set_ylabel(f"Empirical coverage of the {level:.0%} interval")
    ax.set_ylim(max(0.0, lowest - 0.05), 1.02)

    ax.legend(loc="lower left")
    ax.grid(alpha=0.2)

    plt.tight_layout()

    os.makedirs(PLOTS_DIR, exist_ok=True)
    method = df["Method"].iloc[0]
    days = df["Calibration_days"].iloc[0]
    output_path = f"{PLOTS_DIR}/coverage_vs_horizon_{method}_{days}d_{level:.2f}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved {output_path}")


def print_comparison(level):
    """
    Pooled coverage for every band method and calibration length, at the shortest and longest
    outage - the table that says how much each fix is worth.

    Input
    -----
    level: Nominal coverage level to summarise
    """
    df = load_results(level)
    shortest, longest = df["Horizon"].min(), df["Horizon"].max()

    print(f"\nPooled coverage of the {level:.0%} interval, all datasets:\n")
    print(f"  {'method':<11}{'holdout':>9}{HORIZON_LABELS.get(shortest, shortest):>9}"
          f"{HORIZON_LABELS.get(longest, longest):>9}{'width':>9}")

    summary = pooled_coverage(df, ["Method", "Calibration_days", "Horizon"])
    widths = df.groupby(["Method", "Calibration_days"])["Mean_width"].mean()

    for method in sorted(df["Method"].unique()):
        for days in sorted(df["Calibration_days"].unique()):
            rows = summary[(summary["Method"] == method) & (summary["Calibration_days"] == days)]
            short = rows[rows["Horizon"] == shortest]["Coverage"]
            long = rows[rows["Horizon"] == longest]["Coverage"]
            if short.empty or long.empty:
                continue

            print(f"  {method:<11}{str(days) + 'd':>9}{short.iloc[0]:>9.3f}{long.iloc[0]:>9.3f}"
                  f"{widths.loc[(method, days)]:>8.2f}°")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot the empirical coverage of the prediction intervals against outage duration."
    )
    parser.add_argument("--level", type=float, default=0.95,
                        help="Nominal coverage level to plot (default: 0.95).")
    parser.add_argument("--method", type=str, default="conformal",
                        help="Band method to plot (default: conformal).")
    parser.add_argument("--calibration_days", type=int, default=14,
                        help="Holdout length to plot, in days (default: 14).")
    parser.add_argument("--no_band", action="store_true",
                        help="Hide the across-station spread band.")
    args = parser.parse_args()

    results = load_results(args.level, args.method, args.calibration_days)

    plot_coverage_vs_horizon(results, args.level, show_band=not args.no_band)
    print_comparison(args.level)
