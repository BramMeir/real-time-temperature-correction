"""
Plot the ACF and PACF of the residuals written by src/scripts/check_residual_autocorrelation.py: the
stage-one residuals the neighbour regression leaves behind, and the stage-two SARIMA innovations.
The stage-one panels show the structure the regression cannot explain (a decaying lag-1 correlation
plus a bump at 24 h), which is what the residual SARIMA is fitted on; the stage-two panels show what
is left of it afterwards.

The figures are saved in the "plots/residual_autocorrelation" directory.

python -m src.plotting.plot_residual_acf_pacf
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf

# Directory holding the residual csv files and the directory the figures are written to
OUTPUT_DIR = "output/residual_autocorrelation"
PLOT_DIR = "plots/residual_autocorrelation"

# Lags shown on the x axis, two days so the daily peak stands out without squeezing the first lags
MAX_LAG = 48

STAGE_NAMES = {
    "stage_one": "Regression residuals",
    "stage_two": "SARIMA innovations",
}


def plot_residual_correlograms(residuals, title, plot_file):
    """
    Plot the ACF and PACF of both residual stages as a 2x2 grid.

    Input
    -----
    residuals: DataFrame with a stage_one and a stage_two column
    title: Title of the figure, naming the dataset, station and window
    plot_file: Path the figure is written to
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True, sharey="row")

    for row, (stage, stage_name) in enumerate(STAGE_NAMES.items()):
        # zero=False drops the lag-0 spike, which is 1 by construction and flattens everything else
        plot_acf(residuals[stage], lags=MAX_LAG, zero=False, ax=axes[row, 0], title=f"{stage_name} — ACF")
        plot_pacf(residuals[stage], lags=MAX_LAG, zero=False, ax=axes[row, 1], title=f"{stage_name} — PACF")

        # statsmodels fixes the y axis to the full correlation range, which leaves the innovations
        # as a flat line; scale each row to its own correlations instead
        largest = max(abs(acf(residuals[stage], nlags=MAX_LAG, fft=True)[1:]).max(),
                      abs(pacf(residuals[stage], nlags=MAX_LAG)[1:]).max())
        axes[row, 0].set_ylim(-1.3 * largest, 1.3 * largest)

    for ax in axes.flat:
        ax.set_xticks(np.arange(0, MAX_LAG + 1, 12))
        ax.set_xlim(0, MAX_LAG)
        ax.grid(alpha=0.3)

    for ax in axes[1]:
        ax.set_xlabel("Lag (hours)")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(plot_file, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(PLOT_DIR, exist_ok=True)

    for file_name in sorted(f for f in os.listdir(OUTPUT_DIR) if f.startswith("residuals_")):
        dataset, station = file_name.removeprefix("residuals_").removesuffix(".csv").split("_", 1)

        residuals = pd.read_csv(os.path.join(OUTPUT_DIR, file_name),
                                index_col="datetime", parse_dates=True)

        plot_file = os.path.join(PLOT_DIR, f"acf_pacf_{dataset}_{station}.png")
        plot_residual_correlograms(
            residuals,
            title=f"{dataset} — {station}, {residuals.index[0].date()} to {residuals.index[-1].date()}",
            plot_file=plot_file
        )

        print(f"{dataset} {station}: {len(residuals)} hours -> {plot_file}")
