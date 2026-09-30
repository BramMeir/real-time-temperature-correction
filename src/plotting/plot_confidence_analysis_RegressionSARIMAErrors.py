"""
Script: plot_confidence_analysis_RegressionSARIMAErrors.py

Mirrors plot_confidence_analysis.py (ARIMAX) for the two-stage regression with SARIMA errors
model, so the two figures are directly comparable: same layout, same axes, same correlation
annotation, reading from output/confidence_analysis_regression_sarima_errors/ instead of
output/confidence_analysis/.

If the evaluation also kept every single episode (confidence_episodes_station_*.csv), these are drawn in
grey behind the station averages, with their own correlation, since an operator has one validation
episode per station to go on rather than an average over many. The correlation within stations, of each
episode's deviation from its station's average, is printed as well.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

# Output directory
plots_dir = "plots/confidence_analysis_regression_sarima_errors"
os.makedirs(plots_dir, exist_ok=True)

# Load data
files = glob.glob("output/confidence_analysis_regression_sarima_errors/confidence_station_*.csv")
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

# Casino_Oostende's realised error is not representative: its nearest neighbours are too far/dissimilar
# for a meaningful spatial correction, which distorts the scale of the plot without adding information
df = df[df["target_station"] != "Casino_Oostende"]

# Compute correlations
corr_conf = df["confidence_mean"].corr(df["mae_mean"])
corr_interval = df["interval_size_mean"].corr(df["mae_mean"])

print(f"Correlation (confidence vs MAE): {corr_conf:.3f}")
print(f"Correlation (interval size vs MAE): {corr_interval:.3f}")

# Single episodes, if the evaluation kept them
episode_files = glob.glob("output/confidence_analysis_regression_sarima_errors/confidence_episodes_station_*.csv")
episodes = None
if episode_files:
    episodes = pd.concat([pd.read_csv(f) for f in episode_files], ignore_index=True)
    episodes = episodes[episodes["target_station"] != "Casino_Oostende"]

    corr_conf_episodes = episodes["confidence_score"].corr(episodes["mae"])
    corr_interval_episodes = episodes["interval_width"].corr(episodes["mae"])

    # Deviations from each station's own average, so the differences between stations drop out
    columns = ["confidence_score", "interval_width", "mae"]
    deviations = episodes[columns] - episodes.groupby("target_station")[columns].transform("mean")
    corr_conf_within = deviations["confidence_score"].corr(deviations["mae"])
    corr_interval_within = deviations["interval_width"].corr(deviations["mae"])

    print(f"Episodes: {len(episodes)} of {episodes['target_station'].nunique()} stations")
    print(f"Correlation per episode (confidence vs MAE): {corr_conf_episodes:.3f}, "
          f"within stations: {corr_conf_within:.3f}")
    print(f"Correlation per episode (interval size vs MAE): {corr_interval_episodes:.3f}, "
          f"within stations: {corr_interval_within:.3f}")


def _correlation_label(station_r, episode_r):
    """Correlation annotation, with the per-episode value as well if the episodes were kept."""
    if episode_r is None:
        return f"r = {station_r:.2f}"
    return f"r (stations) = {station_r:.2f}\nr (episodes) = {episode_r:.2f}"

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

# Left: Confidence vs MAE
ax = axes[0]

x = df["confidence_mean"]
y = df["mae_mean"]

# Scatter, with the single episodes behind the station averages
if episodes is not None:
    ax.scatter(episodes["confidence_score"], episodes["mae"], s=8, color="lightgrey", alpha=0.6, zorder=0)
ax.scatter(x, y, alpha=0.6)

# Regression line (sorted for clean line)
x_sorted = np.sort(x)
m, b = np.polyfit(x, y, 1)
ax.plot(x_sorted, m * x_sorted + b)

# Labels
ax.set_title("Error-based score")
ax.set_xlabel("Reliability score (1 / (1 + MAE))")
ax.set_ylabel("Mean MAE (°C)")

# Correlation annotation
ax.text(
    0.97, 0.95,
    _correlation_label(corr_conf, corr_conf_episodes if episodes is not None else None),
    transform=ax.transAxes,
    verticalalignment='top',
    horizontalalignment='right',
    fontsize=12,
)

ax.grid(alpha=0.2)


# Right: Interval vs MAE
ax = axes[1]

x = df["interval_size_mean"]
y = df["mae_mean"]

# Scatter, with the single episodes behind the station averages
if episodes is not None:
    ax.scatter(episodes["interval_width"], episodes["mae"], s=8, color="lightgrey", alpha=0.6, zorder=0)
ax.scatter(x, y, alpha=0.6)

# Regression line
x_sorted = np.sort(x)
m, b = np.polyfit(x, y, 1)
ax.plot(x_sorted, m * x_sorted + b)

# Labels
ax.set_title("Prediction-interval width")
ax.set_xlabel("95 % interval width (°C)")

# Correlation annotation
ax.text(
    0.05, 0.95,
    _correlation_label(corr_interval, corr_interval_episodes if episodes is not None else None),
    transform=ax.transAxes,
    verticalalignment='top',
    fontsize=12,
)

ax.grid(alpha=0.2)

# Layout
plt.tight_layout()

# Save
plt.savefig(f"{plots_dir}/confidence_combined_corr.png", dpi=300)
