"""
Script: plot_confidence_analysis_RegressionSARIMAErrors.py

Mirrors plot_confidence_analysis.py (ARIMAX) for the two-stage regression with SARIMA errors
model, so the two figures are directly comparable: same layout, same axes, same correlation
annotation, reading from output/confidence_analysis_regression_sarima_errors/ instead of
output/confidence_analysis/.
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

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

# Left: Confidence vs MAE
ax = axes[0]

x = df["confidence_mean"]
y = df["mae_mean"]

# Scatter
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
    0.825, 0.95,
    f"r = {corr_conf:.2f}",
    transform=ax.transAxes,
    verticalalignment='top',
    fontsize=12,
)

ax.grid(alpha=0.2)


# Right: Interval vs MAE
ax = axes[1]

x = df["interval_size_mean"]
y = df["mae_mean"]

# Scatter
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
    f"r = {corr_interval:.2f}",
    transform=ax.transAxes,
    verticalalignment='top',
    fontsize=12,
)

ax.grid(alpha=0.2)

# Layout
plt.tight_layout()

# Save
plt.savefig(f"{plots_dir}/confidence_combined_corr.png", dpi=300)
