import pandas as pd
import matplotlib.pyplot as plt
import re

importance = pd.Series({
    "ar.L1": 0.682334,
    "ar.S.L24": 0.181585,
    "ar.L2": 0.141316,
    "ar.L3": 0.105307,
    "ar.L4": 0.093636,
    "ar.L5": 0.092482,
    "ar.L6": 0.090474,
    "ar.L7": 0.077535,
    "ar.L9": 0.074064,
    "ar.L8": 0.073742,
    "ar.L10": 0.068182,
})


def extract_lag(name):
    match = re.search(r"L(\d+)", name)
    return int(match.group(1))


df = importance.reset_index()
df.columns = ["parameter", "importance"]

df["lag"] = df["parameter"].apply(extract_lag)
df["seasonal"] = df["parameter"].str.contains("S.")

# Sort by lag to ensure correct order in the plot
df = df.sort_values("lag")

# Custom x-positions to create a gap between regular and seasonal lags
x_positions = []
current_x = 0

for seasonal in df["seasonal"]:
    if seasonal:
        current_x += 1.6   # Extra space for jump between regular and seasonal lags
    else:
        current_x += 1
    x_positions.append(current_x)

df["xpos"] = x_positions

# Plot
plt.figure(figsize=(10, 5))

bars = plt.bar(df["xpos"], df["importance"], width=0.5)

plt.plot(
    df["xpos"],
    df["importance"],
    color="black",
    linewidth=1.8,
    marker="o",
    markersize=5,
    markeredgewidth=1,
)

# Make labels clear
labels = [
    f"{lag} h" if not seasonal else "24 h"
    for lag, seasonal in zip(df["lag"], df["seasonal"])
]

plt.xticks(df["xpos"], labels)

# Add an ellipsis in the gap
gap_center = (df["xpos"].iloc[-2] + df["xpos"].iloc[-1]) / 2

plt.text(
    gap_center,
    0,
    "...",
    ha="center",
    va="top",
    fontsize=18,
    transform=plt.gca().get_xaxis_transform(),
)

plt.xlabel("Autoregressive lag of the target station")
plt.ylabel("Mean absolute coefficient")

plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("autoregressive_lag_importance.png")
