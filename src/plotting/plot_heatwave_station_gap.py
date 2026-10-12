"""
Script: plot_heatwave_station_gap.py
Description: How much the stations of a network differ from one another over the day, during the heat waves of
heatwave_experiment.py and in their control windows. At each hour, the gap is the temperature of the warmest station
minus that of the coolest, computed from the observations alone.

The profile is averaged per window first, the controls of an event are then averaged, and finally the events, so every
heat wave weighs equally, as in summarize_heatwave_errors.py. Only TURCLIM is plotted by default: its six stations lie
in and around one city, so the gap is the contrast between the centre and its surroundings. In the synthetic network,
which spans all of Belgium, the gap is set by the coast and the Ardennes rather than by the cities.

Usage: python -m src.plotting.plot_heatwave_station_gap [--dataset TURKU]
"""
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
from src.scripts.detect_heatwaves import DATASETS, load_hourly
from src.scripts.heatwave_experiment import EXPERIMENT_DATASETS, OUTPUT_DIR, TRAINING_WEEKS

DATASET_LABELS = {"TURKU": "TURCLIM", "SYNTHETIC": "Synthetic"}

PLOT_DIR = "plots/heatwave_experiment"

# Hours in UTC, as the data are stored; Finnish summer time is UTC+3, so the night of 00-05 local time is 21-02 UTC
NIGHT_HOURS = [21, 22, 23, 0, 1, 2]
DAY_HOURS = list(range(6, 18))


def hourly_gap_profiles(dataset):
    """
    Mean gap between the warmest and the coolest station per hour of the day, for the heat waves and their controls.

    Input
    -----
    dataset: Dataset name

    Output
    ------
    profiles: DataFrame indexed by hour of the day, with columns Heatwave and Control
    """
    observations = load_hourly(DATASETS[dataset])
    gap = observations.max(axis=1) - observations.min(axis=1)

    windows = pd.read_csv(
        f"{OUTPUT_DIR}/windows_{TRAINING_WEEKS}_weeks.csv", parse_dates=["Window_start", "Window_end"]
    )
    # The windows of an event are the same for every station and lead
    windows = windows[windows.Dataset == dataset].drop_duplicates(["Event_id", "Control"])

    rows = []

    for window in windows.itertuples():
        window_gap = gap[window.Window_start:window.Window_end + pd.Timedelta(hours=23)]

        for hour, value in window_gap.groupby(window_gap.index.hour).mean().items():
            rows.append([window.Event_id, window.Control > 0, hour, value])

    profiles = pd.DataFrame(rows, columns=["Event_id", "Is_control", "Hour", "Gap"])

    per_event = profiles.groupby(["Is_control", "Event_id", "Hour"]).Gap.mean()

    return pd.DataFrame({
        "Heatwave": per_event.loc[False].groupby("Hour").mean(),
        "Control": per_event.loc[True].groupby("Hour").mean()
    })


def plot_profiles(profiles, dataset):
    """
    Gap between the warmest and the coolest station per hour of the day, heat waves against ordinary weather.

    Input
    -----
    profiles: Output of hourly_gap_profiles
    dataset: Dataset name, for the file name
    """
    plt.figure(figsize=(9, 5))

    plt.fill_between(profiles.index, profiles.Control, profiles.Heatwave, color="#c2410c", alpha=0.12, linewidth=0)
    plt.plot(profiles.index, profiles.Control, color="#475569", linewidth=2, linestyle="--", label="Ordinary weather")
    plt.plot(profiles.index, profiles.Heatwave, color="#c2410c", linewidth=2, label="Heat waves")

    plt.xlabel("Hour of the day (UTC)")
    plt.ylabel("Warmest minus coolest station (°C)")
    plt.xticks(range(0, 24, 3))
    plt.xlim(0, 23)
    plt.ylim(bottom=0)
    plt.grid(True, axis="y", linestyle="-", alpha=0.3)
    plt.legend(frameon=False, loc="upper center")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"station_gap_{dataset}.pdf"))
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=EXPERIMENT_DATASETS, default="TURKU")
    dataset = parser.parse_args().dataset

    os.makedirs(PLOT_DIR, exist_ok=True)
    # Same text sizes and font embedding as the other paper figures, see plot_training_time_impact_RegressionSARIMAErrors.py
    plt.rcParams.update({"axes.labelsize": 13, "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12})
    plt.rcParams["pdf.fonttype"] = 42

    profiles = hourly_gap_profiles(dataset)
    profiles.to_csv(f"{OUTPUT_DIR}/station_gap_profile_{dataset}.csv")
    plot_profiles(profiles, dataset)

    print(f"{DATASET_LABELS[dataset]}: warmest minus coolest station (degC)")
    for name, hours in [("Night (21-02 UTC)", NIGHT_HOURS), ("Day (06-17 UTC)", DAY_HOURS)]:
        heatwave, control = profiles.loc[hours, "Heatwave"].mean(), profiles.loc[hours, "Control"].mean()
        print(f"{name}: heat waves {heatwave:.2f}, ordinary weather {control:.2f}")
