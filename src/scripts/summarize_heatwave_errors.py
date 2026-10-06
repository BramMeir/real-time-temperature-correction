"""
Script: summarize_heatwave_errors.py

Error metrics of the two-stage model during the heat waves of heatwave_experiment.py, and how they differ from its
error in the control windows around the same events.

Errors are reconstruction minus observation, so a negative value is an underestimation. Per window:
- Bias, RMSE: over the reconstructed heat wave hours
- Recovery_bias: over the RECOVERY_DAYS after the heat wave
- Tmax_error, Tmin_error: mean error in the daily maximum and minimum over the reconstructed heat wave days
- Peak_error: error in the highest temperature of the heat wave
- Tmax3_error, Tmin3_error: error in the highest 3-day mean of the daily maximum and minimum, the quantities heat
  warnings are issued on
The last three are computed on the station series as it would exist operationally, observed until the onset and
reconstructed after it, which only differs from the reconstruction when the outage starts during the heat wave.

Three diagnostics come with every window: Sum_beta, the sum of the regression coefficients, Extrapolation, the share of
the reconstructed heat wave hours in which the median of the other stations exceeds its maximum over the training window,
and Station_gap, the warmest minus the coolest station of the network at each reconstructed heat wave hour, averaged
over those hours. Station_gap is computed from the observations alone, so it is the same for every station of a window,
and describes how much the stations differ from one another.

Each heat wave window is compared with the mean of its controls (same station, same lead), and the confidence intervals
come from a bootstrap over events: all stations reconstruct the same heat wave, so their windows are not independent.
The RMSE is also compared per station, to see whether the heat waves affect every station or only a few.

Run:
    python -m src.scripts.summarize_heatwave_errors
"""
import numpy as np
import pandas as pd
from src.scripts.detect_heatwaves import DATASETS, load_hourly
from src.scripts.heatwave_experiment import EXPERIMENT_DATASETS, OUTPUT_DIR, TRAINING_WEEKS

METRICS = [
    "Bias", "RMSE", "Recovery_bias", "Tmax_error", "Tmin_error", "Peak_error", "Tmax3_error", "Tmin3_error",
    "Sum_beta", "Extrapolation", "Station_gap"
]

KEYS = ["Dataset", "Station", "Event_id", "Control", "Lead_days"]

N_BOOTSTRAP = 10000
SEED = 42


def highest_3day_mean(daily):
    """Highest mean over 3 consecutive days of a daily Series."""
    return daily.rolling(3).mean().max()


def window_metrics(scored, window, observed, network_median, station_gap):
    """
    Metrics of one window.

    Input
    -----
    scored: Hourly rows of the window, with Datetime, Observed and Predicted
    window: Row of the window table
    observed: Hourly observations of the target station
    network_median: Hourly median of the other stations
    station_gap: Hourly difference between the warmest and the coolest station of the network

    Output
    ------
    metrics: Dict with one value per name in METRICS
    """
    scored = scored.set_index("Datetime")
    heat_end = window.Window_end + pd.Timedelta(hours=23)

    heat = scored[:heat_end]
    recovery = scored[heat_end + pd.Timedelta(hours=1):]
    error = heat.Predicted - heat.Observed

    heat_days = heat.groupby(heat.index.floor("D"))

    # Observed until the onset and reconstructed after it, over every heat wave day
    actual = observed[window.Window_start:heat_end]
    operational = actual.copy()
    operational[heat.index] = heat.Predicted

    actual_days = actual.groupby(actual.index.floor("D"))
    operational_days = operational.groupby(operational.index.floor("D"))

    training_max = network_median[window.Train_start:window.Onset].max()

    return {
        "Bias": error.mean(),
        "RMSE": np.sqrt((error ** 2).mean()),
        "Recovery_bias": (recovery.Predicted - recovery.Observed).mean(),
        "Tmax_error": (heat_days.Predicted.max() - heat_days.Observed.max()).mean(),
        "Tmin_error": (heat_days.Predicted.min() - heat_days.Observed.min()).mean(),
        "Peak_error": operational.max() - actual.max(),
        "Tmax3_error": highest_3day_mean(operational_days.max()) - highest_3day_mean(actual_days.max()),
        "Tmin3_error": highest_3day_mean(operational_days.min()) - highest_3day_mean(actual_days.min()),
        "Sum_beta": window.Sum_beta,
        "Extrapolation": (network_median[heat.index] > training_max).mean(),
        "Station_gap": station_gap[heat.index].mean()
    }


def bootstrap_ci(values, rng):
    """
    Percentile bootstrap 95 % interval of the mean.

    Input
    -----
    values: Array with one value per event
    rng: Random generator

    Output
    ------
    low, high: Interval bounds
    """
    means = rng.choice(values, size=(N_BOOTSTRAP, len(values))).mean(axis=1)

    return np.percentile(means, [2.5, 97.5])


def pair_windows(metrics):
    """
    Each heat wave window next to the mean of its controls (same station, same lead).

    Input
    -----
    metrics: DataFrame with KEYS and METRICS, one row per window

    Output
    ------
    heatwave, controls: DataFrames with METRICS, indexed by Dataset, Station, Event_id and Lead_days
    """
    pair_keys = ["Dataset", "Station", "Event_id", "Lead_days"]

    heatwave = metrics[metrics.Control == 0].set_index(pair_keys)[METRICS]
    controls = metrics[metrics.Control > 0].groupby(pair_keys)[METRICS].mean()

    return heatwave, controls


def summarise(metrics):
    """
    Heat wave mean, control mean and their difference per dataset, lead and metric, with intervals over events.

    Input
    -----
    metrics: DataFrame with KEYS and METRICS, one row per window

    Output
    ------
    summary: DataFrame with one row per (dataset, lead, metric)
    """
    rng = np.random.default_rng(SEED)

    heatwave, controls = pair_windows(metrics)
    difference = heatwave - controls

    # Every event has the same stations, so the mean over events equals the mean over windows
    per_event = {
        name: frame.groupby(["Dataset", "Lead_days", "Event_id"]).mean()
        for name, frame in [("Heatwave", heatwave), ("Control", controls), ("Difference", difference)]
    }

    rows = []

    for (dataset, lead), events in per_event["Heatwave"].groupby(level=["Dataset", "Lead_days"]):
        for metric in METRICS:
            row = {"Dataset": dataset, "Lead_days": lead, "Metric": metric, "N_events": len(events)}

            for name, frame in per_event.items():
                values = frame.loc[(dataset, lead), metric].values
                low, high = bootstrap_ci(values, rng)
                row.update({name: values.mean(), f"{name}_low": low, f"{name}_high": high})

            rows.append(row)

    return pd.DataFrame(rows)


def summarise_stations(metrics):
    """
    Mean RMSE of every station over its heat wave windows and over its controls, per dataset and lead.

    Input
    -----
    metrics: DataFrame with KEYS and METRICS, one row per window

    Output
    ------
    stations: DataFrame with one row per (dataset, lead, station)
    """
    heatwave, controls = pair_windows(metrics)
    station_keys = ["Dataset", "Lead_days", "Station"]

    stations = pd.DataFrame({
        "RMSE_heatwave": heatwave.RMSE.groupby(station_keys).mean(),
        "RMSE_control": controls.RMSE.groupby(station_keys).mean()
    })
    stations["Difference"] = stations.RMSE_heatwave - stations.RMSE_control

    return stations.reset_index()


def main():
    hourly = pd.read_csv(
        f"{OUTPUT_DIR}/RegressionSARIMAErrors_{TRAINING_WEEKS}_weeks.csv.gz", parse_dates=["Datetime"]
    )
    windows = pd.read_csv(
        f"{OUTPUT_DIR}/windows_{TRAINING_WEEKS}_weeks.csv",
        parse_dates=["Window_start", "Window_end", "Train_start", "Onset"]
    ).set_index(KEYS)

    rows = []

    for dataset in EXPERIMENT_DATASETS:
        observations = load_hourly(DATASETS[dataset])

        network_medians = {
            station: observations.drop(columns=station).median(axis=1) for station in observations.columns
        }
        station_gap = observations.max(axis=1) - observations.min(axis=1)

        for keys, scored in hourly[hourly.Dataset == dataset].groupby(KEYS):
            station = keys[1]
            rows.append({
                **dict(zip(KEYS, keys)),
                **window_metrics(
                    scored, windows.loc[keys], observations[station], network_medians[station], station_gap
                )
            })

    metrics = pd.DataFrame(rows)
    metrics.to_csv(f"{OUTPUT_DIR}/window_metrics_{TRAINING_WEEKS}_weeks.csv", index=False)

    summary = summarise(metrics)
    summary.to_csv(f"{OUTPUT_DIR}/summary_{TRAINING_WEEKS}_weeks.csv", index=False)

    stations = summarise_stations(metrics)
    stations.to_csv(f"{OUTPUT_DIR}/station_summary_{TRAINING_WEEKS}_weeks.csv", index=False)

    with pd.option_context("display.width", 250, "display.float_format", "{:.2f}".format):
        print(summary[["Dataset", "Lead_days", "Metric", "N_events", "Heatwave", "Control", "Difference",
                       "Difference_low", "Difference_high"]].to_string(index=False))

        print("\nStations with a higher RMSE during heat waves than in their controls:")
        print(
            stations.groupby(["Dataset", "Lead_days"]).Difference
            .agg(Higher=lambda difference: (difference > 0).sum(), Stations="size", Smallest_difference="min")
            .to_string()
        )


if __name__ == "__main__":
    main()
