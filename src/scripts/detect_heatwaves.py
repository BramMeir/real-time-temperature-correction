"""
Script: detect_heatwaves.py

Count the heat waves in the three datasets, to see how many events an extreme-conditions experiment can be built on.

The events are defined on the network rather than on any target station, so that selecting them does not condition on
the values a reconstruction is scored against. Two definitions are used:
- "relative": at least MIN_RELATIVE_DAYS consecutive days on which the network-median daily Tmax exceeds a percentile of
  all summer (JJA) days of that dataset. Portable across climates, so it is the one that works for Turku as well.
- "RMI": the RMI definition, a run of at least 5 consecutive days with Tmax >= 25 degC of which at least 3 reach
  Tmax >= 30 degC. Applied to the network median, and for the RMI network also to Uccle, where RMI applies it.

Days are calendar days of the timestamps as stored. Each event is flagged as usable when the record holds the training
window plus the longest lead before it, and the whole event after it.

Run:
    python -m src.scripts.detect_heatwaves
"""
import pandas as pd
from src.evaluation.experiment_utils import write_results_csv

DATASETS = {
    "KMI": "data/Full_AWS/preprocessed_10m_2022_2025.csv",
    "TURKU": "data/Turku/Turku_preprocessed.csv",
    "SYNTHETIC": "data/Synthetic/temperature_data.csv"
}

# Station whose series RMI uses for its official heat wave count, per dataset that contains it
REFERENCE_STATIONS = {"KMI": "UCCLE"}

PERCENTILES = [90, 95]
MIN_RELATIVE_DAYS = 3

# RMI definition
RMI_HOT_THRESHOLD, RMI_MIN_HOT_DAYS = 25, 5
RMI_VERY_HOT_THRESHOLD, RMI_MIN_VERY_HOT_DAYS = 30, 3

# An onset may lie up to MAX_LEAD_DAYS before the event start, with TRAINING_WEEKS of training before it
TRAINING_WEEKS = 8
MAX_LEAD_DAYS = 14

OUTPUT_PATH = "output/heatwaves/heatwave_events.csv"


def load_hourly(file):
    """
    Full-hour observations of every station, as the models see them.

    Input
    -----
    file: Long-format CSV with station_name, datetime and temp_dry_avg_2m

    Output
    ------
    hourly: DataFrame indexed by hour, one column per station
    """
    df = pd.read_csv(file, parse_dates=["datetime"])

    hourly = df.pivot_table(index="datetime", columns="station_name", values="temp_dry_avg_2m")

    # The RMI data is 10-minute; keep the observation at each full hour, as prepare_station_data does
    return hourly[hourly.index.minute == 0]


def daily_extremes(hourly):
    """
    Daily Tmax and Tmin per station.

    Input
    -----
    hourly: DataFrame indexed by hour, one column per station

    Output
    ------
    tmax, tmin: DataFrames indexed by day, one column per station
    """
    days = hourly.groupby(hourly.index.floor("D"))

    # The records start and end part-way through a day (Turku at 22:00, RMI at 15:50), and the extremes of a part day
    # are not that day's extremes
    complete = days.size() == 24

    return days.max()[complete], days.min()[complete]


def runs(mask):
    """
    Consecutive runs of True days.

    Input
    -----
    mask: Boolean Series indexed by consecutive days

    Output
    ------
    runs: List of (first day, last day) tuples
    """
    run_id = (mask != mask.shift()).cumsum()

    return [
        (group.index[0], group.index[-1])
        for _, group in mask[mask].groupby(run_id[mask])
    ]


def relative_events(tmax, percentile):
    """
    Runs of at least MIN_RELATIVE_DAYS days above the given percentile of the summer daily Tmax.

    Input
    -----
    tmax: Daily Tmax Series
    percentile: Percentile of the JJA days that a hot day must exceed

    Output
    ------
    events: List of (first day, last day) tuples
    threshold: The temperature threshold in degC
    """
    summer = tmax[tmax.index.month.isin([6, 7, 8])]
    threshold = summer.quantile(percentile / 100)

    events = [
        (start, end) for start, end in runs(tmax > threshold)
        if (end - start).days + 1 >= MIN_RELATIVE_DAYS
    ]

    return events, threshold


def rmi_events(tmax):
    """
    Heat waves under the RMI definition.

    Input
    -----
    tmax: Daily Tmax Series

    Output
    ------
    events: List of (first day, last day) tuples
    """
    return [
        (start, end) for start, end in runs(tmax >= RMI_HOT_THRESHOLD)
        if (end - start).days + 1 >= RMI_MIN_HOT_DAYS
        and (tmax[start:end] >= RMI_VERY_HOT_THRESHOLD).sum() >= RMI_MIN_VERY_HOT_DAYS
    ]


def describe_events(dataset, definition, series_name, events, tmax, tmin, first_day, last_day):
    """
    One row per event, with its length, intensity and whether the experiment can use it.

    Input
    -----
    dataset, definition, series_name: Labels for the rows
    events: List of (first day, last day) tuples
    tmax, tmin: Daily Tmax and Tmin of the series the events were detected on
    first_day, last_day: First and last complete day of the record

    Output
    ------
    rows: List of rows matching the header in main
    """
    rows = []

    for start, end in events:
        history_days = (start - first_day).days
        usable = (history_days >= TRAINING_WEEKS * 7 + MAX_LEAD_DAYS) and (end <= last_day)

        rows.append([
            dataset,
            definition,
            series_name,
            start.date(),
            end.date(),
            (end - start).days + 1,
            round(tmax[start:end].max(), 1),
            round(tmax[start:end].mean(), 1),
            round(tmin[start:end].mean(), 1),
            history_days,
            usable
        ])

    return rows


def main():
    header = [
        "Dataset", "Definition", "Series", "Start", "End", "Days", "Peak_Tmax", "Mean_Tmax", "Mean_Tmin",
        "History_days", "Usable"
    ]
    rows = []

    for dataset, file in DATASETS.items():
        tmax, tmin = daily_extremes(load_hourly(file))
        first_day, last_day = tmax.index[0], tmax.index[-1]

        print(f"\n== {dataset}: {tmax.shape[1]} stations, {first_day.date()} to {last_day.date()}")

        # The median keeps a single station, including the target of a later reconstruction, from deciding an event
        series = {"network median": (tmax.median(axis=1), tmin.median(axis=1))}

        if dataset in REFERENCE_STATIONS:
            station = REFERENCE_STATIONS[dataset]
            series[station] = (tmax[station], tmin[station])

        for series_name, (series_tmax, series_tmin) in series.items():
            if series_name == "network median":
                for percentile in PERCENTILES:
                    events, threshold = relative_events(series_tmax, percentile)
                    print(f"relative P{percentile} ({threshold:.1f} degC), {series_name}: {len(events)} events")

                    rows += describe_events(
                        dataset, f"relative P{percentile}", series_name, events, series_tmax, series_tmin,
                        first_day, last_day
                    )

            events = rmi_events(series_tmax)
            print(f"RMI, {series_name}: {len(events)} events")

            rows += describe_events(
                dataset, "RMI", series_name, events, series_tmax, series_tmin, first_day, last_day
            )

    write_results_csv(OUTPUT_PATH, header, rows)

    print()
    print(pd.DataFrame(rows, columns=header).to_string(index=False))


if __name__ == "__main__":
    main()
