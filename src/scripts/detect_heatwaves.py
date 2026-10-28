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
from src.utils.experiment_utils import write_results_csv
from src.utils.datasets import DATASETS

PERCENTILE = 90
MIN_RELATIVE_DAYS = 3

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

    # The RMI data is 10-minute; keep the observation at each full hour, as load_station_data in src/utils/datasets.py does
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

    for dataset, dataset_info in DATASETS.items():
        tmax, tmin = daily_extremes(load_hourly(dataset_info["file"]))
        first_day, last_day = tmax.index[0], tmax.index[-1]

        print(f"\n== {dataset}: {tmax.shape[1]} stations, {first_day.date()} to {last_day.date()}")

        # The median keeps a single station, including the target of a later reconstruction, from deciding an event
        median_tmax, median_tmin = tmax.median(axis=1), tmin.median(axis=1)

        events, threshold = relative_events(median_tmax, PERCENTILE)
        print(f"relative P{PERCENTILE} ({threshold:.1f} degC), network median: {len(events)} events")

        rows += describe_events(
            dataset, f"relative P{PERCENTILE}", "network median", events, median_tmax, median_tmin, first_day, last_day
        )

    write_results_csv(OUTPUT_PATH, header, rows)

    print()
    print(pd.DataFrame(rows, columns=header).to_string(index=False))


if __name__ == "__main__":
    main()
