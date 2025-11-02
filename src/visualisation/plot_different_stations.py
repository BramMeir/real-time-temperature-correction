"""
Script: plot_different_stations.py

Plot and compare temperature (or other variable) readings from different weather stations over a specified date range.

Example usage:
python -m src.visualisation.plot_different_stations --path data/weather_stations.csv --start 2025-09-01
        --end 2025-10-01 --station-col station_name --datetime-col datetime --value-col temp_dry_avg_2m --resample 1h
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def plot_stations_data(csv_path, start_date, end_date, station_col, datetime_col, value_col, resample_freq=None):
    """
    Plot and compare the specified variable from different weather stations over a given date range.

    Input
    -----
    csv_path: Path to the CSV file containing the data
    start_date: Start date for the plot (string format, e.g., '2025-09-01')
    end_date: End date for the plot (string format, e.g., '2025-10-01')
    station_col: Column name for station identifiers
    datetime_col: Column name for datetime values
    value_col: Column name for the variable to plot (e.g., 'temp_dry_avg_2m')
    resample_freq: Optional resampling frequency (e.g., '1h', '10min'). If None, no resampling is done.

    Output
    ------
    Displays a plot comparing the specified variable across different stations.
    """
    # Load the data
    df = pd.read_csv(csv_path)
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df = df.set_index(datetime_col)

    # Filter based on date range
    mask = (df.index >= start_date) & (df.index <= end_date)
    df = df.loc[mask]

    # Pivot to have stations as columns
    pivot_df = df.pivot(columns=station_col, values=value_col)

    # Resample if needed
    if resample_freq:
        pivot_df = pivot_df.resample(resample_freq).mean()

    plt.figure(figsize=(12, 6))

    # Plot each station's data
    for col in pivot_df.columns:
        plt.plot(pivot_df.index, pivot_df[col], label=col)

    plt.title(f"{value_col} Comparison ({start_date} to {end_date})")
    plt.xlabel("Time")
    plt.ylabel(value_col)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_difference_stations(csv_path, start_date, end_date, station_col, datetime_col,
                             value_col, base_station=None, resample_freq=None):
    """
    Plot the rolling mean difference between every two stations over time.

    Input
    -----
    csv_path: Path to the CSV file containing the data
    start_date: Start date for the plot (string format, e.g., '2025-09-01')
    end_date: End date for the plot (string format, e.g., '2025-10-01')
    station_col: Column name for station identifiers
    datetime_col: Column name for datetime values
    value_col: Column name for the variable to analyze (e.g., 'temp_dry_avg_2m')
    base_station: Station to use as base for differences, if None, all pairs are compared
    resample_freq: Optional resampling frequency (e.g., '1h', '10min'). If None, no resampling is done.

    Output
    ------
    Plots the rolling mean difference between every two stations.
    """
    # Load the data
    df = pd.read_csv(csv_path)
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df = df.set_index(datetime_col)

    # Pivot to have stations as columns
    pivot_df = df.pivot(columns=station_col, values=value_col)

    # Resample if needed
    if resample_freq:
        pivot_df = pivot_df.resample(resample_freq).mean()

    if base_station and base_station in pivot_df.columns:
        plt.figure(figsize=(12, 6))
        for col in pivot_df.columns:
            if col != base_station:
                diff = pivot_df[base_station] - pivot_df[col]
                mean_rolling = diff.rolling('1h').mean()

                # Filter based on date range
                mask = (mean_rolling.index >= start_date) & (mean_rolling.index <= end_date)
                plt.plot(mean_rolling.index[mask], mean_rolling[mask], label=f"{base_station} - {col}")

    else:
        # Plot the rolling mean difference between every two stations
        for i in range(len(pivot_df.columns)):
            for j in range(i + 1, len(pivot_df.columns)):
                station_a = pivot_df.columns[i]
                station_b = pivot_df.columns[j]
                diff_AB = pivot_df[station_a] - pivot_df[station_b]
                mean_rolling = diff_AB.rolling('3h').mean()

                # Filter based on date range
                mask = (mean_rolling.index >= start_date) & (mean_rolling.index <= end_date)
                plt.plot(mean_rolling.index[mask], mean_rolling[mask], label=f"{station_a} & {station_b}")

    plt.xlabel("Time")
    plt.ylabel("Mean Difference (station A - station B) last 3 hours" if not base_station
               else f"Mean Difference from {base_station} last 3 hours")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot temperature (or other variable) comparisons between weather stations.")
    parser.add_argument("--path", type=str, required=True, help="Path to CSV file containing the data.")
    parser.add_argument("--start", type=str, required=True, help="Start date (e.g., 2025-09-01).")
    parser.add_argument("--end", type=str, required=True, help="End date (e.g., 2025-10-01).")
    parser.add_argument("--station-col", type=str, default="station_name", help="Column name for station identifiers.")
    parser.add_argument("--datetime-col", type=str, default="datetime", help="Column name for datetime values.")
    parser.add_argument("--value-col", type=str, default="temp_dry_avg_2m", help="Column name for the variable to plot.")
    parser.add_argument("--resample", type=str, default="1h", help="Optional resampling frequency (e.g., '1h', '10min').")
    parser.add_argument("--base-station", type=str, default=None,
                        help="Base station for difference plotting. If None, all pairs are compared.")

    args = parser.parse_args()

    # plot_stations_data(
    #     csv_path=args.path,
    #     start_date=args.start,
    #     end_date=args.end,
    #     station_col=args.station_col,
    #     datetime_col=args.datetime_col,
    #     value_col=args.value_col,
    #     resample_freq=args.resample
    # )

    plot_difference_stations(
        csv_path=args.path,
        start_date=args.start,
        end_date=args.end,
        station_col=args.station_col,
        datetime_col=args.datetime_col,
        value_col=args.value_col,
        base_station=args.base_station,
        resample_freq=args.resample
    )
