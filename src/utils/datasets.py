import pandas as pd

# Longest run of missing hours that load_station_data may fill by interpolation inside a window. A longer run would
# enter the window as a straight line instead of a temperature record
MAX_GAP_HOURS = 24

# The file each experiment reads per dataset, and the target stations of the experiments that reconstruct only part of
# a network
DATASETS = {
    "RMI": {
        "file": "data/RMI/observations.csv",
        "stations": ["MELLE", "DIEPENBEEK", "HUMAIN", "SINT-KATELIJNE-WAVER", "ERNAGE"]
    },
    "TURKU": {
        "file": "data/Turku/observations.csv",
        "stations": ["Betel", "Virastotalo", "Ylijoki", "Kurala"]
    },
    "SYNTHETIC": {
        "file": "data/Synthetic/observations.csv",
        "stations": [
            "Stadhuis_Brussel_Grote_Markt",
            "Stadhuis_Antwerpen_Grote_Markt",
            "Grote_Markt_Kortrijk",
            "Tielt",
            "Gembloux",
            "Slag_om_Ardennen_Museum_La_Roche_en_Ardenne",
            "Abdij_Tongerlo"
        ]
    }
}


def load_station_data(df, station):
    """
    Build the hourly data of one target station and the rest of its network.

    Input
    -----
    df: DataFrame of one dataset file, with the columns datetime, station_name and temp_dry_avg_2m
    station: Name of the target station

    Output
    ------
    df_complete: DataFrame with an hourly datetime index over the target's record, the target station as the first
      column and every other station after it, with gaps interpolated
    """
    station_data = df[df['station_name'] == station]

    series = pd.Series(
        station_data['temp_dry_avg_2m'].values,
        index=pd.to_datetime(station_data['datetime'])
    ).asfreq('1h').dropna()

    other_stations = [s for s in df['station_name'].unique() if s != station]
    exog_data = df[df['station_name'].isin(other_stations)]

    exog_pivot = exog_data.pivot_table(
        index='datetime', columns='station_name', values='temp_dry_avg_2m'
    )
    exog_pivot.index = pd.to_datetime(exog_pivot.index)

    # Remove the missing timestamps and align with the target series
    exog_df = exog_pivot.reindex(series.index)

    series = series.resample('1h').mean().interpolate(limit_direction="both")
    exog_df = exog_df.resample('1h').mean().interpolate(limit_direction="both")

    return series.to_frame(name=station).join(exog_df)


def usable_hours(df, station, index):
    """
    Mark which hours of a target's data an outage may cover and which a window may contain.

    Input
    -----
    df: DataFrame of one dataset file, with the columns datetime, station_name, temp_dry_avg_2m and validated
    station: Name of the target station
    index: Hourly datetime index of the target's data, as returned by load_station_data

    Output
    ------
    usable: DataFrame on index with two boolean columns. validated is true where the target has a validated
      observation, so an outage may be scored against it. complete is true where no station lies in a run of more
      than MAX_GAP_HOURS missing hours, so a window holds no long interpolated stretch
    """
    observations = df.assign(datetime=pd.to_datetime(df["datetime"]))
    temperatures = observations.pivot(index="datetime", columns="station_name", values="temp_dry_avg_2m").reindex(index)
    target = observations[observations["station_name"] == station].set_index("datetime")

    # Length of the run of missing hours that every missing hour belongs to
    missing = temperatures.isna()
    run_lengths = missing.apply(
        lambda gaps: gaps.groupby((gaps != gaps.shift()).cumsum()).transform("size").where(gaps, 0)
    )

    return pd.DataFrame({
        "validated": target["validated"].reindex(index),
        "complete": ~(run_lengths > MAX_GAP_HOURS).any(axis=1),
    }, index=index)
