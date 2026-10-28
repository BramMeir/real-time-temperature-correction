import pandas as pd

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
