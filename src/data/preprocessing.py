import argparse
import pandas as pd
import xarray as xr


def prepare_rmi(observations_file, stations_file, output_file):
    """
    Converts the RMI export: adds the station name of every station code, drops the stations with more than 10 %
    missing temperatures and keeps the columns the experiments read.

    Input
    -----
    observations_file: CSV file path with the observations, one row per station code and 10-minute time step, with at
      least the columns code, datetime (YYYY-MM-DD HH:MM:SS) and temp_dry_avg_2m.
    stations_file: CSV file path with the station list, at least the columns code and name.
    output_file: Output CSV file path.

    Output
    ------
    Saves the observations with the columns code, temp_dry_avg_2m, datetime and station_name to output_file.
    """
    df = pd.read_csv(observations_file)
    stations = pd.read_csv(stations_file)[['code', 'name']]

    # Add the station name of every station code
    df = df.merge(stations, on='code', how='left').rename(columns={'name': 'station_name'})

    df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S')

    # Sort by stations and datetime (and reset index so row numbers are in order)
    df = df.sort_values(by=['station_name', 'datetime']).reset_index(drop=True)

    missing_temp_stations = df[df['temp_dry_avg_2m'].isna()]['code'].nunique()
    print(f"Total number of stations: {df['code'].nunique()}")
    print(f"Number of stations with missing temperature data: {missing_temp_stations}")

    # Filter out the rows with a code that have more than 10% missing temperature data
    df = df.groupby('code').filter(lambda x: x['temp_dry_avg_2m'].isna().mean() <= 0.1)

    # Filter out the rows without a station name
    df = df[df['station_name'].notna()]

    temp_stats = df.groupby('station_name')['temp_dry_avg_2m'].agg(['min', 'max'])
    print("Temperature statistics for each station:")
    print(temp_stats)

    df = df[['code', 'temp_dry_avg_2m', 'datetime', 'station_name']]
    df.to_csv(output_file, index=False)


def prepare_turku(input_file, output_file):
    """
    Converts the TURCLIM file from one column per station to one row per station and time step.

    Input
    -----
    input_file: CSV file path with the date and time in the first column and one temperature column per station.
    output_file: Output CSV file path.

    Output
    ------
    Saves the observations with the columns datetime, station_name and temp_dry_avg_2m to output_file.
    """
    df = pd.read_csv(input_file)

    # Convert the first column to datetime
    datetime_col = df.columns[0]
    df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce")

    # Melt the dataframe from wide to long format
    df_long = df.melt(id_vars=[datetime_col], var_name='station_name', value_name='temp_dry_avg_2m')
    df_long = df_long.rename(columns={datetime_col: 'datetime'})

    df_long.to_csv(output_file, index=False)


def prepare_synthetic(input_file, stations_file, output_file):
    """
    Converts the synthetic NetCDF file to one row per station and time step, in °C.

    Input
    -----
    input_file: NetCDF file path with the variable tas (temperature in K) over the dimensions time and point_index.
    stations_file: CSV file path mapping every point to a station, with the columns point_index and station_name.
    output_file: Output CSV file path.

    Output
    ------
    Saves the observations with the columns datetime, temp_dry_avg_2m and station_name to output_file.
    """
    ds = xr.open_dataset(input_file)

    # Convert temperature from Kelvin to Celsius
    tas_celsius = ds["tas"] - 273.15

    # Make sure a row denotes a time step and a column a weather station
    df = tas_celsius.transpose("time", "point_index").to_pandas()

    # Move columns into rows and reset index
    df_long = df.stack().reset_index()
    df_long.columns = ["datetime", "point_index", "temp_dry_avg_2m"]

    # Convert the point_index to station names
    station_mapping = pd.read_csv(stations_file)
    df_long = df_long.merge(station_mapping, on="point_index", how="left")

    df_long = df_long[["datetime", "temp_dry_avg_2m", "station_name"]]
    df_long.to_csv(output_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert the raw data of a dataset to the file the experiments read.")
    parser.add_argument("--dataset", choices=["rmi", "turku", "synthetic"], required=True, help="Dataset to convert.")
    parser.add_argument("--input_file", required=True,
                        help="Raw observations: the RMI export (CSV), the TURCLIM file (CSV) or the synthetic NetCDF file.")
    parser.add_argument("--stations_file",
                        help="Station list (rmi: code to name) or point mapping (synthetic: point_index to station_name).")
    parser.add_argument("--output_file", required=True, help="Path to the output CSV file.")
    args = parser.parse_args()

    if args.dataset == "rmi":
        prepare_rmi(args.input_file, args.stations_file, args.output_file)
    elif args.dataset == "turku":
        prepare_turku(args.input_file, args.output_file)
    else:
        prepare_synthetic(args.input_file, args.stations_file, args.output_file)
