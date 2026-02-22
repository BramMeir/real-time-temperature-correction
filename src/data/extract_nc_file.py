"""
Script: extract_nc_file.py
Description: This script extracts temperature data from a NetCDF file containing synthetic data.
             The NetCDF file is assumed to have a variable named "tas" (temperature at surface) with
                a dimension "point_index" that corresponds to different weather stations. The script converts
                the temperature from Kelvin to Celsius and prints this to a .csv file.
"""
import xarray as xr
import pandas as pd
import argparse


def extract_temperature_data(input_file, output_file, station_file=None):
    """
    Extract temperature data from a NetCDF file and save it as a CSV file.

    Input
    -----
    input_file: Path to the input NetCDF file.
    output_file: Path to the output CSV file.
    station_file: (Optional) Path to a CSV file mapping point_index to station names.

    Output
    ------
    Saves the extracted temperature data to the specified output CSV file.
    """
    # Open the NetCDF dataset
    ds = xr.open_dataset(input_file)

    # Extract the temperature variable (tas)
    tas = ds["tas"]

    # Convert temperature from Kelvin to Celsius
    tas_celsius = tas - 273.15
    tas_celsius.attrs["units"] = "°C"

    # Make sure a row denotes a time step and a column a weather station
    df = tas_celsius.transpose("time", "point_index").to_pandas()

    # Move columns into rows and reset index
    df_long = df.stack().reset_index()
    df_long.columns = ["datetime", "point_index", "temp_dry_avg_2m"]
    # print(df_long)

    # Convert the point_index to station names using the given station mapping
    if station_file:
        station_mapping = pd.read_csv(station_file)
        df_long = df_long.merge(
            station_mapping,
            on="point_index",
            how="left"
        )

    # Select and reorder columns
    df_long = df_long[["datetime", "temp_dry_avg_2m", "station_name"]]

    df_long.to_csv(output_file, index=False)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Extract temperature data from NetCDF file and save as CSV")
    parser.add_argument("--input_file", type=str, default="data/Synthetic/extracted_points_tas.nc", help="Path to the input NetCDF file")
    parser.add_argument("--output", type=str, default="data/Synthetic/temperature_data.csv", help="Path to the output CSV file")
    parser.add_argument("--station_file", type=str, default="data/Synthetic/stations_mapping.csv", help="Path to station mapping CSV file")
    args = parser.parse_args()

    # Extract temperature data and save to CSV
    extract_temperature_data(args.input_file, args.output, args.station_file)
    print(f"Temperature data extracted and saved to {args.output}")
