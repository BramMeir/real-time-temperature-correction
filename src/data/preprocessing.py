"""
Script: preprocessing.py
Description: Loads, preprocesses, and saves raw weather observation data.

Example usage:
python -m src.data.preprocessing --input ./data/Turku_1H_LI.csv --output ./data/Turku_preprocessed.csv --mode turku
"""

import argparse
import pandas as pd


def preprocess(input_file, output_file):
    """
    Loads, preprocesses, and saves weather observation data.

    Input
    -----
    input_file: Input CSV file path.
    output_file: Output CSV file path.

    Output
    ------
    Saves the preprocessed data to the specified output CSV file.
    """
    df = pd.read_csv(input_file, sep=',')

    # Convert 'datetime' column to proper pandas datetime format
    df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S')

    # Sort by stations and datetime (and reset index so row numbers are in order)
    df = df.sort_values(by=['station_name', 'datetime']).reset_index(drop=True)

    # Print the number of stations that have missing temperature data
    missing_temp_stations = df[df['temp_dry_avg_2m'].isna()]['code'].nunique()
    print(f"Number of stations with missing temperature data: {missing_temp_stations}")

    # Filter out the rows with a code that have more than 10% missing temperature data
    df = df.groupby('code').filter(lambda x: x['temp_dry_avg_2m'].isna().mean() <= 0.1)

    # Filter out the rows without a station name
    df = df[df['station_name'].notna()]

    # Show for every station the min and max temperature
    temp_stats = df.groupby('station_name')['temp_dry_avg_2m'].agg(['min', 'max'])
    print("Temperature statistics for each station:")
    print(temp_stats)

    print(df.head())

    # Keep only relevant columns
    df = df[['code', 'temp_dry_avg_2m', 'datetime', 'station_name']]

    # Save the preprocessed data to a new CSV file
    df.to_csv(output_file, index=False)


def convert_turku(input_file, output_file):
    """
    Specific processing for Turku data:
    - Original formatting: "Datatime,Station 1, Station 2,...".
    - Convert to: "datetime, station_name, temp_dry_avg_2m".

    Input
    -----
    input_file: Input CSV file path.
    output_file: Output CSV file path.

    Output
    ------
    Saves the reformatted data to the specified output CSV file.
    """
    df = pd.read_csv(input_file)

    # Convert the first column to datetime
    datetime_col = df.columns[0]
    df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce")

    # Melt the dataframe from wide to long format
    df_long = df.melt(id_vars=[datetime_col], var_name='station_name', value_name='temp_dry_avg_2m')

    # Rename the datetime column
    df_long = df_long.rename(columns={datetime_col: 'datetime'})

    # Save the reformatted data to a new CSV file
    df_long.to_csv(output_file, index=False)


def include_station_metadata(data_file, metadata_file, output_file):
    """
    Merges station metadata into the main dataframe.

    Input
    -----
    df: Main dataframe with weather observations.
    metadata_file: CSV file path containing station metadata.

    Output
    ------
    Merged dataframe with station metadata included.
    """
    # Load the metadata file
    metadata_df = pd.read_csv(metadata_file)

    # Keep only relevant fields
    metadata_df = metadata_df[['code', 'name']]

    # Load the main data file
    df = pd.read_csv(data_file)

    # Merge the data based on the code column
    merged_df = df.merge(metadata_df, on='code', how='left')

    # Rename this new name column to station_name
    merged_df = merged_df.rename(columns={'name': 'station_name'})

    # Save the merged dataframe to a new CSV file
    merged_df.to_csv(output_file, index=False)

    # Print the first few rows of the merged dataframe
    print(merged_df.head())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess weather observation data.")
    parser.add_argument("--input_file", required=True, help="Path to the input CSV file.")
    parser.add_argument("--output", required=True, help="Path to the output CSV file.")
    parser.add_argument("--mode", choices=["standard", "turku", "include_metadata"], required=True, help="Processing mode.")
    parser.add_argument("--metadata", help="Path to the station metadata CSV file (required if mode is include_metadata).")
    args = parser.parse_args()

    if args.mode == "turku":
        convert_turku(args.input_file, args.output)
    elif args.mode == "include_metadata":
        if not args.metadata:
            parser.error("--metadata is required when mode is include_metadata")
        include_station_metadata(args.input_file, args.metadata, args.output)
    else:
        preprocess(args.input_file, args.output)
