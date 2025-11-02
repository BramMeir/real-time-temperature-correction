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

    df = pd.read_csv(input_file, sep=';')

    # Convert 'datetime' column to proper pandas datetime format
    df['datetime'] = pd.to_datetime(df['datetime'], format='%d/%m/%Y %H:%M')

    # Sort by stations and datetime (and reset index so row numbers are in order)
    df = df.sort_values(by=['station_name', 'datetime']).reset_index(drop=True)

    print(df.head())

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess weather observation data.")
    parser.add_argument("--input", required=True, help="Path to the input CSV file.")
    parser.add_argument("--output", required=True, help="Path to the output CSV file.")
    parser.add_argument("--mode", choices=["standard", "turku"], required=True, help="Processing mode.")
    args = parser.parse_args()

    if args.mode == "turku":
        convert_turku(args.input, args.output)
    else:
        preprocess(args.input, args.output)
