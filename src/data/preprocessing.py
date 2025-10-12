"""
Script: preprocessing.py
Description: Loads, preprocesses, and saves raw weather observation data.

Functionality:
1. Load CSV data file with semicolon delimiter.
2. Convert 'datetime' column to pandas datetime format.
3. Sort data by 'station_name' and 'datetime'.
4. Store the preprocessed data in a new CSV file.

Example usage:
python preprocessing.py --input data/raw/weather_data.csv --output data/processed/preprocessed_data.csv
"""

import argparse
import pandas as pd


def preprocess(input_file, output_file):
    """Loads, preprocesses, and saves weather observation data."""

    df = pd.read_csv(input_file, sep=';')

    # Convert 'datetime' column to proper pandas datetime format
    df['datetime'] = pd.to_datetime(df['datetime'], format='%d/%m/%Y %H:%M')

    # Sort by stations and datetime (and reset index so row numbers are in order)
    df = df.sort_values(by=['station_name', 'datetime']).reset_index(drop=True)

    print(df.head())

    # Save the preprocessed data to a new CSV file
    df.to_csv(output_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess weather observation data.")
    parser.add_argument("--input", required=True, help="Path to the input CSV file.")
    parser.add_argument("--output", required=True, help="Path to the output CSV file.")
    args = parser.parse_args()

    preprocess(args.input, args.output)
