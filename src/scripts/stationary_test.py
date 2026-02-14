"""
Module for performing the Augmented Dickey-Fuller test to check if a time series is stationary.

Functions:
- stationary_test: Perform the ADF test and plot rolling statistics.
"""

import statsmodels.api as sm
import pandas as pd
import argparse


def stationary_test(series):
    """
    Perform the Augmented Dickey-Fuller test to check if the time series is stationary.
    If the p-value is less than 0.05, the series is considered stationary (we reject the null hypothesis
    that it is non-stationary).

    Input
    -----
    series: Pandas Series with the time series data

    Output
    ------
    Prints the results of the ADF test, including the test statistic, p-value, and critical values.
    """
    # Perform the ADF test
    adf_result = sm.tsa.stattools.adfuller(series)

    # Print the results
    print('ADF Statistic: %f' % adf_result[0])
    print('p-value: %f' % adf_result[1])
    print('Critical Values:')
    for key, value in adf_result[4].items():
        print('\t%s: %.3f' % (key, value))


if __name__ == "__main__":
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description='Perform the Augmented Dickey-Fuller test on a time series.')
    parser.add_argument('--csv_file', type=str, help='Path to the CSV file containing the time series data.')
    parser.add_argument('--station', type=str, default='Sint_Baafs_Gent', help='Name of the station to analyze (default: Sint_Baafs_Gent).')
    args = parser.parse_args()

    # Load the time series data from the specified CSV file
    data = pd.read_csv(args.csv_file)

    # Select the data points for a specific station
    station_data = data[data['station_name'] == args.station]

    # Select 2 weeks of data between two dates
    station_data = station_data[(station_data['datetime'] >= '2020-05-01') & (station_data['datetime'] <= '2020-12-01')]

    # Perform the ADF test on the selected station's data
    stationary_test(station_data['temp_dry_avg_2m'])
