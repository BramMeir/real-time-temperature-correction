"""
Module for performing the Augmented Dickey-Fuller test to check if a time series is stationary.

Functions:
- stationary_test: Perform the ADF test and plot rolling statistics.
"""

import statsmodels.api as sm


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
