"""
Contains functions for plotting diagnostics for ARIMA models.

Functions:
- arima_plot_diagnostics: Plots ACF and PACF for a given time series.
"""
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import numpy as np


def arima_plot_diagnostics(series):
    """
    Plots the ACF and PACF for a given time series. This is useful for diagnosing
    the properties of the series and for identifying appropriate ARIMA model parameters.

    Input
    -----
    series: 1D array-like time series data

    Output
    ------
    Displays ACF and PACF plots.
    """
    # Create two subplots: one for ACF, one for PACF
    _, axes = plt.subplots(1, 2, figsize=(14, 6))
    max_lag = 100

    # ACF plot
    plot_acf(series, lags=max_lag, ax=axes[0])
    axes[0].set_title('Autocorrelatiefunctie (ACF)')

    # PACF plot
    plot_pacf(series, lags=max_lag, ax=axes[1])
    axes[1].set_title('Partiële Autocorrelatiefunctie (PACF)')

    # Set x-ticks in multiples of 24
    ticks = np.arange(0, max_lag + 1, 24)

    for ax in axes:
        ax.set_xticks(ticks)
        ax.set_xlim(0, max_lag)

    plt.tight_layout()
    plt.savefig('ACF_PACF_example.png')
