"""
Contains functions for plotting diagnostics for ARIMA models.

Functions:
- arima_plot_diagnostics: Plots ACF and PACF for a given time series.
"""
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt


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
    _, axes = plt.subplots(1, 2, figsize=(16, 4))

    # ACF plot
    plot_acf(series, lags=100, ax=axes[0])
    axes[0].set_title('Autocorrelation Function (ACF)')

    # PACF plot
    plot_pacf(series, lags=40, ax=axes[1])
    axes[1].set_title('Partial Autocorrelation Function (PACF)')

    plt.tight_layout()
    plt.show()
