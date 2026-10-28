"""
Script: train.py

SARIMA model implementation for time series forecasting.
Source: https://www.digitalocean.com/community/tutorials/a-guide-to-time-series-forecasting-with-arima-in-python-3
"""
import statsmodels.api as sm


def train_sarima_model(series, exog_df=None, arima_order=(2, 0, 0), seasonal_order=(1, 0, 1, 24), max_iter=1000):
    """
    Fit an SARIMA model to the series.

    Input
    -----
    series: Pandas Series with the time series data
    exog_df: DataFrame with exogenous variables (default is None)
    arima_order: Tuple specifying the (p, d, q) parameters for the SARIMA model (default is (2, 0, 0))
    seasonal_order: Tuple specifying the (P, D, Q, s) seasonal parameters for the SARIMA model (default is (1, 0, 1, 24))
    max_iter: Maximum number of iterations for the model fitting (default is 1000)

    Output
    ------
    Fitted SARIMA model results.
    """
    # Define the SARIMA model with chosen parameters
    model = sm.tsa.statespace.SARIMAX(
        endog=series,
        exog=exog_df,
        order=arima_order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,

        # Always include a constant, so this stays comparable to the two-stage
        # regression-with-SARIMA-errors model, whose regression stage already
        # captures the series' level.
        trend='c'
    )

    # Fit the model to the data
    results = model.fit(maxiter=max_iter, disp=False)

    print(results.summary().tables[1])

    return results
