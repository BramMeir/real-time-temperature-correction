"""
Script: sarima_forecast.py

SARIMA model implementation for time series forecasting.
Source: https://www.digitalocean.com/community/tutorials/a-guide-to-time-series-forecasting-with-arima-in-python-3
"""
import pandas as pd
from pmdarima.arima import ARIMA
import matplotlib.pyplot as plt
from src.evaluation.evaluate import evaluate_forecasts


def sarima_forecast(series, exog_df=None, hours_to_forecast=48, arima_order=(10, 0, 1), seasonal_order=(3, 0, 1, 24),
                    max_iter=1000, plot=True):
    """
    Fit an SARIMA model to the series and forecast values for a specified date range.

    Input
    -----
    series: Pandas Series with the time series data
    exog_df: DataFrame with exogenous variables (default is None)
    hours_to_forecast: Number of hours to forecast into the future (default is 48 = 2 days)
    arima_order: Tuple specifying the (p, d, q) parameters for the SARIMA model (default is (25, 0, 1))
    seasonal_order: Tuple specifying the (P, D, Q, s) seasonal parameters for the SARIMA model (default is (0, 1, 1, 24))
    max_iter: Maximum number of iterations for the model fitting (default is 1000)
    plot: Whether to display the forecast plot (default is True)

    Output
    ------
    Displays a plot comparing the observed values and the forecasted values,
    and returns the MAE and MSE of the forecast.
    """
    # Split the data into training and test sets
    split_date = series.index.max() - pd.DateOffset(hours=hours_to_forecast)
    train = series[series.index <= split_date]
    test = series[series.index > split_date]

    exog_train = exog_df[exog_df.index <= split_date] if exog_df is not None else None
    exog_test = exog_df[exog_df.index > split_date] if exog_df is not None else None

    if exog_df is not None:
        common_idx = train.index.intersection(exog_train.index)
        train = train.loc[common_idx]
        exog_train = exog_train.loc[common_idx]

    # Define the SARIMA model with chosen parameters
    model = ARIMA(
        order=arima_order,
        seasonal_order=seasonal_order,
        maxiter=max_iter,
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    # Fit the model to the data
    results = model.fit(y=train, X=exog_train)

    print(results.summary().tables[1])

    # Get the forecast for the length of the test set
    pred, pred_ci = results.predict(n_periods=len(test), X=exog_test, return_conf_int=True)

    # Create index for forecasted values (same as test index)
    forecast_index = test.index
    y_forecasted = pd.Series(pred.values, index=forecast_index)

    if plot:
        # Plot observed (train + test) and forecasted values
        plt.figure(figsize=(12, 6))
        plt.plot(train.index, train, label='Training data', color='blue')
        plt.plot(test.index, test, label='Real future data', color='green')
        plt.plot(forecast_index, y_forecasted, label='Forecast', color='red')

        # Add confidence intervals
        plt.fill_between(
            forecast_index,
            pred_ci[:, 0],
            pred_ci[:, 1],
            color='gray',
            alpha=0.3,
            label='Confidence interval'
        )

        plt.xlabel('Datetime')
        plt.ylabel('Temperature (°C)')
        plt.legend()
        plt.title('SARIMA Forecast vs Real Data')
        plt.show()

    # Calculate and print the Mean Absolute Error (MAE) and Mean Squared Error (MSE) of the forecast
    errors = evaluate_forecasts(test, y_forecasted)
    print(errors)

    return errors
