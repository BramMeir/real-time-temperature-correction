"""
Script: arima_forecast.py

ARIMA model implementation for time series forecasting.
Source: https://www.digitalocean.com/community/tutorials/a-guide-to-time-series-forecasting-with-arima-in-python-3
"""
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from src.evaluation.evaluate import evaluate_forecasts


def arima_forecast(series, hours_to_forecast=48, arima_order=(25, 0, 1), max_iter=1000):
    """
    Fit an ARIMA model to the series and forecast values for a specified date range.

    Input
    -----
    series: Pandas Series with the time series data
    hours_to_forecast: Number of hours to forecast into the future (default is 48 = 2 days)
    arima_order: Tuple specifying the (p, d, q) parameters for the ARIMA model (default is (25, 0, 1))
    max_iter: Maximum number of iterations for the model fitting (default is 1000)

    Output
    ------
    Displays a plot comparing the observed values and the forecasted values, and prints the MSE of the forecast.
    """
    # Split the data into training and test sets
    split_date = series.index.max() - pd.DateOffset(hours=hours_to_forecast)
    train = series[series.index <= split_date]
    test = series[series.index > split_date]

    # Define the ARIMA model with chosen parameters (p=25, d=0, q=1)
    model = sm.tsa.ARIMA(
        train,
        order=arima_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    # Fit the model to the data
    results = model.fit(method_kwargs={"maxiter": max_iter})

    print(results.summary().tables[1])

    # Get the forecast for the length of the test set
    pred = results.get_forecast(steps=len(test))
    pred_ci = pred.conf_int()

    # Create index for forecasted values (same as test index)
    forecast_index = test.index
    y_forecasted = pd.Series(pred.predicted_mean.values, index=forecast_index)

    # Plot observed (train + test) and forecasted values
    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train, label='Training data', color='blue')
    plt.plot(test.index, test, label='Real future data', color='green')
    plt.plot(forecast_index, y_forecasted, label='Forecast', color='red')

    # Add confidence intervals
    plt.fill_between(
        forecast_index,
        pred_ci.iloc[:, 0],
        pred_ci.iloc[:, 1],
        color='gray',
        alpha=0.3,
        label='Confidence interval'
    )

    plt.xlabel('Datetime')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.title('ARIMA Forecast vs Real Data')
    plt.show()

    # Calculate and print the Mean Squared Error (MSE) of the forecast
    print(evaluate_forecasts(test, y_forecasted))


if __name__ == "__main__":
    # Load the preprocessed data
    df = pd.read_csv("data/preprocessed.csv")

    # Select a single station for analysis
    station = "Melle AWS"
    station_data = df[df['station_name'] == station]

    # Use the temperature column as the time series data
    series = pd.Series(station_data['temp_dry_avg_2m'].values, index=pd.to_datetime(station_data['datetime']))

    # Set the frequency of the time series to 10-minute intervals
    series = series.asfreq('10min')

    # Select only the last month of data
    last_month_start = series.index.max() - pd.DateOffset(months=6)
    series = series[series.index >= last_month_start]

    # Resample to hourly data to reduce noise
    series = series.resample('1h').mean()

    # Perform grid search to find the best ARIMA parameters
    arima_forecast(series)
    # arima_grid_search(series, p_values=range(5, 31, 10), d_values=range(0, 1), q_values=range(0, 2), max_workers=6)
    # arima_plot_diagnostics(series)
