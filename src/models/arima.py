"""
Source: https://www.digitalocean.com/community/tutorials/a-guide-to-time-series-forecasting-with-arima-in-python-3
ARIMA model implementation for time series forecasting.
Functionality:

"""
import itertools
import warnings
import statsmodels.api as sm
import matplotlib.pyplot as plt
from src.evaluation.evaluate import evaluate_forecasts
from concurrent.futures import ProcessPoolExecutor, as_completed


def _fit_arima(params, series):
    """Helper function to fit ARIMA model for a given parameter tuple."""
    try:
        model = sm.tsa.ARIMA(
            series,
            order=params,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        results = model.fit()
        return params, results.aic
    except Exception:
        return params, float("inf")  # invalid combinations return large AIC


def arima_grid_search(series, p_values=range(0, 51, 10), d_values=range(0, 3), q_values=range(0, 3), max_workers=None):
    """
    Perform a parallel grid search to find the best ARIMA model parameters based on AIC.
    """
    pdq = list(itertools.product(p_values, d_values, q_values))
    warnings.filterwarnings("ignore")

    print(f"Starting parallel grid search with {len(pdq)} combinations...")

    best_aic = float("inf")
    best_param = None

    # Use all CPU cores by default (or specify with max_workers)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_fit_arima, param, series): param for param in pdq}

        for i, future in enumerate(as_completed(futures), 1):
            param = futures[future]
            try:
                params, aic = future.result()
                print(f"[{i}/{len(pdq)}] ARIMA{params} - AIC: {aic}")
                if aic < best_aic:
                    best_aic = aic
                    best_param = params
            except Exception as e:
                print(f"Failed for {param}: {e}")

    print(f"\n✅ Best ARIMA{best_param} - AIC: {best_aic:.2f}")
    return best_param


def arima_forecast_one_step_ahead(series):
    """
    Fit an ARIMA model to the series and plot the one-step ahead forecast.

    Input
    -----
    series: Pandas Series with the time series data

    Output
    ------
    Displays a plot comparing the observed values and the one-step ahead forecast.    
    """
    # Define the ARIMA model with chosen parameters (p=3, d=0, q=1)
    model = sm.tsa.ARIMA(
        series,
        order=(3, 0, 1),
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    # Fit the model to the data
    results = model.fit()

    # Plot the diagnostics of the model to assess its fit
    # results.plot_diagnostics(figsize=(20, 14))
    # plt.show()

    print(results.summary().tables[1])

    # Get the forecast from the specified start date, performing a one-step ahead forecast (dynamic=False).
    # This means that each forecasted value is based on all prior observed values, not on prior forecasts.
    # This is useful for evaluating the model's performance.
    pred = results.get_prediction(start=pd.to_datetime("2025-08-01 00:00:00"), dynamic=False)

    # Get the 95% confidence intervals of the forecasts (shows the uncertainty of the forecasts)
    pred_ci = pred.conf_int()

    # Plot the original time series
    ax = series.plot(label='observed')

    # Add the one-step ahead forecast to the plot (alpha is the transparency)
    pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7)

    # Add the confidence intervals to the plot
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.2)

    ax.set_xlabel('Date')
    ax.set_ylabel('Temperature (°C)')
    plt.legend()

    plt.show()

    # Calculate and print the Mean Squared Error (MSE) of the forecast
    y_forecasted = pred.predicted_mean
    y_truth = series[pred_ci.index]
    print(evaluate_forecasts(y_truth, y_forecasted))


def arima_forecast(series):
    """
    Fit an ARIMA model to the series and forecast values for a specified date range.

    Input
    -----
    series: Pandas Series with the time series data

    Output
    ------
    Returns a Pandas Series with the forecasted values for the specified date range.
    """
    # Split the data into training and test sets
    split_date = series.index.max() - pd.DateOffset(days=7)
    train = series[series.index <= split_date]
    test = series[series.index > split_date]

    # Define the ARIMA model with chosen parameters (p=3, d=0, q=1)
    model = sm.tsa.ARIMA(
        train,
        order=(25, 0, 1),
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    # Fit the model to the data
    results = model.fit()

    print(results.summary().tables[1])

    # Get the forecast for the next day
    pred = results.get_forecast(steps=len(test))  # 144 steps of 10 minutes = 1 day
    pred_ci = pred.conf_int()

    # 4. Create index for forecasted values (same as test index)
    forecast_index = test.index
    y_forecasted = pd.Series(pred.predicted_mean.values, index=forecast_index)

    # 5. Plot observed (train + test) and forecasted values
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
    plt.title('ARIMA Forecast vs Real Data (Last 2 Days)')
    plt.show()

    # Calculate and print the Mean Squared Error (MSE) of the forecast
    print(evaluate_forecasts(test, y_forecasted))


if __name__ == "__main__":
    import pandas as pd

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
    arima_grid_search(series)
