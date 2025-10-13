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


def arima_grid_search(series):
    """
    Perform a grid search to find the best ARIMA model parameters based on AIC.

    Input
    -----
    series: Time series data (1D array-like)

    Output
    ------
    Prints the AIC for each combination of (p, d, q) parameters.
    """
    # Define the d and q parameters to take any value between 0 and 1
    q = d = range(0, 2)
    # Define the p parameters to take any value between 0 and 3
    p = range(0, 100, 10)

    # Generate all different combinations of p, q and q triplets
    pdq = list(itertools.product(p, d, q))

    # Some parameter combinations may not work; we will ignore the errors for those
    warnings.filterwarnings("ignore")

    for param in pdq:
        try:
            model = sm.tsa.ARIMA(
                series,
                order=param,                    # (p,d,q) order of the model
                enforce_stationarity=False,
                enforce_invertibility=False
            )

            results = model.fit()

            # Use the AIC (Akaike Information Criterion) to evaluate the model.
            # This measures how well a model fits the data, while taking into account the overall complexity of the model.
            # Lower AIC values indicate a better fit.
            print('ARIMA{} - AIC:{}'.format(param, results.aic))

        except Exception:
            continue


def arima_forecast(series):
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
    pred = results.get_prediction(start=pd.to_datetime("2025-08-01 00:00:00"), dynamic=True, full_results=True)

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

    # Perform grid search to find the best ARIMA parameters
    arima_grid_search(series)
