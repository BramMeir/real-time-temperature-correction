"""
Script: simulate_real_forecast.py


"""
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from src.evaluation.evaluate import evaluate_forecasts


def simulate_real_forecast(series, exog_df, hours_to_forecast=48, arima_order=(25, 0, 0),
                           seasonal_order=(0, 0, 0, 0), max_iter=1000):
    """
    Simulate a real forecast scenario using a SARIMA model to forecast values for a specified date range.
    This includes adding the observed future values into the model to further forecast.

    Input
    -----
    series: Pandas Series with the time series data
    exog_df: DataFrame with exogenous variables (optional, can be None)
    hours_to_forecast: Number of hours to forecast into the future (default is 48 = 2 days)
    arima_order: Tuple specifying the (p, d, q) parameters for the ARIMA model (default is (25, 0, 1))
    seasonal_order: Tuple specifying the (P, D, Q, s) seasonal parameters for the SARIMA model (default is (0, 0, 0, 0))
    max_iter: Maximum number of iterations for the model fitting (default is 1000)

    Output
    ------
    Displays a plot comparing the observed values and the forecasted values,
    and returns the MAE and MSE of the forecast.
    """
    # Split the data into training and test sets
    split_date = series.index.max() - pd.DateOffset(hours=hours_to_forecast)
    train = series[series.index <= split_date]
    test = series[series.index > split_date]

    # Create exogenous variables for training and test sets
    exog_train = exog_df.loc[train.index] if exog_df is not None else None
    exog_test = exog_df.loc[test.index] if exog_df is not None else None

    # Define the SARIMA model with chosen parameters
    model = sm.tsa.statespace.SARIMAX(
        train,
        exog=exog_train,
        order=arima_order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    # Fit the model to the data
    results = model.fit(maxiter=max_iter, disp=False)

    print(results.summary().tables[1])

    forecasts = []
    actuals = []

    # Simulate real-time forecasting
    gap_length = 48             # 48 hours no real data, then again 48 hours real data, etc.
    refit_every = 96    # Refit the model every 'gap_length' steps

    for t in range(len(test)):
        # 1. Forecast one step ahead (use .iloc[[t]] to keep correct shape => (1, n_features))
        exog_next = exog_test.iloc[[t]] if exog_test is not None else None
        pred = results.get_forecast(steps=1, exog=exog_next)
        y_pred = pred.predicted_mean.iloc[0]

        # Store forecast vs actual
        forecasts.append(y_pred)
        actuals.append(test.iloc[t])

        # 2. Dependant on the gap_length, either add the real observed value or the forecasted value
        if (t // gap_length) % 2 == 1:
            # Add the real observed value
            new_obs = pd.Series(test.iloc[t], index=[test.index[t]])
        else:
            # Add the forecasted value
            new_obs = pd.Series(y_pred, index=[test.index[t]])

        results = results.append(new_obs, exog=exog_next, refit=((t + 1) % refit_every == 0),
                                 fit_kwargs={'maxiter': max_iter, 'disp': False} if ((t + 1) % refit_every == 0) else {})

    # Evaluate how well the forecasts performed
    forecast_series = pd.Series(forecasts, index=test.index)
    metrics = evaluate_forecasts(test, forecast_series)
    print(metrics)

    # Plot observed (train + test) and forecasted values
    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train, label='Training data', color='blue')
    plt.plot(test.index, test, label='Real future data', color='green')
    plt.plot(forecast_series, label='Forecast', color='red')

    plt.xlabel('Datetime')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.title('SARIMA Real Forecast Simulation')
    plt.show()

    return metrics
