"""
Script: sarima_forecast.py

SARIMA model implementation for time series forecasting.
Source: https://www.digitalocean.com/community/tutorials/a-guide-to-time-series-forecasting-with-arima-in-python-3
"""
import pandas as pd
import matplotlib.pyplot as plt
from src.evaluation.evaluate_forecasts import evaluate_forecasts
from src.models.arima.train import train_sarima_model


def sarima_forecast(series, exog_df=None, model=None, hours_to_forecast=48, arima_order=(2, 0, 0), seasonal_order=(1, 0, 1, 24),
                    max_iter=1000, plot=True):
    """
    Fit an SARIMA model to the series and forecast values for a specified date range.

    Input
    -----
    series: Pandas Series with the time series data
    exog_df: DataFrame with exogenous variables (default is None)
    model: Optional pre-trained SARIMA model (if None, a new model will be trained)
    hours_to_forecast: Number of hours to forecast into the future (default is 48 = 2 days)
    arima_order: Tuple specifying the (p, d, q) parameters for the SARIMA model (default is (2, 0, 0))
    seasonal_order: Tuple specifying the (P, D, Q, s) seasonal parameters for the SARIMA model (default is (1, 0, 1, 24))
    max_iter: Maximum number of iterations for the model fitting (default is 1000)
    plot: Whether to display the forecast plot (default is True)

    Output
    ------
    Displays a plot comparing the observed values and the forecasted values.

    errors: Dictionary containing MAE and MSE of the forecast
    importance: Series containing feature importance of exogenous variables (if provided)
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

    # Train the SARIMA model on the training data (if no pre-trained model is provided)
    if model is not None:
        results = model
    else:
        results = train_sarima_model(train, exog_train, arima_order, seasonal_order, max_iter)

    # Get the forecast for the length of the test set
    pred = results.get_forecast(steps=len(test), exog=exog_test)
    pred_ci = pred.conf_int()

    # Create index for forecasted values (same as test index)
    forecast_index = test.index
    y_forecasted = pd.Series(pred.predicted_mean.values, index=forecast_index)

    if plot:
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
        plt.title('SARIMA Forecast (1-hour data)')

        # Generate random filename to save the plot
        random_filename = f"sarima_forecast_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(f"plot_results/{random_filename}")

    # Calculate and print the Mean Absolute Error (MAE) and Mean Squared Error (MSE) of the forecast
    errors = evaluate_forecasts(test, y_forecasted)
    print(errors)

    # Compute the feature importance for exogenous variables if provided
    importance = None
    if exog_df is not None:
        # Select the coefficients corresponding to the exogenous variables (only the used ones if LASSO was applied)
        importance = results.params[exog_train.columns].abs().sort_values(ascending=False)

    return errors, importance
