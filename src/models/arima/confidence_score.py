"""
Script: sarima_forecast.py

SARIMA model implementation for time series forecasting.
Source: https://www.digitalocean.com/community/tutorials/a-guide-to-time-series-forecasting-with-arima-in-python-3
"""
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from src.evaluation.evaluate_forecasts import evaluate_forecasts


def sarima_forecast_with_confidence_score(
        series, exog_df=None, hours_to_forecast=48, arima_order=(10, 0, 1),
        seasonal_order=(0, 0, 0, 0), max_iter=1000, plot=True
):
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
    Displays a plot comparing the observed values and the forecasted values.

    errors: Dictionary containing MAE and MSE of the forecast
    importance: Series containing feature importance of exogenous variables (if provided)
    """
    if exog_df is not None:
        if series.index.freq == pd.Timedelta("10min"):
            # Mean of previous full hour (6 × 10-minute intervals)
            window_size = 6
            exog_hour_mean = (
                exog_df.shift(1)
                .rolling(window=window_size, min_periods=window_size)
                .mean()
            )
            exog_hour_mean.columns = [f"{col}_prev_hour_mean" for col in exog_df.columns]

            exog_df = pd.concat([exog_hour_mean], axis=1).dropna()

        elif series.index.freq == pd.Timedelta("1h"):
            # Add the previous hour as lagged exogenous variable
            lags = [1]
            lagged_exogs = []
            for lag in lags:
                lagged = exog_df.shift(lag)
                lagged.columns = [f"{col}_lag{lag}" for col in exog_df.columns]
                lagged_exogs.append(lagged)

            exog_df = pd.concat(lagged_exogs, axis=1).dropna()

        # Align timestamps of target and exogenous data
        common_idx = series.index.intersection(exog_df.index)
        series = series.loc[common_idx]
        exog_df = exog_df.loc[common_idx]

    # Split the data into training and test sets
    split_date = series.index.max() - pd.DateOffset(hours=hours_to_forecast)
    train = series[series.index <= split_date]
    test = series[series.index > split_date]

    # Split training in training set and confidence score set (1 day)
    confidence_score_period = pd.DateOffset(days=1)
    confidence_test = train[train.index > split_date - confidence_score_period]
    train = train[train.index <= split_date - confidence_score_period]

    exog_train = exog_df[exog_df.index <= split_date - confidence_score_period] if exog_df is not None else None
    exog_confidence = exog_df[(exog_df.index > split_date - confidence_score_period) &
                              (exog_df.index <= split_date)] if exog_df is not None else None
    exog_test = exog_df[exog_df.index > split_date] if exog_df is not None else None

    if exog_df is not None:
        common_idx = train.index.intersection(exog_train.index)
        train = train.loc[common_idx]
        exog_train = exog_train.loc[common_idx]

        common_idx_confidence = confidence_test.index.intersection(exog_confidence.index)
        confidence_test = confidence_test.loc[common_idx_confidence]
        exog_confidence = exog_confidence.loc[common_idx_confidence]

    # Define the SARIMA model with chosen parameters
    model = sm.tsa.statespace.SARIMAX(
        endog=train,
        exog=exog_train,
        order=arima_order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,

        # Only include constant for 10-min data
        trend='c' if series.index.freq == pd.Timedelta("10min") else None
    )

    # Fit the model to the data
    results = model.fit(maxiter=max_iter, disp=False)

    print(results.summary().tables[1])

    # Get the forecast for the confidence score period
    conf_pred = results.get_forecast(steps=len(confidence_test), exog=exog_confidence)
    conf_pred_ci = conf_pred.conf_int()
    pred_forecast_index = confidence_test.index
    y_conf_forecasted = pd.Series(conf_pred.predicted_mean.values, index=pred_forecast_index)

    # Calculate the MAE and MSE for the confidence score period
    conf_errors = evaluate_forecasts(confidence_test, y_conf_forecasted)
    print("Formatted confidence score:", 1 / (1 + conf_errors['MAE']))

    # Calculate the average size of the confidence intervals
    conf_interval_sizes = conf_pred_ci.iloc[:, 1] - conf_pred_ci.iloc[:, 0]
    avg_conf_interval_size = conf_interval_sizes.mean()
    print("Average confidence interval size:", avg_conf_interval_size)

    # Add the confidence test data to the model's training data
    train = pd.concat([train, confidence_test])
    if exog_df is not None:
        exog_train = pd.concat([exog_train, exog_confidence])

    # Refit the model with the extended training data
    model = sm.tsa.statespace.SARIMAX(
        endog=train,
        exog=exog_train,
        order=arima_order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
        trend='c' if series.index.freq == pd.Timedelta("10min") else None
    )
    results = model.fit(maxiter=max_iter, disp=False)

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
        plt.title('SARIMA Forecast vs Real Data')
        plt.show()

    # Calculate and print the Mean Absolute Error (MAE) and Mean Squared Error (MSE) of the forecast
    errors = evaluate_forecasts(test, y_forecasted)
    print(errors)

    # Compute the feature importance for exogenous variables if provided
    importance = None
    if exog_df is not None:
        # Exogene parameters do not contain numbers
        importance = results.params[exog_df.columns].abs().sort_values(ascending=False)

    return errors, importance, 1 / (1 + conf_errors['MAE']), avg_conf_interval_size
