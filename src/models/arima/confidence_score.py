"""
Script: sarima_forecast.py

SARIMA model implementation for time series forecasting.
Source: https://www.digitalocean.com/community/tutorials/a-guide-to-time-series-forecasting-with-arima-in-python-3
"""
import pandas as pd
import matplotlib.pyplot as plt
from src.evaluation.evaluate_forecasts import evaluate_forecasts
from src.models.arima.train import train_sarima_model


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
    # Split the data into training and test sets
    split_date = series.index.max() - pd.DateOffset(hours=hours_to_forecast)
    train = series[series.index <= split_date]
    test = series[series.index > split_date]

    # Split training in training set and confidence score set (3 days)
    confidence_score_period = pd.DateOffset(days=3)
    confidence_test = train[train.index > split_date - confidence_score_period]
    train = train[train.index <= split_date - confidence_score_period]

    exog_train, exog_confidence, exog_test = None, None, None
    if exog_df is not None:
        # Split exogenous variables into training, confidence score, and test sets
        exog_train = exog_df[exog_df.index <= split_date - confidence_score_period]
        exog_confidence = exog_df[
            (exog_df.index > split_date - confidence_score_period) & (exog_df.index <= split_date)
        ]
        exog_test = exog_df[exog_df.index > split_date]

        # Ensure that the training and confidence score sets have the same indices as the corresponding target series
        common_idx = train.index.intersection(exog_train.index)
        train = train.loc[common_idx]
        exog_train = exog_train.loc[common_idx]

        common_idx_confidence = confidence_test.index.intersection(exog_confidence.index)
        confidence_test = confidence_test.loc[common_idx_confidence]
        exog_confidence = exog_confidence.loc[common_idx_confidence]

    # Train the SARIMA model on the training data
    results = train_sarima_model(train, exog_train, arima_order, seasonal_order, max_iter)

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
    results = train_sarima_model(train, exog_train, arima_order, seasonal_order, max_iter)

    # Get the forecast for the length of the test set
    pred = results.get_forecast(steps=len(test), exog=exog_test)
    pred_ci = pred.conf_int()

    # Create index for forecasted values (same as test index)
    forecast_index = test.index
    y_forecasted = pd.Series(pred.predicted_mean.values, index=forecast_index)

    # Calculate and print the Mean Absolute Error (MAE) and Mean Squared Error (MSE) of the forecast
    errors = evaluate_forecasts(test, y_forecasted)
    print(errors)

    if plot:
        _, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

        # Top plot: observed vs forecasted values (limit always trained data to last week for better visualization)
        train_vis = train[train.index >= (test.index.min() - pd.DateOffset(days=3))]

        ax1.plot(train_vis.index, train_vis, label='Trainingsdata', color='blue', alpha=0.6, linewidth=2)
        ax1.plot(test.index, test, label='Werkelijke waarden', color='green', linewidth=2)
        ax1.plot(forecast_index, y_forecasted, label='Modelvoorspelling', color='red', linewidth=2)

        # Add confidence intervals
        ax1.fill_between(
            forecast_index,
            pred_ci.iloc[:, 0],
            pred_ci.iloc[:, 1],
            color='gray',
            alpha=0.3,
            label='Betrouwbaarheidsinterval (95%)'
        )

        ax1.axvline(x=test.index.min(), color='grey', linestyle='--', alpha=0.5, label='Start voorspelling')
        ax1.set_ylabel('Temperatuur (°C)')
        ax1.legend(loc='upper left')
        ax1.grid(alpha=0.2)

        # Bottom plot: error between forecasted and actual values
        error = y_forecasted - test

        ax2.fill_between(
            forecast_index,
            0,
            error,
            where=(error >= 0),
            alpha=0.4,
            label='Overschatting'
        )

        ax2.fill_between(
            forecast_index,
            0,
            error,
            where=(error < 0),
            alpha=0.4,
            label='Onderschatting'
        )

        ax2.axhline(0, color='black', linewidth=1)
        ax2.axvline(x=test.index.min(), color='grey', linestyle='--', alpha=0.5, label='Start voorspelling')

        # Plot the error line
        ax2.plot(forecast_index, error, color='grey', linewidth=1)

        ax2.set_ylabel('Fout (°C)')
        ax2.set_xlabel('Tijdstip')
        ax2.legend(loc='upper left')
        ax2.grid(alpha=0.2)

        ax1.axvspan(test.index.min(), test.index.max(), color='grey', alpha=0.05)
        ax2.axvspan(test.index.min(), test.index.max(), color='grey', alpha=0.05)

        # Force symmetry in the error plot by setting the same limits on the y-axis
        max_abs_error = max(abs(error.min()), abs(error.max()))
        ax2.set_ylim(-max_abs_error, max_abs_error)

        plt.tight_layout()

        station_name = series.name
        mae = errors['MAE']
        mse = errors['MSE']

        # Generate filename to save the plot
        filename = f"ARIMAX_{station_name}_{hours_to_forecast}h_MAE_{mae:.2f}_MSE_{mse:.2f}.png"
        plt.savefig(f"plots/forecasts/{filename}", bbox_inches='tight', pad_inches=0.1)

    # Compute the feature importance for exogenous variables if provided
    importance = None
    if exog_df is not None:
        # Select the coefficients corresponding to the exogenous variables (only the used ones if LASSO was applied)
        importance = results.params[exog_train.columns].abs().sort_values(ascending=False)

    return errors, importance, 1 / (1 + conf_errors['MAE']), avg_conf_interval_size
