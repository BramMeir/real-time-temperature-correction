"""
Script: simulate_real_forecast.py


"""
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
from src.evaluation.evaluate import evaluate_forecasts
from concurrent.futures import ProcessPoolExecutor, as_completed


def _simulate_real_forecast(i, series, exog_df, start_date, end_date, hours_to_forecast=48, arima_order=(25, 0, 0),
                            seasonal_order=(0, 0, 0, 0), max_iter=1000, plot=False):
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
    # Extract the sub-series for the current run
    sub_series = series[(series.index >= start_date) & (series.index <= end_date)]
    sub_exog = None

    if exog_df is not None:
        sub_exog = exog_df.loc[sub_series.index]

    print(f"\n🔹 Run {i+1}: using data from {start_date} to {end_date}")

    # Split the data into training and test sets
    split_date = sub_series.index.max() - pd.DateOffset(hours=hours_to_forecast)
    train = sub_series[sub_series.index <= split_date]
    test = sub_series[sub_series.index > split_date]

    # Create exogenous variables for training and test sets
    exog_train = sub_exog.loc[train.index] if sub_exog is not None else None
    exog_test = sub_exog.loc[test.index] if sub_exog is not None else None

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
    refit_every = 96            # Refit the model every 'gap_length' steps

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
                                 fit_kwargs={'maxiter': max_iter} if ((t + 1) % refit_every == 0) else {})

    # Evaluate how well the forecasts performed
    forecast_series = pd.Series(forecasts, index=test.index)
    metrics = evaluate_forecasts(test, forecast_series)
    print(metrics)

    # Plot observed (train + test) and forecasted values
    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(train.index, train, label='Training data', color='blue')
        plt.plot(test.index, test, label='Real future data', color='green')
        plt.plot(forecast_series, label='Forecast', color='red')

        # Plot with vertical lines indicating where the refitting happened
        for t in range(len(test)):
            if t % refit_every == 0:
                plt.axvline(x=test.index[t], color='gray', linestyle='--', alpha=0.5)

        plt.xlabel('Datetime')
        plt.ylabel('Temperature (°C)')
        plt.legend()
        plt.title('SARIMA Real Forecast Simulation')
        plt.show()

    return metrics['MAE'], metrics['MSE']


def repeat_simulate_forecast(series, exog_df, weeks, hours_to_forecast=48, arima_order=(25, 0, 0),
                             seasonal_order=(0, 0, 0, 0), max_iter=1000, n_repeats=10,
                             random_seed=47, n_jobs=10):
    """
    Repeat the simulate_real_forecast function multiple times and average the results.

    Input
    -----
    series: Pandas Series with the time series data
    exog_df: DataFrame with exogenous variables (optional, can be None)
    hours_to_forecast: Number of hours to forecast into the future (default is 48 = 2 days)
    arima_order: Tuple specifying the (p, d, q) parameters for the ARIMA model (default is (25, 0, 1))
    seasonal_order: Tuple specifying the (P, D, Q, s) seasonal parameters for the SARIMA model (default is (0, 0, 0, 0))
    max_iter: Maximum number of iterations for the model fitting (default is 1000)
    n_repeats: Number of times to repeat the simulation (default is 10)

    Output
    ------
    Displays a plot comparing the observed values and the averaged forecasted values,
    and returns the averaged MAE and MSE of the forecasts.
    """
    np.random.seed(random_seed)
    weeks_offset = pd.DateOffset(weeks=weeks)
    hours_offset = pd.DateOffset(hours=hours_to_forecast)
    max_start = series.index.max() - weeks_offset - hours_offset

    possible_starts = series.index[(series.index >= series.index.min()) & (series.index <= max_start)]
    if len(possible_starts) == 0:
        raise ValueError("Series too short for chosen window length and forecast horizon.")

    # Pre-generate all start and end dates
    start_dates = np.random.choice(possible_starts, size=n_repeats, replace=False)
    date_ranges = [(start, start + weeks_offset + hours_offset) for start in start_dates]

    mae_scores, mse_scores = [], []

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = [
            executor.submit(
                _simulate_real_forecast, i, series, exog_df, start, end,
                hours_to_forecast, arima_order, seasonal_order, max_iter
            )
            for i, (start, end) in enumerate(date_ranges)
        ]

        for f in as_completed(futures):
            mae, mse = f.result()
            mae_scores.append(mae)
            mse_scores.append(mse)

    print(f"\nAverage MAE across {len(mae_scores)} runs: {np.mean(mae_scores):.3f}")
    print(f"Average MSE across {len(mse_scores)} runs: {np.mean(mse_scores):.3f}")

    return mae_scores, mse_scores
