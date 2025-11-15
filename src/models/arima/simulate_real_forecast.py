"""
Script: simulate_real_forecast.py

This script contains functions to simulate a real forecast scenario using a SARIMA model.
"""
import time
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
from src.evaluation.evaluate import evaluate_forecasts
from concurrent.futures import ProcessPoolExecutor, as_completed


def _simulate_real_forecast(i, series, exog_df, start_date, end_date, hours_to_forecast=48, arima_order=(25, 0, 0),
                            seasonal_order=(0, 0, 0, 0), max_iter=1000, retrain_step=pd.Timedelta(days=3), plot=False):
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
    training_period = pd.DateOffset(weeks=2)
    start_of_forecast = sub_series.index.max() - pd.DateOffset(hours=hours_to_forecast)
    train = sub_series[sub_series.index < (start_date + training_period)]
    test = sub_series[sub_series.index > start_of_forecast]

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

    # Add the data to the model that was not part of training and not part of testing
    # This is done in increments defined by retrain_step
    retrain_count = 0
    retrain_time = 0.0
    if retrain_step is None:
        # No retraining, all data added at once
        new_part = sub_series.loc[train.index.max() + pd.Timedelta(hours=1): test.index.min() - pd.Timedelta(hours=1)]
        new_exog_train = sub_exog.loc[new_part.index] if sub_exog is not None else None
        results = results.append(new_part, exog=new_exog_train, refit=False)

    else:
        # Add data starting from the end of training to the start of testing in increments
        current_end = train.index.max()
        final_train_end = test.index.min()

        while current_end + retrain_step < final_train_end:
            current_end += retrain_step

            # Select new training window
            new_train = sub_series.loc[:current_end]
            new_exog_train = sub_exog.loc[new_train.index] if sub_exog is not None else None

            # Refit using previous parameters as starting values
            model = sm.tsa.statespace.SARIMAX(
                new_train,
                exog=new_exog_train,
                order=arima_order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )

            # Fit the model using previous parameters as starting values (incremental retraining)
            start_time = time.time()
            results = model.fit(
                # start_params=results.params,
                maxiter=max_iter,
                disp=False
            )
            retrain_time += time.time() - start_time
            retrain_count += 1

        # If there is any remaining data before the test set, add it with refitting
        if current_end + pd.Timedelta(hours=1) < final_train_end:
            new_part = sub_series.loc[current_end + pd.Timedelta(hours=1): (final_train_end - pd.Timedelta(hours=1))]
            new_exog_train = sub_exog.loc[new_part.index] if sub_exog is not None else None

            results = results.append(new_part, exog=new_exog_train, refit=True)

    for actual_time, actual_value in test.items():
        # 1. Forecast one step ahead (use .iloc[[t]] to keep correct shape => (1, n_features))
        exog_next = exog_test.loc[[actual_time]] if exog_test is not None else None
        pred = results.get_forecast(steps=1, exog=exog_next)
        y_pred = pred.predicted_mean.iloc[0]

        # Store forecast vs actual
        forecasts.append(y_pred)
        actuals.append(actual_value)

        # # 2. Dependant on the gap_length, either add the real observed value or the forecasted value
        # if (t // gap_length) % 2 == 1:
        #     # Add the real observed value
        #     new_obs = pd.Series(test.iloc[t], index=[test.index[t]])
        # else:
        #     # Add the forecasted value
        #     new_obs = pd.Series(y_pred, index=[test.index[t]])

        # results = results.append(new_obs, exog=exog_next, refit=((t + 1) % refit_every == 0),
        #                          fit_kwargs={'maxiter': max_iter} if ((t + 1) % refit_every == 0) else {})
        results = results.append(pd.Series(y_pred, index=pd.DatetimeIndex([actual_time], freq=train.index.freq)),
                                 exog=exog_next, refit=False)

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
        # for t in range(len(test)):
        #     if t % refit_every == 0:
        #         plt.axvline(x=test.index[t], color='gray', linestyle='--', alpha=0.5)

        plt.xlabel('Datetime')
        plt.ylabel('Temperature (°C)')
        plt.legend()
        plt.title('SARIMA Real Forecast Simulation')
        plt.show()

    return metrics['MAE'], metrics['MSE'], retrain_count, retrain_time


def repeat_simulate_forecast(series, exog_df, weeks, hours_to_forecast=48, arima_order=(25, 0, 0),
                             seasonal_order=(0, 0, 0, 0), max_iter=1000, n_repeats=10, retrain_step=pd.Timedelta(days=3),
                             gap_between_training_forecasting=pd.DateOffset(months=1), random_seed=47, n_jobs=10):
    """
    Repeat the _simulate_real_forecast function multiple times and average the results.

    Input
    -----
    series: Pandas Series with the time series data
    exog_df: DataFrame with exogenous variables (optional, can be None)
    hours_to_forecast: Number of hours to forecast into the future (default is 48 = 2 days)
    arima_order: Tuple specifying the (p, d, q) parameters for the ARIMA model (default is (25, 0, 1))
    seasonal_order: Tuple specifying the (P, D, Q, s) seasonal parameters for the SARIMA model (default is (0, 0, 0, 0))
    max_iter: Maximum number of iterations for the model fitting (default is 1000)
    n_repeats: Number of times to repeat the simulation (default is 10)
    random_seed: Random seed for reproducibility (default is 47)
    n_jobs: Number of parallel jobs to run (default is 10)

    Output
    ------
    Returns lists of MAE and MSE scores from each simulation run.
    """
    np.random.seed(random_seed)
    weeks_offset = pd.DateOffset(weeks=weeks)
    hours_offset = pd.DateOffset(hours=hours_to_forecast)
    max_start = series.index.max() - weeks_offset - hours_offset - gap_between_training_forecasting

    possible_starts = series.index[(series.index >= series.index.min()) & (series.index <= max_start)]
    if len(possible_starts) == 0:
        raise ValueError("Series too short for chosen window length and forecast horizon.")

    # Pre-generate all start and end dates
    start_dates = np.random.choice(possible_starts, size=n_repeats, replace=False)
    date_ranges = [(start, start + weeks_offset + hours_offset + gap_between_training_forecasting) for start in start_dates]

    mae_scores, mse_scores = [], []
    retrain_times, retrain_counts = [], []

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = [
            executor.submit(
                _simulate_real_forecast, i, series, exog_df, start, end,
                hours_to_forecast, arima_order, seasonal_order, max_iter, retrain_step
            )
            for i, (start, end) in enumerate(date_ranges)
        ]

        for f in as_completed(futures):
            mae, mse, retrain_count, retrain_time = f.result()
            mae_scores.append(mae)
            mse_scores.append(mse)
            retrain_times.append(retrain_time)
            retrain_counts.append(retrain_count)

    print(f"\nAverage MAE across {len(mae_scores)} runs: {np.mean(mae_scores):.3f}")
    print(f"Average MSE across {len(mse_scores)} runs: {np.mean(mse_scores):.3f}")

    return mae_scores, mse_scores, retrain_counts, retrain_times


def experiment_retrain_frequency(series, exog_df, weeks, hours_to_forecast=48, arima_order=(25, 0, 0),
                                 seasonal_order=(0, 0, 0, 0), max_iter=1000, n_repeats=10, random_seed=47, n_jobs=10):
    """
    Experiment with different retraining frequencies to see their effect on forecast accuracy.

    Input
    -----
    series: Pandas Series with the time series data
    exog_df: DataFrame with exogenous variables (optional, can be None)

    Output
    ------
    None
    """
    retrain_steps = [
        pd.Timedelta(days=14),
        pd.Timedelta(days=7),
        pd.Timedelta(days=3),
        None,    # No retraining
        # pd.Timedelta(days=14),
    ]

    results = []

    for step in retrain_steps:
        print(f"\n=== Experimenting with retrain step: {step} ===")
        mae_scores, mse_scores, retrain_counts, retrain_times = repeat_simulate_forecast(
            series, exog_df, weeks=weeks, hours_to_forecast=hours_to_forecast, n_repeats=n_repeats, retrain_step=step,
            gap_between_training_forecasting=pd.DateOffset(months=1), arima_order=arima_order, seasonal_order=seasonal_order,
            max_iter=max_iter, random_seed=random_seed, n_jobs=1
        )

        results.append({
            'retrain_step': step,
            'avg_mae': np.mean(mae_scores),
            'avg_mse': np.mean(mse_scores),
            'avg_retrain_count': np.mean(retrain_counts),
            'avg_retrain_time': np.mean(retrain_times)
        })

    print("\nRetraining Frequency Experiment Results:")

    for res in results:
        print(f"Retrain Step: {res['retrain_step']}, Avg MAE: {res['avg_mae']:.3f}, Avg MSE: {res['avg_mse']:.3f}, "
              f"Avg Retrain Count: {res['avg_retrain_count']:.2f}, Avg Retrain Time: {res['avg_retrain_time']:.2f}s")

    plt.figure(figsize=(12, 6))
    plt.scatter(results["avg_retrain_time"], results["avg_mae"])

    for idx, row in results.iterrows():
        plt.annotate(str(row["retrain_step"]), (row["avg_retrain_time"], row["avg_mae"]))

    plt.xlabel("Total Retraining Time (seconds)")
    plt.ylabel("MAE")
    plt.title("Cost vs Benefit of Retraining Frequency")
    plt.show()
