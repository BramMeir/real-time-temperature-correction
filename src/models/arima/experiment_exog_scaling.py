"""
Module: experiment_exog_scaling.py

Description:
This module contains a function to measure the runtime behaviour of the ARIMA forecasting process when varying the
number of exogenous stations included in the model. For each k in k_values, it selects the top k exogenous stations
(based on correlation with the target station) and evaluates the timing of the forecasting process over multiple
repeats.

Functionality:
- experiment_exog_scaling: Main function to run the experiment with different numbers of exogenous stations and collect results.
"""
import pandas as pd
import matplotlib.pyplot as plt
from src.models.arima.repeat_forecast import repeat_forecasts


def experiment_exog_scaling(
    series,
    full_exog_df,
    k_values,
    weeks=2,
    hours_to_forecast=48,
    arima_order=(2, 0, 0),
    seasonal_order=(1, 0, 1, 24),
    n_repeats=10,
    max_iter=1000,
    n_jobs=4,
):
    """
    Measures the runtime behaviour of the ARIMA forecasting process when varying the number of
    exogenous stations included in the model. For each k in k_values, it selects the top k exogenous stations
    (based on correlation with the target station) and evaluates the timing of the forecasting process over multiple repeats.

    Input:
    - series: The target time series to forecast.
    - full_exog_df: A DataFrame containing the exogenous variables (other stations).
    - k_values: A list of integers specifying the number of exogenous stations to include in the model.
    - weeks: Number of weeks to use for training in each repeat.
    - hours_to_forecast: Number of hours to forecast in each repeat.
    - arima_order: The (p, d, q) order of the ARIMA model.
    - seasonal_order: The (P, D, Q, s) seasonal order of the ARIMA model.
    - n_repeats: Number of times to repeat the forecasting process for each k.
    - max_iter: Maximum number of iterations for the ARIMA model fitting.
    - n_jobs: Number of parallel jobs to run for the repeats.

    Output:
    A DataFrame containing the average MAE and MSE for each k in k_values.
    """
    results = []

    for k in k_values:
        print(f"Running experiment with {k} exogenous stations")

        if k == 0:
            exog_subset = None
        else:
            selected_cols = full_exog_df.columns[:k]
            exog_subset = full_exog_df[selected_cols]

        result_metrices = repeat_forecasts(
            series,
            exog_df=exog_subset,
            weeks=weeks,
            hours_to_forecast=hours_to_forecast,
            arima_order=arima_order,
            seasonal_order=seasonal_order,
            n_repeats=n_repeats,
            max_iter=max_iter,
            n_jobs=n_jobs,
        )

        results.append({
            "k_exog": k,
            "mae_mean": result_metrices['mae_mean'],
            "mse_mean": result_metrices['mse_mean'],
            "duration_mean_seconds": result_metrices['duration_mean_seconds']
        })

    return pd.DataFrame(results)


def plot_exog_scaling_results(csv_path, save_path="output/experiment_exog_scaling.png", log_scale=False):
    """
    Plot runtime behaviour as a function of the number of exogenous stations.

    Input:
    - csv_path: Path to the CSV file containing the results of the experiment.
    - save_path: Path to save the generated plot.
    - log_scale: Whether to use a logarithmic scale for the y-axis.
    """
    df = pd.read_csv(csv_path)

    plt.figure()

    plt.plot(df["k_exog"], df["duration_mean_seconds"])
    plt.xlabel("Aantal exogene stations")
    plt.ylabel("Gemiddelde duur (seconden)")
    plt.title("Runtime vs Aantal exogene stations")

    if log_scale:
        plt.yscale("log")

    plt.tight_layout()
    plt.savefig(save_path)
