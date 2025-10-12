"""
Script: forecast_baseline.py
Description: Generates baseline temperature forecasts per station, based on historical data.

Functionality:
- Load the preprocessed data
- Generate forecasts using simple methods (persistence and moving average)
- Evaluate performance using MSE and MAE

Voorbeeld gebruik:
python forecast_baseline.py --input data/processed/preprocessed_data.csv --output results/baseline_forecast.csv 
        --lookback 1440 --horizon 144 --nr_periods_to_plot 2
"""

import pandas as pd
import numpy as np
import argparse
from evaluation.evaluate import evaluate_forecasts
from data.create_dataset import create_dataset
from visualisation.plot_forecast import plot_forecast_comparison


# ----------------------------
# Baseline models
# ----------------------------

def persistence_forecast(X, horizon=1):
    """
    Predict the future values as the last observed value in the input window.

    Input
    -----
    X: Numpy array with shape (num_series, lookback) for which to make predictions
    horizon: Number of future points to predict

    Output
    ------
    Numpy array with shape (num_series, horizon) with the forecasts
    """
    return np.array([np.repeat(x[-1], horizon) for x in X])


def moving_average_forecast(X, horizon=1, window=6):
    """
    Predict the future values as the average of the last 'window' values.

    Input
    -----
    X: Numpy array with shape (num_series, lookback) for which to make predictions
    horizon: Number of future points to predict

    Output
    ------
    Numpy array with shape (num_series, horizon) with the forecasts
    """
    forecasts = []

    for x in X:
        preds = []
        history = list(x[-window:])

        for _ in range(horizon):
            next_pred = np.mean(history[-window:])
            preds.append(next_pred)
            history.append(next_pred)  # include prediction, so it can be used for next step
        forecasts.append(preds)

    return np.array(forecasts)

# ----------------------------
# Main script
# ----------------------------


def main(input_file, output_file, lookback=4320, forecast_horizon=1, nr_periods_to_plot=0):
    """
    Applies baseline forecasting methods to each station in the dataset and evaluates their performance.

    Input
    -----
    input_file: Path to the preprocessed CSV file
    output_file: Path to save the CSV file with results
    lookback: Number of past time steps to use for making predictions (4320 = 30 days)
    forecast_horizon: Number of time steps to forecast ahead (1 = 10 minutes)
    nr_periods_to_plot: Number of latest forecasting periods to plot (0 = no plots)

    Output
    ------
    Saves a CSV file with the evaluation results for each station and model.
    """
    df = pd.read_csv(input_file)
    df['datetime'] = pd.to_datetime(df['datetime'])

    results = []

    for station, group in df.groupby('station_name'):
        group = group.sort_values('datetime').reset_index(drop=True)
        series = group['temp_dry_avg_2m'].values

        # Make input-output pairs for forecasting
        X, y_true = create_dataset(series, lookback, forecast_horizon)

        # Baseline 1: Persistence forecast
        y_pred_pers = persistence_forecast(X, horizon=forecast_horizon)
        metrics_pers = evaluate_forecasts(y_true, y_pred_pers)

        # Baseline 2: Moving average forecast
        y_pred_ma = moving_average_forecast(X, horizon=forecast_horizon, window=6)
        metrics_ma = evaluate_forecasts(y_true, y_pred_ma)

        results.append({
            "station": station,
            "model": "persistence",
            **metrics_pers
        })
        results.append({
            "station": station,
            "model": "moving_average",
            **metrics_ma
        })

        if nr_periods_to_plot:
            print(f"Plotting results for station: {station}")
            plot_forecast_comparison(
                timestamps=group['datetime'].values[-(nr_periods_to_plot * (lookback + forecast_horizon)):],
                X=X[-nr_periods_to_plot:],
                y_true=y_true[-nr_periods_to_plot:],
                y_pred=y_pred_pers[-nr_periods_to_plot:],
                title=f"{station} - Persistence Forecast",
                horizon=forecast_horizon
            )
            plot_forecast_comparison(
                timestamps=group['datetime'].values[-(nr_periods_to_plot * (lookback + forecast_horizon)):],
                X=X[-nr_periods_to_plot:],
                y_true=y_true[-nr_periods_to_plot:],
                y_pred=y_pred_ma[-nr_periods_to_plot:],
                title=f"{station} - Moving Average Forecast",
                horizon=forecast_horizon
            )

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print(f"Results stored in {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline forecasting")
    parser.add_argument("--input", type=str, required=True, help="Path to the preprocessed CSV")
    parser.add_argument("--output", type=str, required=True, help="Path to the output CSV with results")
    parser.add_argument("--lookback", type=int, default=4320,
                        help="Number of past time steps to use for making predictions (4320 = 30 days)")
    parser.add_argument("--horizon", type=int, default=1,
                        help="Number of time steps to forecast ahead (1 = 10 minutes)")
    parser.add_argument("--nr_periods_to_plot", type=int, default=0,
                        help="Number of latest forecasting periods to plot (0 = no plots)")
    args = parser.parse_args()

    main(args.input, args.output, lookback=args.lookback, forecast_horizon=args.horizon, 
         nr_periods_to_plot=args.nr_periods_to_plot)
