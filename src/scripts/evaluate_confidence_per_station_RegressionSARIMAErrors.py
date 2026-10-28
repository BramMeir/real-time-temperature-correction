"""
Script: evaluate_confidence_per_station_RegressionSARIMAErrors.py

Description:
Evaluates model performance and confidence score for a single target station, for the two-stage
regression with SARIMA errors model. Mirrors evaluate_confidence_per_station.py (ARIMAX) so the
two models' results are directly comparable with plot_confidence_analysis.py /
plot_confidence_analysis_RegressionSARIMAErrors.py: same station ordering, same output CSV columns.

For the selected station:
- Runs repeated forecasts
- Computes average MAE/MSE
- Computes average confidence score
- Computes average bootstrapped interval width
"""
import argparse
import os
import pandas as pd
from src.utils.datasets import DATASETS, load_station_data
from src.models.regsarima.repeat_forecast import repeat_forecasts

# Orders selected on the stage-one residuals, as used elsewhere for this model (see
# short_horizon_comparison_experiment.py, test_aging_model_RegressionSARIMAErrors.py)
RESIDUAL_ORDER, RESIDUAL_SEASONAL_ORDER = (3, 0, 0), (1, 0, 0, 24)


def run_station_experiment(df, target_station, weeks, hours_to_forecast, calibration_days):
    """
    Run confidence evaluation for a single station.

    Input:
    -----
    df: pandas DataFrame containing the preprocessed data
    target_station: Name of the target weather station
    weeks: Number of weeks of training data to include per repeat, the calibration window included
    hours_to_forecast: Number of hours to forecast into the future
    calibration_days: Length of the held-out calibration window in days

    Output:
    ------
    result: Dictionary containing the evaluation results for the target station
    """
    df_wide = load_station_data(df, target_station)
    exog_cols = list(df_wide.columns[1:])

    # Run repeated forecasts with confidence
    results = repeat_forecasts(
        df_wide,
        target_station,
        exog_cols=exog_cols,
        weeks=weeks,
        hours_to_forecast=hours_to_forecast,
        calibration_days=calibration_days,
        arima_order=RESIDUAL_ORDER,
        seasonal_order=RESIDUAL_SEASONAL_ORDER,
        n_repeats=20,
        random_seed=47,
        max_iter=1000,
        n_jobs=10
    )

    result = {
        "target_station": target_station,
        "mae_mean": results["mae_mean"],
        "mse_mean": results["mse_mean"],
        "confidence_mean": results["confidence_score_mean"],
        "interval_size_mean": results["interval_width_mean"],
        "duration_seconds": results["duration_mean_seconds"]
    }

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate confidence score for a single station, for the two-stage regression"
                     " with SARIMA errors model."
    )
    parser.add_argument("--weeks", type=int, default=8,
                        help="Number of weeks of training data per repeat, the calibration window included (default: 8).")
    parser.add_argument("--hours_to_forecast", type=int, default=168,
                        help="Number of hours to forecast into the future (default: 168).")
    parser.add_argument("--calibration_days", type=int, default=14,
                        help="Length of the held-out calibration window in days (default: 14).")
    parser.add_argument("--station_index", type=int, default=0,
                        help="Index of the target station to analyze (default: 0).")
    args = parser.parse_args()

    # Read the preprocessed data
    df = pd.read_csv(DATASETS["SYNTHETIC"]["file"])

    # Get the list of unique stations and select the target station based on the provided index
    stations = sorted(df['station_name'].unique())
    target_station = stations[args.station_index]

    # Run experiment
    result = run_station_experiment(
        df,
        target_station,
        weeks=args.weeks,
        hours_to_forecast=args.hours_to_forecast,
        calibration_days=args.calibration_days
    )

    # Make sure the output directory exists
    os.makedirs("output/confidence_analysis_regression_sarima_errors", exist_ok=True)

    # Save the result to a CSV file
    pd.DataFrame([result]).to_csv(
        f"output/confidence_analysis_regression_sarima_errors/confidence_station_{args.station_index}.csv",
        index=False
    )
