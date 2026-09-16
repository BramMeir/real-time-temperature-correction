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
from src.models.regression_sarima_errors.repeat_forecast import repeat_forecasts

# Orders selected on the stage-one residuals, as used elsewhere for this model (see
# short_horizon_comparison_experiment.py, test_aging_model_RegressionSARIMAErrors.py)
RESIDUAL_ORDER, RESIDUAL_SEASONAL_ORDER = (3, 0, 0), (1, 0, 0, 24)


def run_station_experiment(df, target_station, weeks, resample, hours_to_forecast):
    """
    Run confidence evaluation for a single station.

    Input:
    -----
    df: pandas DataFrame containing the preprocessed data
    target_station: Name of the target weather station
    weeks: Number of weeks of training data to include per repeat
    resample: Resampling interval
    hours_to_forecast: Number of hours to forecast into the future

    Output:
    ------
    result: Dictionary containing the evaluation results for the target station
    """
    # Pivot the long-format data into a wide, hourly DataFrame: target station plus every neighbour
    other_stations = [s for s in df['station_name'].unique() if s != target_station]

    station_data = df[df['station_name'] == target_station]
    series = pd.Series(
        station_data['temp_dry_avg_2m'].values,
        index=pd.to_datetime(station_data['datetime'])
    ).asfreq('1h').dropna()

    exog_data = df[df['station_name'].isin(other_stations)]
    exog_pivot = exog_data.pivot_table(
        index='datetime', columns='station_name', values='temp_dry_avg_2m'
    )
    exog_pivot.index = pd.to_datetime(exog_pivot.index)
    exog_df = exog_pivot.reindex(series.index)

    series = series.resample(resample).mean().interpolate(limit_direction="both")
    exog_df = exog_df.resample(resample).mean().interpolate(limit_direction="both")

    df_wide = series.to_frame(name=target_station).join(exog_df)
    exog_cols = [c for c in df_wide.columns if c != target_station]

    # Run repeated forecasts with confidence
    results = repeat_forecasts(
        df_wide,
        target_station,
        exog_cols=exog_cols,
        weeks=weeks,
        hours_to_forecast=hours_to_forecast,
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
                        help="Number of weeks of training data per repeat (default: 8).")
    parser.add_argument("--resample", type=str, default="1h",
                        help="Resampling interval, e.g. '10min', '30min', '1h' (default: '1h').")
    parser.add_argument("--hours_to_forecast", type=int, default=168,
                        help="Number of hours to forecast into the future (default: 168).")
    parser.add_argument("--station_index", type=int, default=0,
                        help="Index of the target station to analyze (default: 0).")
    parser.add_argument("--input_file", type=str, default="data/Synthetic/temperature_data.csv",
                        help="Path to the preprocessed data CSV file (default: 'data/Synthetic/temperature_data.csv').")
    args = parser.parse_args()

    # Read the preprocessed data
    df = pd.read_csv(args.input_file)

    # Get the list of unique stations and select the target station based on the provided index
    stations = sorted(df['station_name'].unique())
    target_station = stations[args.station_index]

    # Run experiment
    result = run_station_experiment(
        df,
        target_station,
        weeks=args.weeks,
        resample=args.resample,
        hours_to_forecast=args.hours_to_forecast
    )

    # Make sure the output directory exists
    os.makedirs("output/confidence_analysis_regression_sarima_errors", exist_ok=True)

    # Save the result to a CSV file
    pd.DataFrame([result]).to_csv(
        f"output/confidence_analysis_regression_sarima_errors/confidence_station_{args.station_index}.csv",
        index=False
    )
