"""
Script: forecast_with_filtered_stations.py

Description:
This script tries to determine the optimal number of stations to include as exogenous variables in an ARIMA forecasting model
for a target station. It does this by:
1. Ranking all the stations based on their importance for forecasting the target station.
2. Iteratively testing the forecast performance (MAE, MSE) of the ARIMA model using the top k stations as exogenous variables.

Example usage:
python -m src.models.arima.forecast_with_filtered_stations
       --target_station "Sint_Baafs_Gent" --input_file "data/Synthetic/temperature_data.csv""
"""

import argparse
import pandas as pd
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from src.models.arima.repeat_forecast import repeat_forecasts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ARIMA forecast utilities.")
    parser.add_argument("--ranking_method", type=str, default="none",
                        choices=["pearson", "LASSO", "elasticnet", "none"],
                        help="Method to rank stations by importance (default: 'none', so all stations are used).")
    parser.add_argument("--weeks", type=int, default=2,
                        help="Number of weeks of data to include (default: 2).")
    parser.add_argument("--resample", type=str, default="1h",
                        help="Resampling interval, e.g. '10min', '30min', '1h' (default: '1h').")
    parser.add_argument("--hours_to_forecast", type=int, default=48,
                        help="Number of hours to forecast into the future (default: 48).")
    parser.add_argument("--target_station", type=str, default="Melle AWS",
                        help="Name of the target weather station (default: 'Melle AWS').")
    parser.add_argument("--input_file", type=str, default="data/Part_AWS/preprocessed.csv",
                        help="Path to the preprocessed data CSV file (default: 'data/Part_AWS/preprocessed.csv').")
    args = parser.parse_args()

    # Read the preprocessed data
    df = pd.read_csv(args.input_file)

    # Select target station
    station_data = df[df['station_name'] == args.target_station]

    # Create target time series with a datetime index
    series = pd.Series(
        station_data['temp_dry_avg_2m'].values,
        index=pd.to_datetime(station_data['datetime'])
    ).asfreq('1h').dropna()

    # Create exogenous DataFrame
    other_stations = [s for s in df['station_name'].unique() if s != args.target_station]
    exog_data = df[df['station_name'].isin(other_stations)]

    # Pivot to get each station as a separate column
    exog_pivot = exog_data.pivot_table(
        index='datetime', columns='station_name', values='temp_dry_avg_2m'
    )

    # Create datetime index
    exog_pivot.index = pd.to_datetime(exog_pivot.index)

    # Remove the missing timestamps and align with the main series
    exog_df = exog_pivot.reindex(series.index)

    # Resample data by taking the mean
    series = series.resample(args.resample).mean().interpolate(limit_direction="both")
    exog_df = exog_df.resample(args.resample).mean().interpolate(limit_direction="both")

    # DF to hold the top-k exogenous variables after ranking
    top_exog_df = None

    # Rank stations based on the chosen method
    if args.ranking_method == "pearson":
        ranking = exog_df.corrwith(series).abs().sort_values(ascending=False)

        # Select top-k stations
        top_k = 23
        top_exog_df = exog_df[ranking.index[:top_k]]

    elif args.ranking_method == "LASSO":
        # Timing for LASSO ranking
        start_time = pd.Timestamp.now()

        # Align and drop NaNs
        X = exog_df.copy()
        y = series.copy()

        valid_idx = X.dropna().index.intersection(y.dropna().index)
        X = X.loc[valid_idx]
        y = y.loc[valid_idx]

        # Standardize features (important for LASSO)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Cross-validated LASSO
        lasso = LassoCV(
            cv=5,
            random_state=47,
            n_jobs=10,
            max_iter=10000
        )
        lasso.fit(X_scaled, y.values)

        coefs = pd.Series(lasso.coef_, index=X.columns)
        ranking = coefs.abs().sort_values(ascending=False)

        # Select the stations with non-zero coefficients
        ranking = ranking[ranking > 0]
        top_exog_df = exog_df[ranking.index]

        print("Station ranking based on LASSO coefficients:")
        print(ranking)
        print(f"Time taken for LASSO ranking: {(pd.Timestamp.now() - start_time).total_seconds():.2f} seconds")

    elif args.ranking_method == "elasticnet":
        start_time = pd.Timestamp.now()

        X = exog_df.copy()
        y = series.copy()

        valid_idx = X.dropna().index.intersection(y.dropna().index)
        X = X.loc[valid_idx]
        y = y.loc[valid_idx]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Elastic Net with cross-validation
        elastic = ElasticNetCV(
            l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
            cv=5,
            random_state=47,
            n_jobs=10,
            max_iter=20000
        )

        elastic.fit(X_scaled, y.values)

        coefs = pd.Series(elastic.coef_, index=X.columns)
        ranking = coefs.abs().sort_values(ascending=False)

        ranking = ranking[ranking > 0]
        top_exog_df = exog_df[ranking.index]

        print("Station ranking based on Elastic Net coefficients:")
        print(ranking)
        print(f"Best l1_ratio selected: {elastic.l1_ratio_}")
        print(f"Best alpha selected: {elastic.alpha_}")
        print(f"Time taken for Elastic Net ranking: {(pd.Timestamp.now() - start_time).total_seconds():.2f} seconds")

    else:
        top_exog_df = exog_df

    results_all = repeat_forecasts(
        series,
        exog_df=top_exog_df,
        weeks=args.weeks,
        hours_to_forecast=args.hours_to_forecast,
        arima_order=(2, 0, 0),
        seasonal_order=(1, 0, 1, 24),
        confidence_score=False,
        n_repeats=30,
        random_seed=47,
        max_iter=1000,
        n_jobs=10,
    )

    print(f"Average MAE across 30 runs: {results_all['mae_mean']:.4f}")
    print(f"Average MSE across 30 runs: {results_all['mse_mean']:.4f}")
    print(f"Average duration per run: {results_all['duration_mean_seconds']:.2f}")
    print(f"\nAverage feature importance top-{len(top_exog_df.columns)} stations:")
    print(results_all['importance'])
