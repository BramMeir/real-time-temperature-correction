import argparse
import pandas as pd
import numpy as np
from src.models.sarima.grid_search import sarima_grid_search
from src.models.sarima.plot_diagnostics import arima_plot_diagnostics
from src.models.sarima.stationarity_test import stationarity_test

# Orders searched for the paper. The differencing is not searched, see sarima_grid_search: the augmented Dickey-Fuller
# test (--mode stationarity) rejects a unit root, so d = D = 0
AR_ORDERS = range(0, 49)
MA_ORDERS = range(0, 6)
SEASONAL_AR_ORDERS = range(0, 4)
SEASONAL_MA_ORDERS = range(0, 4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select the orders of SARIMA and SARIMAX.")
    parser.add_argument("--mode", choices=["grid_search", "diagnostics", "stationarity"], default="grid_search",
                        help="Grid search over the orders, plot the ACF and PACF of the series, or test it for a unit "
                             "root.")
    parser.add_argument("--weeks", type=int, default=2,
                        help="Number of weeks of data to include (default: 2).")
    parser.add_argument("--seed", type=int, default=47,
                        help="Seed of the random choice of the weeks of data (default: 47).")
    parser.add_argument("--resample", type=str, default="1h",
                        help="Resampling interval, e.g. '10min', '30min', '1h' (default: '1h').")
    parser.add_argument("--exog", action="store_true",
                        help="Include the other stations as exogenous variables (SARIMAX) if set, else SARIMA.")
    parser.add_argument("--target_station", type=str, default="MELLE",
                        help="Name of the target weather station (default: 'MELLE').")
    parser.add_argument("--input_file", type=str, default="data/RMI/observations.csv",
                        help="Path to the preprocessed data CSV file (default: 'data/RMI/observations.csv').")
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
    series.name = args.target_station

    # Create exogenous DataFrame if requested
    exog_df = None
    if args.exog:
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

    # Shorten the data to the specified number of weeks
    total_weeks = (series.index.max().year - series.index.min().year) * 52 + \
                  (series.index.max().month - series.index.min().month) * 4 + \
                  (series.index.max().day - series.index.min().day) // 7

    if total_weeks > args.weeks:
        start_weeks = np.random.default_rng(args.seed).integers(0, total_weeks - args.weeks + 1)
        start_date = series.index.min() + pd.DateOffset(weeks=start_weeks)
        end_date = start_date + pd.DateOffset(weeks=args.weeks)
        series = series[(series.index >= start_date) & (series.index < end_date)]

        if exog_df is not None:
            exog_df = exog_df[(exog_df.index >= start_date) & (exog_df.index < end_date)]

    # Resample data by taking the mean
    series = series.resample(args.resample).mean().interpolate(limit_direction="both")

    # Resample exogenous data if provided
    if exog_df is not None:
        exog_df = exog_df.resample(args.resample).mean().interpolate(limit_direction="both")

    if args.mode == "grid_search":
        sarima_grid_search(series, exog_df=exog_df, p_values=AR_ORDERS, q_values=MA_ORDERS, P_values=SEASONAL_AR_ORDERS,
                           Q_values=SEASONAL_MA_ORDERS, S=24, max_workers=10)

    elif args.mode == "diagnostics":
        arima_plot_diagnostics(series)

    elif args.mode == "stationarity":
        stationarity_test(series)
