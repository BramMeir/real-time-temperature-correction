import pandas as pd
from src.evaluation.evaluate_forecasts import evaluate_forecasts
from src.models.sarima.train import train_sarima_model


def run_single_forecast(number, series, exog_df, model, start_date, end_date, hours_to_forecast,
                        arima_order=(2, 0, 0), seasonal_order=(1, 0, 1, 24), max_iter=1000):
    """
    Runs a single forecast with SARIMA (exog_df None) or SARIMAX (the neighbouring stations as exog_df) over the last
    hours_to_forecast hours of [start_date, end_date]; the hours before them are the training window.

    Input
    -----
    number: An identifier number for the forecast run (for logging purposes)
    series: Pandas Series with the target time series data
    exog_df: DataFrame with the neighbouring stations as exogenous variables, or None for SARIMA
    model: Optional pre-trained model (if None, a new model will be trained on the training window)
    start_date: Start date of the training window
    end_date: End date of the forecast window
    hours_to_forecast: Number of hours to forecast
    arima_order: Tuple specifying the (p, d, q) parameters (default is (2, 0, 0))
    seasonal_order: Tuple specifying the (P, D, Q, S) parameters (default is (1, 0, 1, 24))
    max_iter: Maximum number of iterations for the model fitting (default is 1000)

    Output
    ------
    mae: Mean Absolute Error on the forecast window
    mse: Mean Squared Error on the forecast window
    y_forecasted: Series with the forecast for every hour of the forecast window
    """
    print(f"\nRun {number + 1}: using data from {start_date} to {end_date}")

    sub_series = series[(series.index >= start_date) & (series.index <= end_date)]
    sub_exog = exog_df.loc[sub_series.index] if exog_df is not None else None

    # Split the data into training and test sets
    split_date = sub_series.index.max() - pd.DateOffset(hours=hours_to_forecast)
    train = sub_series[sub_series.index <= split_date]
    test = sub_series[sub_series.index > split_date]

    exog_train = sub_exog[sub_exog.index <= split_date] if sub_exog is not None else None
    exog_test = sub_exog[sub_exog.index > split_date] if sub_exog is not None else None

    if sub_exog is not None:
        common_idx = train.index.intersection(exog_train.index)
        train = train.loc[common_idx]
        exog_train = exog_train.loc[common_idx]

    if model is None:
        model = train_sarima_model(train, exog_train, arima_order, seasonal_order, max_iter)

    pred = model.get_forecast(steps=len(test), exog=exog_test)
    y_forecasted = pd.Series(pred.predicted_mean.values, index=test.index)

    errors = evaluate_forecasts(test, y_forecasted)
    print(errors)

    return errors["MAE"], errors["MSE"], y_forecasted
