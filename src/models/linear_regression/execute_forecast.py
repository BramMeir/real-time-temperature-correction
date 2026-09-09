import pandas as pd
from src.models.linear_regression.train import train_neighbour_regression
from src.evaluation.evaluate_baseline_forecast import evaluate_baseline_forecast


def run_single_forecast(df, target_station, model=None, exog_cols=None, start=None, train_end=None,
                        test_end=None, mode="forecast"):
    """
    Runs a single forecast using a linear regression of the target station on the contemporaneous
    observations of the neighbouring stations. The forecast assumes the neighbours' future observations
    are known (the same assumption made for the exogenous variables of ARIMAX and the Random Forest),
    predicting the target station directly from the neighbours' observed values over the test window.

    This baseline has no hyperparameter search, so every mode behaves the same as "forecast" — the
    mode argument is only kept for interface compatibility with the other models.

    Input
    -----
    df: DataFrame with a datetime index, one column per station, hourly frequency
    target_station: The target station for forecasting
    model: Optional pre-trained model to use for forecasting (if None, a new model will be trained)
    exog_cols: List of neighbouring station column names used as regressors (if None, every column
              of df except the target station is used)
    start: Start date for the training window
    train_end: End date for the training window / start of the forecast horizon
    test_end: End date for the forecast horizon
    mode: Mode of operation (kept only for interface compatibility, unused)

    Output
    ------
    mae: Mean Absolute Error on the test set
    mse: Mean Squared Error on the test set
    best_hp: Always None, this baseline has no hyperparameters to search
    coefficients: DataFrame with the regression coefficients ('Feature' and 'Coefficient' columns)
    """
    # Print the date range being used
    print(f"Running forecast from {start} to {test_end} with training until {train_end}")

    if exog_cols is None:
        exog_cols = [col for col in df.columns if col != target_station]

    if model is None:
        train_df = df.loc[start:train_end]
        model, coefficients = train_neighbour_regression(train_df, target_station, exog_cols)
    else:
        coefficients = pd.DataFrame({
            'Feature': exog_cols,
            'Coefficient': model.coef_
        })
        coefficients = coefficients.reindex(
            coefficients['Coefficient'].abs().sort_values(ascending=False).index
        ).reset_index(drop=True)

    # Test index excludes train_end so the forecast starts one step after the training window
    test_index = df.loc[train_end:test_end].index[1:]

    # The neighbour observations are assumed to be known over the forecast horizon; interpolate
    # any short gaps so a missing regressor value does not crash the prediction
    X_test = df.loc[test_index, exog_cols].interpolate(limit_direction="both")
    y_test = df.loc[test_index, target_station]

    predictions = pd.Series(model.predict(X_test), index=test_index)

    mae, rmse = evaluate_baseline_forecast(predictions, y_test, model_name="linear_regression", plot=False)
    mse = rmse ** 2

    return mae, mse, None, coefficients
