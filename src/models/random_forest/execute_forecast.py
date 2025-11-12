"""
Module to execute Random Forest forecasts.
"""
from src.data.create_supervised import create_supervised_dataset
from src.models.random_forest.bayes_search import bayes_search_random_forest
from src.models.random_forest.train import train_random_forest
from src.models.random_forest.evaluate_forecast import evaluate_forecast


def run_single_forecast(df, target_station, previous_time_steps, exog_cols, start, train_end, end, mode):
    """
    Runs a single training and evaluation of the Random Forest model.

    Input
    -----
    X_train: Training features DataFrame
    y_train: Training target Series
    X_test: Test features DataFrame
    y_test: Test target Series
    start: Start datetime for the training and test split
    train_end: End datetime for the training set
    end: End datetime for the test set
    mode: Mode of operation for the task

    Output
    ------
    mae: Mean Absolute Error on the test set
    mse: Mean Squared Error on the test set
    """
    # Print the date range being used
    print(f"Running forecast from {start} to {end} with training until {train_end}")

    # Create supervised dataset
    X, y = create_supervised_dataset(df, target_station=target_station,
                                     previous_time_steps=previous_time_steps,
                                     exog_cols=exog_cols, exog_lags=2)

    # Select the data based on the provided date ranges
    X_train, y_train = X.loc[start:train_end], y.loc[start:train_end]
    X_test, y_test = X.loc[train_end:end], y.loc[train_end:end]

    # Dependant on the mode, train the model using the selected parameters or Bayesian search
    if mode == "bayes_search":
        model, _ = bayes_search_random_forest(X_train, y_train, X_test, y_test)
    else:
        model = train_random_forest(X_train, y_train)

    # Evaluate using recursive multi-step forecasting
    mae, rmse = evaluate_forecast(model, y_train, X_test, y_test, plot=False)
    mse = rmse ** 2

    return mae, mse
