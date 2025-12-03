from src.data.create_supervised import create_supervised_dataset
from src.models.random_forest.bayes_search import bayes_search_random_forest
from src.models.random_forest.train import train_random_forest
from src.models.random_forest.evaluate_forecast import evaluate_forecast


def run_single_forecast(df, target_station, previous_time_steps=24, exog_cols=None,
                        start=None, train_end=None, test_end=None, mode="bayes_search"):
    """
    Runs a single forecast using Random Forest, either with Bayesian hyperparameter search or
    with the default parameters for training. Evaluates the model using recursive multi-step forecasting.


    Input
    -----
    df: DataFrame with the complete dataset
    target_station: The target station for forecasting
    previous_time_steps: Number of previous time steps to include as lags
    exog_cols: List of exogenous feature column names
    start: Start date for the dataset
    train_end: End date for the training set
    test_end: End date for the test set
    mode: Mode of operation ("bayes_search" or "forecast")


    Output
    ------
    mae: Mean Absolute Error on the test set
    mse: Mean Squared Error on the test set
    """
    # Print the date range being used
    print(f"Running forecast from {start} to {test_end} with training until {train_end}")

    # Create supervised dataset
    X, y = create_supervised_dataset(df, target_station=target_station,
                                     previous_time_steps=previous_time_steps,
                                     exog_cols=exog_cols, exog_lags=0)

    # Select the data based on the provided date ranges
    X_train, y_train = X.loc[start:train_end], y.loc[start:train_end]
    X_test, y_test = X.loc[train_end:test_end], y.loc[train_end:test_end]

    # Dependant on the mode, train the model using the selected parameters or Bayesian search
    importances = None
    if mode == "bayes_search":
        model, _ = bayes_search_random_forest(X_train, y_train, X_test, y_test)
    else:
        model, importances = train_random_forest(X_train, y_train)

    # Evaluate using recursive multi-step forecasting
    mae, rmse = evaluate_forecast(model, y_train, X_test, y_test, plot=False)
    mse = rmse ** 2

    return mae, mse, importances
