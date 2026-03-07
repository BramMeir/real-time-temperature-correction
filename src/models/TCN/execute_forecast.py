from src.data.create_3d_dataset import create_3d_dataset
from src.models.TCN.bayes_search import bayes_search_tcn
from src.models.TCN.train import train_tcn_model
from src.models.TCN.evaluate_forecast import evaluate_forecast


def run_single_forecast(df, target_station, previous_time_steps=24, exog_cols=None,
                        start=None, train_end=None, test_end=None, mode="bayes_search"):
    """
    Runs a single forecast using TCN, either with Bayesian hyperparameter search or
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
    best_hp: Best hyperparameters found by Bayesian search (if mode is "bayes_search"), otherwise None
    """
    # Print the date range being used
    print(f"Running forecast from {start} to {test_end} with training until {train_end}")

    # Create supervised dataset
    X, y, dates = create_3d_dataset(df, target_station=target_station,
                                    previous_time_steps=previous_time_steps,
                                    exog_cols=exog_cols)

    # Select the data based on the provided date ranges
    # Boolean masks for date filtering
    train_mask = (dates >= start) & (dates <= train_end)
    test_mask = (dates > train_end) & (dates <= test_end)

    X_train = X[train_mask]
    y_train = y[train_mask]

    X_test = X[test_mask]
    y_test = y[test_mask]

    # Dependant on the mode, train the model using the selected parameters or Bayesian search
    best_hp = None
    if mode == "bayes_search":
        model, best_hp = bayes_search_tcn(X_train, y_train)
    else:
        model = train_tcn_model(X_train, y_train)

    # Evaluate using recursive multi-step forecasting
    mae, rmse = evaluate_forecast(model, y_train, X_test, y_test, dates[train_mask], dates[test_mask],
                                  previous_time_steps, target_station, plot=False)
    mse = rmse ** 2

    return mae, mse, best_hp
