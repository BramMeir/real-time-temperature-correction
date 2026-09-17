import pandas as pd
from src.data.create_supervised import create_supervised_dataset
from src.models.xgboost_model.bayes_search import bayes_search_xgboost
from src.models.xgboost_model.train import train_xgboost
from src.models.random_forest.evaluate_forecast import evaluate_forecast


def run_single_forecast(df, target_station, model=None, previous_time_steps=24, exog_cols=None,
                        start=None, train_end=None, test_end=None, mode="forecast"):
    """
    Runs a single forecast using XGBoost and evaluates it with recursive multi-step forecasting.

    Input
    -----
    df: DataFrame with the complete dataset
    target_station: The target station for forecasting
    model: Optional pre-trained model to use for forecasting (if None, a new model will be trained)
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
    best_hp: Best hyperparameters found (only for "bayes_search" mode)
    importances: Feature importances from the trained model
    """
    print(f"Running forecast from {start} to {test_end} with training until {train_end}")

    # Limit the dataframe so only the relevant range is used for creating the supervised dataset
    history_start = pd.to_datetime(start) - pd.Timedelta(hours=previous_time_steps)
    df_subset = df.loc[history_start:test_end]

    # Create supervised dataset
    X, y = create_supervised_dataset(df_subset, target_station=target_station,
                                     previous_time_steps=previous_time_steps,
                                     exog_cols=exog_cols, exog_lags=0)

    # Select the data based on the provided date ranges
    X_train, y_train = X.loc[start:train_end], y.loc[start:train_end]
    X_test, y_test = X.loc[train_end:test_end], y.loc[train_end:test_end]

    # Dependant on the mode, train the model using the selected parameters or Bayesian search
    importances = None
    best_hp = None
    if mode == "bayes_search":
        model, best_hp = bayes_search_xgboost(X_train, y_train)
    else:
        if model is None:
            model, importances = train_xgboost(X_train, y_train)
        else:
            importances = model.feature_importances_

    # Evaluate using recursive multi-step forecasting (the evaluator only calls model.predict)
    mae, rmse = evaluate_forecast(model, y_train, X_test, y_test, plot=False)
    mse = rmse ** 2

    return mae, mse, best_hp, importances
