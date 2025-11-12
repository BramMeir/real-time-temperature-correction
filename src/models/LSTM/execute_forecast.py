from src.models.LSTM.train import train_LSTM_model
from src.models.LSTM.evaluate_forecast import evaluate_LSTM_forecast


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

    # Split the data into a training and test set
    df_train = df.loc[start:train_end].copy()
    df_test = df.loc[train_end:test_end].copy()
    df_test = df_test.iloc[1:]  # Remove the first row to avoid overlap (loc slicing is inclusive)

    # Dependant on the mode, train the model using the selected parameters or Bayesian search
    model = train_LSTM_model(df_train, target_station, previous_time_steps=previous_time_steps)

    # Evaluate using recursive multi-step forecasting
    mae, rmse = evaluate_LSTM_forecast(
        model, df_train, df_test, previous_time_steps, target_station, plot=True
    )
    mse = rmse ** 2

    return mae, mse
