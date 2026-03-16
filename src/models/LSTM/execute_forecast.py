from src.models.LSTM.train import train_LSTM_model
from src.models.LSTM.evaluate_forecast import evaluate_LSTM_forecast
from src.models.LSTM.bayes_search import bayesian_search_LSTM


def run_single_forecast(df, number, target_station, model=None, previous_time_steps=24,
                        start=None, train_end=None, test_end=None, mode="bayes_search"):
    """
    Runs a single forecast using Random Forest, either with Bayesian hyperparameter search or
    with the default parameters for training. Evaluates the model using recursive multi-step forecasting.


    Input
    -----
    df: DataFrame with the complete dataset
    number: An identifier number for the forecast run
    target_station: The target station for forecasting
    model: Optional pre-trained model to use for forecasting (if mode is "forecast")
    previous_time_steps: Number of previous time steps to include as lags
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
    best_hp = None
    if mode == "bayes_search":
        model, best_hp, _ = bayesian_search_LSTM(df_train, target_station, number, previous_time_steps=previous_time_steps)
    else:
        if model is None:
            model = train_LSTM_model(df_train, target_station, previous_time_steps=previous_time_steps)

    # Evaluate using recursive multi-step forecasting
    mae, rmse = evaluate_LSTM_forecast(
        model, number, df_train, df_test, previous_time_steps, target_station, plot=False
    )
    mse = rmse ** 2

    return mae, mse, best_hp
