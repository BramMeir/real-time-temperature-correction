from src.models.transformer.evaluate_forecast import evaluate_forecast
from src.models.transformer.bayes_search import bayesian_search_transformer


def run_single_forecast(df, number, target_station, previous_time_steps=24,
                        start=None, train_end=None, test_end=None, mode="bayes_search"):
    """
    Runs a single forecast using transformer model, either with Bayesian hyperparameter search or
    with the default parameters for training. Evaluates the model using recursive multi-step forecasting.

    Input
    -----
    df: DataFrame with the complete dataset
    number: An identifier number for the run
    target_station: The target station for forecasting
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
    print(f"Running forecast from {start} to {test_end} with training until {train_end}")

    df_train = df.loc[start:train_end].copy()
    df_test = df.loc[train_end:test_end].copy()
    df_test = df_test.iloc[1:]  # Remove the first row to avoid overlap (loc slicing is inclusive)

    best_hp = None

    if mode == "bayes_search":
        model, best_hp = bayesian_search_transformer(
            df_train, target_station, number, previous_time_steps=previous_time_steps
        )
    # else:
    #     # Train with default parameters if not searching
    #     model = train_Transformer_model(df_train, target_station, previous_time_steps=previous_time_steps)

    # Evaluate
    mae, rmse = evaluate_forecast(
        model, number, df_train, df_test, previous_time_steps, target_station, plot=True
    )
    mse = rmse ** 2

    return mae, mse, best_hp
