import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


def evaluate_forecast(model, y_train, X_test, y_test, dates_train, dates_test, previous_time_steps, target_station, plot=False):
    """
    Evaluate the TCN model using recursive multi-step forecasting.

    Input
    -----
    model: Trained TCN model (ScaledTCN, or the KerasRegressor from the Bayesian search)
    X_test: Full test features, (Samples, Timesteps, Features)
    y_test: Test target variable for evaluation (1D NumPy array)
    dates_test: Datetime index corresponding to y_test
    previous_time_steps: The size of the lookback window (L)
    target_station: Name of the target station column to predict.
    plot: Boolean to indicate whether to plot the results (default is False)

    Output
    ------
    mae: Mean Absolute Error on the test set
    rmse: Root Mean Squared Error on the test set
    """
    predictions = []

    # We must start with a complete, single sample window: (1, Timesteps, Features)
    # The test set (X_test) already begins with the correct history for y_test[0].
    X_current_window = X_test[0:1].copy()

    # Identify the index of the target station in the feature set (where X_test equals the target)
    TARGET_COL_INDEX = 0  # Assuming the target station is the first feature

    # Recursive Forecasting Loop
    for i in range(len(y_test)):

        # Predict the next step (y_pred for time t+i)
        y_pred = model.predict(X_current_window, verbose=0)[0]
        predictions.append(y_pred)

        # B. Prepare for the next iteration (t+i+1)
        if i < len(y_test) - 1:

            # SHIFT: Drop the oldest time step (t+i - L) from the sequence.
            # Shifting the whole sequence one step to the left
            X_current_window[0, 0:previous_time_steps - 1, :] = X_current_window[0, 1:previous_time_steps, :]

            # The exogenous data for the new step is taken from the next sequence's last time step.
            # X_test[i+1, L-1, :] gives the features (exog and time) available at the time of prediction t+i+1.
            next_step_features = X_test[i + 1, previous_time_steps - 1, :].copy()

            # Replace the Target feature (index 0) with the new prediction
            next_step_features[TARGET_COL_INDEX] = y_pred

            # Place the complete feature vector (new prediction + known exog/time)
            # into the last position (L-1) of the current window.
            X_current_window[0, previous_time_steps - 1, :] = next_step_features

    # Convert predictions to a Series for plotting and indexing
    predictions = pd.Series(predictions, index=dates_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = root_mean_squared_error(y_test, predictions)

    if plot:
        # Convert numpy arrays back to a Pandas Series using the passed indices
        y_train_series = pd.Series(y_train, index=dates_train)
        y_test_series = pd.Series(y_test, index=dates_test)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(y_train_series.index, y_train_series, label='Training data', color='blue', alpha=0.6)
        ax.plot(y_test_series.index, y_test_series, label='Real future data', color='green')
        ax.plot(y_test_series.index, predictions, label='Forecast', color='red')
        ax.legend()
        ax.set_title('TCN Recursive Forecast')

        random_filename = f"TCN_forecast_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png"
        fig.savefig(f'plots/forecasts/{random_filename}')
        plt.close(fig)

    return mae, rmse
