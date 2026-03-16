import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


def evaluate_LSTM_forecast(model, number, df_train, df_test, previous_time_steps, target_station, plot=False):
    """
    Evaluate the LSTM model using recursive multi-step forecasting.

    Input
    -----
    model: Trained TensorFlow/Keras LSTM model.
    number: An identifier number for the forecast run.
    df_train: The DataFrame used for training the model.
    df_test: The DataFrame with the test data (true future values).
    previous_time_steps: The number of past time steps the model uses (e.g., 24).
    target_station: The name of the column being predicted (e.g., 'Melle AWS').
    plot: Boolean to indicate whether to plot the results (default is False).

    Output
    ------
    mae: Mean Absolute Error on the test set
    rmse: Root Mean Squared Error on the test set
    """
    # Get the column index of the target and exogenous variables
    all_cols = list(df_train.columns)
    target_col_index = all_cols.index(target_station)
    exog_cols = [col for col in all_cols if col != target_station]

    # The initial input sequence is the last `previous_time_steps` steps of the training data
    last_known_sequence = df_train.values[-previous_time_steps:]

    predictions = []

    for i in range(len(df_test)):
        # Reshape the sequence to (1, previous_time_steps, num_features) for the model
        current_sequence_reshaped = np.expand_dims(last_known_sequence, axis=0)

        # Predict next step
        # The model outputs a nested array, so we get the scalar value with [0][0]
        y_pred = model.predict(current_sequence_reshaped, verbose=0)[0][0]
        predictions.append(y_pred)

        # Construct the new input sequence for the next prediction

        # Get the known future values for the exogenous variables at this time step
        exog_values_next_step = df_test.iloc[i][exog_cols].values

        # Create a new feature vector for the next time step
        next_input_features = np.zeros(len(all_cols))

        # Set the target variable to the predicted value
        next_input_features[target_col_index] = y_pred

        # Set the exogenous variables to their known future values
        for j, col in enumerate(exog_cols):
            col_index = all_cols.index(col)
            next_input_features[col_index] = exog_values_next_step[j]

        # Append the new input features to the sequence and remove the oldest step
        last_known_sequence = np.vstack([last_known_sequence[1:], next_input_features])

    # Convert predictions to series for evaluation
    predictions = pd.Series(predictions, index=df_test.index)

    mae = mean_absolute_error(df_test[target_station], predictions)
    rmse = root_mean_squared_error(df_test[target_station], predictions)

    # Plot results
    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(df_train.index, df_train[target_station], label='Training data', color='blue', alpha=0.6)
        plt.plot(df_test.index, df_test[target_station], label='Real future data', color='green')
        plt.plot(df_test.index, predictions, label='Forecast', color='red')
        plt.legend()
        plt.title('Recursive Forecast')

        # Save the plot with a unique name
        random_filename = f"LSTM_forecast_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(f"plot_results/{random_filename}")
        plt.close()

    return mae, rmse
