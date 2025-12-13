import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


def evaluate_forecast(model, number, df_train, df_test, previous_time_steps, target_station, plot=False):
    """
    Evaluate the Transformer model using recursive multi-step forecasting.

    Input
    -----
    model: Trained Transformer model
    number: An identifier number for the run
    df_train: DataFrame with training data
    df_test: DataFrame with test data
    previous_time_steps: Number of previous time steps to include as lags
    target_station: Name of the target station column to predict.
    plot: Boolean to indicate whether to plot the results (default is False)

    Output
    ------
    mae: Mean Absolute Error on the test set
    rmse: Root Mean Squared Error on the test set
    """
    all_cols = list(df_train.columns)
    target_col_index = all_cols.index(target_station)
    exog_cols = [col for col in all_cols if col != target_station]

    last_known_sequence = df_train.values[-previous_time_steps:]
    predictions = []

    for i in range(len(df_test)):
        current_sequence_reshaped = np.expand_dims(last_known_sequence, axis=0)

        # Predict
        y_pred = model.predict(current_sequence_reshaped, verbose=0)[0][0]
        predictions.append(y_pred)

        # Update Sequence
        exog_values_next_step = df_test.iloc[i][exog_cols].values
        next_input_features = np.zeros(len(all_cols))
        next_input_features[target_col_index] = y_pred

        for j, col in enumerate(exog_cols):
            col_index = all_cols.index(col)
            next_input_features[col_index] = exog_values_next_step[j]

        last_known_sequence = np.vstack([last_known_sequence[1:], next_input_features])

    predictions = pd.Series(predictions, index=df_test.index)
    mae = mean_absolute_error(df_test[target_station], predictions)
    rmse = root_mean_squared_error(df_test[target_station], predictions)

    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(df_train.index, df_train[target_station], label='Training data', color='blue', alpha=0.6)
        plt.plot(df_test.index, df_test[target_station], label='Real future data', color='green')
        plt.plot(df_test.index, predictions, label='Forecast', color='red')
        plt.legend()
        plt.title(f'Recursive Forecast (Transformer) #{number}')
        plt.savefig(f'Transformer_recursive_forecast_{number}.png')

    return mae, rmse
