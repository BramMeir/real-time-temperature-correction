import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


def evaluate_forecast(model, number, df_train, df_test, previous_time_steps, target_station,
                      x_scaler, y_scaler, plot=False):
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

    # Standardize train data
    train_values_scaled = x_scaler.transform(df_train.values)
    test_values_scaled = x_scaler.transform(df_test.values)

    last_known_sequence = train_values_scaled[-previous_time_steps:]
    predictions_scaled = []

    for i in range(len(df_test)):
        current_sequence_reshaped = np.expand_dims(last_known_sequence, axis=0)

        # Predict
        y_pred_scaled = model.predict(current_sequence_reshaped, verbose=0)[0][0]
        predictions_scaled.append(y_pred_scaled)

        # Build next timestep input (scaled)
        next_input = np.zeros(len(all_cols))

        # Target (predicted, scaled)
        next_input[target_col_index] = y_pred_scaled

        # Exogenous variables (already scaled)
        for col in exog_cols:
            col_idx = all_cols.index(col)
            next_input[col_idx] = test_values_scaled[i, col_idx]

        last_known_sequence = np.vstack([last_known_sequence[1:], next_input])

    # Inverse transform predictions
    predictions_scaled = np.array(predictions_scaled).reshape(-1, 1)
    predictions = y_scaler.inverse_transform(predictions_scaled).ravel()

    y_true = df_test[target_station].values

    mae = mean_absolute_error(y_true, predictions)
    rmse = root_mean_squared_error(y_true, predictions)

    if plot:
        plt.figure(figsize=(12, 6))
        # Select last part of training data for better visualization
        train_part = df_train[target_station].iloc[-24 * 14:]
        plt.plot(train_part.index, train_part, label='Training data', color='blue', alpha=0.6)
        plt.plot(df_test.index, df_test[target_station], label='Real future data', color='green')
        plt.plot(df_test.index, predictions, label='Forecast', color='red')
        plt.legend()
        plt.title(f'Recursive Forecast (Transformer) #{number}')

        # Generate random filename to avoid overwriting
        random_filename = f"Transformer_forecast_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(f"plot_results/{random_filename}")
        plt.close()

    return mae, rmse
