import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


def evaluate_forecast(model, y_train, X_test, y_test, plot=False):
    """
    Evaluate the RF model using recursive multi-step forecasting.

    Input
    -----
    model: Trained Random Forest model
    y_train: Target variable for training (real temperature values)
    X_test: Test features for forecasting (follows on X_train)
    y_test: Test target variable for evaluation (follows on y_train)
    plot: Boolean to indicate whether to plot the results (default is False)

    Output
    ------
    mae: Mean Absolute Error on the test set
    rmse: Root Mean Squared Error on the test set
    """
    # Set the first test input as the startpoint
    X_input = X_test.iloc[[0]].copy()

    # Detect which columns belong to the target lags vs exogenous features
    lag_cols = [col for col in X_input.columns if 'lag' in col]
    exog_cols = [col for col in X_input.columns if col not in lag_cols]

    predictions = []

    for i in range(len(y_test)):
        # Predict next step
        y_pred = model.predict(X_input)[0]
        predictions.append(y_pred)

        # Shift lag columns to the left
        X_input[lag_cols] = X_input[lag_cols].shift(-1, axis=1)

        # Prediction becomes the most recent lag
        X_input.iloc[0, X_input.columns.get_loc(lag_cols[-1])] = y_pred

        # Update exogenous columns from the actual future values (known ahead)
        if i < len(X_test):
            X_input[exog_cols] = X_test.iloc[i][exog_cols].values

    # Convert predictions to series for evaluation
    predictions = pd.Series(predictions, index=y_test.index)

    mae = mean_absolute_error(y_test, predictions)
    rmse = root_mean_squared_error(y_test, predictions)

    # Plot results
    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(y_train.index, y_train, label='Training data', color='blue', alpha=0.6)
        plt.plot(y_test.index, y_test, label='Real future data', color='green')
        plt.plot(y_test.index, predictions, label='Forecast', color='red')
        plt.legend()
        plt.title('Recursive Forecast')
        plt.show()

    return mae, rmse
