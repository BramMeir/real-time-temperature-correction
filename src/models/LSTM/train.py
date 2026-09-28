import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow logging
from tensorflow import keras
from sklearn.preprocessing import StandardScaler


class ScaledLSTM:
    """
    A trained LSTM together with the scalers of its training window.

    The network is fitted on standardized inputs and targets, but predict() takes and returns
    temperatures in degrees, so the recursive forecast can keep feeding its own predictions back
    into an unscaled window.
    """

    def __init__(self, model, x_scaler, y_scaler):
        self.model = model
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler

    def predict(self, X, verbose=0):
        # The scaler works per feature, so fold the windows into rows of feature vectors and back
        X_scaled = self.x_scaler.transform(X.reshape(-1, X.shape[2])).reshape(X.shape)

        return self.y_scaler.inverse_transform(self.model.predict(X_scaled, verbose=verbose))


def train_LSTM_model(df, target_station, previous_time_steps=24, number_of_neurons=512,
                     nr_epochs=200, learning_rate=0.001):
    """
    Train a LSTM model with specified hyperparameters.

    Inputs and target are standardized with the statistics of the training window, so the network
    works on values around zero instead of raw temperatures.

    Input
    -----
    df: DataFrame with time series data, indexed by datetime
    target_station: Name of the target station column to predict
    previous_time_steps: Number of previous time steps to include as features (default is 24)
    number_of_neurons: Number of neurons in the LSTM layer (default is 512)
    nr_epochs: Number of training epochs (default is 200)
    learning_rate: Learning rate for the Adam optimizer (default is 0.001)

    Output
    ------
    model: Trained LSTM model, wrapped with its scalers
    """
    # Standardize each feature and the target with the statistics of the training window
    x_scaler = StandardScaler().fit(df.values)
    y_scaler = StandardScaler().fit(df[[target_station]].values)

    # The features are the data from the station itself
    features = x_scaler.transform(df.values)

    # The target is the data from the target station
    targets = y_scaler.transform(df[[target_station]].values).ravel()

    # Split the data into training and validation sets (80/20 split)
    split_index = int(0.8 * len(df))
    X_train, y_train = features[:split_index], targets[:split_index]
    X_val, y_val = features[split_index:], targets[split_index:]

    dataset_train = keras.preprocessing.timeseries_dataset_from_array(
        data=X_train[:-previous_time_steps],         # Stop early so we have matching targets
        targets=y_train[previous_time_steps:],       # Start late to predict future
        sequence_length=previous_time_steps,
        batch_size=64,
        shuffle=True,
    )

    dataset_val = keras.preprocessing.timeseries_dataset_from_array(
        data=X_val[:-previous_time_steps],
        targets=y_val[previous_time_steps:],
        sequence_length=previous_time_steps,
        batch_size=64
    )

    for batch in dataset_train.take(1):
        inputs, targets = batch

    # Build the LSTM model
    # The input shape should be (sequence_length, num_features)
    num_features = X_train.shape[1]
    inputs = keras.layers.Input(shape=(previous_time_steps, num_features))
    lstm_out = keras.layers.LSTM(number_of_neurons)(inputs)
    outputs = keras.layers.Dense(1)(lstm_out)

    model = keras.Model(name="Temperature_forecast", inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")

    # Add early stopping to prevent overfitting
    early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    # Train the model using the training/validation datasets
    model.fit(
        dataset_train,
        epochs=nr_epochs,
        validation_data=dataset_val,
        callbacks=[early_stopping],
        verbose=0
    )

    return ScaledLSTM(model, x_scaler, y_scaler)
