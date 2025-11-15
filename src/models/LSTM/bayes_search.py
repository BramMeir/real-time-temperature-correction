import os
import keras_tuner as kt
from tensorflow import keras

# Suppress TensorFlow logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def build_LSTM_model(hp, num_features, sequence_length):
    """
    Build and compile an LSTM model based on hyperparameters.
    """
    model = keras.Sequential()

    # Add seperate input layer to the LSTM model
    model.add(keras.layers.Input(shape=(sequence_length, num_features)))

    model.add(
        keras.layers.LSTM(
            units=hp.Int("units", min_value=32, max_value=256, step=32),
            dropout=hp.Float("dropout", 0.0, 0.5, step=0.1),
        )
    )
    model.add(keras.layers.Dense(1))

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=hp.Choice("learning_rate", [0.001, 0.005, 0.01, 0.02])
        ),
        loss="mse",
    )

    return model


def bayesian_search_LSTM(df, target_station, number, previous_time_steps=24):
    """
    Perform Bayesian hyperparameter tuning for LSTM forecasting.
    """
    features = df.values
    targets = df[target_station].values

    # Split into training and validation
    split_index = int(0.8 * len(df))
    X_train, y_train = features[:split_index], targets[:split_index]
    X_val, y_val = features[split_index:], targets[split_index:]

    num_features = X_train.shape[1]
    sequence_length = previous_time_steps

    # Convert into time series datasets
    dataset_train = keras.preprocessing.timeseries_dataset_from_array(
        data=X_train,
        targets=y_train,
        sequence_length=sequence_length,
        batch_size=64,
    )
    dataset_val = keras.preprocessing.timeseries_dataset_from_array(
        data=X_val,
        targets=y_val,
        sequence_length=sequence_length,
        batch_size=64,
    )

    # Create the Bayesian tuner
    tuner = kt.BayesianOptimization(
        lambda hp: build_LSTM_model(hp, num_features, sequence_length),
        objective="val_loss",
        max_trials=20,                      # number of different configs to try
        executions_per_trial=1,
        directory="tuner_results",
        project_name=f"bayesian_lstm_{number}",
        overwrite=True
    )

    # Early stopping to prevent overfitting (if val_loss doesn't improve for 10 epochs)
    stop_early = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    try:
        tuner.search(
            dataset_train,
            epochs=200,
            validation_data=dataset_val,
            callbacks=[stop_early],
            verbose=1
        )
    except Exception as e:
        print(f"Error during hyperparameter tuning: {e}")
        return None, None, None

    # Print the best hyperparameters
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("\nBest hyperparameters found:")
    print(f"Units: {best_hp.get('units')}")
    print(f"Dropout: {best_hp.get('dropout')}")
    print(f"Learning rate: {best_hp.get('learning_rate')}")

    # Retrieve and train the best model
    best_model = tuner.hypermodel.build(best_hp)
    history = best_model.fit(
        dataset_train,
        validation_data=dataset_val,
        epochs=500,
        callbacks=[stop_early],
        verbose=1
    )

    optimal_epochs = len(history.history['loss'])
    print(f"\nOptimal number of epochs for the best model: {optimal_epochs}")

    return best_model, best_hp, optimal_epochs
