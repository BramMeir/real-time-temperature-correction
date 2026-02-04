import keras_tuner as kt
from tensorflow import keras
from src.models.transformer.build_model import build_bayes_transformer_model
from sklearn.preprocessing import StandardScaler


def bayesian_search_transformer(df, target_station, number, previous_time_steps=24):
    """
    Perform Bayesian hyperparameter tuning for Transformer forecasting.
    """
    print(df.head())
    features = df.values
    targets = df[target_station].values

    num_features = df.shape[1]

    # Split into training and validation
    split_index = int(0.8 * len(df))
    X_train_raw, y_train_raw = features[:split_index], targets[:split_index]
    X_val_raw, y_val_raw = features[split_index:], targets[split_index:]

    # Standardize features
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train = x_scaler.fit_transform(X_train_raw)
    X_val = x_scaler.transform(X_val_raw)

    y_train = y_scaler.fit_transform(y_train_raw.reshape(-1, 1)).flatten()
    y_val = y_scaler.transform(y_val_raw.reshape(-1, 1)).flatten()

    # CRITICAL: Shift the Targets
    # We want input[i:i+24] to predict target[i+24]
    # timeseries_dataset pairs data[i:i+seq] with targets[i].
    # So we must offset the targets array by sequence_length.

    sequence_length = previous_time_steps

    # Train set
    # Inputs: 0 to end-seq_len
    # Targets: seq_len to end
    dataset_train = keras.preprocessing.timeseries_dataset_from_array(
        data=X_train[:-sequence_length],         # Stop early so we have matching targets
        targets=y_train[sequence_length:],       # Start late to predict future
        sequence_length=sequence_length,
        batch_size=64,
        shuffle=True
    )

    dataset_val = keras.preprocessing.timeseries_dataset_from_array(
        data=X_val[:-sequence_length],
        targets=y_val[sequence_length:],
        sequence_length=sequence_length,
        batch_size=64
    )

    # Create the Bayesian tuner
    tuner = kt.BayesianOptimization(
        lambda hp: build_bayes_transformer_model(hp, num_features, sequence_length),
        objective="val_loss",
        max_trials=20,
        executions_per_trial=1,
        directory="tuner_results_transformer",
        project_name=f"bayesian_transformer_{number}",
        overwrite=True
    )

    # Early stopping
    stop_early = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=20,
        restore_best_weights=True
    )

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

    # Get best hyperparameters
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]

    # Retrain best model
    best_model = tuner.hypermodel.build(best_hp)
    best_model.fit(
        dataset_train,
        validation_data=dataset_val,
        epochs=500,
        callbacks=[stop_early],
        verbose=1
    )

    return best_model, best_hp, x_scaler, y_scaler
