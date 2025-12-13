import keras_tuner as kt
from tensorflow import keras
from src.models.transformer.build_model import build_transformer_model


def bayesian_search_transformer(df, target_station, number, previous_time_steps=24):
    """
    Perform Bayesian hyperparameter tuning for Transformer forecasting.
    """
    print(df.head())
    features = df.values
    targets = df[target_station].values

    # Split into training and validation
    split_index = int(0.8 * len(df))
    X_train, y_train = features[:split_index], targets[:split_index]
    X_val, y_val = features[split_index:], targets[split_index:]

    sequence_length = previous_time_steps

    # Convert into time series datasets
    # This makes sure that all the previous time steps (sequence_length) are included from all the stations as input,
    # where the target is the target_station's value at the next time step.
    dataset_train = keras.preprocessing.timeseries_dataset_from_array(
        data=X_train, targets=y_train, sequence_length=sequence_length, batch_size=64
    )
    dataset_val = keras.preprocessing.timeseries_dataset_from_array(
        data=X_val, targets=y_val, sequence_length=sequence_length, batch_size=64
    )

    # Create the Bayesian tuner
    tuner = kt.BayesianOptimization(
        lambda hp: build_transformer_model(hp, sequence_length),
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

    return best_model, best_hp
