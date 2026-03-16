import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow logging
from tensorflow import keras
from src.models.transformer.build_model import build_transformer_model
from sklearn.preprocessing import StandardScaler


def train_transformer_model(df, target_station, previous_time_steps=8, num_layers=4,
                            d_model=448, num_heads=2, dff=64, dropout_rate=0.1, learning_rate=1e-4, batch_size=64):
    """
    Train a Transformer model with specified hyperparameters.

    Input
    -----
    df: DataFrame with time series data, indexed by datetime
    target_station: Name of the target station column to predict
    previous_time_steps: Number of previous time steps to include as features (default is 24)
    num_layers: Number of Transformer blocks (default is 2)
    d_model: Embedding dimension (default is 256)
    num_heads: Number of attention heads (default is 2)
    dff: Dimension of the feed-forward network (default is 32)
    dropout_rate: Dropout rate for regularization (default is 0.1)
    learning_rate: Learning rate for the optimizer (default is 1e-4)
    batch_size: Batch size for training (default is 64)

    Output
    ------
    model: Trained LSTM model
    """
    features = df.values
    targets = df[target_station].values

    # Split the data into training and validation sets (80/20 split)
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
        batch_size=batch_size,
        shuffle=True,
    )

    dataset_val = keras.preprocessing.timeseries_dataset_from_array(
        data=X_val[:-sequence_length],
        targets=y_val[sequence_length:],
        sequence_length=sequence_length,
        batch_size=batch_size
    )

    # Build the Transformer model
    model = build_transformer_model(
        embed_dim=d_model,
        num_heads=num_heads,
        ff_dim=dff,
        num_blocks=num_layers,
        dropout_rate=dropout_rate,
        num_features=X_train.shape[1],
        sequence_length=sequence_length,
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )

    # Train the model
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    model.fit(
        dataset_train,
        validation_data=dataset_val,
        epochs=200,
        callbacks=[early_stopping],
        verbose=0,
    )

    return model, x_scaler, y_scaler
