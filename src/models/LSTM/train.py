import matplotlib.pyplot as plt
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow logging
from tensorflow import keras


def train_LSTM_model(df, target_station, previous_time_steps=24, random_seed=42):
    """
    Train a LSTM model with specified hyperparameters.

    Input
    -----
    X_train: DataFrame with training features
    y_train: Series with training target variable
    random_seed: Random seed for reproducibility (default is 42)
    n_estimators: Number of trees in the forest (default is 350)
    max_depth: Maximum depth of the trees (default is 20)
    min_samples_leaf: Minimum number of samples required to be at a leaf node (default is 1)
    min_samples_split: Minimum number of samples required to split an internal node (default is 3)

    Output
    ------
    model: Trained Random Forest model
    """
    # The features are the data from all the stations
    features = df.values

    # The target is the data from the target station
    targets = df[target_station].values

    # Split the data into training and validation sets (80/20 split)
    split_index = int(0.8 * len(df))
    X_train, y_train = features[:split_index], targets[:split_index]
    X_val, y_val = features[split_index:], targets[split_index:]

    print(f"Training data shape: {X_train.shape}, {y_train.shape}")

    # Define the training and validation datasets
    # This creates automatically for each window of size sequence_length a sample
    # with the corresponding target value at the end of the window
    dataset_train = keras.preprocessing.timeseries_dataset_from_array(
        data=X_train,
        targets=y_train,
        sequence_length=previous_time_steps,
        batch_size=64
    )

    dataset_val = keras.preprocessing.timeseries_dataset_from_array(
        data=X_val,
        targets=y_val,
        sequence_length=previous_time_steps,
        batch_size=64
    )

    for batch in dataset_train.take(1):
        inputs, targets = batch

        print("--- Sample batch from training dataset ---")
        print("Inputs:", inputs)
        print("Targets:", targets)

    print("Input batch shape:", inputs.shape)
    print("Target batch shape:", targets.shape)

    # Build the LSTM model
    # The input shape should be (sequence_length, num_features)
    num_features = X_train.shape[1]
    inputs = keras.layers.Input(shape=(previous_time_steps, num_features))
    lstm_out = keras.layers.LSTM(128)(inputs)
    outputs = keras.layers.Dense(1)(lstm_out)

    model = keras.Model(name="Temperature_forecast", inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss="mse")
    model.summary()

    # Train the model using the training/validation datasets
    history = model.fit(
        dataset_train,
        epochs=500,
        validation_data=dataset_val
    )

    loss = history.history["loss"]
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, "b", label="Training loss")
    plt.title("Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig("training_loss.png")

    return model
