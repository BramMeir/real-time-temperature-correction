from tensorflow import keras


def train_LSTM_model(X_train, y_train, random_seed=42):
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
    print(X_train.shape, y_train.shape)
    inputs = keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2]))
    lstm_out = keras.layers.LSTM(32)(inputs)
    outputs = keras.layers.Dense(1)(lstm_out)

    model = keras.Model(name="Weather_forcaster", inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    model.summary()
