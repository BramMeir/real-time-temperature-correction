from scikeras.wrappers import KerasRegressor
from keras.callbacks import EarlyStopping
from src.models.TCN.skoptTCN import create_tcn_model
from sklearn.model_selection import train_test_split


def train_tcn_model(X_train, y_train, nb_filters=128, kernel_size=2, dilations='(1, 2)',
                    learning_rate=0.01, epochs=100, random_seed=42) -> KerasRegressor:
    """
    Trains the TCN model using the optimal hyperparameters found by BayesSearchCV.

    This function uses validation data and early stopping for robust final training.

    Input
    -----
    X_train: 3D NumPy array with training features (samples, timesteps, features)
    y_train: 1D NumPy array with training target variable
    nb_filters: Number of filters in convolutional layers
    kernel_size: Size of the convolutional kernel
    dilations: Tuple of dilation rates for TCN layers
    learning_rate: Learning rate for the optimizer
    epochs: Maximum number of training epochs
    random_seed: Random seed for reproducibility

    Output
    ------
    final_model: The fully trained TCN model
    """
    # Determine the input shape from the data
    # (Timesteps, Features)
    input_shape = (X_train.shape[1], X_train.shape[2])

    X_t, X_val, y_t, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=random_seed
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )

    final_estimator = KerasRegressor(
        model=create_tcn_model,
        input_shape=input_shape,
        nb_filters=nb_filters,
        kernel_size=kernel_size,
        dilations=dilations,
        learning_rate=learning_rate,
        verbose=1,
        epochs=epochs,
        batch_size=32,
        random_state=random_seed,
        callbacks=[early_stopping]
    )

    # 2. Pass validation_data as a tuple
    final_estimator.fit(
        X_t,
        y_t,
        callbacks=[early_stopping],
        validation_data=(X_val, y_val),
        verbose=1
    )

    return final_estimator
