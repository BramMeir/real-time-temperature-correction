from scikeras.wrappers import KerasRegressor
from keras.callbacks import EarlyStopping
from src.models.TCN.skoptTCN import create_tcn_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class ScaledTCN:
    """
    A trained TCN together with the scalers of its training window.

    The network is fitted on standardized inputs and targets, but predict() takes and returns
    temperatures in degrees, so the recursive forecast can keep feeding its own predictions back
    into an unscaled window.
    """

    def __init__(self, estimator, x_scaler, y_scaler):
        self.estimator = estimator
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler

    def predict(self, X, verbose=0):
        # The scaler works per feature, so fold the windows into rows of feature vectors and back
        X_scaled = self.x_scaler.transform(X.reshape(-1, X.shape[2])).reshape(X.shape)
        y_scaled = self.estimator.predict(X_scaled, verbose=verbose)

        return self.y_scaler.inverse_transform(y_scaled.reshape(-1, 1)).ravel()


def train_tcn_model(X_train, y_train, nb_filters=166, kernel_size=2, dilations='(1, 2, 4, 8)',
                    learning_rate=0.01, epochs=100, random_seed=42) -> ScaledTCN:
    """
    Trains the TCN model using the optimal hyperparameters found by BayesSearchCV.

    This function uses validation data and early stopping for robust final training. Inputs and
    target are standardized with the statistics of the training window, so the ReLU network
    works on values around zero instead of raw temperatures.

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
    final_model: The fully trained TCN model, wrapped with its scalers
    """
    # Determine the input shape from the data
    # (Timesteps, Features)
    input_shape = (X_train.shape[1], X_train.shape[2])

    # Standardize each feature and the target with the statistics of the training window
    x_scaler = StandardScaler().fit(X_train.reshape(-1, X_train.shape[2]))
    y_scaler = StandardScaler().fit(y_train.reshape(-1, 1))

    X_scaled = x_scaler.transform(X_train.reshape(-1, X_train.shape[2])).reshape(X_train.shape)
    y_scaled = y_scaler.transform(y_train.reshape(-1, 1)).ravel()

    X_t, X_val, y_t, y_val = train_test_split(
        X_scaled, y_scaled, test_size=0.15, random_state=random_seed
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
        verbose=0,
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
        verbose=0
    )

    return ScaledTCN(final_estimator, x_scaler, y_scaler)
