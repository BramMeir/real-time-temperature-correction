from skopt import BayesSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from skopt.space import Integer, Real, Categorical
from src.models.TCN.skoptTCN import create_tcn_model
from scikeras.wrappers import KerasRegressor
from keras.callbacks import EarlyStopping


def bayes_search_tcn(X_train, y_train, X_test, y_test, random_seed=42):
    """
    Perform Bayesian hyperparameter optimization for a TCN model
    using TimeSeriesSplit cross-validation.

    Input
    -----
    X_train: DataFrame with training features
    y_train: Series with training target variable
    X_test: DataFrame with test features
    y_test: Series with test target variable
    random_seed: Random seed for reproducibility (default is 42)

    Output
    ------
    best_tcn_model: Trained TCN model with best hyperparameters
    metrics: Dictionary with MAE and RMSE on the test set
    """
    # Define the input shape from your training data
    input_shape = (X_train.shape[1], X_train.shape[2])

    # Define early stopping callback (stop if validation loss doesn't improve for 10 epochs)
    early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

    # Wrap the Keras model so it can be used by scikit-learn/skopt.
    # Pass the static `input_shape` parameter here.
    tcn_estimator = KerasRegressor(
        model=create_tcn_model,
        input_shape=input_shape,
        verbose=0,
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping]
    )

    # Hyperparameter search space
    search_space = {
        "model__nb_filters": Integer(16, 128),                             # Number of filters in convolutional layers (similar as #units LSTM)
        "model__kernel_size": Integer(2, 8),                               # Size of the convolutional kernel
        "model__learning_rate": Real(1e-4, 1e-2, prior="log-uniform"),
        "model__dilations": Categorical([                                  # List/Tuple of dilation rates for TCN layers
            "(1, 2)",
            "(1, 2, 4)",
            "(1, 2, 4, 8)",
        ]),
    }

    # Same cross-validation structure as RF
    tscv = TimeSeriesSplit(n_splits=5)

    # Bayesian search
    bayes_search = BayesSearchCV(
        estimator=tcn_estimator,
        search_spaces=search_space,
        n_iter=20,
        cv=tscv,
        scoring="neg_mean_absolute_error",
        n_jobs=8,
        random_state=random_seed,
        verbose=2,
    )

    # Fit the model to the training data
    bayes_search.fit(X_train, y_train)
    best_tcn = bayes_search.best_estimator_

    print("Best hyperparameters:", bayes_search.best_params_)

    # Make predictions on the test set (this means how good the model fits this data, not real recursive forecasting)
    y_pred = best_tcn.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)

    return best_tcn, {"MAE": mae, "RMSE": rmse}
