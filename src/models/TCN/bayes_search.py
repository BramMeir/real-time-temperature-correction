from skopt import BayesSearchCV
from sklearn.model_selection import TimeSeriesSplit
from skopt.space import Integer, Categorical
from src.models.TCN.skoptTCN import create_tcn_model
from scikeras.wrappers import KerasRegressor
from keras.callbacks import EarlyStopping


def bayes_search_tcn(X_train, y_train, random_seed=42):
    """
    Perform Bayesian hyperparameter optimization for a TCN model
    using TimeSeriesSplit cross-validation.

    Input
    -----
    X_train: DataFrame with training features
    y_train: Series with training target variable
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
        learning_rate=0.005,
        callbacks=[early_stopping]
    )

    # Hyperparameter search space
    search_space = {
        "model__nb_filters": Integer(32, 256),                             # Number of filters in convolutional layer
        "model__kernel_size": Integer(2, 8),                               # Size of the convolutional kernel
        "model__dilations": Categorical([                                  # List/Tuple of dilation rates for TCN layers
            "(1, 2)",
            "(1, 2, 4)",
            "(1, 2, 4, 8)",
        ]),
    }

    # Same cross-validation structure as RF
    tscv = TimeSeriesSplit(n_splits=4)

    # Bayesian search
    bayes_search = BayesSearchCV(
        estimator=tcn_estimator,
        search_spaces=search_space,
        n_iter=15,
        cv=tscv,
        scoring="neg_mean_absolute_error",
        n_jobs=1,
        random_state=random_seed,
        verbose=2,
    )

    # Fit the model to the training data
    bayes_search.fit(X_train, y_train)
    best_tcn = bayes_search.best_estimator_

    return best_tcn, bayes_search.best_params_
