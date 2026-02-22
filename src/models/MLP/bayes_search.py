from skopt import BayesSearchCV
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from skopt.space import Integer, Real, Categorical
from src.models.MLP.skoptMLP import SkoptMLP


def bayes_search_mlp(X_train, y_train, X_test, y_test, random_seed=42):
    """
    Perform Bayesian hyperparameter optimization for an MLP model
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
    best_mlp_model: Trained MLP model with best hyperparameters
    metrics: Dictionary with MAE and RMSE on the test set
    """
    # Initialize the MLP model (using SkoptMLP to handle tuple parameters)
    mlp = SkoptMLP(
        random_state=random_seed,
        solver="adam",
        early_stopping=True,
        n_iter_no_change=20,
        validation_fraction=0.1,
        max_iter=500,
        activation='relu',
        learning_rate_init=0.005,
        batch_size=16,
    )

    # Hyperparameter search space
    # Remark: hidden_layer_sizes are represented as strings to be compatible with skopt Categorical
    # and will be converted back to tuples in the SkoptMLP subclass. Otherwise, skopt has issues with tuples in Categorical.
    search_space = {
        "hidden_layer_sizes": Integer(128, 1024),  # Number of neurons in a single hidden layer
        "activation": Categorical(["relu", "tanh", "logistic"]),
        "learning_rate_init": Real(1e-4, 1e-2, prior="log-uniform"),
        "batch_size": Integer(4, 64),
    }

    # Same cross-validation structure as RF
    tscv = TimeSeriesSplit(n_splits=5)

    # Bayesian search
    bayes_search = BayesSearchCV(
        estimator=mlp,
        search_spaces=search_space,
        n_iter=25,
        cv=tscv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        random_state=random_seed,
        verbose=2,
    )

    # Fit the model to the training data
    bayes_search.fit(X_train, y_train)
    best_mlp = bayes_search.best_estimator_

    print("Best hyperparameters:", bayes_search.best_params_)

    # Make predictions on the test set (this means how good the model fits this data, not real recursive forecasting)
    y_pred = best_mlp.predict(X_test)

    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)

    return best_mlp, {"MAE": mae, "RMSE": rmse}
