from sklearn.ensemble import RandomForestRegressor
from skopt import BayesSearchCV
from skopt.space import Integer
from sklearn.model_selection import TimeSeriesSplit


def bayes_search_random_forest(X_train, y_train, random_seed=42):
    """
    Perform Bayesian hyperparameter optimization for a Random Forest model
    using TimeSeriesSplit cross-validation.

    Input
    -----
    X_train: DataFrame with training features
    y_train: Series with training target variable
    random_seed: Random seed for reproducibility (default is 42)

    Output
    ------
    best_rf_model: Trained Random Forest model with best hyperparameters
    best_params: Dictionary of the best hyperparameters found
    """
    # Define the search space for hyperparameter tuning
    search_space = {
        'n_estimators': Integer(200, 1000),
        'max_depth': Integer(10, 40),
        'min_samples_split': Integer(2, 6),
        'min_samples_leaf': Integer(1, 3)
    }

    # Initialize the base Random Forest model (n_jobs=-1 to use all available cores)
    rf_model = RandomForestRegressor(random_state=random_seed, n_jobs=-1)

    # Use TimeSeriesSplit for cross-validation to respect the temporal order of data
    tscv = TimeSeriesSplit(n_splits=4)

    # Bayesian optimization with cross-validation to find the best hyperparameters
    bayes_search = BayesSearchCV(
        estimator=rf_model,
        search_spaces=search_space,
        n_iter=15,
        cv=tscv,
        n_jobs=-1,
        scoring='neg_mean_absolute_error',
        random_state=random_seed,
        verbose=2
    )

    # Fit the model to the training data
    bayes_search.fit(X_train, y_train)
    best_rf_model = bayes_search.best_estimator_

    print(f"Best hyperparameters: {bayes_search.best_params_}")

    return best_rf_model, bayes_search.best_params_
