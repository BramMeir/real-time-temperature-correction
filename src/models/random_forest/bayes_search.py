from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from skopt import BayesSearchCV
from skopt.space import Integer
from sklearn.model_selection import TimeSeriesSplit


def bayes_search_random_forest(X_train, y_train, X_test, y_test, random_seed=42):
    """
    Perform Bayesian hyperparameter optimization for a Random Forest model
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
    best_rf_model: Trained Random Forest model with best hyperparameters
    metrics: Dictionary with MAE and RMSE on the test set
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
    tscv = TimeSeriesSplit(n_splits=5)

    # Bayesian optimization with cross-validation to find the best hyperparameters
    bayes_search = BayesSearchCV(
        estimator=rf_model,
        search_spaces=search_space,
        n_iter=20,
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

    # Make predictions on the test set (this means how good the model fits this data, not real recursive forecasting)
    y_pred = best_rf_model.predict(X_test)

    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)

    return best_rf_model, {'MAE': mae, 'RMSE': rmse}
