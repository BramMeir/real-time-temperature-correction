from xgboost import XGBRegressor
from skopt import BayesSearchCV
from skopt.space import Integer, Real
from sklearn.model_selection import TimeSeriesSplit


def bayes_search_xgboost(X_train, y_train, random_seed=42):
    """
    Perform Bayesian hyperparameter optimization for an XGBoost model
    using TimeSeriesSplit cross-validation.

    Input
    -----
    X_train: DataFrame with training features
    y_train: Series with training target variable
    random_seed: Random seed for reproducibility (default is 42)

    Output
    ------
    best_xgb_model: Trained XGBoost model with best hyperparameters
    best_params: Dictionary of the best hyperparameters found
    """
    # Define the search space for hyperparameter tuning
    search_space = {
        'n_estimators': Integer(200, 1000),
        'max_depth': Integer(3, 10),
        'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
        'subsample': Real(0.6, 1.0),
        'colsample_bytree': Real(0.6, 1.0),
        'min_child_weight': Integer(1, 10)
    }

    # Initialize the base XGBoost model (n_jobs=-1 to use all available cores)
    xgb_model = XGBRegressor(objective="reg:squarederror", random_state=random_seed, n_jobs=-1)

    # Use TimeSeriesSplit for cross-validation to respect the temporal order of data
    tscv = TimeSeriesSplit(n_splits=4)

    # Bayesian optimization with cross-validation to find the best hyperparameters
    bayes_search = BayesSearchCV(
        estimator=xgb_model,
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
    best_xgb_model = bayes_search.best_estimator_

    print(f"Best hyperparameters: {bayes_search.best_params_}")

    return best_xgb_model, bayes_search.best_params_
