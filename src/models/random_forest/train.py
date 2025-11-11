from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


def train_random_forest(X_train, y_train, X_test, y_test, random_seed=42):
    """
    Train a Random Forest model and evaluate its performance.

    Input
    -----
    X_train: DataFrame with training features
    y_train: Series with training target variable
    X_test: DataFrame with test features
    y_test: Series with test target variable
    random_seed: Random seed for reproducibility (default is 42)

    Output
    ------
    model: Trained Random Forest model
    metrics: Dictionary with MAE and RMSE on the test set
    """
    # Define the parameter grid for hyperparameter tuning
    param_grid = {
        'n_estimators': [200, 500],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],       # Minimum samples required to split an internal node (to justify creating a split)
        'min_samples_leaf': [1, 2, 4]          # Minimum samples required to be at a leaf node (to prevent overfitting)
    }
    # param_grid = {
    #     'n_estimators': [200],
    #     'max_depth': [10],
    #     'min_samples_split': [2],
    #     'min_samples_leaf': [4]
    # }

    # Initialize the base Random Forest model (n_jobs=-1 to use all available cores)
    rf_model = RandomForestRegressor(random_state=random_seed, n_jobs=-1)

    # Grid search with cross-validation to find the best hyperparameters
    grid_search = GridSearchCV(
        estimator=rf_model,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        scoring='neg_mean_absolute_error',
        verbose=2
    )

    # Fit the model to the training data
    grid_search.fit(X_train, y_train)
    best_rf_model = grid_search.best_estimator_

    print(f"Best hyperparameters: {grid_search.best_params_}")

    # Make predictions on the test set (this means how good the model fits this data, not real recursive forecasting)
    y_pred = best_rf_model.predict(X_test)

    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)

    return best_rf_model, {'MAE': mae, 'RMSE': rmse}
