from sklearn.ensemble import RandomForestRegressor
import pandas as pd


def train_random_forest(X_train, y_train, random_seed=42, n_estimators=200, max_depth=10,
                        min_samples_leaf=1, min_samples_split=2):
    """
    Train a Random Forest model with specified hyperparameters.

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
    # Initialize the base Random Forest model (n_jobs=1 to prevent issues with parallel processing in some environments)
    rf_model = RandomForestRegressor(random_state=random_seed, n_jobs=1,
                                     n_estimators=n_estimators, max_depth=max_depth,
                                     min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split)

    # Fit the model to the training data
    rf_model.fit(X_train, y_train)

    # Get the feature importances
    importances = rf_model.feature_importances_

    # Map the feature importances to their corresponding feature names
    feature_importance_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    # print("Feature importances:")
    # print(feature_importance_df.head(10))

    return rf_model, feature_importance_df
