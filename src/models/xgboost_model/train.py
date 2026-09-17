import pandas as pd
from xgboost import XGBRegressor


def train_xgboost(X_train, y_train, random_seed=42, n_estimators=618, max_depth=7,
                  learning_rate=0.018, subsample=0.638, colsample_bytree=0.925, min_child_weight=8):
    """
    Train an XGBoost model with specified hyperparameters.

    The defaults are the most frequent outcome of the Bayesian search of
    src.models.xgboost_model.main on MELLE, at the 24 lags and 8 training weeks of the comparison.

    Input
    -----
    X_train: DataFrame with training features
    y_train: Series with training target variable
    random_seed: Random seed for reproducibility (default is 42)
    n_estimators: Number of boosting rounds (default is 618)
    max_depth: Maximum depth of the trees (default is 7)
    learning_rate: Shrinkage applied to each boosting round (default is 0.018)
    subsample: Fraction of the training rows sampled per tree (default is 0.638)
    colsample_bytree: Fraction of the features sampled per tree (default is 0.925)
    min_child_weight: Minimum sum of instance weights needed in a child (default is 8)

    Output
    ------
    model: Trained XGBoost model
    feature_importance_df: DataFrame with the gain importance per feature
    """
    # Initialize the base XGBoost model (n_jobs=1 to prevent issues with parallel processing in some environments)
    xgb_model = XGBRegressor(objective="reg:squarederror", random_state=random_seed, n_jobs=1,
                             n_estimators=n_estimators, max_depth=max_depth,
                             learning_rate=learning_rate, subsample=subsample,
                             colsample_bytree=colsample_bytree, min_child_weight=min_child_weight)

    # Fit the model to the training data
    xgb_model.fit(X_train, y_train)

    # Map the feature importances to their corresponding feature names
    feature_importance_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': xgb_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    return xgb_model, feature_importance_df
