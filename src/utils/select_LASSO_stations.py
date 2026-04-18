"""
- select_LASSO_stations: Select useful exogenous variables using LASSO regression. This function takes a
  time series and a DataFrame of exogenous variables, standardizes the features,
  and fits a LassoCV model to identify which exogenous variables have non-zero coefficients.
"""


import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler


def select_LASSO_stations(
    series,
    exog_df,
    random_state=47
):
    """
    Select the most useful exogenous variables using LASSO regression. Only the variables with non-zero coefficients
    will be selected.

    Input
    -----
    series: Pandas Series with the target time series data
    exog_df: DataFrame with the exogenous variables (must have the same index as the series)
    random_state: Random seed for reproducibility (default is 42)

    Output
    ------
    selected_exog_df: DataFrame containing only the selected exogenous variables
    selected_columns: List of the names of the selected exogenous variables
    selected: Series with the absolute values of the coefficients of the selected variables, sorted in descending order
    """
    # Align indices
    common_idx = series.index.intersection(exog_df.index)
    y = series.loc[common_idx]
    X = exog_df.loc[common_idx]

    # Standardize features (important for LASSO)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Cross-validated LASSO
    lasso = LassoCV(
        cv=5,
        random_state=random_state,
        n_jobs=-1,
        max_iter=10000
    )
    lasso.fit(X_scaled, y.values)

    coefs = pd.Series(lasso.coef_, index=X.columns)
    ranking = coefs.abs().sort_values(ascending=False)

    # Keep non-zero coefficients
    ranking = ranking[ranking > 0]

    selected_exog_df = exog_df[ranking.index]

    return selected_exog_df, ranking.index.tolist(), ranking
