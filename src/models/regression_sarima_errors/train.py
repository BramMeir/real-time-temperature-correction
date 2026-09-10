"""
Script: train.py

Regression on the neighbouring stations with SARIMA errors, estimated in two stages.

The model is the same specification as ARIMAX:

    y_t = beta' x_t + eta_t,   with eta_t following a SARIMA process

but beta is estimated by ordinary least squares instead of jointly by maximum likelihood
together with the error parameters. Joint (one-stage) estimation of ARIMAX maximises the
one-step-ahead likelihood, which for autoregressive errors is a GLS fit on the
quasi-differenced series, so beta ends up explaining hour-to-hour changes rather than the
level of the target. Multi-step forecasts converge to beta' x_t once the autoregressive
contribution has decayed, so long-lead accuracy depends only on how well beta reproduces
the level. Estimating beta by OLS and fitting the SARIMA to its residuals keeps each
component fitted against the quantity it is used to predict.

Functions:
- stage_one_residuals: Compute the residuals the neighbouring stations cannot explain.
- train_regression_sarima_errors: Fit the two-stage model.
"""
import pandas as pd
import statsmodels.api as sm
from src.models.linear_regression.train import train_neighbour_regression


def stage_one_residuals(df, target_station, exog_cols, regression=None):
    """
    Compute the stage-one residuals: what the neighbouring stations cannot explain about the target.

    Shared by the training and the grid search, so both see an identical series.

    Input
    -----
    df: DataFrame with a datetime index and one column per station, hourly frequency
    target_station: Name of the target station column to predict
    exog_cols: List of neighbouring station column names used as regressors
    regression: Trained LinearRegression model to reuse (default is None, which fits a new one)

    Output
    ------
    regression: The regression used to produce the residuals
    residuals: Pandas Series with the residuals, indexed by the rows the regression was fitted on
    """
    if regression is None:
        regression = train_neighbour_regression(df, target_station, exog_cols)

    # Restrict to the rows the regression was fitted on, so both stages see the same window
    data = df[[target_station] + list(exog_cols)].dropna()

    residuals = pd.Series(
        data[target_station].values - regression.predict(data[exog_cols]),
        index=data.index
    )

    return regression, residuals


def train_regression_sarima_errors(df, target_station, exog_cols, arima_order=(2, 0, 0),
                                   seasonal_order=(1, 0, 1, 24), max_iter=1000):
    """
    Fit a regression on the neighbouring stations with SARIMA errors in two stages: an OLS
    regression of the target station on the contemporaneous neighbour observations, followed
    by a SARIMA model of the regression residuals.

    Input
    -----
    df: DataFrame with a datetime index and one column per station, hourly frequency
    target_station: Name of the target station column to predict
    exog_cols: List of neighbouring station column names used as regressors
    arima_order: Tuple specifying the (p, d, q) parameters for the residual SARIMA (default is (2, 0, 0))
    seasonal_order: Tuple specifying the (P, D, Q, s) parameters for the residual SARIMA (default is (1, 0, 1, 24))
    max_iter: Maximum number of iterations for fitting the residual SARIMA (default is 1000)

    Output
    ------
    regression: Trained LinearRegression model mapping the neighbour observations to the target
    residual_model: Fitted SARIMA results object for the regression residuals
    """
    # Stage 1: the spatial part, an OLS regression of the target on the neighbouring stations
    # Stage 2: the local part, what the neighbours cannot explain
    regression, residuals = stage_one_residuals(df, target_station, exog_cols)

    # The residual series carries no exogenous variables, so this fit is independent of the number
    # of stations: 5 parameters instead of 5 + len(exog_cols) as in one-stage ARIMAX
    residual_model = sm.tsa.statespace.SARIMAX(
        endog=residuals,
        order=arima_order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit(maxiter=max_iter, disp=False)

    return regression, residual_model
