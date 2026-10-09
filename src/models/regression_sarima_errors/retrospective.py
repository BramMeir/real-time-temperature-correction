"""
Script: retrospective.py

Retrospective use of the two-stage regression with SARIMA errors: filling a gap afterwards, when the
target's observations after the gap are available as well.

Two things change with respect to the real-time correction of execute_forecast.py, and nothing else:
- the regression is fitted on a window around the gap rather than on the weeks before it, like the
  symmetric learning period of the ERA5 debiasing of Jacobs et al. (2024);
- the residual inside the gap is estimated by the Kalman smoother from the residuals on both sides
  of it, rather than forecast from the residuals before it. Near either end of the gap the estimate
  follows the residual observed there, and in the middle of a long gap it decays to zero, so that
  the reconstruction reduces to the regression, as in real time.

The SARIMA is fitted on the residual series of the whole window with the gap left missing, which the
state-space likelihood handles directly. Unlike in real time, the fit is restricted to stationary
parameters. With the gap missing, the optimiser occasionally ends in an explosive AR polynomial
(22 of 78000 fits in the comparison with Jacobs et al., all with gaps of a week or more), which the
smoother then propagates over every missing hour into errors of up to 1e70 degC^2. The restriction
matches the model, whose errors are stationary, and leaves the other fits practically unchanged.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from src.models.regression_sarima_errors.train import stage_one_residuals


def fill_gap_retrospective(df, target_station, exog_cols, gap_start, gap_end, window_days=28,
                           arima_order=(3, 0, 0), seasonal_order=(1, 0, 0, 24), max_iter=1000):
    """
    Reconstruct the target over a gap from the neighbours and from the target's observations on both
    sides of the gap.

    Input
    -----
    df: DataFrame with an hourly datetime index and one column per station
    target_station: Name of the target station column
    exog_cols: List of neighbouring station column names used as regressors
    gap_start: First missing hour
    gap_end: Last missing hour
    window_days: Days of observations used on each side of the gap (default 28, so that the window
      holds eight weeks, as the training window of the real-time correction)
    arima_order: (p, d, q) of the residual SARIMA (default as in the model comparison)
    seasonal_order: (P, D, Q, s) of the residual SARIMA (default as in the model comparison)
    max_iter: Maximum number of iterations for fitting the residual SARIMA

    Output
    ------
    predictions: Series with the reconstruction for every hour of the gap
    spatial: Series with the regression part alone, the reconstruction without the residual model
    """
    window = df.loc[
        gap_start - pd.Timedelta(days=window_days):gap_end + pd.Timedelta(days=window_days),
        [target_station] + list(exog_cols)
    ].copy()
    window.loc[gap_start:gap_end, target_station] = np.nan

    # Stage 1 drops the missing rows, so the regression is fitted on both sides of the gap only
    regression, _ = stage_one_residuals(window, target_station, exog_cols)

    # Stage 2 on the residuals of the whole window, missing inside the gap
    residuals = window[target_station] - regression.predict(window[exog_cols])
    residual_model = sm.tsa.statespace.SARIMAX(
        endog=residuals,
        order=arima_order,
        seasonal_order=seasonal_order,
        enforce_stationarity=True,
        enforce_invertibility=False
    ).fit(maxiter=max_iter, disp=False)

    gap_index = window.loc[gap_start:gap_end].index
    spatial = pd.Series(regression.predict(window.loc[gap_index, exog_cols]), index=gap_index)
    local = residual_model.get_prediction(
        start=gap_start, end=gap_end, information_set="smoothed"
    ).predicted_mean

    return spatial + local.values, spatial
