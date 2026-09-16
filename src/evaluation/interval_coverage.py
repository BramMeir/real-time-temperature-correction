"""
Module: interval_coverage.py

Empirical coverage of the two-stage model's prediction intervals (experiment P1.5 of the paper).

The existing reliability indicators (src/models/*/confidence_score.py) only show that a station's
interval width correlates with the error realised later. This checks whether the intervals are
honest: how often the truth falls inside a nominal 95 % interval.

Two bands are compared, both built from the same calibration-window errors:
- "quantile": the interpolated error quantiles, as regression_sarima_errors/confidence_score.py
  builds them today. Biased narrow - a band between order statistics of n points covers about
  (hi_rank - lo_rank)/(n + 1), which is ~0.92 for a nominal 0.95 at n = 72.
- "conformal": the finite-sample rank of the absolute errors, which removes that bias. Symmetric
  about the forecast, so it assumes the errors are roughly unbiased.

Functions:
- conformal_rank: Order statistic of the split-conformal quantile.
- interval_score: Per-point Winkler interval score.
- summarise_interval: Coverage, width and interval score over one window.
- empirical_bounds: Bounds from interpolated error quantiles.
- conformal_bounds: Bounds from the conformal rank of the absolute errors.
"""
import numpy as np


def conformal_rank(n, level):
    """
    Order statistic of the split-conformal quantile. The k-th largest of n absolute errors leaves k
    of the n + 1 gaps between them inside the band, so a new error falls inside with probability
    k/(n + 1).

    Input
    -----
    n: Number of calibration points
    level: Nominal coverage level

    Output
    ------
    k: 1-based rank into the sorted absolute errors
    """
    return int(np.ceil((n + 1) * level))


def interval_score(lower, upper, actual, level):
    """
    Winkler interval score per point: width plus a penalty for missing. Reported alongside
    coverage, which on its own is bought by widening.

    Input
    -----
    lower, upper: Interval bounds
    actual: Observed values
    level: Nominal coverage level

    Output
    ------
    score: Per-point scores, in the units of the observations
    """
    lower, upper, actual = np.asarray(lower), np.asarray(upper), np.asarray(actual)
    alpha = 1 - level

    width = upper - lower
    below = np.where(actual < lower, (2 / alpha) * (lower - actual), 0.0)
    above = np.where(actual > upper, (2 / alpha) * (actual - upper), 0.0)

    return width + below + above


def summarise_interval(lower, upper, actual, level):
    """
    Summarise one interval over one reconstruction window. Counts are returned so windows of
    different lengths pool correctly.

    Input
    -----
    lower, upper: Interval bounds over the window
    actual: Observed values over the window
    level: Nominal coverage level

    Output
    ------
    summary: Dict with n_total, n_inside, coverage, mean_width and mean_interval_score
    """
    lower, upper, actual = np.asarray(lower), np.asarray(upper), np.asarray(actual)
    inside = (actual >= lower) & (actual <= upper)

    return {
        "n_total": actual.size,
        "n_inside": int(inside.sum()),
        "coverage": float(inside.mean()),
        "mean_width": float(np.mean(upper - lower)),
        "mean_interval_score": float(np.mean(interval_score(lower, upper, actual, level)))
    }


def empirical_bounds(predictions, calibration_errors, level):
    """
    Interval bounds from the interpolated error quantiles, added to the point forecast. Constant in
    width, so it says nothing about how the error grows past the calibration window.

    Input
    -----
    predictions: Point forecast over the reconstruction window
    calibration_errors: Errors (actual - predicted) over the calibration window
    level: Nominal coverage level

    Output
    ------
    lower, upper: Series with the interval bounds
    """
    alpha = 1 - level

    lower_quantile = np.quantile(calibration_errors, alpha / 2)
    upper_quantile = np.quantile(calibration_errors, 1 - alpha / 2)

    return predictions + lower_quantile, predictions + upper_quantile


def conformal_bounds(predictions, calibration_errors, level):
    """
    Interval bounds at the conformal rank of the absolute calibration errors. Same inputs as
    empirical_bounds, but without its finite-sample narrowing.

    Input
    -----
    predictions: Point forecast over the reconstruction window
    calibration_errors: Errors (actual - predicted) over the calibration window
    level: Nominal coverage level

    Output
    ------
    lower, upper: Series with the interval bounds
    """
    absolute_errors = np.sort(np.abs(calibration_errors))
    half_width = absolute_errors[conformal_rank(absolute_errors.size, level) - 1]

    return predictions - half_width, predictions + half_width
