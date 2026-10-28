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


def summarise_interval(lower, upper, actual):
    """
    Summarise one interval over one reconstruction window. Counts are returned so windows of
    different lengths pool correctly.

    Input
    -----
    lower, upper: Interval bounds over the window
    actual: Observed values over the window

    Output
    ------
    summary: Dict with n_total, n_inside, coverage and mean_width
    """
    lower, upper, actual = np.asarray(lower), np.asarray(upper), np.asarray(actual)
    inside = (actual >= lower) & (actual <= upper)

    return {
        "n_total": actual.size,
        "n_inside": int(inside.sum()),
        "coverage": float(inside.mean()),
        "mean_width": float(np.mean(upper - lower))
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
