import numpy as np

# Columns identifying one forecast window, which every model forecasts exactly once
WINDOW = ["Dataset", "Station", "Train_start"]

NUMBER_OF_RESAMPLES = 10_000
NUMBER_OF_SIGN_FLIPS = 10_000
SEED = 42


def bootstrap_interval(differences, rng, resamples=NUMBER_OF_RESAMPLES):
    """
    Percentile confidence interval for the mean of the paired differences, by resampling the windows
    with replacement. Nothing is assumed about the shape of the distribution of the differences.

    Input
    -----
    differences: Array with the per window difference between two models
    rng: Random generator
    resamples: Number of resamples to draw (default is NUMBER_OF_RESAMPLES)

    Output
    ------
    lower: Lower bound of the 95% interval
    upper: Upper bound of the 95% interval
    """
    values = np.asarray(differences, dtype=float)

    means = values[rng.integers(0, len(values), (resamples, len(values)))].mean(axis=1)

    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def sign_flip_test(differences, rng, flips=NUMBER_OF_SIGN_FLIPS):
    """
    Two sided p value for the mean of the paired differences. If the two models forecast equally well
    the sign of a difference carries no information, so flipping signs at random builds the
    distribution of the mean under that null.

    Input
    -----
    differences: Array with the per window difference between two models
    rng: Random generator
    flips: Number of sign patterns to draw (default is NUMBER_OF_SIGN_FLIPS)

    Output
    ------
    Returns the share of sign patterns reaching the observed mean, never zero so it stays reportable.
    """
    values = np.asarray(differences, dtype=float)
    observed = abs(values.mean())

    signs = rng.choice([-1.0, 1.0], size=(flips, len(values)))
    under_null = np.abs((signs * values).mean(axis=1))

    return float((np.count_nonzero(under_null >= observed) + 1) / (flips + 1))


def holm(p_values):
    """
    Holm adjusted p values, controlling the chance of any false positive over a family of tests.

    Input
    -----
    p_values: Array with the raw p values of the family

    Output
    ------
    Returns an array with the adjusted p values, in the order of the input.
    """
    values = np.asarray(p_values, dtype=float)
    adjusted = np.empty_like(values)
    running = 0.0

    # Holm walks the p values from small to large, each one scaled by the number still untested
    for rank, position in enumerate(np.argsort(values)):
        running = max(running, (len(values) - rank) * values[position])
        adjusted[position] = min(running, 1.0)

    return adjusted
