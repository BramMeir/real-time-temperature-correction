"""
Group of functions to compare the short horizon forecast models on their per window error difference
(src/scripts/short_horizon_comparison_experiment.py). Every model forecasts exactly the same windows,
so the comparison pairs the windows and works on the difference instead of on the error levels, which
removes the variation between stations and between forecast periods.
"""
import numpy as np
import pandas as pd

# Columns identifying one forecast window, which every model forecasts exactly once
WINDOW = ["Dataset", "Station", "Train_start"]

NUMBER_OF_RESAMPLES = 10_000
NUMBER_OF_SIGN_FLIPS = 10_000
SEED = 42


def window_losses(df, leads):
    """
    Mean absolute error per window and model, over the given forecast hours.

    Input
    -----
    df: DataFrame with one row per forecast hour and an Absolute_error column
    leads: Forecast hours to average over, a single hour for the error at that lead and a range for
      the error averaged over the whole forecast up to that lead

    Output
    ------
    Returns a DataFrame with the windows as rows and the models as columns.
    """
    selected = df[df["Lead_hours"].isin(list(leads))]

    return selected.groupby(WINDOW + ["Model"])["Absolute_error"].mean().unstack("Model")


def block_labels(windows, block_hours):
    """
    Label windows of the same dataset that start within block_hours of each other, so a bootstrap can
    resample them together. Used to check that overlapping forecast periods do not affect the result.

    Input
    -----
    windows: MultiIndex of the windows, as returned by window_losses
    block_hours: Maximum gap between two starts of the same block

    Output
    ------
    Returns an array with one label per window, in the order of the index.
    """
    frame = windows.to_frame(index=False)
    frame["Train_start"] = pd.to_datetime(frame["Train_start"])
    labels = np.empty(len(frame), dtype=object)

    for dataset, group in frame.groupby("Dataset"):
        ordered = group.sort_values("Train_start")
        new_block = ordered["Train_start"].diff() > pd.Timedelta(hours=block_hours)
        labels[ordered.index] = dataset + "_" + new_block.cumsum().astype(str)

    return labels


def bootstrap_interval(differences, rng, blocks=None, resamples=NUMBER_OF_RESAMPLES):
    """
    Percentile confidence interval for the mean of the paired differences, by resampling the windows
    with replacement. Nothing is assumed about the shape of the distribution of the differences.

    Input
    -----
    differences: Array with the per window difference between two models
    rng: Random generator
    blocks: Optional array with one block label per window, to resample whole blocks instead of
      single windows (default is None, which resamples single windows)
    resamples: Number of resamples to draw (default is NUMBER_OF_RESAMPLES)

    Output
    ------
    lower: Lower bound of the 95% interval
    upper: Upper bound of the 95% interval
    """
    values = np.asarray(differences, dtype=float)

    if blocks is None:
        # Every window is drawn on its own, so the resample is a single index operation
        means = values[rng.integers(0, len(values), (resamples, len(values)))].mean(axis=1)
    else:
        grouped = [values[blocks == label] for label in pd.unique(blocks)]
        means = np.array([
            np.concatenate([grouped[i] for i in rng.integers(0, len(grouped), len(grouped))]).mean()
            for _ in range(resamples)
        ])

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


def compare(df, contrasts, lead_groups, block_hours=None, seed=SEED):
    """
    Compare every pair of models over every group of forecast hours.

    Input
    -----
    df: DataFrame with one row per forecast hour and an Absolute_error column
    contrasts: List of (model, reference) tuples to compare
    lead_groups: List of (label, leads) tuples, the forecast hours to average over
    block_hours: Optional gap used to resample windows in blocks (default is None)
    seed: Seed of the random generator (default is SEED)

    Output
    ------
    Returns a DataFrame with one row per contrast and group of forecast hours, holding both error
    levels, the mean difference with its confidence interval, the raw and adjusted p value and the
    share of windows in which the model beats its reference.
    """
    rng = np.random.default_rng(seed)
    rows = []

    for label, leads in lead_groups:
        losses = window_losses(df, leads)
        blocks = block_labels(losses.index, block_hours) if block_hours else None

        for model, reference in contrasts:
            differences = (losses[model] - losses[reference]).values
            lower, upper = bootstrap_interval(differences, rng, blocks)

            rows.append({
                "Leads": label,
                "Model": model,
                "Reference": reference,
                "Model_MAE": losses[model].mean(),
                "Reference_MAE": losses[reference].mean(),
                "Difference": differences.mean(),
                "Lower": lower,
                "Upper": upper,
                "P_value": sign_flip_test(differences, rng),
                "Win_rate": float((differences < 0).mean()),
                "Windows": len(differences),
            })

    result = pd.DataFrame(rows)

    # Every contrast is tested once per group of forecast hours, so each contrast is its own family
    result["P_holm"] = result.groupby(["Model", "Reference"])["P_value"].transform(
        lambda values: holm(values.values)
    )

    return result
