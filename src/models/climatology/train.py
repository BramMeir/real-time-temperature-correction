def train_hourly_climatology(series):
    """
    Train an hourly climatology model by averaging the observed temperature per hour of day over the
    training window. This is the whole "training" step for this baseline.

    Input
    -----
    series: Series with observed temperature values, indexed by datetime

    Output
    ------
    climatology: Series indexed by hour of day (0-23) holding the mean observed temperature for that
        hour of day over the training window. Hours of day absent from the training window are filled
        with the overall training mean so the forecast never produces NaN.
    """
    if series is None or series.dropna().empty:
        raise ValueError("Cannot train hourly climatology: the training series is empty or all-NaN")

    overall_mean = series.mean()

    # Group the observations by hour of day and average them (NaNs are ignored by groupby/mean)
    climatology = series.groupby(series.index.hour).mean()

    # Make sure every hour of day (0-23) is present, falling back to the overall mean if missing
    climatology = climatology.reindex(range(24), fill_value=overall_mean)
    climatology = climatology.fillna(overall_mean)

    return climatology
