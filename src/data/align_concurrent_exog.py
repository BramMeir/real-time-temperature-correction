def align_concurrent_exog(df, target_station):
    """
    Shift the neighbouring stations one step forward so the sequence models see x_t.

    The window built for LSTM/Transformer/TCN spans [t-L, t-1] to predict y_t, so it misses
    the neighbours' reading at t itself - the observation real-time correction is built
    around, and the one ARIMAX and the tabular models do get. Shifting puts (y_{t-1}, x_t)
    in the last window position. The target column is untouched, so no future y leaks in.

    Input
    -----
    df: Target station and neighbouring stations as columns, datetime index.
    target_station: Name of the target column, left unshifted.

    Output
    ------
    DataFrame with the same columns, one row shorter.
    """
    shifted = df.copy()

    exog_cols = [c for c in shifted.columns if c != target_station]
    shifted[exog_cols] = shifted[exog_cols].shift(-1)

    # The last row has no x_{t+1} to take. Dropping it costs the final hour of the series;
    # onsets are drawn from the unshifted series, so they stay identical across runs.
    return shifted.iloc[:-1]
