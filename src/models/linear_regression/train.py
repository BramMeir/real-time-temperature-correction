from sklearn.linear_model import LinearRegression
import pandas as pd


def train_neighbour_regression(df, target_station, exog_cols):
    """
    Fit an ordinary least squares linear regression of the target station on the contemporaneous
    observations of the neighbouring stations.

    Input
    -----
    df: DataFrame with a datetime index and one column per station
    target_station: Name of the target station column to predict
    exog_cols: List of neighbouring station column names used as regressors

    Output
    ------
    model: Trained LinearRegression model
    coefficient_df: DataFrame with columns 'Feature' and 'Coefficient', sorted by absolute
                    coefficient value descending
    """
    # Drop the timestamps where the target or any neighbour is missing before fitting
    data = df[[target_station] + list(exog_cols)].dropna()

    if data.empty:
        raise ValueError(
            f"No overlapping, non-missing observations for target '{target_station}' and "
            f"neighbours {list(exog_cols)} to fit the linear regression on"
        )

    X_train = data[exog_cols]
    y_train = data[target_station]

    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    coefficient_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Coefficient': lr_model.coef_
    })
    coefficient_df = coefficient_df.reindex(
        coefficient_df['Coefficient'].abs().sort_values(ascending=False).index
    ).reset_index(drop=True)

    return lr_model, coefficient_df
