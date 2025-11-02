import pandas as pd
import numpy as np
from src.evaluation.comparison_amber.evaluation_gf_techniques import Test_techniques_differentgaplengths
from src.models.arima.sarima_forecast import sarima_forecast


def test_different_gf_techniques_amber():
    # Read the Turku data and the ERA5 data into dataframes
    df_Turku = pd.read_csv("data/Turku_1H_LI.csv", index_col="DateTime", parse_dates=True)
    df_ERA5 = pd.read_csv("data/Turku_ERA5.csv", index_col="DateTime", parse_dates=True)

    # Rename the ERA5 station names to station name + '_ERA5'
    df_ERA5.columns = [col + "_ERA5" for col in df_ERA5.columns if col != "DateTime"]

    # Make sure the Turku data is complete by filling missing timestamps using Linear Interpolation
    df_Turku = df_Turku.resample("h").mean().interpolate()

    # Join both datasets on their timestamps
    df_all = df_Turku.join(df_ERA5, how="inner")

    df_errors, df_stderr = Test_techniques_differentgaplengths(
        df_all,
        name_fulldata='Ylijoki',
        name_model='Ylijoki_ERA5',
        dictionarytechniques={
            "debmodelReg": [60, 1, 'both'],
            "debmodelMeanbias": [60, 1, 'both'],
            "debmodelTvar": [60, 1, 'both'],
        },
        par_slicedates=30,
        error="MSE",
        range_gaplengths=[5, 7, 12, 24, 48, 168, 336],
        repetitions=250,
        check=50,
        plot=True,
    )

    print("Mean Errors:")
    print(df_errors)

    print("\nStandard Errors:")
    print(df_stderr)


def test_SARIMA_approach(seed=47):
    # Set seed for reproducibility
    np.random.seed(seed)

    # Read the Turku data into a dataframe
    df_Turku = pd.read_csv("data/Turku_1H_LI.csv", index_col="DateTime", parse_dates=True)

    # Make sure the Turku data is complete by filling missing timestamps using Linear Interpolation
    df_Turku = df_Turku.resample("h").mean().interpolate()

    # Select the target station time series
    series_full = df_Turku["Ylijoki"]

    # Select the exogenous data (other stations in the Turku dataset)
    exog_df_full = df_Turku.drop(columns=["Ylijoki"])

    # Keep track of the forecast errors
    mse_list = []
    mae_list = []

    for hours_to_forecast in [5, 7, 12, 24, 48, 168, 336]:
        print(f"\nForecasting {hours_to_forecast} hours into the future:")

        # Repeat the forecasting multiple times to get an average error
        temp_mse_list = []
        temp_mae_list = []
        for _ in range(100):
            # Select random 2 weeks + forecast horizon for testing from series and exog_df
            max_start = series_full.index.max() - pd.DateOffset(hours=hours_to_forecast + 24 * 14)
            min_start = series_full.index.min()
            random_start = min_start + (max_start - min_start) * np.random.random()

            print(random_start)

            series = series_full[random_start: random_start + pd.DateOffset(hours=hours_to_forecast + 24 * 14)]
            exog_df = exog_df_full[random_start: random_start + pd.DateOffset(hours=hours_to_forecast + 24 * 14)]

            errors = sarima_forecast(
                series,
                exog_df=exog_df,
                hours_to_forecast=hours_to_forecast,
                arima_order=(25, 0, 0),
                seasonal_order=(0, 0, 0, 0),
                max_iter=1000,
                plot=False,
            )
            temp_mse_list.append(errors["MSE"])
            temp_mae_list.append(errors["MAE"])

        average_mse = sum(temp_mse_list) / len(temp_mse_list)
        mse_list.append(average_mse)

        average_mae = sum(temp_mae_list) / len(temp_mae_list)
        mae_list.append(average_mae)

    print("\nAverage MSE for different forecast horizons:")
    for hours, mse in zip([5, 7, 12, 24, 48, 168, 336], mse_list):
        print(f"{hours} hours: MSE = {mse}")

    print("\nAverage MAE for different forecast horizons:")
    for hours, mae in zip([5, 7, 12, 24, 48, 168, 336], mae_list):
        print(f"{hours} hours: MAE = {mae}")


if __name__ == "__main__":
    test_different_gf_techniques_amber()
    # test_SARIMA_approach()
