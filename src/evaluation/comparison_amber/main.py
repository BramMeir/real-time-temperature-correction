"""
Main file to run the comparison of the different gap-filling techniques presented in Amber's paper with the SARIMA approach.
The comparisons are done on the Turku dataset.

Example usage:
python -m src.evaluation.comparison_amber.main --
"""
import argparse
import pandas as pd
import numpy as np
from src.evaluation.comparison_amber.evaluation_gf_techniques import Test_techniques_differentgaplengths
from src.models.arima.sarima_forecast import sarima_forecast
from concurrent.futures import ProcessPoolExecutor, as_completed
from src.models.random_forest.execute_forecast import run_single_forecast
from src.models.TCN.evaluate_forecast import evaluate_forecast as evaluate_tcn_forecast
from src.models.LSTM.evaluate_forecast import evaluate_LSTM_forecast
from src.models.transformer.evaluate_forecast import evaluate_forecast as evaluate_transformer_forecast
from src.models.LSTM.train import train_LSTM_model
from src.models.MLP.train import train_mlp_model
from src.models.TCN.train import train_tcn_model
from src.models.transformer.train import train_transformer_model
from src.data.create_supervised import create_supervised_dataset
from src.data.create_3d_dataset import create_3d_dataset


def test_different_gf_techniques_amber(input_file="data/Turku/Turku_1H_LI.csv"):
    """
    Test different gap-filling techniques from Amber's paper on the Turku dataset using the Test_techniques_differentgaplengths function.
    The default parameters are set to the ones preferred in Amber's paper.
    """
    # Read the Turku data and the ERA5 data into dataframes
    df_Turku = pd.read_csv(input_file, index_col="DateTime", parse_dates=True)
    df_ERA5 = pd.read_csv("data/Turku/Turku_ERA5.csv", index_col="DateTime", parse_dates=True)

    # Rename the ERA5 station names to station name + '_ERA5'
    df_ERA5.columns = [col + "_ERA5" for col in df_ERA5.columns if col != "DateTime"]

    # Make sure the Turku data is complete by filling missing timestamps using Linear Interpolation
    df_Turku = df_Turku.resample("h").mean().interpolate()

    # Join both datasets on their timestamps
    df_all = df_Turku.join(df_ERA5, how="inner")

    # Run the evaluation of different gap-filling techniques
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


def test_SARIMA_approach(input_file="data/Turku/Turku_1H_LI.csv", seed=47):
    """
    Test the SARIMA approach on the Turku dataset for different forecast horizons.

    Input
    -----
    seed : Random seed for reproducibility.
    """
    # Set seed for reproducibility
    np.random.seed(seed)

    # Read the Turku data into a dataframe
    df_Turku = pd.read_csv(input_file, index_col="DateTime", parse_dates=True)

    # Make sure the Turku data is complete by filling missing timestamps using Linear Interpolation
    df_Turku = df_Turku.resample("h").mean().interpolate()

    # Select the target station time series
    series_full = df_Turku["Betel"]

    # Select the exogenous data (other stations in the Turku dataset)
    exog_df_full = df_Turku.drop(columns=["Betel", "Puutori", "Virastotalo"])

    # Keep track of the forecast errors
    mse_list = []
    mae_list = []

    for hours_to_forecast in [5, 7, 12, 24, 48, 168, 336]:
        print(f"\nForecasting {hours_to_forecast} hours into the future:")

        # Repeat the forecasting multiple times to get an average error
        temp_mse_list = []
        temp_mae_list = []

        max_start = series_full.index.max() - pd.DateOffset(hours=hours_to_forecast + 24 * 14)
        min_start = series_full.index.min()

        with ProcessPoolExecutor(max_workers=10) as executor:
            futures = []
            for _ in range(100):
                # Select random 2 weeks (training) + forecast horizon from series and exog_df
                random_start = min_start + (max_start - min_start) * np.random.random()

                series = series_full[random_start: random_start + pd.DateOffset(hours=hours_to_forecast + 24 * 14)]
                exog_df = exog_df_full[random_start: random_start + pd.DateOffset(hours=hours_to_forecast + 24 * 14)]

                futures.append(executor.submit(
                    sarima_forecast,
                    series,
                    exog_df,
                    hours_to_forecast,
                    (25, 0, 0),
                    (0, 0, 0, 0),
                    1000,
                    False,
                ))

            for future in as_completed(futures):
                errors = future.result()
                temp_mse_list.append(errors["MSE"])
                temp_mae_list.append(errors["MAE"])

        mse_list.append(np.mean(temp_mse_list))
        mae_list.append(np.mean(temp_mae_list))

    print("\nAverage MSE for different forecast horizons:")
    for hours, mse in zip([5, 7, 12, 24, 48, 168, 336], mse_list):
        print(f"{hours} hours: MSE = {mse}")

    print("\nAverage MAE for different forecast horizons:")
    for hours, mae in zip([5, 7, 12, 24, 48, 168, 336], mae_list):
        print(f"{hours} hours: MAE = {mae}")


def test_RF_approach(input_file="data/Turku/Turku_1H_LI.csv", seed=47):
    """
    Test the RF approach on the Turku dataset for different forecast horizons.

    Input
    -----
    seed : Random seed for reproducibility.
    """
    # Set seed for reproducibility
    np.random.seed(seed)

    # Read the Turku data into a dataframe
    df_Turku = pd.read_csv(input_file, index_col="DateTime", parse_dates=True)

    # Make sure the Turku data is complete by filling missing timestamps using Linear Interpolation
    df_Turku = df_Turku.resample("h").mean().interpolate()

    # Select the target station time series
    series_full = df_Turku["Betel"]

    # Select the exogenous data (other stations in the Turku dataset)
    exog_df_full = df_Turku.drop(columns=["Betel"])

    # Keep track of the forecast errors
    mse_list = []
    mae_list = []

    for hours_to_forecast in [5, 7, 12, 24, 48, 168, 336]:
        print(f"\nForecasting {hours_to_forecast} hours into the future:")

        # Repeat the forecasting multiple times to get an average error
        temp_mse_list = []
        temp_mae_list = []

        max_start = series_full.index.max() - pd.DateOffset(hours=hours_to_forecast + 24 * 14)
        min_start = series_full.index.min()

        with ProcessPoolExecutor(max_workers=10) as executor:
            futures = []
            for _ in range(50):
                # Select random 2 weeks (training) + forecast horizon from series and exog_df
                random_start = min_start + (max_start - min_start) * np.random.random()
                train_end = random_start + pd.DateOffset(hours=24 * 14)
                forecast_end = train_end + pd.DateOffset(hours=hours_to_forecast)

                # Take the slice of the dataframe for the selected period
                df_slice = df_Turku.loc[random_start:forecast_end].copy()

                futures.append(executor.submit(
                    run_single_forecast,
                    df_slice,
                    target_station="Betel",
                    previous_time_steps=3,
                    exog_cols=exog_df_full.columns.tolist(),
                    start=random_start,
                    train_end=train_end,
                    test_end=forecast_end,
                    mode="repeat_forecast",
                ))

            for future in as_completed(futures):
                mae, mse, _ = future.result()
                temp_mse_list.append(mse)
                temp_mae_list.append(mae)

        mse_list.append(np.mean(temp_mse_list))
        mae_list.append(np.mean(temp_mae_list))

    print("\nAverage MSE for different forecast horizons:")
    for hours, mse in zip([5, 7, 12, 24, 48, 168, 336], mse_list):
        print(f"{hours} hours: MSE = {mse}")

    print("\nAverage MAE for different forecast horizons:")
    for hours, mae in zip([5, 7, 12, 24, 48, 168, 336], mae_list):
        print(f"{hours} hours: MAE = {mae}")


def test_LSTM_approach(input_file="data/Turku/Turku_1H_LI.csv", seed=47):
    """
    Test the LSTM approach on the Turku dataset for different forecast horizons.

    Input
    -----
    seed : Random seed for reproducibility.
    """
    # Set seed for reproducibility
    np.random.seed(seed)

    # Read the Turku data into a dataframe
    df_Turku = pd.read_csv(input_file, index_col="DateTime", parse_dates=True)

    # Make sure the Turku data is complete by filling missing timestamps using Linear Interpolation
    df_Turku = df_Turku.resample("h").mean().interpolate()

    # Select the target station time series
    series_full = df_Turku["Betel"]

    # Select only the target station and the rural stations as exogenous data
    df_partly = df_Turku.drop(columns=["Puutori", "Virastotalo"])

    # Keep track of the forecast errors
    mse_dir = {}
    mae_dir = {}

    max_start = series_full.index.max() - pd.DateOffset(hours=336 + 24 * 21)
    min_start = series_full.index.min()

    # Keep a list of the X_test and y_test
    temp_train = {}
    temp_test = {}

    with ProcessPoolExecutor(max_workers=10) as executor:
        futures = []
        for i in range(30):

            # Select random 3 weeks (training) + forecast horizon from series and exog_df
            random_start = min_start + (max_start - min_start) * np.random.random()
            train_end = random_start + pd.DateOffset(hours=24 * 7 * 3)
            max_forecast_end = train_end + pd.DateOffset(hours=336)

            # Take the slice of the dataframe for the selected period
            df_slice = df_partly.loc[random_start:max_forecast_end].copy()

            # Select the training and test parts of the df_slice
            df_train = df_slice.loc[random_start:train_end].copy()
            df_test = df_slice.loc[train_end:max_forecast_end].copy()
            df_test = df_test.iloc[1:]  # Remove the first row to avoid overlap (loc slicing is inclusive)

            # Store the test sets for evaluation later
            temp_train[i] = df_train
            temp_test[i] = df_test

            futures.append(executor.submit(
                execute_train_LSTM_model,
                i,
                df_train,
                "Betel",
            ))

        for future in as_completed(futures):
            # Get both the model and the corresponding index
            model, index = future.result()

            for hours_to_forecast in [5, 7, 12, 24, 48, 168, 336]:
                print(f"\nForecasting {hours_to_forecast} hours into the future (index {index}):")

                # Get the corresponding test set
                df_train = temp_train[index]
                df_test = temp_test[index].iloc[:hours_to_forecast]

                # Make predictions
                mae, rmse = evaluate_LSTM_forecast(model, index, df_train, df_test,
                                                   previous_time_steps=5, target_station="Betel", plot=True)
                mse = rmse ** 2

                # Store the errors
                if hours_to_forecast not in mse_dir:
                    mse_dir[hours_to_forecast] = []
                    mae_dir[hours_to_forecast] = []

                mse_dir[hours_to_forecast].append(mse)
                mae_dir[hours_to_forecast].append(mae)

    # Calculate average errors for each forecast horizon
    mse_list = []
    mae_list = []

    for hours_to_forecast in [5, 7, 12, 24, 48, 168, 336]:
        mse_list.append(np.mean(mse_dir[hours_to_forecast]))
        mae_list.append(np.mean(mae_dir[hours_to_forecast]))

    print("\nAverage MSE for different forecast horizons:")
    for hours, mse in zip([5, 7, 12, 24, 48, 168, 336], mse_list):
        print(f"{hours} hours: MSE = {mse}")

    print("\nAverage MAE for different forecast horizons:")
    for hours, mae in zip([5, 7, 12, 24, 48, 168, 336], mae_list):
        print(f"{hours} hours: MAE = {mae}")


def execute_train_LSTM_model(i, df_train, target_station):
    model = train_LSTM_model(df_train, target_station, previous_time_steps=5)
    return model, i


def test_MLP_approach(input_file="data/Turku/Turku_1H_LI.csv", seed=47):
    """
    Test the MLP approach on the Turku dataset for different forecast horizons.

    Input
    -----
    seed : Random seed for reproducibility.
    """
    # Set seed for reproducibility
    np.random.seed(seed)

    # Read the Turku data into a dataframe
    df_Turku = pd.read_csv(input_file, index_col="DateTime", parse_dates=True)

    # Make sure the Turku data is complete by filling missing timestamps using Linear Interpolation
    df_Turku = df_Turku.resample("h").mean().interpolate()

    # Select the target station time series
    series_full = df_Turku["Betel"]

    # Select the exogenous data (other stations in the Turku dataset)
    exog_df_full = df_Turku.drop(columns=["Betel"])

    # Keep track of the forecast errors
    mse_dir = {}
    mae_dir = {}

    max_start = series_full.index.max() - pd.DateOffset(hours=336 + 24 * 21)
    min_start = series_full.index.min()

    # Keep a list of the X_test and y_test
    temp_y_train = {}
    temp_x_test = {}
    temp_y_test = {}

    with ProcessPoolExecutor(max_workers=10) as executor:
        futures = []
        for i in range(100):

            # Select random 6 weeks (training) + forecast horizon from series and exog_df
            random_start = min_start + (max_start - min_start) * np.random.random()
            train_end = random_start + pd.DateOffset(hours=24 * 7 * 3)
            max_forecast_end = train_end + pd.DateOffset(hours=336)

            # Take the slice of the dataframe for the selected period
            df_slice = df_Turku.loc[random_start:max_forecast_end].copy()

            # Create supervised dataset
            X, y = create_supervised_dataset(df_slice, target_station="Betel",
                                             previous_time_steps=5,
                                             exog_cols=exog_df_full.columns.tolist(), exog_lags=2)

            # Select the data based on the provided date ranges
            X_train, y_train = X.loc[random_start:train_end], y.loc[random_start:train_end]
            X_test = X.loc[train_end + pd.DateOffset(hours=1):max_forecast_end]
            y_test = y.loc[train_end + pd.DateOffset(hours=1):max_forecast_end]

            # Store the test sets for evaluation later
            temp_y_train[i] = y_train
            temp_x_test[i] = X_test
            temp_y_test[i] = y_test

            futures.append(executor.submit(
                execute_train_mlp_model,
                i,
                X_train,
                y_train,
                seed
            ))

        for future in as_completed(futures):
            # Get both the model and the corresponding index
            model, index = future.result()

            for hours_to_forecast in [5, 7, 12, 24, 48, 168, 336]:
                print(f"\nForecasting {hours_to_forecast} hours into the future (index {index}):")

                # Get the corresponding test set
                y_train = temp_y_train[index]
                X_test = temp_x_test[index].iloc[:hours_to_forecast]
                y_test = temp_y_test[index].iloc[:hours_to_forecast]

                # Make predictions
                mae, rmse = evaluate_tcn_forecast(model, y_train, X_test, y_test, True)
                mse = rmse ** 2

                # Store the errors
                if hours_to_forecast not in mse_dir:
                    mse_dir[hours_to_forecast] = []
                    mae_dir[hours_to_forecast] = []

                mse_dir[hours_to_forecast].append(mse)
                mae_dir[hours_to_forecast].append(mae)

    # Calculate average errors for each forecast horizon
    mse_list = []
    mae_list = []

    for hours_to_forecast in [5, 7, 12, 24, 48, 168, 336]:
        mse_list.append(np.mean(mse_dir[hours_to_forecast]))
        mae_list.append(np.mean(mae_dir[hours_to_forecast]))

    print("\nAverage MSE for different forecast horizons:")
    for hours, mse in zip([5, 7, 12, 24, 48, 168, 336], mse_list):
        print(f"{hours} hours: MSE = {mse}")

    print("\nAverage MAE for different forecast horizons:")
    for hours, mae in zip([5, 7, 12, 24, 48, 168, 336], mae_list):
        print(f"{hours} hours: MAE = {mae}")


def execute_train_mlp_model(i, X_train, y_train, random_seed):
    model = train_mlp_model(X_train, y_train, random_seed=random_seed)
    return model, i


def test_TCN_approach(input_file="data/Turku/Turku_1H_LI.csv", seed=47):
    """
    Test the TCN approach on the Turku dataset for different forecast horizons.

    Input
    -----
    seed : Random seed for reproducibility.
    """
    # Set seed for reproducibility
    np.random.seed(seed)

    # Read the Turku data into a dataframe
    df_Turku = pd.read_csv(input_file, index_col="DateTime", parse_dates=True)

    # Make sure the Turku data is complete by filling missing timestamps using Linear Interpolation
    df_Turku = df_Turku.resample("h").mean().interpolate()

    # Select the target station time series
    series_full = df_Turku["Betel"]

    # Select the exogenous data (other stations in the Turku dataset)
    exog_df_full = df_Turku.drop(columns=["Betel"])

    # Keep track of the forecast errors
    mse_dir = {}
    mae_dir = {}

    max_start = series_full.index.max() - pd.DateOffset(hours=336 + 24 * 14)
    min_start = series_full.index.min()

    # Keep a list of the X_test and y_test
    temp_y_train = {}
    temp_x_test = {}
    temp_y_test = {}
    temp_dates_train = {}
    temp_dates_test = {}

    with ProcessPoolExecutor(max_workers=10) as executor:
        futures = []
        for i in range(50):

            # Select random 2 weeks (training) + forecast horizon from series and exog_df
            random_start = min_start + (max_start - min_start) * np.random.random()
            train_end = random_start + pd.DateOffset(hours=24 * 7 * 2)
            max_forecast_end = train_end + pd.DateOffset(hours=336)

            # Take the slice of the dataframe for the selected period
            df_slice = df_Turku.loc[random_start:max_forecast_end].copy()

            # Create supervised dataset
            X, y, dates = create_3d_dataset(df_slice, target_station="Betel",
                                            previous_time_steps=24 * 3,
                                            exog_cols=exog_df_full.columns.tolist())

            # Select the data based on the provided date ranges
            train_mask = (dates >= random_start) & (dates <= train_end)
            test_mask = (dates > train_end) & (dates <= max_forecast_end)

            X_train = X[train_mask]
            y_train = y[train_mask]

            X_test = X[test_mask]
            y_test = y[test_mask]

            # Store the test sets for evaluation later
            temp_y_train[i] = y_train
            temp_x_test[i] = X_test
            temp_y_test[i] = y_test
            temp_dates_train[i] = dates[train_mask]
            temp_dates_test[i] = dates[test_mask]

            futures.append(executor.submit(
                execute_train_tcn_model,
                i,
                X_train,
                y_train,
                seed
            ))

        for future in as_completed(futures):
            # Get both the model and the corresponding index
            model, index = future.result()

            for hours_to_forecast in [5, 7, 12, 24, 48, 168, 336]:
                print(f"\nForecasting {hours_to_forecast} hours into the future (index {index}):")

                # Get the corresponding test set
                y_train = temp_y_train[index]
                X_test = temp_x_test[index][:hours_to_forecast]
                y_test = temp_y_test[index][:hours_to_forecast]
                dates_train = temp_dates_train[index]
                dates_test = temp_dates_test[index][:hours_to_forecast]

                # Make predictions
                mae, rmse = evaluate_tcn_forecast(model, y_train, X_test, y_test,
                                                  dates_train, dates_test,
                                                  previous_time_steps=24 * 3,
                                                  target_station="Betel", plot=True)

                mse = rmse ** 2

                # Store the errors
                if hours_to_forecast not in mse_dir:
                    mse_dir[hours_to_forecast] = []
                    mae_dir[hours_to_forecast] = []

                mse_dir[hours_to_forecast].append(mse)
                mae_dir[hours_to_forecast].append(mae)

    # Calculate average errors for each forecast horizon
    mse_list = []
    mae_list = []

    for hours_to_forecast in [5, 7, 12, 24, 48, 168, 336]:
        mse_list.append(np.mean(mse_dir[hours_to_forecast]))
        mae_list.append(np.mean(mae_dir[hours_to_forecast]))

    print("\nAverage MSE for different forecast horizons:")
    for hours, mse in zip([5, 7, 12, 24, 48, 168, 336], mse_list):
        print(f"{hours} hours: MSE = {mse}")

    print("\nAverage MAE for different forecast horizons:")
    for hours, mae in zip([5, 7, 12, 24, 48, 168, 336], mae_list):
        print(f"{hours} hours: MAE = {mae}")


def execute_train_tcn_model(i, X_train, y_train, random_seed):
    model = train_tcn_model(X_train, y_train, random_seed=random_seed)
    return model, i


def test_transformer_approach(input_file="data/Turku/Turku_1H_LI.csv", seed=47):
    """
    Test the Transformer approach on the Turku dataset for different forecast horizons.

    Input
    -----
    seed : Random seed for reproducibility.
    """
    # Set seed for reproducibility
    np.random.seed(seed)

    # Read the Turku data into a dataframe
    df_Turku = pd.read_csv(input_file, index_col="DateTime", parse_dates=True)

    # Make sure the Turku data is complete by filling missing timestamps using Linear Interpolation
    df_Turku = df_Turku.resample("h").mean().interpolate()

    # Select the target station time series
    series_full = df_Turku["Betel"]

    # Select only the target station and the rural stations as exogenous data
    # df_partly = df_Turku.drop(columns=["Puutori", "Virastotalo"])

    # Keep track of the forecast errors
    mse_dir = {}
    mae_dir = {}

    max_start = series_full.index.max() - pd.DateOffset(hours=336 + 24 * 7 * 6)
    min_start = series_full.index.min()

    # Keep a list of the X_test and y_test
    temp_train = {}
    temp_test = {}

    with ProcessPoolExecutor(max_workers=10) as executor:
        futures = []
        for i in range(30):

            # Select random 3 weeks (training) + forecast horizon from series and exog_df
            random_start = min_start + (max_start - min_start) * np.random.random()
            train_end = random_start + pd.DateOffset(hours=24 * 7 * 6)
            max_forecast_end = train_end + pd.DateOffset(hours=336)

            # Take the slice of the dataframe for the selected period
            df_slice = df_Turku.loc[random_start:max_forecast_end].copy()

            # Select the training and test parts of the df_slice
            df_train = df_slice.loc[random_start:train_end].copy()
            df_test = df_slice.loc[train_end:max_forecast_end].copy()
            df_test = df_test.iloc[1:]  # Remove the first row to avoid overlap (loc slicing is inclusive)

            # Store the test sets for evaluation later
            temp_train[i] = df_train
            temp_test[i] = df_test

            futures.append(executor.submit(
                execute_train_transformer_model,
                i,
                df_train,
                "Betel",
            ))

        for future in as_completed(futures):
            # Get both the model and the corresponding index
            model, x_scaler, y_scaler, index = future.result()

            for hours_to_forecast in [5, 7, 12, 24, 48, 168, 336]:
                print(f"\nForecasting {hours_to_forecast} hours into the future (index {index}):")

                # Get the corresponding test set
                df_train = temp_train[index]
                df_test = temp_test[index].iloc[:hours_to_forecast]

                # Make predictions
                mae, rmse = evaluate_transformer_forecast(model, index, df_train, df_test, previous_time_steps=5,
                                                          target_station="Betel", x_scaler=x_scaler, y_scaler=y_scaler, plot=True)
                mse = rmse ** 2

                # Store the errors
                if hours_to_forecast not in mse_dir:
                    mse_dir[hours_to_forecast] = []
                    mae_dir[hours_to_forecast] = []

                mse_dir[hours_to_forecast].append(mse)
                mae_dir[hours_to_forecast].append(mae)

    # Calculate average errors for each forecast horizon
    mse_list = []
    mae_list = []

    for hours_to_forecast in [5, 7, 12, 24, 48, 168, 336]:
        mse_list.append(np.mean(mse_dir[hours_to_forecast]))
        mae_list.append(np.mean(mae_dir[hours_to_forecast]))

    print("\nAverage MSE for different forecast horizons:")
    for hours, mse in zip([5, 7, 12, 24, 48, 168, 336], mse_list):
        print(f"{hours} hours: MSE = {mse}")

    print("\nAverage MAE for different forecast horizons:")
    for hours, mae in zip([5, 7, 12, 24, 48, 168, 336], mae_list):
        print(f"{hours} hours: MAE = {mae}")


def execute_train_transformer_model(i, df_train, target_station):
    model, x_scaler, y_scaler = train_transformer_model(df_train, target_station, previous_time_steps=5)
    return model, x_scaler, y_scaler, i


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run gap-filling technique comparisons.")
    parser.add_argument("--model", choices=["amber", "sarima", "RF", "LSTM", "MLP", "TCN", "transformer"], default="sarima",
                        help="Specify which model to use.")
    parser.add_argument("--input", type=str, default="data/Turku/Turku_1H_LI.csv", help="Path to the input CSV file.")
    args = parser.parse_args()

    if args.model == "amber":
        test_different_gf_techniques_amber(input_file=args.input)
    elif args.model == "RF":
        test_RF_approach(input_file=args.input)
    elif args.model == "LSTM":
        test_LSTM_approach(input_file=args.input)
    elif args.model == "MLP":
        test_MLP_approach(input_file=args.input)
    elif args.model == "TCN":
        test_TCN_approach(input_file=args.input)
    elif args.model == "transformer":
        test_transformer_approach(input_file=args.input)
    else:
        test_SARIMA_approach(input_file=args.input)
