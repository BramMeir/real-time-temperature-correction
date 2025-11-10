"""
Main script for Random Forest model training.

Example usage:
python -m src.models.random_forest.main --input ./data/preprocessed.csv
"""
import argparse
import pandas as pd
from src.data.create_supervised import create_supervised_dataset
from src.models.random_forest.train import train_random_forest


if __name__ == "__main__":
    # Define the arguments for the main script
    parser = argparse.ArgumentParser(description="Random Forest Model Training Script")
    parser.add_argument('--input', type=str, required=True, help='Path to the input CSV file')
    args = parser.parse_args()

    # Read the dataset
    df = pd.read_csv(args.input, index_col='datetime', parse_dates=True)

    # Pivot the DataFrame to have every station as a separate column
    df_pivot = df.pivot_table(index='datetime', columns='station_name', values='temp_dry_avg_2m')

    # Remove the multi-index on columns if present (to simplify column access)
    df_pivot.columns.name = None

    # Sort by datetime index
    df_pivot = df_pivot.sort_index()

    # Define the target station
    target_station = "Melle AWS"

    # Define the other stations as exogenous variables
    exog_cols = [col for col in df_pivot.columns if col != target_station]

    # Select the first month of data for testing
    df_pivot = df_pivot.loc[df_pivot.index < '2022-11-01']

    # Create supervised dataset
    X, y = create_supervised_dataset(df_pivot, target_station=target_station, previous_time_steps=24, exog_cols=exog_cols)

    # Chronologically split the data into training, validation, and test sets (70% train, 15% val, 15% test)
    total_samples = len(X)
    train_end = int(0.7 * total_samples)
    val_end = int(0.85 * total_samples)

    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]

    print(f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")

    # Train the Random Forest model
    model, metrics = train_random_forest(X_train, y_train, X_test, y_test, random_seed=42)
    print(f"Test MAE: {metrics['MAE']:.4f}")
    print(f"Test RMSE: {metrics['RMSE']:.4f}")