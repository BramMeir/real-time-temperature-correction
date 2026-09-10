"""
Functionality:
- evaluate_baseline_forecast: Function to compute the evaluation metrics (MAE and RMSE) for the baseline models
  (persistence, hourly climatology, IDW and linear regression of the neighbours). These models predict the whole
  forecast horizon at once, so no recursive multi-step evaluation is needed (as is the case for e.g. the Random
  Forest in src/models/random_forest/evaluate_forecast.py).
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


def evaluate_baseline_forecast(predictions, y_test, y_train=None, model_name="baseline", plot=False):
    """
    Evaluate a baseline forecast by comparing the predictions with the observed values.

    Input
    -----
    predictions: Series with the forecasted values, indexed by datetime
    y_test: Series with the observed values over the forecast horizon, indexed by datetime
    y_train: Optional Series with the observed values over the training period (only used for plotting)
    model_name: Name of the model, used in the plot title and filename (default is "baseline")
    plot: Boolean to indicate whether to plot the results (default is False)

    Output
    ------
    mae: Mean Absolute Error on the test set
    rmse: Root Mean Squared Error on the test set
    """
    # Align the predictions with the observations and drop the timestamps that are missing in either of them
    predictions = pd.Series(predictions, index=y_test.index) if not isinstance(predictions, pd.Series) else predictions
    combined = pd.concat([y_test.rename("observed"), predictions.rename("predicted")], axis=1).dropna()

    if combined.empty:
        raise ValueError(f"No overlapping timestamps between the {model_name} forecast and the observations")

    mae = mean_absolute_error(combined["observed"], combined["predicted"])
    rmse = root_mean_squared_error(combined["observed"], combined["predicted"])

    # Plot results
    if plot:
        plt.figure(figsize=(12, 6))

        if y_train is not None:
            plt.plot(y_train.index, y_train, label='Training data', color='blue', alpha=0.6)

        plt.plot(combined.index, combined["observed"], label='Real future data', color='green')
        plt.plot(combined.index, combined["predicted"], label='Forecast', color='red')
        plt.legend()
        plt.title(f'{model_name} forecast')

        # Generate a filename with a timestamp to avoid overwriting
        filename = f"{model_name}_forecast_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(f'plots/forecasts/{filename}')
        plt.close()

    return mae, rmse
