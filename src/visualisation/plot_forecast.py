"""
Functionality:
- plot_forecast_comparison: Function that plots the true vs predicted values for visual comparison.
"""

import matplotlib.pyplot as plt


def plot_forecast_comparison(timestamps, X, y_true, y_pred, title, horizon):
    """
    Plots the sequence of an input window (X, which is the data used to make the forecast), the
    predicted future values (y_pred) and the true future values (y_true)
    sequentially for visual comparison, using the actual timestamps.

    So we get X1, y1_pred, y1_true, X2, y2_pred, y2_true, ...

    Parameters
    ----------
    timestamps : array-like, shape (series length,)
        Full timestamps corresponding to the original series.
    X : array-like, shape (num_segments, lookback)
        Input windows used for forecasting.
    y_true : array-like, shape (num_segments, horizon)
        True future values for each window.
    y_pred : array-like, shape (num_segments, horizon)
        Predicted values for each window.
    title : str
        Title of the plot.
    horizon : int
        Number of time steps forecasted (in 10-minute intervals).
    """
    plt.figure(figsize=(12, 8))

    for i in range(X.shape[0]):
        # Indices for the current segment in the full timestamp array
        start_idx = i * (X.shape[1] + horizon)
        end_idx_X = start_idx + X.shape[1]
        end_idx_y = end_idx_X + horizon

        # Plot the data that was used as input (data before the forecast horizon)
        plt.plot(timestamps[start_idx:end_idx_X], X[i], color='black', label='Input X' if i == 0 else "")

        # Plot the predicted future values
        plt.plot(timestamps[end_idx_X:end_idx_y], y_pred[i], color='blue', label='Predicted y' if i == 0 else "")

        # Plot the true future values
        plt.plot(timestamps[end_idx_X:end_idx_y], y_true[i], color='green', linestyle='dashed', label='True y' if i == 0 else "")

    plt.title(f"{title} (forecast horizon = {horizon*10} min)")
    plt.xlabel("Datetime")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
