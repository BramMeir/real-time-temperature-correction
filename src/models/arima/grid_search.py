"""
Module for performing a parallel grid search to find the best ARIMA model parameters based on AIC.

Functions:
- _fit_arima: Helper function to fit ARIMA model for a given parameter tuple.
- arima_grid_search: Perform a parallel grid search to find the best ARIMA model parameters based on AIC.
"""
import itertools
import warnings
import statsmodels.api as sm
from concurrent.futures import ProcessPoolExecutor, as_completed


def _fit_arima(params, series):
    """
    Fit an ARIMA model with given parameters and return the AIC.

    Input
    -----
    params: Tuple specifying the (p, d, q) parameters for the ARIMA model
    series: Pandas Series with the time series data

    Output
    ------
    Returns a tuple of (params, AIC) where AIC is the Akaike Information Criterion of the fitted model.
    """
    try:
        model = sm.tsa.ARIMA(
            series,
            order=params,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        results = model.fit()
        return params, results.aic
    except Exception:
        return params, float("inf")  # invalid combinations return large AIC


def arima_grid_search(series, p_values=range(0, 51, 10), d_values=range(0, 3), q_values=range(0, 3), max_workers=None):
    """
    Perform a parallel grid search to find the best ARIMA model parameters based on AIC.

    Input
    -----
    series: Pandas Series with the time series data
    p_values: Range of p values to try (default is range(0, 51, 10))
    d_values: Range of d values to try (default is range(0, 3))
    q_values: Range of q values to try (default is range(0, 3))
    max_workers: Maximum number of parallel workers (default is None, which uses all available cores)

    Output
    ------
    Returns the best (p, d, q) tuple and its corresponding AIC.
    """
    pdq = list(itertools.product(p_values, d_values, q_values))

    # Suppress warnings from statsmodels
    warnings.filterwarnings("ignore")

    print(f"Starting parallel grid search with {len(pdq)} combinations...")

    best_aic = float("inf")
    best_param = None

    # Use all CPU cores by default (or specify with max_workers)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_fit_arima, param, series): param for param in pdq}

        for i, future in enumerate(as_completed(futures), 1):
            param = futures[future]
            try:
                params, aic = future.result()
                print(f"[{i}/{len(pdq)}] ARIMA{params} - AIC: {aic}")
                if aic < best_aic:
                    best_aic = aic
                    best_param = params
            except Exception as e:
                print(f"Failed for {param}: {e}")

    print(f"\nBest ARIMA{best_param} - AIC: {best_aic:.2f}")
    return best_param
