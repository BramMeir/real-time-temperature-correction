import itertools
import warnings
import statsmodels.api as sm
from concurrent.futures import ProcessPoolExecutor, as_completed


def _fit_arima(series, exog_df, params, seasonal_params=(0, 0, 0, 0)):
    """
    Fit an ARIMA model with given parameters and return the AIC.

    Input
    -----
    series: Pandas Series with the time series data
    exog_df: DataFrame with exogenous variables
    params: Tuple specifying the (p, d, q) parameters for the ARIMA model
    seasonal_params: Tuple specifying the (P, D, Q, S) parameters for the SARIMA model (default is (0, 0, 0, 0))

    Output
    ------
    Returns a tuple of (params, AIC) where AIC is the Akaike Information Criterion of the fitted model.
    """
    try:
        model = sm.tsa.statespace.SARIMAX(
            series,
            exog=exog_df,
            order=params,
            seasonal_order=seasonal_params,
            enforce_stationarity=False,
            enforce_invertibility=False,
            # The constant of train_sarima_model, so the search ranks the model that is actually fitted
            trend='c'
        )
        results = model.fit(maxiter=1000)
        return params, results.aic
    except Exception:
        return params, float("inf")  # invalid combinations return large AIC


def sarima_grid_search(series, exog_df=None, p_values=range(0, 3), q_values=range(0, 3), P_values=range(0, 3),
                       Q_values=range(0, 3), S=24, d=0, D=0, max_workers=None):
    """
    Perform a parallel grid search to find the best SARIMA model parameters based on AIC.

    The differencing orders d and D are fixed for the whole search rather than searched: a model with more
    differencing is fitted to a different series, so its AIC cannot be compared with that of the others.

    Input
    -----
    series: Pandas Series with the time series data
    exog_df: DataFrame with exogenous variables, for SARIMAX (default is None, SARIMA)
    p_values: Range of p values to try (default is range(0, 3))
    q_values: Range of q values to try (default is range(0, 3))
    P_values: Range of P values to try (default is range(0, 3))
    Q_values: Range of Q values to try (default is range(0, 3))
    S: Seasonal period (default is 24)
    d: Order of differencing of every candidate (default is 0)
    D: Order of seasonal differencing of every candidate (default is 0)
    max_workers: Maximum number of parallel workers (default is None, which uses all available cores)

    Output
    ------
    Returns the best (p, d, q) and (P, D, Q, S) tuples and their corresponding AIC.
    """
    pdq = [(p, d, q) for p, q in itertools.product(p_values, q_values)]
    seasonal_pdq = [(P, D, Q) for P, Q in itertools.product(P_values, Q_values)]

    # Suppress warnings from statsmodels
    warnings.filterwarnings("ignore")

    print(f"Starting parallel SARIMA grid search with {len(pdq) * len(seasonal_pdq)} combinations...")

    best_aic = float("inf")
    best_param = None
    best_seasonal_param = None

    # Use all CPU cores by default (or specify with max_workers)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for param in pdq:
            for seasonal_param in seasonal_pdq:
                seasonal_order = seasonal_param + (S,)
                futures[executor.submit(_fit_arima, series, exog_df, param, seasonal_order)] = (param, seasonal_order)

        for i, future in enumerate(as_completed(futures), 1):
            param, seasonal_param = futures[future]
            try:
                params, aic = future.result()
                print(f"[{i}/{len(futures)}] SARIMA{params}x{seasonal_param} - AIC: {aic}")
                if aic < best_aic:
                    best_aic = aic
                    best_param = params
                    best_seasonal_param = seasonal_param
            except Exception as e:
                print(f"Failed for SARIMA{param}x{seasonal_param}: {e}")

    print(f"\nBest SARIMA{best_param}x{best_seasonal_param} - AIC: {best_aic:.2f}")
    return best_param, best_seasonal_param
