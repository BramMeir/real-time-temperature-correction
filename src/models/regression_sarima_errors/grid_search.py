"""
Module for performing a parallel grid search to find the best SARIMA parameters for the residuals
of the stage-one regression on the neighbouring stations.

Functions:
- build_candidate_grid: Build the list of (order, seasonal_order) candidates to evaluate.
- fit_residual_sarima: Fit one candidate on a residual series and return its information criteria.
- residual_sarima_grid_search: Perform a parallel grid search over the candidates for one residual series.
"""
import itertools
import warnings
import pandas as pd
import statsmodels.api as sm
from concurrent.futures import ProcessPoolExecutor, as_completed


def build_candidate_grid(p_values=range(0, 4), d_values=(0,), q_values=range(0, 3),
                         P_values=range(0, 2), D_values=(0,), Q_values=range(0, 2), S=24):
    """
    Build the list of (order, seasonal_order) candidates to evaluate.

    The differencing orders default to zero, as the residuals are already stationary and mean zero.

    Input
    -----
    p_values: Range of p values to try (default is range(0, 4))
    d_values: Range of d values to try (default is (0,))
    q_values: Range of q values to try (default is range(0, 3))
    P_values: Range of P values to try (default is range(0, 2))
    D_values: Range of D values to try (default is (0,))
    Q_values: Range of Q values to try (default is range(0, 2))
    S: Seasonal period (default is 24)

    Output
    ------
    Returns a list of (order, seasonal_order) tuples.
    """
    orders = itertools.product(p_values, d_values, q_values)
    seasonal_orders = list(itertools.product(P_values, D_values, Q_values))

    return [(order, seasonal + (S,)) for order, seasonal in itertools.product(orders, seasonal_orders)]


def fit_residual_sarima(residuals, order, seasonal_order):
    """
    Fit a SARIMA model on a residual series and return its information criteria.

    No trend term is included, as the residuals have a mean of exactly zero.

    Input
    -----
    residuals: Pandas Series with the stage-one regression residuals
    order: Tuple specifying the (p, d, q) parameters
    seasonal_order: Tuple specifying the (P, D, Q, s) parameters

    Output
    ------
    Returns a dictionary with the order, seasonal order, number of parameters, number of
    observations, log likelihood, AIC, BIC, convergence flag and fit duration in seconds. Failed
    fits return infinite information criteria so they never win the search.
    """
    start_time = pd.Timestamp.now()

    try:
        results = sm.tsa.statespace.SARIMAX(
            endog=residuals,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(maxiter=1000, disp=False)

        return {
            "order": order,
            "seasonal_order": seasonal_order,
            "n_params": len(results.params),
            "nobs": int(results.nobs),
            "loglikelihood": float(results.llf),
            "aic": float(results.aic),
            "bic": float(results.bic),
            "converged": bool(results.mle_retvals.get("converged", False)),
            "fit_duration_seconds": (pd.Timestamp.now() - start_time).total_seconds()
        }
    except Exception:
        # Invalid combinations return infinite information criteria instead of raising, so one
        # unfittable candidate does not abort the search
        return {
            "order": order,
            "seasonal_order": seasonal_order,
            "n_params": 0,
            "nobs": 0,
            "loglikelihood": float("-inf"),
            "aic": float("inf"),
            "bic": float("inf"),
            "converged": False,
            "fit_duration_seconds": (pd.Timestamp.now() - start_time).total_seconds()
        }


def residual_sarima_grid_search(residuals, candidates=None, criterion="aic", max_workers=None, verbose=True):
    """
    Perform a parallel grid search over the candidate orders for one residual series.

    Input
    -----
    residuals: Pandas Series with the stage-one regression residuals
    candidates: List of (order, seasonal_order) tuples to evaluate (default is build_candidate_grid())
    criterion: Information criterion to minimise, either "aic" or "bic" (default is "aic")
    max_workers: Maximum number of parallel workers (default is None, which uses all available cores)
    verbose: Whether to print the result of every candidate (default is True)

    Output
    ------
    best_order: The (p, d, q) tuple with the lowest value of the criterion
    best_seasonal_order: The (P, D, Q, s) tuple with the lowest value of the criterion
    rows: List of result dictionaries, one per candidate, as returned by fit_residual_sarima
    """
    if criterion not in ("aic", "bic"):
        raise ValueError(f"Unknown criterion {criterion!r}, expected 'aic' or 'bic'")

    if candidates is None:
        candidates = build_candidate_grid()

    # Suppress warnings from statsmodels
    warnings.filterwarnings("ignore")

    if verbose:
        print(f"Starting parallel grid search with {len(candidates)} combinations...")

    rows = []

    # Use all CPU cores by default (or specify with max_workers)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(fit_residual_sarima, residuals, order, seasonal_order): (order, seasonal_order)
            for order, seasonal_order in candidates
        }

        for i, future in enumerate(as_completed(futures), 1):
            candidate = futures[future]
            try:
                row = future.result()
                rows.append(row)
                if verbose:
                    print(f"[{i}/{len(candidates)}] SARIMA{row['order']}x{row['seasonal_order']} - "
                          f"AIC: {row['aic']:.2f} - BIC: {row['bic']:.2f}")
            except Exception as e:
                print(f"Failed for SARIMA{candidate[0]}x{candidate[1]}: {e}")

    best = min(rows, key=lambda row: row[criterion])

    if verbose:
        print(f"\nBest SARIMA{best['order']}x{best['seasonal_order']} - "
              f"{criterion.upper()}: {best[criterion]:.2f}")

    return best["order"], best["seasonal_order"], rows
