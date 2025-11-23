from skopt import gp_minimize
from skopt.space import Integer
from skopt.utils import use_named_args
import warnings
import statsmodels.api as sm

warnings.filterwarnings("ignore")


def sarima_bayes_search(series, exog_df=None, S=1440,
                        p_range=(0, 10), d_range=(0, 2), q_range=(0, 3),
                        P_range=(0, 3), D_range=(0, 1), Q_range=(0, 3)):

    # Define search space
    search_space = [
        Integer(*p_range, name="p"),
        Integer(*d_range, name="d"),
        Integer(*q_range, name="q"),
        Integer(*P_range, name="P"),
        Integer(*D_range, name="D"),
        Integer(*Q_range, name="Q"),
    ]

    @use_named_args(search_space)
    def objective(p, d, q, P, D, Q):
        try:
            model = sm.tsa.statespace.SARIMAX(
                endog=series,
                exog=exog_df,
                order=(p, d, q),
                seasonal_order=(P, D, Q, S),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            res = model.fit(disp=False, maxiter=1000)
            return res.aic

        except Exception:
            return 1e6  # Return a large AIC value on failure

    print("Starting Bayesian SARIMA optimization...")

    result = gp_minimize(
        objective,
        search_space,
        n_calls=40,           # number of evaluations
        n_initial_points=10,  # random initial points
        random_state=57,
        verbose=True
    )

    best = result.x
    best_aic = result.fun

    best_nonseasonal = tuple(best[:3])
    best_seasonal = tuple(best[3:] + [S])

    print("\nBest model:")
    print(f"SARIMA{best_nonseasonal}x{best_seasonal} - AIC: {best_aic:.2f}")

    return best_nonseasonal, best_seasonal, best_aic
