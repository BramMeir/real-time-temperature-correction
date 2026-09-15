"""
Main script to determine which part of the two-stage model needs retraining: the stage-one regression,
the residual SARIMA, or both. Uses the shared retraining_experiment engine (also used by
compare_retraining_strategies.py for ARIMAX), reusing the hooks from the aging experiment
(test_aging_model_RegressionSARIMAErrors.py). Every arm retrains from scratch at the same fixed
frequency, differing only in which component(s) get refit ("lr_only", "sarima_only", "both").

Unlike ARIMAX's script, which varies retraining frequency and full-vs-incremental (it has no separable
stages), this fixes frequency and only varies what's refit. Incremental (warm-started) retraining of
the residual SARIMA is a separate follow-up experiment, not covered here.

python -m src.scripts.compare_retraining_strategies_RegressionSARIMAErrors --dataset TURKU
"""
import argparse
from src.models.linear_regression.train import train_neighbour_regression
from src.models.regression_sarima_errors.train import stage_one_residuals, fit_residual_sarima
from src.scripts.test_aging_model_RegressionSARIMAErrors import (
    train_two_stage, age_two_stage, forecast_two_stage, RESIDUAL_ORDER, RESIDUAL_SEASONAL_ORDER
)
from src.evaluation.retraining_experiment import run_retraining_experiment
from src.evaluation.aging_experiment import DATASETS

RETRAIN_GAP_DAYS = 3


def full_retrain_two_stage(model, df_window, station, exog_cols):
    """Full retrain from scratch on the trailing training window; the current model is unused."""
    return train_two_stage(df_window, station, exog_cols)


def retrain_lr_only(model, df_window, station, exog_cols):
    """Refit the regression on the training window; the residual SARIMA is left untouched."""
    _, residual_model = model
    regression = train_neighbour_regression(df_window, station, exog_cols)
    return (regression, residual_model), exog_cols


def retrain_sarima_only(model, df_window, station, exog_cols):
    """Refit the residual SARIMA on the training window, under the frozen, original regression."""
    regression, _ = model

    _, window_residuals = stage_one_residuals(df_window, station, exog_cols, regression=regression)
    residual_model = fit_residual_sarima(window_residuals, RESIDUAL_ORDER, RESIDUAL_SEASONAL_ORDER, max_iter=1000)

    return (regression, residual_model), exog_cols


# (model_type, retrain_gap_days, window_kind, retrain_fn). Every arm retrains from scratch on the same
# trailing training window at the same frequency, differing only in which component(s) get refit
MODEL_CONFIGS = [
    ("no_retrain", None, None, None),
    ("lr_only", RETRAIN_GAP_DAYS, "trailing", retrain_lr_only),
    ("sarima_only", RETRAIN_GAP_DAYS, "trailing", retrain_sarima_only),
    ("both", RETRAIN_GAP_DAYS, "trailing", full_retrain_two_stage),
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Determine which part of the two-stage regression with"
                                                 " SARIMA errors model needs retraining (the regression,"
                                                 " the residual SARIMA, or both) across all stations of"
                                                 " one dataset. The results are saved to a CSV file for"
                                                 " later analysis.")
    parser.add_argument("--dataset", type=str, choices=DATASETS.keys(), required=True,
                        help="The dataset to run the experiment on.")
    parser.add_argument("--max_workers", type=int, default=None,
                        help="Number of worker processes (default: one per core). Lower it if the"
                             " full core count runs into memory pressure")
    args = parser.parse_args()

    run_retraining_experiment(
        dataset_name=args.dataset,
        training_weeks=8,
        train_fn=train_two_stage,
        age_fn=age_two_stage,
        forecast_fn=forecast_two_stage,
        model_configs=MODEL_CONFIGS,
        output_path=f"output/retraining_results_regression_sarima_errors/results_{args.dataset}.csv",
        max_workers=args.max_workers
    )
