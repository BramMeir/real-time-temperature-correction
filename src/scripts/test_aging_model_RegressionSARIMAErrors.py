"""
Main script to determine how much the performance of the two-stage regression with SARIMA errors
model degrades with aging, i.e. when it keeps forecasting without ever being retrained. Uses the
shared aging_experiment engine (see src/evaluation/aging_experiment.py), which also runs this
experiment for ARIMAX (test_aging_model.py) so the two models' aging curves are directly comparable.

The stage-one regression has no internal state to age (it only ever needs the current neighbour
readings), so aging is only applied to the residual SARIMA: for every "age gap", the true stage-one
residuals over the gap are appended to the residual SARIMA (refit=False, updating its state without
re-estimating its parameters) before forecasting ahead from that aged state. Since nothing is ever
retrained, this still captures any staleness in the stage-one regression weights too, exactly as it
does for ARIMAX's coefficients.

python -m src.scripts.test_aging_model_RegressionSARIMAErrors --dataset SYNTHETIC
"""
import argparse
from src.models.regression_sarima_errors.train import train_regression_sarima_errors, stage_one_residuals
from src.models.regression_sarima_errors.execute_forecast import run_single_forecast as run_regression_sarima_errors
from src.evaluation.aging_experiment import run_aging_experiment, DATASETS

# Orders selected on the stage-one residuals, as used in short_horizon_comparison_experiment.py
RESIDUAL_ORDER, RESIDUAL_SEASONAL_ORDER = (3, 0, 0), (1, 0, 0, 24)


def train_two_stage(df_window, station, exog_cols):
    """
    Train the two-stage model on the training window. LASSO station selection is deliberately not
    applied here: it does not improve this model's accuracy even on large networks (see PR #42), so
    every neighbouring station is always kept.
    """
    model = train_regression_sarima_errors(
        df=df_window,
        target_station=station,
        exog_cols=exog_cols,
        arima_order=RESIDUAL_ORDER,
        seasonal_order=RESIDUAL_SEASONAL_ORDER,
        max_iter=1000
    )
    return model, exog_cols


def age_two_stage(model, df_gap, station, exog_cols):
    """Advance the residual SARIMA's state with the gap's true stage-one residuals, without refitting."""
    regression, residual_model = model

    _, gap_residuals = stage_one_residuals(df_gap, station, exog_cols, regression=regression)

    return regression, residual_model.append(endog=gap_residuals, refit=False)


def forecast_two_stage(repeat_id, df_complete, station, model, exog_cols, train_start, test_begin, test_end):
    result = run_regression_sarima_errors(
        df=df_complete,
        target_station=station,
        model=model,
        exog_cols=exog_cols,
        start=train_start,
        train_end=test_begin,
        test_end=test_end,
        arima_order=RESIDUAL_ORDER,
        seasonal_order=RESIDUAL_SEASONAL_ORDER,
        mode="forecast"
    )
    return result[:2]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the aging experiment for the two-stage regression"
                                                 " with SARIMA errors model, across all stations and age"
                                                 " gaps of one dataset. The results are saved to a CSV"
                                                 " file for later analysis.")
    parser.add_argument("--dataset", type=str, choices=DATASETS.keys(), required=True,
                        help="The dataset to run the experiment on.")
    parser.add_argument("--max_workers", type=int, default=None,
                        help="Number of worker processes (default: one per core). Lower it if the"
                             " full core count runs into memory pressure")
    args = parser.parse_args()

    run_aging_experiment(
        dataset_name=args.dataset,
        training_weeks=8,
        train_fn=train_two_stage,
        age_fn=age_two_stage,
        forecast_fn=forecast_two_stage,
        output_path=f"output/aging_model_regression_sarima_errors/results_{args.dataset}_100.csv",
        max_workers=args.max_workers
    )
