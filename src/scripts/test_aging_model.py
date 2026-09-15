"""
Main script to determine the how much the performance of the ARIMAX model degrades with aging. Uses
the shared aging_experiment engine (see src/evaluation/aging_experiment.py), which also runs this
experiment for the two-stage regression with SARIMA errors model
(test_aging_model_RegressionSARIMAErrors.py) so the two models' aging curves are directly comparable.

python -m src.scripts.test_aging_model --dataset SYNTHETIC
"""
import argparse
from src.utils.select_LASSO_stations import select_LASSO_stations
from src.models.arima.train import train_sarima_model
from src.models.arima.repeat_forecast import _run_single_forecast as run_arimax
from src.evaluation.aging_experiment import run_aging_experiment, DATASETS

ARIMA_ORDER, SEASONAL_ORDER = (2, 0, 0), (1, 0, 1, 24)

# Above this many candidate stations, narrow them with LASSO before the joint MLE fit: ARIMAX
# estimates every exogenous coefficient jointly with the SARIMA parameters, so a large network (e.g.
# the Synthetic dataset's 48 stations) makes that optimisation slow and unstable
LASSO_STATION_THRESHOLD = 20


def train_arimax(df_window, station, exog_cols):
    """
    Train the ARIMAX model on the training window, narrowing the candidate stations with LASSO first
    when there are many of them (see LASSO_STATION_THRESHOLD).
    """
    series_window = df_window[station]
    exog_window = df_window[exog_cols]

    if len(exog_cols) > LASSO_STATION_THRESHOLD:
        _, exog_cols, _ = select_LASSO_stations(series_window, exog_window)
        exog_window = df_window[exog_cols]

    model = train_sarima_model(
        series=series_window,
        exog_df=exog_window,
        arima_order=ARIMA_ORDER,
        seasonal_order=SEASONAL_ORDER,
        max_iter=10000
    )
    return model, exog_cols


def age_arimax(model, df_gap, station, exog_cols):
    """Advance the model's state with the gap's true observations, without refitting."""
    return model.append(endog=df_gap[station], exog=df_gap[exog_cols], refit=False)


def forecast_arimax(repeat_id, df_complete, station, model, exog_cols, train_start, test_begin, test_end):
    hours_to_forecast = int((test_end - test_begin).total_seconds() // 3600)

    result = run_arimax(
        repeat_id,
        series=df_complete[station],
        exog_df=df_complete[exog_cols],
        model=model,
        start_date=train_start,
        end_date=test_end,
        hours_to_forecast=hours_to_forecast,
        arima_order=ARIMA_ORDER,
        seasonal_order=SEASONAL_ORDER,
        confidence_score=False,
        use_LASSO_selection=False,  # Already narrowed once in train_arimax; reused for every age gap
        max_iter=1000,
        plot=False
    )
    return result[:2]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full experiment with the given number of training weeks"
                                                 " across all datasets, stations, and age gaps. The results"
                                                 " are saved to a CSV file for later analysis.")
    parser.add_argument("--dataset", type=str, choices=DATASETS.keys(), required=True,
                        help="The dataset to run the experiment on.")
    parser.add_argument("--max_workers", type=int, default=None,
                        help="Number of worker processes (default: one per core). Lower it if the"
                             " full core count runs into memory pressure")
    args = parser.parse_args()

    run_aging_experiment(
        dataset_name=args.dataset,
        training_weeks=8,
        train_fn=train_arimax,
        age_fn=age_arimax,
        forecast_fn=forecast_arimax,
        output_path=f"output/aging_model/results_{args.dataset}_100.csv",
        max_workers=args.max_workers
    )
