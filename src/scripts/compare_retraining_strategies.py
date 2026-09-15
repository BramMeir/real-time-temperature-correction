"""
Main script to compare retraining strategies for the ARIMAX model: no retraining, full retraining, and
incremental (warm-started) retraining, at various frequencies. Uses the shared retraining_experiment
engine, reusing the hooks from the aging experiment (test_aging_model.py).

Note: ported to the shared engine, which also fixes a per-station seed collision the original script
had, so rerunning this will NOT reproduce the numbers already published for figure 10 — don't rerun it
against output/retraining_results/ without deciding that regenerating that figure is intended.

python -m src.scripts.compare_retraining_strategies --dataset TURKU
"""
import argparse
from src.scripts.test_aging_model import train_arimax, age_arimax, forecast_arimax
from src.evaluation.retraining_experiment import run_retraining_experiment
from src.evaluation.aging_experiment import DATASETS


def full_retrain_arimax(model, df_window, station, exog_cols):
    """Full retrain from scratch on the trailing training window; the current model is unused."""
    return train_arimax(df_window, station, exog_cols)


def incremental_retrain_arimax(model, df_window, station, exog_cols):
    """Warm-started retraining: update the model's parameters from its current fit, not from scratch."""
    return model.append(endog=df_window[station], exog=df_window[exog_cols], refit=True)


# (model_type, retrain_gap_days, window_kind, retrain_fn). "full" retrains from scratch on the trailing
# training window; "inc" warm-starts from the current fit on the window since the last update
MODEL_CONFIGS = [
    ("no_retrain", None, None, None),
    ("full_3d", 3, "trailing", full_retrain_arimax),
    ("full_15d", 15, "trailing", full_retrain_arimax),
    ("inc_3d", 3, "delta", incremental_retrain_arimax),
    ("inc_15d", 15, "delta", incremental_retrain_arimax),
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare retraining strategies for the ARIMAX model"
                                                 " across all stations of one dataset. The results are"
                                                 " saved to a CSV file for later analysis.")
    parser.add_argument("--dataset", type=str, choices=DATASETS.keys(), required=True,
                        help="The dataset to run the experiment on.")
    parser.add_argument("--max_workers", type=int, default=None,
                        help="Number of worker processes (default: one per core). Lower it if the"
                             " full core count runs into memory pressure")
    args = parser.parse_args()

    run_retraining_experiment(
        dataset_name=args.dataset,
        training_weeks=8,
        train_fn=train_arimax,
        age_fn=age_arimax,
        forecast_fn=forecast_arimax,
        model_configs=MODEL_CONFIGS,
        output_path=f"output/retraining_results/results_{args.dataset}.csv",
        max_workers=args.max_workers
    )
