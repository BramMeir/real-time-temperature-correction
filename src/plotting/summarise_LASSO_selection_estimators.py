"""
Summarise what LASSO station selection does to the one-stage SARIMAX fit and to the two-stage regression
with SARIMA errors model, based on the results of compare_LASSO_selection_estimators.py. Prints one table
with a column per estimator: the selected stations, the selection and training times, the net effect on
the training time, and the effect on the RMSE averaged over all horizons, with a paired sign-flip test
over the training periods. The table is also saved as a CSV file in the "plots" directory.

python -m src.plotting.summarise_LASSO_selection_estimators
"""
import os
import numpy as np
import pandas as pd
from src.evaluation.paired_comparison import sign_flip_test, SEED

RESULTS_FILE = "output/LASSO_selection_estimators/results_8_weeks.csv"
OUTPUT_DIR = "plots/LASSO_selection_estimators"

# Columns identifying one fit of one estimator: every fit is repeated over the horizons in the results
FIT = ["dataset", "station", "repeat_id", "estimator", "arm"]


def summarise_estimator(df, rng):
    """
    Summarise the effect of the LASSO selection for one estimator.

    Input
    -----
    df: Results of one estimator, one row per (station, repeat, arm, horizon)
    rng: Random generator for the sign-flip test

    Output
    ------
    Returns a dictionary with one entry per row of the table.
    """
    fits = df.drop_duplicates(subset=FIT)
    all_fits = fits[fits["arm"] == "all"]
    lasso_fits = fits[fits["arm"] == "lasso"]

    selection_time = lasso_fits["lasso_selection_duration"].mean()
    training_time_all = all_fits["training_duration"].mean()
    training_time_lasso = lasso_fits["training_duration"].mean()

    # RMSE of every forecast, averaged over the horizons per training period, as in the model comparison
    per_period = (
        df.assign(rmse=np.sqrt(df["mse"]))
        .groupby(["station", "repeat_id", "arm"])["rmse"].mean()
        .unstack("arm")
    )
    differences = (per_period["lasso"] - per_period["all"]).values

    per_station = per_period.groupby("station").mean()

    return {
        "Candidate stations": int(all_fits["nr_stations"].max()),
        "Selected stations, mean": lasso_fits["nr_stations"].mean(),
        "Selected stations, min": int(lasso_fits["nr_stations"].min()),
        "Selected stations, max": int(lasso_fits["nr_stations"].max()),
        "Selection time (s)": selection_time,
        "Training time, all stations (s)": training_time_all,
        "Training time, selected stations (s)": training_time_lasso,
        "Speed-up incl. selection": training_time_all / (selection_time + training_time_lasso),
        "RMSE, all stations": per_period["all"].mean(),
        "RMSE, selected stations": per_period["lasso"].mean(),
        "RMSE difference (selected - all)": differences.mean(),
        "p value": sign_flip_test(differences, rng),
        "Training periods": len(differences),
        "Stations improved by selection": int((per_station["lasso"] < per_station["all"]).sum()),
        "Target stations": len(per_station),
    }


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(RESULTS_FILE)
    rng = np.random.default_rng(SEED)

    table = pd.DataFrame({
        estimator: summarise_estimator(df[df["estimator"] == estimator], rng)
        for estimator in ["SARIMAX", "RegSARIMA"]
    })

    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 120):
        print(table)

    table.to_csv(os.path.join(OUTPUT_DIR, "LASSO_selection_summary.csv"))
