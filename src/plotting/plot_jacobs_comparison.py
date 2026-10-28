import argparse
import numpy as np
import pandas as pd
from src.scripts.jacobs_comparison_experiment import OUTPUT_DIR, REPETITIONS

# Stations pooled per part of the table. The urban stations are those of the UHI evaluation of Jacobs et al., the
# other three lie outside the city centre
PANELS = {"Rural stations": ["Kurala", "Tuorla", "Ylijoki"], "Urban stations": ["Betel", "Puutori", "Virastotalo"]}

# Switches of their algorithm, in hours
LI_UNTIL = 5
SEPARATE_FROM = 15 * 24

# Methods of the table: their algorithm and both RegSARIMA variants
METHODS = ["Jacobs", "RegSARIMA_realtime", "RegSARIMA_retrospective"]


def add_jacobs_algorithm(results):
    """
    Add their algorithm as a method, taking for every gap the technique it selects for the gap length, as in Table 1
    of their paper: linear interpolation below 5 h, mean-bias debiased ERA5 with both halves of the learning period
    weighted together from 5 h, and weighted separately from 15 days.
    """
    technique = np.select(
        [results["gap_length"] < LI_UNTIL, results["gap_length"] < SEPARATE_FROM], ["LI", "MB_both"], "MB_separate"
    )
    algorithm = results[results["method"] == technique].assign(method="Jacobs")
    return pd.concat([results, algorithm])


def rmse_summary(results, panels):
    """Mean of the per-gap RMSE, per panel, method and gap length."""
    panel_of = {station: panel for panel, stations in panels.items() for station in stations}
    results = results[results["station"].isin(panel_of)]
    results = results.assign(rmse=np.sqrt(results["mse"]), panel=results["station"].map(panel_of))
    return results.groupby(["panel", "method", "gap_length"])["rmse"].mean().reset_index()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print the comparison of RegSARIMA with Jacobs et al. (2024).")
    parser.add_argument("--repetitions", type=int, default=REPETITIONS, help="Repetitions of the run to read")
    args = parser.parse_args()

    stations = sorted(station for group in PANELS.values() for station in group)
    results = add_jacobs_algorithm(pd.concat(
        pd.read_csv(f"{OUTPUT_DIR}/jacobs_comparison_{station}_rep{args.repetitions}.csv") for station in stations
    ))

    summary = rmse_summary(results, PANELS)
    pd.set_option("display.width", 200)
    print(summary[summary["method"].isin(METHODS)]
          .pivot_table(index="gap_length", columns=["panel", "method"], values="rmse").round(3).to_string())
