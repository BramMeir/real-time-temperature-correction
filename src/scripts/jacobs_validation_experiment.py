"""
Script: jacobs_validation_experiment.py

Reproduction of the gap-filling technique evaluation of Jacobs et al. (2024), Fig. 6, as a check
that their code is applied correctly before it serves as the retrospective reference for RegSARIMA.

Everything follows their Execution_Evaluation_GFtechniques.py: the TURCLIM observations joined with
the ERA5 series of the nearest land grid point, the five techniques (LI, original ERA5, and the LR,
MB and WMB debiasing) with a 60-day learning period placed symmetrically around the gap and a time
variation of 1 h, gap lengths of 1 h to 2 weeks, 1000 single gaps per length with a uniformly drawn
start that stays 60 days away from both ends of the record, and the error E, the MSE of each gap
averaged over the gaps. Their draws were unseeded, so only agreement within the confidence
intervals can be expected, not identical values.

Run:
    python -m src.scripts.jacobs_validation_experiment
    python -m src.scripts.jacobs_validation_experiment --stations Betel Kurala Puutori Tuorla Virastotalo Ylijoki
"""
import argparse
import os
import random
import zlib
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from src.evaluation.comparison_amber.evaluation_gf_techniques import Test_techniques_differentgapdates
from src.evaluation.comparison_amber.turku_era5 import load_turku_with_era5

OUTPUT_DIR = "output/jacobs_validation"

SEED = 42

# Selection parameters held constant in their technique comparison (Sect. 2.3.1)
SEASONAL_SPAN = 60
TIME_VARIATION = 1
POSITIONING = "both"

# Techniques with their arguments, as in their dictionary of Execution_Evaluation_GFtechniques.py
TECHNIQUES = {
    "linint": np.nan,
    "fillmodel": np.nan,
    "debmodelReg": [SEASONAL_SPAN, TIME_VARIATION, POSITIONING, 1, False],
    "debmodelMeanbias": [SEASONAL_SPAN, TIME_VARIATION, POSITIONING, False],
    "debmodelTvar": [SEASONAL_SPAN, TIME_VARIATION, POSITIONING],
}

GAP_LENGTHS = [1, 3, 5, 7, 10, 20, 30, 336]
REPETITIONS = 1000


def run_single_length(df, station, gap_length, repetitions):
    """
    Evaluate all techniques on `repetitions` random single gaps of one length in one station, and
    return one row per technique with E, its standard error and the number of gaps.
    """
    # Their gap starts come from the random module, so seed it per station and length
    random.seed(SEED + zlib.crc32(station.encode()) + gap_length)

    df_errors = Test_techniques_differentgapdates(
        df[[station, station + "_ERA5"]],
        name_fulldata=station,
        name_model=station + "_ERA5",
        gap_length=gap_length,
        dictionarytechniques=TECHNIQUES,
        par_slicedates=SEASONAL_SPAN,
        listerrors=["MSE"],
        repetitions=repetitions,
        check=repetitions,
    )[0]

    return [
        [station, gap_length, technique,
         float(df_errors.loc["mean", technique]),
         float(df_errors.loc["sterr", technique]),
         int(df_errors.loc["len", technique])]
        for technique in TECHNIQUES
    ]


def run_all(stations, repetitions, max_workers):
    """Run every station and gap length in a worker pool and save the errors to OUTPUT_DIR."""
    df = load_turku_with_era5()

    rows = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(run_single_length, df, station, gap_length, repetitions)
            for station in stations
            for gap_length in GAP_LENGTHS
        ]
        for f in as_completed(futures):
            rows.extend(f.result())
            print(f"Finished {rows[-1][0]}, {rows[-1][1]} h")

    results = pd.DataFrame(rows, columns=["station", "gap_length", "technique", "E", "sterr", "n"])
    results = results.sort_values(["station", "gap_length", "technique"])

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = f"{OUTPUT_DIR}/jacobs_techniques_rep{repetitions}.csv"
    results.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reproduce the gap-filling technique evaluation of Jacobs et al. (2024), Fig. 6."
    )
    parser.add_argument("--stations", nargs="+", default=["Ylijoki", "Betel"],
                        help="TURCLIM stations to evaluate (default: the two of their Fig. 6)")
    parser.add_argument("--repetitions", type=int, default=REPETITIONS,
                        help="Number of random gaps per gap length")
    parser.add_argument("--max_workers", type=int, default=None,
                        help="Parallel worker processes (default lets the pool decide)")
    args = parser.parse_args()

    run_all(args.stations, args.repetitions, args.max_workers)
