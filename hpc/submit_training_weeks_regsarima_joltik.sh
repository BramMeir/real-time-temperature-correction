#!/bin/bash
# Rerun the two-stage model on training windows of 1 to 52 weeks on joltik, with the allocation of the 8-week model
# comparison (submit_model.slurm: 16 cores, 1 GPU), so its training times are comparable with that benchmark. All
# windows share their forecast starts, which leave room for 52 weeks of history. The RMI record is too short for
# that, so only TURCLIM and the synthetic network are run.
# Run from the repo root: bash hpc/submit_training_weeks_regsarima_joltik.sh

set -euo pipefail

mkdir -p hpc/logs

module swap cluster/joltik

MODEL=RegressionSARIMAErrors
WEEKS_LIST=(1 2 4 8 12 26 52)
ONSET_HISTORY_WEEKS=52
DATASETS="TURKU SYNTHETIC"
OUTPUT_DIR=output/training_weeks_regression_sarima_errors_joltik

# The 26-week run on accelgor took under 4 minutes, so half an hour leaves ample room for 52 weeks
for WEEKS in "${WEEKS_LIST[@]}"; do
    sbatch --job-name="training_weeks_joltik_${MODEL}_${WEEKS}" --time=00:30:00 \
        --export=MODEL="$MODEL",WEEKS="$WEEKS",ONSET_HISTORY_WEEKS="$ONSET_HISTORY_WEEKS",DATASETS="$DATASETS",OUTPUT_DIR="$OUTPUT_DIR" \
        hpc/submit_model.slurm
done
