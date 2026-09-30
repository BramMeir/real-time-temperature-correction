#!/bin/bash
# Rerun every model that learns from its training window on windows of 1 to 26 weeks, one job per model and
# window. Persistence and IDW are left out, since nothing they do depends on the window.
#
# All windows share their forecast starts: the starts leave room for the longest window (ONSET_HISTORY_WEEKS),
# so a difference between two windows is never a difference between the periods they reconstruct. The RMI record
# (Nov 2022 - Jul 2023) leaves too little room for 26 weeks of history plus a 30-day outage, so only TURCLIM and
# the synthetic network are run.
#
# Only the accuracy is used, not the training time, so the cluster does not have to match the benchmark's.
# accelgor only accepts GPU jobs, so every job takes the GPU share of one node (12 cores). The time budgets are
# rough estimates scaled from the 8-week benchmark, adjust if a job times out.
# Run from the repo root: bash hpc/submit_training_weeks.sh

set -euo pipefail

mkdir -p hpc/logs

module swap cluster/accelgor

WEEKS_LIST=(1 2 4 8 12 16 26)
ONSET_HISTORY_WEEKS=26
DATASETS="TURKU SYNTHETIC"
OUTPUT_DIR=output/training_weeks_comparison

# Time budget per model for the longest window
declare -A TIME=(
    [Climatology]=01:00:00
    [LinearRegression]=01:00:00
    [RegressionSARIMAErrors]=01:00:00
    [ARIMA]=01:00:00
    [ARIMAX]=04:00:00
    [RF]=03:00:00
    [MLP]=03:00:00
    [LSTM]=12:00:00
    [TCN]=12:00:00
    [Transformer]=24:00:00
)

for MODEL in Climatology LinearRegression RegressionSARIMAErrors ARIMA ARIMAX RF MLP LSTM TCN Transformer; do
    for WEEKS in "${WEEKS_LIST[@]}"; do
        sbatch --job-name="training_weeks_${MODEL}_${WEEKS}" --time="${TIME[$MODEL]}" --cpus-per-task=12 \
            --export=MODEL="$MODEL",WEEKS="$WEEKS",ONSET_HISTORY_WEEKS="$ONSET_HISTORY_WEEKS",DATASETS="$DATASETS",OUTPUT_DIR="$OUTPUT_DIR" \
            hpc/submit_model.slurm
    done
done
