#!/bin/bash
# Submit one exclusive-node job per model in the comparison.
# Run from the repo root: bash hpc/submit_all.sh

set -euo pipefail

mkdir -p hpc/logs

# Cluster selection must happen in this shell, before sbatch is called - sbatch itself
# talks to whichever cluster's scheduler is currently active via the module system.
module swap cluster/joltik

MODELS=(
    ARIMA
    ARIMAX
    RF
    XGBoost
    MLP
    Persistence
    Climatology
    IDW
    LinearRegression
    RegressionSARIMAErrors
    LSTM
    Transformer
    TCN
)

for MODEL in "${MODELS[@]}"; do
    sbatch --job-name="models_comparison_${MODEL}" --export=MODEL="$MODEL" hpc/submit_model.slurm
done
