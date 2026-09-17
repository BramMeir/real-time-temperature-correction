#!/bin/bash
# Submit one exclusive-node job per model in the comparison.
# Run from the repo root: bash hpc/submit_all.sh

set -euo pipefail

mkdir -p hpc/logs

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
