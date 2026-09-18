#!/bin/bash
# Submit one job per model in the comparison, each with a time budget sized to
# that model's expected cost - untested estimates, adjust if a model times out
# or a short budget turns out too tight.
# Run from the repo root: bash hpc/submit_all.sh

set -euo pipefail

mkdir -p hpc/logs

# Cluster selection must happen in this shell, before sbatch is called - sbatch itself
# talks to whichever cluster's scheduler is currently active via the module system.
module swap cluster/joltik

# Trivial - no real training (Persistence/Climatology just copy/average the
# last observations, IDW only fits spatial weights).
FAST_MODELS=(Persistence Climatology IDW)
FAST_TIME=01:00:00

# Fits a scikit-learn/xgboost model per (station, repeat) via ProcessPoolExecutor.
MEDIUM_MODELS=(LinearRegression RF XGBoost MLP)
MEDIUM_TIME=04:00:00

# Iterative SARIMA fitting (max_iter=1000), still parallelized across repeats.
SLOW_MODELS=(ARIMA ARIMAX RegressionSARIMAErrors)
SLOW_TIME=08:00:00

# Deep learning models - run sequentially (not parallelized) to avoid GPU OOM,
# so these are the most expensive by far. Keep the script's own 24h default.
NN_MODELS=(LSTM Transformer TCN)

for MODEL in "${FAST_MODELS[@]}"; do
    sbatch --job-name="models_comparison_${MODEL}" --time="$FAST_TIME" --export=MODEL="$MODEL" hpc/submit_model.slurm
done

for MODEL in "${MEDIUM_MODELS[@]}"; do
    sbatch --job-name="models_comparison_${MODEL}" --time="$MEDIUM_TIME" --export=MODEL="$MODEL" hpc/submit_model.slurm
done

for MODEL in "${SLOW_MODELS[@]}"; do
    sbatch --job-name="models_comparison_${MODEL}" --time="$SLOW_TIME" --export=MODEL="$MODEL" hpc/submit_model.slurm
done

for MODEL in "${NN_MODELS[@]}"; do
    sbatch --job-name="models_comparison_${MODEL}" --export=MODEL="$MODEL" hpc/submit_model.slurm
done
