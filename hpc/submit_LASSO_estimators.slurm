#!/bin/bash
#
# LASSO selection for both estimators of the regression with SARIMA errors model
# (src/scripts/compare_LASSO_selection_estimators.py). Same node type and core
# allocation as submit_model.slurm, so the training durations are comparable with
# those of the model comparison. The GPU is not used, it is requested only to land
# on the same node type.
#
# Usage, from the repo root:
#   module swap cluster/joltik
#   sbatch hpc/submit_LASSO_estimators.slurm

#SBATCH --job-name=LASSO_estimators
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --error=hpc/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --gpus=1
#SBATCH --time=12:00:00

set -euo pipefail

module purge
module load poetry/1.8.3-GCCcore-13.3.0

# One core per worker, as in submit_model.slurm, so nested BLAS/OpenMP threading
# doesn't oversubscribe the reserved cores
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "$SLURM_SUBMIT_DIR"

# Job-unique venv and no keyring, for the reasons given in submit_model.slurm
export POETRY_VIRTUALENVS_IN_PROJECT=false
export POETRY_VIRTUALENVS_PATH="${TMPDIR:-/tmp}/poetry-venv-$SLURM_JOB_ID"
export POETRY_KEYRING_ENABLED=false

poetry install --no-interaction --no-root
poetry run python -m src.scripts.compare_LASSO_selection_estimators --training_weeks 8 --repeats 15
