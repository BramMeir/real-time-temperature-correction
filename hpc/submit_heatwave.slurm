#!/bin/bash
#
# Heat wave experiment of the two-stage model (src/scripts/heatwave_experiment.py), followed by its summary. Only
# statsmodels and scikit-learn fits, so it runs on the CPU cluster doduo without a GPU.
#
# Usage: module swap cluster/doduo
#        sbatch hpc/submit_heatwave.slurm

#SBATCH --job-name=heatwave_experiment
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --error=hpc/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16gb
#SBATCH --time=02:00:00

# After the #SBATCH block, see submit_model.slurm
set -euo pipefail

module purge
module load poetry/1.8.3-GCCcore-13.3.0

# One BLAS thread per worker process, as in submit_model.slurm
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "$SLURM_SUBMIT_DIR"

# A job-unique venv off the shared repo path, see submit_model.slurm
export POETRY_VIRTUALENVS_IN_PROJECT=false
export POETRY_VIRTUALENVS_PATH="${TMPDIR:-/tmp}/poetry-venv-$SLURM_JOB_ID"
export POETRY_KEYRING_ENABLED=false

poetry install --no-interaction --no-root

# The pool would otherwise size itself to every core of the node rather than the cores of this job
poetry run python -m src.scripts.heatwave_experiment --max_workers "$SLURM_CPUS_PER_TASK"
poetry run python -m src.scripts.summarize_heatwave_errors
