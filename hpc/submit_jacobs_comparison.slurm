#!/bin/bash
#
# Comparison of RegSARIMA with the ERA5 debiasing of Jacobs et al. (2024) (src/scripts/jacobs_comparison_experiment.py),
# one array task per TURCLIM station. Only statsmodels and scikit-learn fits, so it runs on the CPU cluster doduo.
# Each task fills 13 gap lengths x 1000 gaps, about 3.5 CPU-seconds per gap.
#
# Usage: module swap cluster/doduo
#        sbatch hpc/submit_jacobs_comparison.slurm

#SBATCH --job-name=jacobs_comparison
#SBATCH --output=hpc/logs/%x_%A_%a.out
#SBATCH --error=hpc/logs/%x_%A_%a.err
#SBATCH --array=0-5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32gb
#SBATCH --time=03:00:00

# After the #SBATCH block, see submit_model.slurm
set -euo pipefail

STATIONS=(Betel Kurala Puutori Tuorla Virastotalo Ylijoki)
STATION=${STATIONS[$SLURM_ARRAY_TASK_ID]}

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
poetry run python -m src.scripts.jacobs_comparison_experiment --stations "$STATION" --max_workers "$SLURM_CPUS_PER_TASK"
