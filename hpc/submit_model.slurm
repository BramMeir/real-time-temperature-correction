#!/bin/bash
set -euo pipefail
#
# Every model (not just the NN ones) runs on the same GPU cluster/node type,
# with the same fixed core/GPU allocation, so Train_duration_seconds /
# Forecast_duration_seconds are measured on comparable hardware across models.
# Cores are cgroup-reserved (not shared with other jobs), so this doesn't need
# a whole exclusive node - which would mean a much longer queue wait for little
# extra timing consistency.
#
# Usage: sbatch --export=MODEL=XGBoost hpc/submit_model.slurm
# (submit_all.sh loops this over every model)

#SBATCH --job-name=models_comparison_%x
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --error=hpc/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --gpus=1
#SBATCH --time=24:00:00

if [ -z "$MODEL" ]; then
    echo "MODEL is not set - submit with --export=MODEL=<name>" >&2
    exit 1
fi

WEEKS=8

# NOTE: cluster selection (module swap cluster/joltik) must happen in the shell that
# calls sbatch, not here - by the time this script runs, the job is already scheduled
# on whichever cluster was active at submission time. See submit_all.sh.

module purge
# poetry/1.8.3 pulls in a compatible Python (3.12.3-GCCcore-13.3.0) as a module
# dependency - no separate Python module load needed.
module load poetry/1.8.3-GCCcore-13.3.0
module load CUDA/12.6.0
module load cuDNN/9.5.0.50-CUDA-12.6.0

# These 16 cores are reserved for this job only (SLURM enforces this via cgroups,
# even though other jobs may run elsewhere on the same node), so nested BLAS/OpenMP
# threading inside numpy/xgboost/tensorflow doesn't oversubscribe them.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "$SLURM_SUBMIT_DIR"

# Isolate the venv per job: all 13 models run against the same shared repo path,
# and Poetry defaults to one shared .venv inside it - concurrent jobs installing
# into that same venv at once corrupts it. $TMPDIR here is plain /tmp, not a
# job-unique path, so two of our own jobs landing on the same (non-exclusive)
# node would still collide there - $SLURM_JOB_ID makes it unique regardless.
# Code, data and output CSVs still read/write directly against shared storage.
export POETRY_VIRTUALENVS_IN_PROJECT=false
export POETRY_VIRTUALENVS_PATH="${TMPDIR:-/tmp}/poetry-venv-$SLURM_JOB_ID"

# This is a headless compute node with no DBus session - poetry's keyring
# dependency tries to reach one anyway and intermittently aborts installing
# whatever package it's on when it fails, silently leaving that package
# missing (confirmed on HPC: different package each time, e.g. pandas,
# wcwidth). Disabling the keyring backend avoids it reaching for DBus at all.
export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring

poetry install --no-interaction --no-root
mkdir -p output/models_comparison
poetry run python -m src.scripts.models_comparison_experiment --model "$MODEL" --training_weeks "$WEEKS"
