#!/bin/bash
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
#SBATCH --gpus=1
#SBATCH --time=24:00:00

if [ -z "$MODEL" ]; then
    echo "MODEL is not set - submit with --export=MODEL=<name>" >&2
    exit 1
fi

# NOTE: cluster selection (module swap cluster/joltik) must happen in the shell that
# calls sbatch, not here - by the time this script runs, the job is already scheduled
# on whichever cluster was active at submission time. See submit_all.sh.

module purge
module load Python/3.11.5-GCCcore-13.2.0
module load poetry/1.6.1-GCCcore-13.2.0

# These 16 cores are reserved for this job only (SLURM enforces this via cgroups,
# even though other jobs may run elsewhere on the same node), so nested BLAS/OpenMP
# threading inside numpy/xgboost/tensorflow doesn't oversubscribe them.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "$SLURM_SUBMIT_DIR"
poetry install --no-interaction --no-root
poetry run python -m src.scripts.models_comparison_experiment --model "$MODEL"
