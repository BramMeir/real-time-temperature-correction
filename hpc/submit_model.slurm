#!/bin/bash
#
# Every model (not just the NN ones) runs on the same GPU cluster/node type,
# exclusively, so Train_duration_seconds / Forecast_duration_seconds are
# measured on identical hardware across models and nothing else on the node
# competes for cores.
#
# Usage: sbatch --export=MODEL=XGBoost hpc/submit_model.slurm
# (submit_all.sh loops this over every model)

#SBATCH --job-name=models_comparison_%x
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --error=hpc/logs/%x_%j.err
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks=1
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

# Single job per node reserved exclusively for this model, so nested BLAS/OpenMP
# threading inside numpy/xgboost/tensorflow doesn't oversubscribe the cores that
# ProcessPoolExecutor is also using.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "$SLURM_SUBMIT_DIR"
poetry install --no-interaction --no-root
poetry run python -m src.scripts.models_comparison_experiment --model "$MODEL"
