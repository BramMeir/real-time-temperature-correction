#!/bin/bash
#
# Reliability indicators of the two-stage model for every synthetic station, one after another
# (src/scripts/evaluate_confidence_per_station_RegressionSARIMAErrors.py), keeping every episode as
# well as the station averages. Only the indicators and the realised errors are used, not the run
# times, so a CPU cluster will do.
#
# Usage, from the repo root:
#   module swap cluster/doduo
#   sbatch hpc/submit_confidence_RegressionSARIMAErrors.slurm

#SBATCH --job-name=confidence_RegressionSARIMAErrors
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --error=hpc/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=16gb
#SBATCH --time=01:00:00

set -euo pipefail

module purge
module load poetry/1.8.3-GCCcore-13.3.0

# One core per worker, the script runs 10 repeats in parallel
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "$SLURM_SUBMIT_DIR"

# Job-unique venv and no keyring, for the reasons given in submit_model.slurm
export POETRY_VIRTUALENVS_IN_PROJECT=false
export POETRY_VIRTUALENVS_PATH="${TMPDIR:-/tmp}/poetry-venv-$SLURM_JOB_ID"
export POETRY_KEYRING_ENABLED=false

poetry install --no-interaction --no-root

INPUT_FILE=data/Synthetic/temperature_data.csv
N_STATIONS=$(poetry run python -c "import pandas as pd; print(pd.read_csv('$INPUT_FILE', usecols=['station_name'])['station_name'].nunique())")

for ((i = 0; i < N_STATIONS; i++)); do
    poetry run python -m src.scripts.evaluate_confidence_per_station_RegressionSARIMAErrors \
        --station_index "$i" --input_file "$INPUT_FILE"
done
