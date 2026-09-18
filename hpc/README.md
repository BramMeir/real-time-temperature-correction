# Running the model comparison on UGent HPC (VSC)

Every model runs on the same GPU cluster (`joltik`) with the same fixed
allocation (16 cores + 1 GPU), including the models that don't use the GPU,
so `Train_duration_seconds` / `Forecast_duration_seconds` in the output CSVs
are measured on comparable hardware across models. Jobs don't request a whole
exclusive node — SLURM still reserves those 16 cores for the job alone via
cgroups even if another job runs elsewhere on the same node, and skipping
`--exclusive` avoids a much longer queue wait for an entire idle node.

## Setup (once, on a login node)

The repo is already cloned at `/data/gent/466/vsc46666/real-time-temperature-correction`,
with `data/` already in place. There's no HPC `poetry` module compatible with
`Python/3.11.5-GCCcore-13.2.0` (available `poetry` modules only pair with
newer GCCcore toolchains, and mixing toolchains isn't allowed), so poetry is
installed via `pip install --user` under that same Python module instead —
it's pure Python with no compiled dependencies, so the toolchain mismatch
doesn't matter for it specifically:

```bash
cd /data/gent/466/vsc46666/real-time-temperature-correction
module swap cluster/joltik
module purge
module load Python/3.11.5-GCCcore-13.2.0
python -m pip install --user 'poetry==1.8.2'
~/.local/bin/poetry install --no-interaction --no-root
```

## Submit all models

```bash
bash hpc/submit_all.sh
```

This submits one `sbatch` job per model (`hpc/submit_model.slurm`), each
requesting 16 cores plus one GPU on `joltik`. Logs land in `hpc/logs/`.

## Submitting a single model by hand

`sbatch` talks to whichever cluster is active in your current shell, so
`module swap cluster/joltik` must be run before `sbatch`, not inside the
job script (by the time the script runs, the job is already scheduled):

```bash
module swap cluster/joltik
sbatch --export=MODEL=XGBoost hpc/submit_model.slurm
```

## Notes

- All 13 models target `joltik` and reserve a GPU, even the ones that never
  touch it (RF, XGBoost, ARIMA, ...) — that's deliberate, so every model's
  timing comes from the same node type instead of mixing CPU-only and
  GPU-cluster hardware.
- `OMP_NUM_THREADS=1` etc. are set so nested BLAS/OpenMP threading inside
  numpy/xgboost/tensorflow doesn't oversubscribe the cores that
  `ProcessPoolExecutor` is already parallelizing across.
- Results still land in `output/models_comparison/` as before, on the same
  filesystem the repo is cloned on — no copy-back step needed.
