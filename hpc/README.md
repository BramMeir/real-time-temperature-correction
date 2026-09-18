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
with `data/` already in place. `poetry/1.8.3-GCCcore-13.3.0` pulls in a
matching Python (3.12.3) as a module dependency, so there's no separate
Python module to load:

```bash
cd /data/gent/466/vsc46666/real-time-temperature-correction
module swap cluster/joltik
module purge
module load poetry/1.8.3-GCCcore-13.3.0
poetry install --no-interaction --no-root
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
- Each job `rsync`s the repo (minus `.git`/`output`/`plots`) to node-local
  scratch (`$TMPDIR`) and runs there, copying just the resulting
  `<model>_<weeks>_weeks.csv` back to `output/models_comparison/` on the
  shared filesystem when done — avoids running training/inference against
  the shared filesystem directly.
- `CUDA/12.6.0` + `cuDNN/9.5.0.50-CUDA-12.6.0` are loaded for every model
  (even CPU-only ones) so all jobs share the same loaded module set.
