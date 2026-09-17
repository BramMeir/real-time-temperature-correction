# Running the model comparison on UGent HPC (VSC)

Every model runs on the same GPU cluster (`joltik`) on its own exclusive node,
including the models that don't use the GPU, so `Train_duration_seconds` /
`Forecast_duration_seconds` in the output CSVs are measured on identical
hardware across models and aren't affected by other jobs sharing the node.

## Setup (once, on a login node)

The repo is already cloned at `/data/gent/466/vsc46666/real-time-temperature-correction`,
with `data/` already in place. Just install the environment:

```bash
cd /data/gent/466/vsc46666/real-time-temperature-correction
module swap cluster/joltik
module purge
module load Python/3.11.5-GCCcore-13.2.0
pip install --user poetry
```

## Submit all models

```bash
bash hpc/submit_all.sh
```

This submits one `sbatch` job per model (`hpc/submit_model.slurm`), each
requesting `--exclusive` plus one GPU on `joltik`. Logs land in `hpc/logs/`.

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
