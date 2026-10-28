# src/evaluation

How a reconstruction is scored.

| File | What it does |
|---|---|
| `metrics.py` | MAE and MSE of a reconstruction over the hours at which the target was observed. |
| `paired_comparison.py` | Paired tests between two models on the same windows: bootstrap confidence interval, sign-flip permutation test and Holm adjustment. |
| `interval_coverage.py` | Prediction intervals from held-out errors (conformal rank or interpolated quantiles), and their coverage, width and interval score. |

`aging_experiment.py` and `retraining_experiment.py` move to `src/scripts` with the cleanup of that directory.
