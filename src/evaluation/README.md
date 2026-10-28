# src/evaluation

How a reconstruction is scored. Everything here takes numbers and returns numbers; reading data, running models and
writing results is left to the experiments in `src/scripts`.

| File | What it does |
|---|---|
| `metrics.py` | MAE and MSE of a reconstruction over the hours at which the target was observed. Every model's `execute_forecast.py` scores its reconstruction with it; the RMSE in the paper is the square root of this MSE, per window. |
| `paired_comparison.py` | The paired tests of the paired-comparison tables: a percentile bootstrap interval for the mean difference between two models over the same windows, a sign-flip permutation test for it, and the Holm adjustment of the p values over the outage durations. |
| `interval_coverage.py` | Coverage of the prediction intervals: the conformal rank of the held-out errors, interval bounds from that rank or from interpolated error quantiles, and per window the coverage, width and Winkler interval score. |

The two kinds of interval in `interval_coverage.py` are built from the same held-out errors. The interpolated
quantiles are biased narrow: a band between order statistics of n points covers about (hi_rank - lo_rank)/(n + 1),
which is about 0.92 for a nominal 0.95 at n = 72. The conformal rank removes that bias, and is what
`src/models/regsarima/confidence_score.py` uses. It is symmetric about the correction, so it assumes the errors are
roughly unbiased.

`aging_experiment.py` and `retraining_experiment.py` run the ageing and retraining experiments rather than score
anything. They move to `src/scripts` with the cleanup of that directory.
