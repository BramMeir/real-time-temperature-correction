# src/plotting

The figures and tables of the paper, made from the results the experiments in `src/scripts` write to `output/`.
`plot_*.py` draws a figure, `table_*.py` prints a table. Run every file from the repository root, e.g.
`python -m src.plotting.plot_aging`. The plots that take `--metric` report RMSE by default and MAE with
`--metric MAE`.

## Figures

| Figure | File | Needs | Written to |
|---|---|---|---|
| Station maps | `plot_station_maps.py` | the LCZ map (read over the internet) | `plots/station_maps/` |
| Error against outage duration | `plot_model_comparison.py` | model comparison | `plots/performance_vs_horizon_zoomed_rmse.pdf` |
| Station gap during heat waves | `plot_heatwave_station_gap.py` | heat-wave experiment | `plots/heatwave_experiment/` |
| Station weight maps | `plot_station_weights.py` | station weights | `plots/stage_one_weights/` |
| Error against training time | `plot_training_history.py` | RegSARIMA training-window runs | `plots/training_time_impact_regression_sarima_errors_joltik/` |
| Reliability score and interval width | `plot_reliability.py` | reliability experiment | `plots/confidence_analysis_regression_sarima_errors/` |
| Model ageing | `plot_aging.py` | ageing experiment | `plots/aging_model_regression_sarima_errors/` |
| Retraining strategies | `plot_retraining.py` | retraining experiment | `plots/retraining_results_regression_sarima_errors/` |

## Tables

| Table | File | Needs |
|---|---|---|
| Error per outage duration, overall and per network | `plot_model_comparison.py` (printed as LaTeX) | model comparison |
| Paired comparison, in the results and in full | `table_paired_comparison.py` (printed as LaTeX) | model comparison |
| Comparison with Jacobs et al. (2024) | `table_jacobs_comparison.py` | comparison with Jacobs et al. |
| Heat waves | `table_heatwaves.py` (printed as LaTeX) | heat-wave experiment |
| LASSO pre-selection | `table_lasso.py` | LASSO experiment |
| Interval coverage | `table_interval_coverage.py` | interval-coverage experiment |
| Length of the training window | `table_training_window.py` | training-window runs of all models |

The model comparison tables and figure report the outages from 4 h on; the 1 h outage is only in the paired
comparison. The reliability figure leaves out Casino_Oostende, whose neighbours are too far away for a meaningful
correction.
