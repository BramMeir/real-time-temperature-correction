# src/scripts

One script per experiment of the paper. Each one simulates the outages, runs the model(s) and writes its results to
`output/`; the figures and tables are then made from those results by `src/plotting`. Run every script from the
repository root, after preparing the data (`src/data/README.md`). `--help` lists the options of each one.

## Reconstruction accuracy

| Experiment | Command | Results in |
|---|---|---|
| Model comparison | `python -m src.scripts.model_comparison_experiment --model <key>`, once per model key of `src/models/README.md` | `output/models_comparison/` |
| Comparison with Jacobs et al. (2024) | `python -m src.scripts.jacobs_comparison_experiment` | `output/jacobs_comparison/` |
| Heat waves | `python -m src.scripts.heatwave_experiment` | `output/heatwave_experiment/` |

`detect_heatwaves.py` lists the heat waves of every network under three definitions, the 90th and 95th percentile of
the network's summer days and the RMI definition (`output/heatwaves/`), which were compared to choose the one of the
paper. The heat-wave experiment uses the 90th percentile and detects the events itself.

## Drivers of accuracy

| Experiment | Command | Results in |
|---|---|---|
| Station weights | `python -m src.scripts.station_weights_experiment` | `output/stage_one_weights/` |
| Weights and local climate zones | `python -m src.scripts.lcz_weights_test`, after the station weights | `output/urban_context_weights/` |

The LCZ test reads the global LCZ map of Demuzere et al. (2022) over the internet.

## Operational readiness

| Experiment | Command | Results in |
|---|---|---|
| LASSO pre-selection | `python -m src.scripts.lasso_experiment` | `output/LASSO_selection_estimators/` |
| Reliability score | `python -m src.scripts.reliability_experiment` | `output/confidence_analysis_regression_sarima_errors/` |
| Interval coverage | `python -m src.scripts.interval_coverage_experiment` | `output/interval_coverage/` |
| Model ageing | `python -m src.scripts.aging_experiment --dataset <RMI\|TURKU\|SYNTHETIC>` | `output/aging_model_regression_sarima_errors/` |
| Retraining | `python -m src.scripts.retraining_experiment --dataset <RMI\|TURKU\|SYNTHETIC>` | `output/retraining_results_regression_sarima_errors/` |

## Training window

Both comparisons of training windows rerun the model comparison with forecast starts that leave room for the longest
window, so every window reconstructs the same outages.

- All models (appendix): for every model key except `Persistence` and `IDW`, and for `<w>` in 1, 2, 4, 8, 12, 16 and 26:

  ```bash
  python -m src.scripts.model_comparison_experiment --model <key> --training_weeks <w> --onset_history_weeks 26 --datasets TURKU SYNTHETIC --output_dir output/training_weeks_comparison
  ```

- RegSARIMA up to a year: for `<w>` in 1, 2, 4, 8, 12, 26 and 52:

  ```bash
  python -m src.scripts.model_comparison_experiment --model RegressionSARIMAErrors --training_weeks <w> --onset_history_weeks 52 --datasets TURKU SYNTHETIC --output_dir output/training_weeks_regression_sarima_errors_joltik
  ```

## Residual model of RegSARIMA

| Script | What it does | Results in |
|---|---|---|
| `residual_sarima_order.py` | Grid search over the orders of the stage-two SARIMA | `output/residual_sarima_order_grid_search/` |
| `residual_stationarity.py` | Augmented Dickey-Fuller test on the stage-one residuals | `output/residual_stationarity/` |
| `residual_autocorrelation.py` | Autocorrelation of the residuals before and after the SARIMA stage | `output/residual_autocorrelation/` |
