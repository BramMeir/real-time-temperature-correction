# src/models

One folder per model of the paper. Every folder but `jacobs2024` has the same two entry points, which the experiments
in `src/scripts` call:

- `train.py` fits the model on a training window;
- `execute_forecast.py` reconstructs the target over an outage with a fitted model (`run_single_forecast`), and
  returns the MAE and MSE (`src/evaluation/metrics.py`) together with the hourly reconstruction.

The models that have orders or hyperparameters to choose also have a `tune.py`, which runs that search on the RMI
station of Melle and reads `data/RMI/observations.csv` by default. Its result is fixed in the code of the model and
listed in Table A.1 of the paper.

## Names

The result files and the `--model` option of the experiments use short keys rather than the names of the paper:

| Folder | Model in the paper | Key |
|---|---|---|
| `persistence` | Persistence | `Persistence` |
| `climatology` | Hourly climatology | `Climatology` |
| `sarima` | SARIMA (no network inputs) and SARIMAX (the network as exogenous inputs, one-stage fit) | `ARIMA`, `ARIMAX` |
| `idw` | Inverse distance weighting | `IDW` |
| `neighbour_regression` | Neighbour regression | `LinearRegression` |
| `regsarima` | RegSARIMA, the two-stage corrector | `RegressionSARIMAErrors` |
| `random_forest` | Random forest | `RF` |
| `MLP` | Multilayer perceptron | `MLP` |
| `LSTM` | Long short-term memory network | `LSTM` |
| `transformer` | Transformer | `Transformer` |
| `TCN` | Temporal convolutional network | `TCN` |

## regsarima

The corrector of the paper: a regression of the target on the concurrent readings of the other stations, with a
SARIMA model of its residuals, fitted in two stages.

| File | What it does |
|---|---|
| `train.py` | The two-stage fit: the regression (stage one), its residuals, and the SARIMA on those residuals (stage two). |
| `execute_forecast.py` | The real-time reconstruction: the regression at the current network readings, plus the SARIMA forecast of the residual from the last hour before the outage. |
| `grid_search.py` | Grid search over the orders of the stage-two SARIMA, with fixed differencing (`d`, `D`), run by `src/scripts/residual_sarima_order.py`. |
| `confidence_score.py` | Reliability score and prediction interval: the model is fitted without the last part of the training window, reconstructs that part as if the target had failed, and is refitted on the full window for the actual correction. |
| `repeat_forecast.py` | Runs `confidence_score.py` over random periods of one station, so the score can be compared with the error realised afterwards. |
| `retrospective.py` | The retrospective variant of the comparison with Jacobs et al. (2024): the regression is fitted on a window around the gap, and the residual inside the gap is estimated by the Kalman smoother from both sides of it rather than forecast from before it. |

## sarima

| File | What it does |
|---|---|
| `train.py` | Fits SARIMA, or SARIMAX when the other stations are passed as exogenous inputs, with a constant. |
| `execute_forecast.py` | Reconstructs the target over an outage. |
| `grid_search.py` | Grid search over the orders by AIC, with fixed differencing (`d`, `D`). |
| `plot_diagnostics.py` | ACF and PACF of a series. |
| `stationarity_test.py` | Augmented Dickey-Fuller test of a series for a unit root, which decides the differencing `d`. |
| `tune.py` | Runs the grid search (default), the diagnostics (`--mode diagnostics`) or the stationarity test (`--mode stationarity`) on a random training window of Melle, with `--exog` for SARIMAX: `python -m src.models.sarima.tune --weeks 8 --exog` |

## Machine-learning models

`random_forest`, `MLP`, `LSTM`, `transformer` and `TCN` share one layout:

| File | What it does |
|---|---|
| `train.py` | Fits the model with the hyperparameters of Table A.1. |
| `execute_forecast.py` | Reconstructs the target over an outage, step by step, feeding each estimate back as the target's previous value. |
| `evaluate_forecast.py` | That step-by-step reconstruction. The MLP uses the one of `random_forest`. |
| `bayes_search.py` | Bayesian optimisation of the hyperparameters on one training window. |
| `tune.py` | Runs `bayes_search.py` over random training windows of Melle and reports the most frequent hyperparameters, e.g. `python -m src.models.LSTM.tune`. The Transformer and TCN take `--previous_time_steps` and `--weeks`, the others loop over a range of both. |

Model-specific helpers: `MLP/skoptMLP.py` lets scikit-optimize search over tuples of layer sizes,
`TCN/skoptTCN.py` builds the TCN network (for both the search and the final fit), and `transformer/build_model.py`
builds the Transformer network.

## jacobs2024

The gap-filling methods of Jacobs et al. (2024), the retrospective reference of the paper. This is their code, with
its own layout; see `jacobs2024/README.md`.

## Baselines

`persistence`, `climatology`, `idw` and `neighbour_regression` have no orders or hyperparameters, only `train.py` and
`execute_forecast.py`. IDW needs the station coordinates, read by `src/utils/load_station_coordinates.py`.
