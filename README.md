# Real-time correction of temperature measurements in urban sensor networks

Code of the paper *Real-time correction of temperature measurements in urban sensor networks using regression with
seasonal ARIMA errors*.

When a station of a temperature network fails, its readings are reconstructed in real time from the concurrent
readings of the rest of the network. The corrector of the paper, RegSARIMA, is a regression on the other stations
with a SARIMA model of its residuals, fitted in two stages. This repository contains that corrector, the eleven
other models it is compared with (its one-stage fit SARIMAX and ten reference approaches), and every experiment,
figure and table of the paper.

## Installation

The code needs Python 3.11 or later and [Poetry](https://python-poetry.org/). From the repository root:

```bash
poetry install
```

Run every command below from the repository root inside that environment, e.g.
`poetry run python -m src.scripts.model_comparison_experiment --model RegressionSARIMAErrors`.

## Data

The data is not part of the repository; see the data availability statement of the paper. The experiments use
three networks:

- **RMI**: automatic weather stations of the Royal Meteorological Institute of Belgium, 2022-2023;
- **TURCLIM**: the urban network of Turku, Finland, 2012-2021, with ERA5 at its stations for the comparison with
  Jacobs et al. (2024);
- **Synthetic**: simulated hourly temperatures at 49 locations across Belgium, 2018-2022, from dense urban centres
  such as Antwerp, Brussels and Ghent to rural sites. Since the simulation sets where the stations are and how
  densely they are spaced, this network is the one used for the experiments on large networks (station weights,
  LASSO pre-selection, reliability score).

`src/data/README.md` describes the files each network needs under `data/` and how to prepare them.

## Reproducing the paper

1. Prepare the data (`src/data/README.md`).
2. Run the experiments, which write their results to `output/` (`src/scripts/README.md`).
3. Make the figures and tables from those results, which are written to `plots/` or printed
   (`src/plotting/README.md`).

The order and hyperparameter searches behind the fixed configuration of every model (Table A.1) are in the
`tune.py` of each model folder (`src/models/README.md`). The experiments were run on the UGent HPC; the neural
networks of the model comparison are run one at a time and benefit from a GPU.

## Repository structure

```
src/
├── data/         Data preparation and the inputs of the machine-learning models
├── models/       One folder per model: training, reconstruction and, where needed, tuning
├── evaluation/   Error metrics, paired tests and prediction-interval coverage
├── scripts/      One script per experiment of the paper
├── plotting/     The figures and tables of the paper
└── utils/        Shared helpers: datasets, outage onsets, station coordinates, LASSO selection
```

Every directory has a README that describes its files. `src/models/README.md` also maps the keys used in the code
and the result files (e.g. `RegressionSARIMAErrors`) to the names of the paper (e.g. RegSARIMA).

## License

MIT, see `LICENSE`. The gap-filling code of Jacobs et al. (2024) in `src/models/jacobs2024/` is theirs, under its
own MIT licence.
