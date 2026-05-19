# Real-time Correction of Temperature Measurements in Urban Sensor Networks.

This repository contains the implementation for the master thesis “Real-Time Correction of Temperature Measurements in Urban Sensor Networks”.

## Project Setup
### Requirements
Make sure you have **Python 3.11** or higher and **Poetry** installed. You can install Poetry via:
```bash
pip install poetry
```

### Virtual Environment
Create a virtual environment and install the requirements:
```bash
poetry install
```

Activate the environment with:
```bash
poetry shell
```

### Running the Project
You can run the various scripts using for example:
```bash
python -m src.models.arima.main --mode forecast --model arima --weeks 6 --resample 1h --hours_to_forecast 48
```

## Directory Structure
```
├── data/                  # Raw datasets (not in GitHub, too large)
│
├── src/
│   ├── data/              # Data loading and preprocessing modules
│   ├── evaluation/        # Evaluation metrics and functions
│   ├── models/            # Model implementations (ARIMA, LSTM, etc.)
|   ├── plotting/          # Plotting utilities for visualisation
│   ├── scripts/           # Some standalone scripts for experiments (model comparison, retraining strategies evaluation, etc.)
│   └── utils/             # Utility functions
│
├── pyproject.toml         # Poetry configuration (dependencies, Python version)
├── poetry.lock            # Exact version lock of dependencies
├── .gitignore             # Git ignore file
└── README.md              # This manual
```
