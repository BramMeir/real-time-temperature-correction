# Masterproef
Real-time correction of temperature measurements in urban sensor networks.

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
│   ├── models/            # Model implementations (ARIMA, LSTM, etc.)
│   ├── evaluation/        # Evaluation metrics and functions
│   ├── scripts/           # Some standalone scripts for experiments (stationarity tests, etc.)
│   └── visualisation/     # Plots and visualisation utilities
│
├── hpc_interactive.sh     # Script to load necessary modules on HPC
├── pyproject.toml         # Poetry configuration (dependencies, Python version)
├── poetry.lock            # Exact version lock of dependencies
└── README.md              # This manual
```