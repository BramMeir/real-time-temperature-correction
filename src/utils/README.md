# src/utils

Helpers shared by the experiments.

| File | What it does |
|---|---|
| `datasets.py` | The file and the target stations of each network (`DATASETS`), and `load_station_data`, which builds the hourly data of a target and the rest of its network: the observation at each full hour, with gaps interpolated. |
| `generate_forecast_start.py` | Draws a reproducible random outage onset for a station, leaving enough history before it for training and enough observations after it for the longest outage. |
| `load_station_coordinates.py` | Latitude and longitude of every station of a dataset, used by IDW, the station weights and the maps. RMI coordinates come from `data/RMI/stations.csv`, TURCLIM from `data/Turku/stations.csv` and the synthetic network from `data/Synthetic/stations.csv` with `point_names.csv`. |
| `select_LASSO_stations.py` | LASSO pre-selection of the network stations for one target, on the training window only. |
| `experiment_utils.py` | Runs the tasks of one station in a pool of worker processes, sending the station's data to each worker once rather than with every task, and writes rows to a CSV file. |
