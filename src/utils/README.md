# src/utils

Helpers shared by the experiments.

| File | What it does |
|---|---|
| `generate_forecast_start.py` | Draws a reproducible random outage onset for a station, leaving enough history before it for training and enough observations after it for the longest outage. |
| `load_station_coordinates.py` | Latitude and longitude of every station of a dataset, used by IDW, the station weights and the maps. RMI coordinates come from `data/Full_AWS/metadata.csv`, TURCLIM from `data/Turku/Turku_coordinates.csv` and the synthetic network from `data/Synthetic/extracted_points_lat_lon.csv` with `stations_mapping.csv`. |
| `select_LASSO_stations.py` | LASSO pre-selection of the network stations for one target, on the training window only. |
