# src/data

Preparation of the three datasets, and the conversion of a time series into the inputs of the machine-learning models.

## Preparing the datasets

The data is not part of the repository. The experiments read one CSV file per dataset from a fixed path under
`data/`, with one row per station and time step and the columns `datetime`, `station_name` and `temp_dry_avg_2m`.
`preprocessing.py` converts the raw data of each dataset to that file, run from the repository root:

```bash
python -m src.data.preprocessing --dataset <rmi|turku|synthetic> --input_file <raw data> [--stations_file <station list>] --output_file <path below>
```

| `--dataset` | `--input_file` | `--stations_file` | `--output_file` |
|---|---|---|---|
| `rmi` | RMI export (CSV), one row per station code and 10-minute time step, with the columns `code`, `datetime` (`YYYY-MM-DD HH:MM:SS`) and `temp_dry_avg_2m` | RMI station list (CSV) with the columns `code` and `name` | `data/RMI/observations.csv` |
| `turku` | TURCLIM observations (CSV), the date and time in the first column (`DateTime`) and one column per station | – | `data/Turku/observations.csv` |
| `synthetic` | Synthetic network (NetCDF), the variable `tas` in kelvin over the dimensions `time` and `point_index` | Point mapping (CSV) with the columns `point_index` and `station_name` | `data/Synthetic/observations.csv` |

For RMI, the stations with more than 10 % missing temperatures are dropped. The data stays at 10 minutes, the
experiments themselves keep the observation at every full hour.

Besides these files, some experiments read a few more files directly:

| File | Content | Read by |
|---|---|---|
| `data/RMI/stations.csv` | RMI stations, with the columns `station_name`, `latitude` and `longitude` | station coordinates |
| `data/Turku/stations.csv` | TURCLIM stations, with the columns `Site_name`, `Latitude` and `Longitude` | station coordinates |
| `data/Synthetic/stations.csv` | Synthetic points, with the columns `point_index`, `lat` and `lon` | station coordinates |
| `data/Synthetic/point_names.csv` | The point mapping used above | station coordinates |
| `data/Turku/raw_observations.csv` | The TURCLIM observations, the `--input_file` of `turku` above | comparison with Jacobs et al. (2024) |
| `data/Turku/era5.csv` | ERA5 temperature at the nearest land grid point of every TURCLIM station, in the same layout as the observations | comparison with Jacobs et al. (2024) |

## Files

| File | What it does |
|---|---|
| `preprocessing.py` | Converts the raw data of each dataset, see above. |
| `turku_era5.py` | Joins the TURCLIM observations with ERA5 in the layout of Jacobs et al. (2024), one column per station and one `<station>_ERA5` column per station, for the comparison with their gap-filling. |
| `create_supervised.py` | Input-output pairs of the tabular models (RF, MLP): the target's past 24 hours, the network at the current hour and the hour of day. |
| `create_3d_dataset.py` | Input sequences of the sequence models (LSTM, Transformer, TCN): the preceding hours of all stations, plus the hour of day. |
| `align_concurrent_exog.py` | Shifts the network readings one hour ahead of the target, so the sequence models see the readings of the hour they correct. |
