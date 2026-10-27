# src/data

Preparation of the three datasets, and the conversion of a time series into the inputs of the machine-learning models.

## Preparing the datasets

Every experiment reads one long-format CSV file per dataset, with one row per station and time step and the columns
`datetime`, `station_name` and `temp_dry_avg_2m`. The raw files go in `data/` (not in the repository) and are converted
as follows, run from the repository root.

**RMI.** The raw export has one row per station code and 10-minute time step. First add the station names, then
filter the stations:

```bash
python -m src.data.preprocessing --mode include_metadata --input_file <raw RMI export>.csv --metadata <RMI station list>.csv --output_file <with names>.csv
python -m src.data.preprocessing --mode standard --input_file <with names>.csv --output_file data/Full_AWS/preprocessed_10m_2022_2025.csv
```

The second step drops every station with more than 10 % missing temperatures. The data stays at 10 minutes, the
experiments themselves keep the observation at every full hour.

**TURCLIM.** The source file has one column per station. It is converted to long format with:

```bash
python -m src.data.preprocessing --mode turku --input_file data/Turku/Turku_1H_LI.csv --output_file data/Turku/Turku_preprocessed.csv
```

The comparison with Jacobs et al. (2024) reads `Turku_1H_LI.csv` directly, together with `data/Turku/Turku_ERA5.csv`.

**Synthetic.** The NetCDF file holds the temperature (`tas`, in kelvin) per grid point. It is converted to °C and
long format, with the station name of every point taken from `stations_mapping.csv`:

```bash
python -m src.data.extract_nc_file --input_file data/Synthetic/extracted_points_tas.nc --station_file data/Synthetic/stations_mapping.csv --output data/Synthetic/temperature_data.csv
```

The station coordinates are read separately, by `src/utils/load_station_coordinates.py`.

## Files

| File | What it does |
|---|---|
| `preprocessing.py` | Converts the raw RMI and TURCLIM files, see above. |
| `extract_nc_file.py` | Converts the synthetic NetCDF file, see above. |
| `create_supervised.py` | Input-output pairs of the tabular models (RF, MLP, XGBoost): the target's past 24 hours, the network at the current hour and the hour of day. |
| `create_3d_dataset.py` | Input sequences of the sequence models (LSTM, Transformer, TCN): the preceding hours of all stations, plus the hour of day. |
| `align_concurrent_exog.py` | Shifts the network readings one hour ahead of the target, so the last step of a sequence pairs the target's previous value with the concurrent network readings. |
