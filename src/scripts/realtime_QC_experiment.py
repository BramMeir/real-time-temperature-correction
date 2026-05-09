"""
Script: realtime_QC_experiment.py
Description: Simulates real-time quality control using a rolling buffer approach.
             New observations are appended to a short buffer, QC is run on the tail,
             and consecutive outliers are used to detect sensor malfunctions.

Usage:
    python -m src.scripts.realtime_QC_experiment --dataset data/Full_AWS/preprocessed_10m_2022_2025.csv
"""
import os
import argparse
import pandas as pd
import metobs_toolkit

# How many hours to keep in the rolling buffer for QC
BUFFER_HOURS = 48

# How many consecutive outlier steps trigger a malfunction alert
MALFUNCTION_THRESHOLD = 2

# Step size for the simulated real-time loop (hours between new batches)
STEP_HOURS = 1


def build_dataset_from_buffer(
    buffer: pd.DataFrame,
    template_file: str,
    metadata_file: str,
) -> metobs_toolkit.Dataset:
    """
    Recreate a metobs_toolkit Dataset from a pandas DataFrame buffer.
    The buffer must have columns: datetime, station_name, temp_dry_avg_2m

    Input:
    ------
    buffer: DataFrame with the recent observations to be QC'd
    template_file: Path to the metobs_toolkit template JSON file
    metadata_file: Path to the metadata CSV file describing the stations

    Output:
    -------
    A metobs_toolkit Dataset object ready for QC checks.
    """
    # Write buffer to a temp CSV so metobs_toolkit can import it
    tmp_path = "/tmp/qc_buffer.csv"
    buffer.to_csv(tmp_path, index=False)

    dataset = metobs_toolkit.Dataset()
    dataset.import_data_from_file(
        input_data_file=tmp_path,
        input_metadata_file=metadata_file,
        template_file=template_file,
    )

    dataset.resample(target_freq="1h")
    return dataset


def run_qc_checks(dataset: metobs_toolkit.Dataset) -> pd.DataFrame:
    """
    Run all QC checks on a dataset and return the outlier DataFrame.
    Returns an empty DataFrame if no outliers were detected.

    Input:
    ------
    dataset: A metobs_toolkit Dataset object with the observations to be checked

    Output:
    -------
    A DataFrame containing the outliers detected by the QC checks.
    """
    # Checks contain some placeholder parameters — these can be tuned based on the specific use case and data characteristics
    dataset.gross_value_check(
        obstype="temp",
        lower_threshold=-20.0,
        upper_threshold=45.0
    )
    dataset.persistence_check(
        obstype="temp",
        timewindow="6h",
        min_records_per_window=3
    )
    dataset.repetitions_check(
        obstype="temp",
        max_N_repetitions=2,
    )
    dataset.step_check(
        obstype="temp",
        max_increase_per_second=4.0 / 3600,
        max_decrease_per_second=-4.0 / 3600
    )
    dataset.buddy_check(
        obstype="temp",
        spatial_buddy_radius=50000.0,
        spatial_z_threshold=2.0,
        min_sample_size=2,
        N_iter=3,
        instantaneous_tolerance="4min",
        min_std=0.1,
    )

    return dataset.outliersdf if dataset.outliersdf is not None else pd.DataFrame()


def count_recent_outliers(
    outlier_history: dict,
    station: str,
    window: int = MALFUNCTION_THRESHOLD
) -> int:
    """
    Count how many of the last `window` steps had at least one outlier
    for the given station.

    Input:
    ------
    outlier_history: Dictionary mapping station names to lists of 0/1 flags indicating outlier presence in recent steps
    station: Name of the station to check
    window: Number of recent steps to consider (default is MALFUNCTION_THRESHOLD)

    Output:
    -------
    The count of how many of the last `window` steps had at least one outlier for the station.
    """
    history = outlier_history.get(station, [])
    recent = history[-window:]
    return sum(recent)


def update_outlier_history(
    outlier_history: dict,
    stations: list,
    outliers_df: pd.DataFrame,
) -> dict:
    """
    For each station, record whether the current step had any outlier.
    Keeps only the last MALFUNCTION_THRESHOLD entries.

    Input:
    ------
    outlier_history: Dictionary mapping station names to lists of 0/1 flags indicating outlier presence in recent steps
    stations: List of station names
    outliers_df: DataFrame containing the outliers detected in the current step

    Output:
    -------
    Updated outlier_history dictionary
    """
    # Determine which stations had at least one outlier in this step
    flagged_stations = set()
    if not outliers_df.empty:
        try:
            flagged_stations = set(outliers_df.index.get_level_values("name").unique())
        except KeyError:
            if "name" in outliers_df.columns:
                flagged_stations = set(outliers_df["name"].unique())

    for station in stations:
        if station not in outlier_history:
            outlier_history[station] = []

        outlier_history[station].append(1 if station in flagged_stations else 0)

        # Keep only what we need
        outlier_history[station] = outlier_history[station][-MALFUNCTION_THRESHOLD:]

    return outlier_history


# Main simulation loop
def run_realtime_qc(
    dataset_file: str,
    template_file: str,
    metadata_file: str,
    sim_start: str = None,
    sim_end: str = None,
):
    """
    Simulate real-time QC over a historical dataset.

    Each step:
      1. Slide the buffer window forward by STEP_HOURS
      2. Rebuild Dataset from the buffer (metobs_toolkit batch approach)
      3. Run all QC checks
      4. Track consecutive outliers per station
      5. Raise a malfunction alert if threshold is exceeded

    Input:
    ------
    dataset_file: Path to the full dataset CSV file (must contain datetime, station_name, temp_dry_avg_2m)
    template_file: Path to the metobs_toolkit template JSON file
    metadata_file: Path to the metadata CSV file describing the stations
    sim_start: Optional start datetime for the simulation (YYYY-MM-DD HH:MM). If None, starts after the initial buffer period.
    sim_end: Optional end datetime for the simulation (YYYY-MM-DD HH:MM). If None, simulates for 1 week after sim_start.
    """
    print("Loading full dataset...")
    df = pd.read_csv(dataset_file, parse_dates=["datetime"])
    df = df.sort_values("datetime")

    stations = df["station_name"].unique().tolist()
    print(f"Stations found: {stations}")

    # Define simulation window
    if sim_start is None:
        # Start after enough data exists for the buffer
        sim_start = df["datetime"].min() + pd.Timedelta(hours=BUFFER_HOURS)
    else:
        sim_start = pd.Timestamp(sim_start)

    if sim_end is None:
        sim_end = sim_start + pd.Timedelta(days=7)   # Simulate 1 week by default
    else:
        sim_end = pd.Timestamp(sim_end)

    current_time = sim_start
    outlier_history: dict = {}
    alert_log = []

    print(f"\nStarting real-time QC simulation: {sim_start} → {sim_end}")
    print(f"Buffer size: {BUFFER_HOURS}h | Malfunction threshold: {MALFUNCTION_THRESHOLD} consecutive flags\n")

    step = 0
    while current_time <= sim_end:
        # 1. Slice buffer window
        buffer_start = current_time - pd.Timedelta(hours=BUFFER_HOURS)
        buffer = df[(df["datetime"] >= buffer_start) & (df["datetime"] <= current_time)].copy()

        if buffer.empty:
            current_time += pd.Timedelta(hours=STEP_HOURS)
            continue

        print(f"[{current_time}] Buffer: {len(buffer)} rows across {buffer['station_name'].nunique()} stations")

        # 2. Build Dataset from buffer
        try:
            dataset = build_dataset_from_buffer(buffer, template_file, metadata_file)
        except Exception as e:
            print(f"  [ERROR] Could not build dataset: {e}")
            current_time += pd.Timedelta(hours=STEP_HOURS)
            continue

        # 3. Run QC checks
        try:
            outliers_df = run_qc_checks(dataset)
            n_outliers = len(outliers_df)
            print(f"  QC complete — {n_outliers} outlier records flagged")
        except Exception as e:
            print(f"  [ERROR] QC checks failed: {e}")
            outliers_df = pd.DataFrame()

        # 4. Update outlier history
        outlier_history = update_outlier_history(
            outlier_history, stations, outliers_df
        )

        # 5. Check for malfunction alerts
        for station in stations:
            consecutive = count_recent_outliers(outlier_history, station)
            if consecutive >= MALFUNCTION_THRESHOLD:
                alert = {
                    "time": current_time,
                    "station": station,
                    "consecutive_flags": consecutive,
                }
                alert_log.append(alert)
                print(f"  *** MALFUNCTION ALERT: {station} flagged {consecutive} consecutive steps! "
                      f"Consider triggering ARIMAX retraining. ***")

        current_time += pd.Timedelta(hours=STEP_HOURS)
        step += 1

    # Summary
    print(f"\nSimulation complete. {step} steps processed.")
    print(f"Total malfunction alerts: {len(alert_log)}")
    if alert_log:
        alert_df = pd.DataFrame(alert_log)
        print(alert_df.to_string(index=False))
        os.makedirs("output", exist_ok=True)
        alert_df.to_csv("output/qc_alerts.csv", index=False)
        print("\nAlerts saved to output/qc_alerts.csv")

    return alert_log


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time QC simulation using metobs_toolkit")
    parser.add_argument("--dataset", default="data/Full_AWS/preprocessed_10m_2022_2025.csv")
    parser.add_argument("--template", default="data/Full_AWS/template.json")
    parser.add_argument("--metadata", default="data/Full_AWS/metadata.csv")
    parser.add_argument("--start", default=None, help="Simulation start (YYYY-MM-DD HH:MM)")
    parser.add_argument("--end", default=None, help="Simulation end   (YYYY-MM-DD HH:MM)")
    args = parser.parse_args()

    # Check if template already exists, otherwise build it (this is needed for the Dataset import to work)
    if not os.path.exists(args.template):
        print("Template file not found — building prompt...")
        metobs_toolkit.build_template_prompt()

    run_realtime_qc(
        dataset_file=args.dataset,
        template_file=args.template,
        metadata_file=args.metadata,
        sim_start=args.start,
        sim_end=args.end,
    )
