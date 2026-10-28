"""
Module: experiment_utils.py

Helpers shared by the per-station experiment engines (aging_experiment.py, retraining_experiment.py):
running tasks for one station in a worker pool, and writing results to a CSV file.
"""
import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

# Set by _init_worker to the station's df_complete, once per worker process rather than once per
# task, since it is identical for every task submitted for a station
_worker_df_complete = None


def _init_worker(df_complete):
    """ProcessPoolExecutor initializer: stash the station's df_complete once per worker process."""
    global _worker_df_complete
    _worker_df_complete = df_complete


def get_worker_df_complete():
    """Return the current worker process's df_complete, set by _init_worker."""
    return _worker_df_complete


def run_tasks_for_station(df_complete, tasks, run_fn, run_args, max_workers=None):
    """
    Run run_fn(task, *run_args) for every task in a worker pool, sending df_complete to each worker
    once via the pool initializer instead of once per task. Returns every result list, concatenated.
    """
    results = []

    with ProcessPoolExecutor(
        max_workers=max_workers, initializer=_init_worker, initargs=(df_complete,)
    ) as executor:
        futures = [executor.submit(run_fn, t, *run_args) for t in tasks]

        for f in as_completed(futures):
            results.extend(f.result())

    return results


def write_results_csv(output_path, header, rows):
    """Write header and rows to output_path as UTF-8 CSV, creating the output directory if needed."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
