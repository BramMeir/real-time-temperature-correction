import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from src.utils.load_station_coordinates import load_station_coordinates
from src.models.idw.train import _haversine_distance

WEIGHTS_FILE = "output/stage_one_weights/weights.csv"

OUTPUT_DIR = "output/urban_context_weights"

# Global LCZ map, read through HTTP range requests so the raster is never downloaded in full
LCZ_COG_URL = "/vsicurl/https://lcz-generator.rub.de/cogs/lcz_filter_v1_cog.tif"

# Radius around a station within which its LCZ is the majority class
LCZ_RADIUS_M = 500

# Compact midrise, the class of the urban targets in this network
URBAN_CLASS = 2

N_PERMUTATIONS = 10_000
N_BOOTSTRAP = 10_000
SEED = 42


def load_lcz_labels(coords, radius_m=LCZ_RADIUS_M):
    """
    Label every station with the majority LCZ class within radius_m, ignoring pixels without a class.

    Input
    -----
    coords: Dict mapping station_name to a (latitude, longitude) tuple
    radius_m: Radius of the neighbourhood in metres

    Output
    ------
    Returns a DataFrame indexed by station with the majority class in the column lcz.
    """
    rows = []
    with rasterio.open(LCZ_COG_URL) as src:
        for station, (lat, lon) in coords.items():
            row, col = src.index(lon, lat)
            # Enough pixels to cover the radius in both directions, with one to spare
            half_rows = int(np.ceil(radius_m / 111_320 / abs(src.res[1]))) + 1
            half_cols = int(np.ceil(radius_m / (111_320 * np.cos(np.radians(lat))) / abs(src.res[0]))) + 1
            window = Window(col - half_cols, row - half_rows, 2 * half_cols + 1, 2 * half_rows + 1)
            classes = src.read(1, window=window)

            # Pixel centres of the window, to keep the pixels within the radius
            cols, rows_ = np.meshgrid(np.arange(window.col_off, window.col_off + window.width),
                                      np.arange(window.row_off, window.row_off + window.height))
            lons, lats = rasterio.transform.xy(src.transform, rows_.ravel(), cols.ravel())
            distance_m = 1000 * _haversine_distance(lat, lon, np.asarray(lats), np.asarray(lons)).reshape(classes.shape)
            inside = classes[(distance_m <= radius_m) & (classes != src.nodata)]

            values, counts = np.unique(inside, return_counts=True)
            rows.append({"station": station, "lcz": int(values[counts.argmax()])})

    return pd.DataFrame(rows).set_index("station")


def demean_by_target(values, targets):
    """Subtract the mean per target, which absorbs the target fixed effect."""
    codes, uniques = pd.factorize(targets)
    means = np.bincount(codes, weights=values, minlength=len(uniques)) / np.bincount(codes, minlength=len(uniques))
    return values - means[codes]


def context_effect(y, log_distance, same, targets):
    """
    Estimate the coefficient of the same-context indicator in log(weight) ~ log(distance) + same + target effect.

    Input
    -----
    y, log_distance, same: Arrays with one value per pair, y and log_distance already demeaned by target
    targets: Array with the target of every pair

    Output
    ------
    Returns the coefficient of the same-context indicator.
    """
    X = np.column_stack([log_distance, demean_by_target(same.astype(float), targets)])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef[1]


def test_context(pairs, labels, same_context, permute_targets=True, rng=None):
    """
    Estimate how much more weight a same-context neighbour gets at the same distance, with a permutation p-value and
    a cluster bootstrap confidence interval.

    Input
    -----
    pairs: DataFrame with target, neighbour, weight and distance_km per pair
    labels: Series with the LCZ class per station
    same_context: Function (target class, neighbour class) -> bool arrays, which defines a match
    permute_targets: Whether the permutation also relabels the targets (False keeps the target classes as observed,
      for a test restricted to targets of one class)
    rng: NumPy random generator

    Output
    ------
    Returns a dict with the ratio exp(coefficient), its 95 % bootstrap interval, the permutation p-value, and the
    number of pairs and same-context pairs.
    """
    stations = labels.index.to_numpy()
    index = {station: i for i, station in enumerate(stations)}
    target_idx = pairs["target"].map(index).to_numpy()
    neighbour_idx = pairs["neighbour"].map(index).to_numpy()
    targets = pairs["target"].to_numpy()
    classes = labels.to_numpy()

    y = demean_by_target(np.log(pairs["weight"].to_numpy()), targets)
    log_distance = demean_by_target(np.log(pairs["distance_km"].to_numpy()), targets)

    same = same_context(classes[target_idx], classes[neighbour_idx])
    observed = context_effect(y, log_distance, same, targets)

    # Permutation: the classes are shuffled over the stations, which keeps their locations and weights
    permuted = np.empty(N_PERMUTATIONS)
    for i in range(N_PERMUTATIONS):
        shuffled = rng.permutation(classes)
        target_classes = shuffled[target_idx] if permute_targets else classes[target_idx]
        permuted[i] = context_effect(y, log_distance, same_context(target_classes, shuffled[neighbour_idx]), targets)
    p_value = (1 + np.sum(np.abs(permuted) >= abs(observed))) / (1 + N_PERMUTATIONS)

    # Cluster bootstrap: whole targets are resampled, so the dependence between the pairs of a target is kept
    unique_targets = np.unique(targets)
    rows_of = {t: np.flatnonzero(targets == t) for t in unique_targets}
    bootstrap = np.empty(N_BOOTSTRAP)
    for i in range(N_BOOTSTRAP):
        drawn = rng.choice(unique_targets, size=len(unique_targets), replace=True)
        rows = np.concatenate([rows_of[t] for t in drawn])
        # Every draw of a target is its own cluster, so a target drawn twice gets two fixed effects
        clusters = np.repeat(np.arange(len(drawn)), [len(rows_of[t]) for t in drawn])
        bootstrap[i] = context_effect(y[rows], log_distance[rows], same[rows], clusters)

    low, high = np.exp(np.percentile(bootstrap, [2.5, 97.5]))
    return {
        "ratio": np.exp(observed),
        "ci_low": low,
        "ci_high": high,
        "p_permutation": p_value,
        "pairs": len(pairs),
        "same_context_pairs": int(same.sum()),
    }


def run_context_test():
    """
    Label the stations, run the context test over all targets, over the LCZ 2 targets and over the other targets,
    and save the labels and the results to OUTPUT_DIR.
    """
    weights = pd.read_csv(WEIGHTS_FILE)
    coords = load_station_coordinates("SYNTHETIC")

    print(f"Sampling the LCZ within {LCZ_RADIUS_M} m of {len(coords)} stations from the WUDAPT map...", flush=True)
    labels = load_lcz_labels(coords)
    print(labels["lcz"].value_counts().sort_index().to_string())

    rng = np.random.default_rng(SEED)
    lcz = labels["lcz"]
    urban_pairs = weights[weights["target"].map(lcz) == URBAN_CLASS]
    other_pairs = weights[weights["target"].map(lcz) != URBAN_CLASS]
    tests = {
        "same LCZ class, all targets": (weights, lambda t, n: t == n, True),
        f"LCZ {URBAN_CLASS} neighbour, LCZ {URBAN_CLASS} targets": (
            urban_pairs, lambda t, n: n == URBAN_CLASS, False),
        f"same LCZ class, targets outside LCZ {URBAN_CLASS}": (other_pairs, lambda t, n: t == n, False),
    }

    results = []
    for name, (pairs, same_context, permute_targets) in tests.items():
        result = test_context(pairs, lcz, same_context, permute_targets, rng)
        results.append({"test": name, **result})
        print(f"\n{name}: a same-context neighbour gets {result['ratio']:.2f} times the weight at the same distance "
              f"(95 % CI {result['ci_low']:.2f} to {result['ci_high']:.2f}, permutation p = "
              f"{result['p_permutation']:.4f}; {result['same_context_pairs']} of {result['pairs']} pairs)", flush=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    labels.to_csv(os.path.join(OUTPUT_DIR, "lcz_labels.csv"))
    pd.DataFrame(results).to_csv(os.path.join(OUTPUT_DIR, "results.csv"), index=False)
    print(f"\nResults written to {OUTPUT_DIR}")


if __name__ == "__main__":
    run_context_test()
