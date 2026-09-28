"""
Script to test whether the first stage of the two-stage model gives more weight to neighbours in the same urban
context as the target than their distance explains. It replaces the single-window check of
test_urban_context_vs_distance.py with a test over every target and neighbour of the synthetic network, on the
weights averaged over 50 training periods (src/scripts/determine_stage_one_weights.py, which has to run first).

The context of a station is its local climate zone (LCZ), the majority class of the global WUDAPT LCZ map
(Demuzere et al. 2022) within LCZ_RADIUS_M of the station. A zone covers several hundred metres, so a single 100 m
pixel under the station would depend on the road or park it happens to fall on.

Over all target and neighbour pairs, log(weight) is regressed on log(distance) and an indicator for a neighbour in
the same LCZ as the target, with a fixed effect per target. exp(coefficient) is then how many times the weight of a
same-zone neighbour is, at the same distance. The pairs are not independent, since every station appears in about a
hundred of them, so the p-value comes from permuting the LCZ labels over the stations and the confidence interval
from resampling whole targets.

python -m src.scripts.test_urban_context_weights
"""
import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from scipy.stats import spearmanr
from src.utils.load_station_coordinates import load_station_coordinates
from src.models.idw.train import _haversine_distance

WEIGHTS_FILE = "output/stage_one_weights/weights.csv"

COEFFICIENTS_FILE = "output/stage_one_weights/coefficients_per_period.csv"

OUTPUT_DIR = "output/urban_context_weights"

# Global LCZ map, read through HTTP range requests so the raster is never downloaded in full
LCZ_COG_URL = "/vsicurl/https://lcz-generator.rub.de/cogs/lcz_filter_v1_cog.tif"

# Radius around a station within which its LCZ is the majority class
LCZ_RADIUS_M = 500

# LCZ classes 1 to 10 are built types, 11 to 17 land cover types
BUILT_CLASSES = set(range(1, 11))

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
    Returns a DataFrame indexed by station with the majority class, the share of the neighbourhood it covers, and
    the class of the single pixel under the station.
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
            pixel = int(classes[half_rows, half_cols])
            rows.append({
                "station": station,
                "lcz": int(values[counts.argmax()]),
                "lcz_share": counts.max() / counts.sum(),
                "lcz_pixel": pixel if pixel != src.nodata else None,
            })

    return pd.DataFrame(rows).set_index("station")


def split_half_stability(coefficients):
    """
    Check whether the ranking of the neighbours depends on the training periods, by averaging the weights over the
    first and the second half of the periods separately and correlating them per target.

    Input
    -----
    coefficients: DataFrame with one row per target, training period and neighbour, as written by
    determine_stage_one_weights.py

    Output
    ------
    Returns a Series with the Spearman correlation between the two halves per target.
    """
    coefficients = coefficients.assign(weight=coefficients["std_coef"].abs())
    half = coefficients["repeat_id"] < coefficients["repeat_id"].nunique() / 2
    first = coefficients[half].groupby(["target", "neighbour"])["weight"].mean()
    second = coefficients[~half].groupby(["target", "neighbour"])["weight"].mean()
    halves = pd.concat({"first": first, "second": second}, axis=1).reset_index()

    return halves.groupby("target").apply(lambda g: spearmanr(g["first"], g["second"]).statistic)


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
    Label the stations, check the stability of the weights, run the context test for every definition of a match
    and save the labels and the results to OUTPUT_DIR.
    """
    weights = pd.read_csv(WEIGHTS_FILE)
    coords = load_station_coordinates("SYNTHETIC")

    print(f"Sampling the LCZ within {LCZ_RADIUS_M} m of {len(coords)} stations from the WUDAPT map...", flush=True)
    labels = load_lcz_labels(coords)
    print(labels["lcz"].value_counts().sort_index().to_string())
    changed = labels[labels["lcz"] != labels["lcz_pixel"]]
    print(f"{len(changed)} stations get a different class than the pixel under them: "
          f"{', '.join(changed.index)}")

    stability = split_half_stability(pd.read_csv(COEFFICIENTS_FILE))
    print(f"\nSplit-half Spearman correlation of the weights per target: median {stability.median():.2f}, "
          f"lowest {stability.min():.2f} ({stability.idxmin()})")

    rng = np.random.default_rng(SEED)
    lcz = labels["lcz"]
    urban_pairs = weights[weights["target"].map(lcz) == URBAN_CLASS]
    tests = {
        "same LCZ class, all targets": (weights, lambda t, n: t == n, True),
        "same built or land cover type, all targets": (
            weights, lambda t, n: np.isin(t, list(BUILT_CLASSES)) == np.isin(n, list(BUILT_CLASSES)), True),
        f"LCZ {URBAN_CLASS} neighbour, LCZ {URBAN_CLASS} targets": (
            urban_pairs, lambda t, n: n == URBAN_CLASS, False),
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
    stability.rename("spearman").to_csv(os.path.join(OUTPUT_DIR, "split_half_stability.csv"))
    pd.DataFrame(results).to_csv(os.path.join(OUTPUT_DIR, "results.csv"), index=False)
    print(f"\nResults written to {OUTPUT_DIR}")


if __name__ == "__main__":
    run_context_test()
