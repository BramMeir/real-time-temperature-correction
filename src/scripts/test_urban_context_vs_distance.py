"""
Test: are urban-context (same-LCZ) stations over-represented among the highest-weighted
neighbours, even though they're under-represented among the closest ones?

For every station in the Synthetic network as a target: fit target ~ all other stations
(standardized, RidgeCV-regularized so correlated near-duplicate stations share weight instead
of producing unstable, sign-flipping coefficients), then compare the count of context-matched
(same WUDAPT LCZ class) stations in the nearest-5-by-distance vs. the top-5-by-|weight|.
Paired Wilcoxon signed-rank test across targets, reported for all targets and for the urban
(LCZ 2, compact midrise) subset separately.

LCZ labels are sampled live from the global WUDAPT LCZ COG (Demuzere et al. 2022) via HTTP range
reads - no full raster download needed.

Casino_Oostende is dropped: its coordinate lands on an LCZ nodata pixel (likely too close to open
water for the 100m raster to classify), so its context label isn't valid.
"""
import numpy as np
import pandas as pd
import rasterio
from scipy.stats import wilcoxon
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from src.utils.load_station_coordinates import load_station_coordinates
from src.models.idw.train import _haversine_distance

EXCLUDE_STATIONS = ["Casino_Oostende"]
TRAIN_WEEKS = 8
TOP_N = 5
URBAN_LCZ = 2  # compact midrise - the LCZ class of every urban target in this network
LCZ_COG_URL = "/vsicurl/https://lcz-generator.rub.de/cogs/lcz_filter_v1_cog.tif"
ALPHAS = np.logspace(-2, 4, 25)


def load_wide_data():
    df = pd.read_csv("data/Synthetic/temperature_data.csv")
    df["datetime"] = pd.to_datetime(df["datetime"])
    wide = df.pivot(index="datetime", columns="station_name", values="temp_dry_avg_2m")
    return wide.asfreq("1h")


def load_lcz_labels(coords):
    stations = list(coords.keys())
    pts = [(coords[s][1], coords[s][0]) for s in stations]  # rasterio.sample expects (lon, lat)
    with rasterio.open(LCZ_COG_URL) as src:
        vals = [int(v[0]) for v in src.sample(pts)]
    return dict(zip(stations, vals))


def fit_weights(train, target, neighbours):
    sub = train[[target] + list(neighbours)].dropna()
    X = sub[neighbours].values
    y = sub[target].values

    Xs = StandardScaler().fit_transform(X)
    ys = (y - y.mean()) / y.std()

    model = RidgeCV(alphas=ALPHAS, cv=5).fit(Xs, ys)
    return dict(zip(neighbours, model.coef_))


def run_for_target(train, coords, lcz, target_station, candidates):
    weights = fit_weights(train, target_station, candidates)

    target_lat, target_lon = coords[target_station]
    target_lcz = lcz[target_station]

    rows = []
    for n in candidates:
        dist_km = _haversine_distance(target_lat, target_lon, coords[n][0], coords[n][1])
        rows.append({
            "target": target_station,
            "neighbour": n,
            "distance_km": dist_km,
            "context_matched": lcz[n] == target_lcz,
            "weight": weights[n],
            "abs_weight": abs(weights[n]),
        })
    return pd.DataFrame(rows)


def report(label, summary):
    nonzero = summary[summary["diff"] != 0]
    print(f"\n--- {label} (n={len(summary)} targets) ---")
    print(f"Mean matched-in-nearest{TOP_N}: {summary['matched_in_nearest5'].mean():.2f} / {TOP_N}")
    print(f"Mean matched-in-top{TOP_N}-weight: {summary['matched_in_top5_weight'].mean():.2f} / {TOP_N}")
    print(f"Mean diff: {summary['diff'].mean():.2f}  "
          f"({(summary['diff'] > 0).sum()} positive / {(summary['diff'] < 0).sum()} negative / "
          f"{(summary['diff'] == 0).sum()} tied)")
    if len(nonzero) >= 5:
        stat, p = wilcoxon(nonzero["diff"])
        print(f"Wilcoxon signed-rank (non-zero diffs, n={len(nonzero)}): stat={stat:.3f}, p={p:.5f}")
    else:
        print("Too few non-zero diffs for a signed-rank test.")


def main():
    wide = load_wide_data()
    coords = load_station_coordinates("SYNTHETIC")
    for s in EXCLUDE_STATIONS:
        coords.pop(s, None)

    print("Sampling LCZ labels for", len(coords), "stations from the global LCZ COG...", flush=True)
    lcz = load_lcz_labels(coords)
    print("LCZ label counts:", pd.Series(lcz).value_counts().to_dict(), flush=True)

    start = wide.index.min()
    train_end = start + pd.Timedelta(weeks=TRAIN_WEEKS)
    train = wide.loc[start:train_end]

    all_stations = list(coords.keys())
    all_pairs = []
    summary_rows = []

    for i, target in enumerate(all_stations):
        candidates = [c for c in all_stations if c != target]
        result = run_for_target(train, coords, lcz, target, candidates)
        all_pairs.append(result)

        nearest5 = result.nsmallest(TOP_N, "distance_km")
        top5_weight = result.nlargest(TOP_N, "abs_weight")

        summary_rows.append({
            "target": target,
            "lcz": lcz[target],
            "matched_in_nearest5": nearest5["context_matched"].sum(),
            "matched_in_top5_weight": top5_weight["context_matched"].sum(),
        })
        print(f"[{i + 1}/{len(all_stations)}] {target}: "
              f"nearest{TOP_N}_matched={nearest5['context_matched'].sum()}, "
              f"top{TOP_N}_weight_matched={top5_weight['context_matched'].sum()}", flush=True)

    pairs = pd.concat(all_pairs, ignore_index=True)
    pairs.to_csv("output/urban_context_vs_distance/full_network_weights.csv", index=False)

    summary = pd.DataFrame(summary_rows)
    summary["diff"] = summary["matched_in_top5_weight"] - summary["matched_in_nearest5"]
    summary.to_csv("output/urban_context_vs_distance/full_network_summary.csv", index=False)

    print("\n=== Summary across all", len(summary), "targets ===")
    print(summary.to_string(index=False))

    report("All targets", summary)

    urban = summary[summary["lcz"] == URBAN_LCZ]
    report(f"Urban targets only (LCZ {URBAN_LCZ})", urban)


if __name__ == "__main__":
    main()
