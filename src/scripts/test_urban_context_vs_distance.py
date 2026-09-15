"""
Evidence check: does the two-stage model's actual stage-1 regression ever rely on a
context-similar (same WUDAPT LCZ class) station well beyond the nearest ones, rather than only
picking the closest stations available?

This is deliberately a modest, descriptive check, not a network-wide significance claim: it
reports, per urban target, whether the model's own fitted weights put a context-matched station
in the top-5 despite that station being absent from the nearest-5 by distance - and shows the
single most concrete example per target (the actual top-weighted station vs. the actual nearest
station) so every case is directly inspectable.

Uses the REAL project model unmodified: train_neighbour_regression (plain OLS, all stations, no
LASSO pre-selection - confirmed not to help for this two-stage model). Standardized-equivalent
coefficients are computed post-hoc (std_coef = raw_coef * std(x_i) / std(y)) WITHOUT refitting,
purely to make weights comparable across stations of different natural variance.

LCZ labels are sampled live from the global WUDAPT LCZ COG (Demuzere et al. 2022) via HTTP range
reads - no full raster download needed.

Casino_Oostende is dropped: its coordinate lands on an LCZ nodata pixel (likely too close to open
water for the 100m raster to classify), so its context label isn't valid.
"""
import pandas as pd
import rasterio

from src.models.linear_regression.train import train_neighbour_regression
from src.utils.load_station_coordinates import load_station_coordinates
from src.models.idw.train import _haversine_distance

EXCLUDE_STATIONS = ["Casino_Oostende"]
TRAIN_WEEKS = 8
TOP_N = 5
URBAN_LCZ = 2  # compact midrise - the LCZ class of every urban target in this network
LCZ_COG_URL = "/vsicurl/https://lcz-generator.rub.de/cogs/lcz_filter_v1_cog.tif"


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


def run_for_target(train, coords, lcz, target_station, candidates):
    # The actual project model, unmodified: plain OLS, fed all stations
    regression = train_neighbour_regression(train, target_station, candidates)

    data = train[[target_station] + list(candidates)].dropna()
    y_std = data[target_station].std()
    x_stds = data[candidates].std()

    target_lat, target_lon = coords[target_station]
    target_lcz = lcz[target_station]

    rows = []
    for i, n in enumerate(candidates):
        raw_coef = regression.coef_[i]
        std_coef = raw_coef * x_stds[n] / y_std  # post-hoc rescale for comparability, not a refit
        dist_km = _haversine_distance(target_lat, target_lon, coords[n][0], coords[n][1])
        rows.append({
            "target": target_station,
            "neighbour": n,
            "distance_km": dist_km,
            "context_matched": lcz[n] == target_lcz,
            "std_coef": std_coef,
            "abs_std_coef": abs(std_coef),
        })
    return pd.DataFrame(rows)


def main():
    wide = load_wide_data()
    coords = load_station_coordinates("SYNTHETIC")
    for s in EXCLUDE_STATIONS:
        coords.pop(s, None)

    print("Sampling LCZ labels for", len(coords), "stations from the global LCZ COG...", flush=True)
    lcz = load_lcz_labels(coords)

    start = wide.index.min()
    train_end = start + pd.Timedelta(weeks=TRAIN_WEEKS)
    train = wide.loc[start:train_end]

    all_stations = list(coords.keys())
    urban_targets = [s for s in all_stations if lcz[s] == URBAN_LCZ]
    print(f"Urban (LCZ {URBAN_LCZ}) targets: {urban_targets}", flush=True)

    summary_rows = []
    detail_rows = []

    for target in urban_targets:
        candidates = [c for c in all_stations if c != target]
        result = run_for_target(train, coords, lcz, target, candidates)

        nearest_n = result.nsmallest(TOP_N, "distance_km")
        top_n = result.nlargest(TOP_N, "abs_std_coef")
        summary_rows.append({
            "target": target,
            f"matched_in_nearest{TOP_N}": nearest_n["context_matched"].sum(),
            f"matched_in_top{TOP_N}_weight": top_n["context_matched"].sum(),
        })

        top_weighted = result.loc[result["abs_std_coef"].idxmax()]
        nearest = result.loc[result["distance_km"].idxmin()]
        detail_rows.append({
            "target": target,
            "top_weighted_station": top_weighted["neighbour"],
            "top_weighted_dist_km": round(top_weighted["distance_km"], 1),
            "top_weighted_matched": bool(top_weighted["context_matched"]),
            "nearest_station": nearest["neighbour"],
            "nearest_dist_km": round(nearest["distance_km"], 1),
            "nearest_matched": bool(nearest["context_matched"]),
        })

    summary = pd.DataFrame(summary_rows)
    detail = pd.DataFrame(detail_rows)

    summary.to_csv("output/urban_context_vs_distance/urban_summary.csv", index=False)
    detail.to_csv("output/urban_context_vs_distance/urban_top_station_detail.csv", index=False)

    print(f"\n=== Evidence table: matched-context stations in nearest-{TOP_N} vs top-{TOP_N}-weighted ===")
    print(summary.to_string(index=False))

    n_with_more_matches = (
        summary[f"matched_in_top{TOP_N}_weight"] > summary[f"matched_in_nearest{TOP_N}"]
    ).sum()
    print(f"\nIn {n_with_more_matches} of {len(summary)} urban targets, the model's top-{TOP_N} "
          f"weighted stations include MORE context-matched stations than the nearest-{TOP_N} by "
          f"distance does.")

    print("\n=== Top-weighted station vs. nearest station, per urban target ===")
    print(detail.to_string(index=False))


if __name__ == "__main__":
    main()
