"""
Maps of the stations the first stage of the two-stage model weights most, for a few representative targets of the
synthetic network (the station selection maps of the paper). Every map shows the whole network on the same
extent, so a highly weighted station far from the target stands out. The weights come from
src/scripts/determine_stage_one_weights.py, which has to run first.

python -m src.plotting.plot_stage_one_weight_maps
"""
import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from src.utils.load_station_coordinates import load_station_coordinates

WEIGHTS_FILE = "output/stage_one_weights/weights.csv"

PLOTS_DIR = "plots/stage_one_weights"

# Targets shown in the paper, from rural to urban, with the file name of their map
MAP_TARGETS = {
    "Vielsalm": "Vielsalm",
    "Vesten_Geraardsbergen": "Geraardsbergen",
    "Station_Eeklo": "Eeklo",
    "Stadhuis_Antwerpen_Grote_Markt": "Antwerp",
    "Stadhuis_Brussel_Grote_Markt": "Brussels",
    "Sint_Baafs_Gent": "Ghent",
}

# Number of highest-weighted neighbours that are marked on every map
TOP_N = 4

# Margin around the network, as a fraction of its extent
MARGIN = 0.06


def load_station_points():
    """
    Load the coordinates of the synthetic stations as points in Web Mercator, the projection of the basemap.

    Output
    ------
    Returns a GeoDataFrame with one row per station, indexed by station name.
    """
    coords = load_station_coordinates("SYNTHETIC")
    names = list(coords)
    lats, lons = zip(*(coords[name] for name in names))

    return gpd.GeoDataFrame(
        index=pd.Index(names, name="station"),
        geometry=gpd.points_from_xy(lons, lats),
        crs="EPSG:4326"
    ).to_crs(epsg=3857)


def plot_weight_map(points, weights, target, filename):
    """
    Plot the network with the target and its TOP_N highest-weighted neighbours marked, numbered by rank.

    Input
    -----
    points: GeoDataFrame with the station locations, indexed by station name
    weights: DataFrame with the averaged weights, as written by determine_stage_one_weights.py
    target: Name of the target station
    filename: Path of the PNG file to write
    """
    top = weights[(weights["target"] == target) & (weights["weight_rank"] <= TOP_N)].sort_values("weight_rank")

    fig, ax = plt.subplots(figsize=(6, 5))

    others = points.drop(index=[target] + top["neighbour"].tolist())
    ax.scatter(others.geometry.x, others.geometry.y, s=28, color="0.45", edgecolor="white", linewidth=0.6,
               zorder=3)

    selected = points.loc[top["neighbour"]]
    ax.scatter(selected.geometry.x, selected.geometry.y, s=150, color="tab:green", edgecolor="white",
               linewidth=1.0, zorder=4)
    for rank, point in zip(top["weight_rank"], selected.geometry):
        ax.annotate(str(rank), (point.x, point.y), ha="center", va="center", fontsize=8, fontweight="bold",
                    color="white", zorder=5)

    target_point = points.loc[target].geometry
    ax.scatter(target_point.x, target_point.y, s=150, marker="s", color="gold", edgecolor="black",
               linewidth=1.2, zorder=5)

    # Same extent on every map, set before the basemap so the tiles cover it
    xmin, ymin, xmax, ymax = points.total_bounds
    dx, dy = (xmax - xmin) * MARGIN, (ymax - ymin) * MARGIN
    ax.set_xlim(xmin - dx, xmax + dx)
    ax.set_ylim(ymin - dy, ymax + dy)

    # Esri's grey canvas: key-free and unobtrusive under the markers (see plot_spatial_mae_and_confidence.py)
    ctx.add_basemap(ax, source=ctx.providers.Esri.WorldGrayCanvas, alpha=0.75, attribution_size=5)
    ax.set_axis_off()

    plt.savefig(filename, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(PLOTS_DIR, exist_ok=True)

    weights = pd.read_csv(WEIGHTS_FILE)
    points = load_station_points()

    for target, name in MAP_TARGETS.items():
        top = weights[(weights["target"] == target) & (weights["weight_rank"] <= TOP_N)]
        ranks = ", ".join(f"{row.neighbour} (distance rank {row.distance_rank})" for row in top.itertuples())
        print(f"{name}: {ranks}")

        plot_weight_map(points, weights, target, os.path.join(PLOTS_DIR, f"{name}.png"))

    print(f"\nMaps written to {PLOTS_DIR}")
