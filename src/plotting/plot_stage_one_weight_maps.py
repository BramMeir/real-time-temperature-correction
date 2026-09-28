"""
Maps of the stations the first stage of the two-stage model weights most, for a few representative targets of the
synthetic network (the station selection maps of the paper). Every map is framed on the target and its marked
neighbours, and shows the other stations in that area, so it can be seen which nearby stations the model passes
over. The weights come from src/scripts/determine_stage_one_weights.py, which has to run first.

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

# Margin around the target and its marked neighbours, as a fraction of the span they cover
MARGIN = 0.25

# Smallest span of a map in Web Mercator metres (about 60 km in Belgium), so a target whose marked neighbours
# all lie close by still shows its surroundings
MIN_SPAN = 100_000

# Width to height ratio of every map
ASPECT = 6 / 5

# OpenStreetMap's standard style, which names places in their local language (Antwerpen, Gent, Namur). Its tile
# policy blocks requests that do not identify the application, so they are sent with TILE_HEADERS. Carto watermarks
# its tiles without an API key, and the French OpenStreetMap style translates the names (Anvers, Gand)
BASEMAP = ctx.providers.OpenStreetMap.Mapnik

TILE_HEADERS = {
    "User-Agent": "real-time-temperature-correction (+https://github.com/BramMeir/real-time-temperature-correction)"
}

# Opacity of the basemap, slightly faded so the station markers stand out
BASEMAP_ALPHA = 0.7


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


def map_extent(points):
    """
    Frame a set of points with MARGIN around them, at least MIN_SPAN wide or high, in the ASPECT ratio.

    Input
    -----
    points: GeoDataFrame with the points that have to be on the map

    Output
    ------
    Returns (xmin, xmax, ymin, ymax) of the map.
    """
    xmin, ymin, xmax, ymax = points.total_bounds
    width = max((xmax - xmin) * (1 + 2 * MARGIN), MIN_SPAN)
    height = max((ymax - ymin) * (1 + 2 * MARGIN), MIN_SPAN / ASPECT)

    # Widen the narrower side to the aspect ratio, around the centre of the points
    width, height = max(width, height * ASPECT), max(height, width / ASPECT)
    x, y = (xmin + xmax) / 2, (ymin + ymax) / 2

    return x - width / 2, x + width / 2, y - height / 2, y + height / 2


def plot_weight_map(points, weights, target, filename, basemap=BASEMAP):
    """
    Plot the target and its TOP_N highest-weighted neighbours, numbered by rank, among the other stations
    around them.

    Input
    -----
    points: GeoDataFrame with the station locations, indexed by station name
    weights: DataFrame with the averaged weights, as written by determine_stage_one_weights.py
    target: Name of the target station
    filename: Path of the PNG file to write
    basemap: Tile provider of the background map (default is BASEMAP)
    """
    top = weights[(weights["target"] == target) & (weights["weight_rank"] <= TOP_N)].sort_values("weight_rank")
    selected = points.loc[top["neighbour"]]
    target_point = points.loc[target].geometry

    fig, ax = plt.subplots(figsize=(6, 6 / ASPECT))

    # The extent is set before the basemap so the tiles cover it
    xmin, xmax, ymin, ymax = map_extent(points.loc[[target] + top["neighbour"].tolist()])
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    others = points.drop(index=[target] + top["neighbour"].tolist())
    ax.scatter(others.geometry.x, others.geometry.y, s=60, color="0.2", edgecolor="white", linewidth=1.2,
               zorder=3)

    # The target below the marked neighbours, so a neighbour next to it stays readable
    ax.scatter(target_point.x, target_point.y, s=70, marker="s", color="gold", edgecolor="black",
               linewidth=1.0, zorder=4)

    ax.scatter(selected.geometry.x, selected.geometry.y, s=170, color="tab:green", edgecolor="white",
               linewidth=1.0, zorder=5)
    for rank, point in zip(top["weight_rank"], selected.geometry):
        ax.annotate(str(rank), (point.x, point.y), ha="center", va="center", fontsize=8, fontweight="bold",
                    color="white", zorder=6)

    ctx.add_basemap(ax, source=basemap, headers=TILE_HEADERS, alpha=BASEMAP_ALPHA, attribution_size=4)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
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
