"""
Maps of the three station networks (the station maps in the appendix of the paper). Every map shows the stations of
one dataset as they enter the model comparison, on the basemap of the stage-one weight maps
(src/plotting/plot_stage_one_weight_maps.py), coloured by the local climate zone (LCZ) group of their surroundings, with
the LCZ labels of src/scripts/test_urban_context_weights.py. The two Belgian networks share one frame, with the
national borders drawn, so they can be compared directly. The three TURCLIM stations in the centre of Turku lie within
a kilometre of each other, so that map has an inset of the centre. The legend is saved as a separate figure, to sit
under all the maps.

python -m src.plotting.plot_station_maps
"""
import math
import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from matplotlib.lines import Line2D
from matplotlib_scalebar.scalebar import ScaleBar
from src.utils.load_station_coordinates import load_station_coordinates
from src.scripts.models_comparison_experiment import DATASETS
from src.scripts.test_urban_context_weights import load_lcz_labels
from src.plotting.plot_stage_one_weight_maps import (
    ASPECT, BASEMAP, JPEG_QUALITY, MAP_DPI, MAP_WIDTH, TILE_HEADERS
)

PLOTS_DIR = "plots/station_maps"

# File name of every map
MAP_NAMES = {"KMI": "RMI", "TURKU": "TURCLIM", "SYNTHETIC": "Synthetic"}

# Margin around the stations, as a fraction of the span they cover: wider for TURCLIM, whose stations are named
MARGIN = 0.12
TURKU_MARGIN = 0.25

# Stations shown in the inset of the TURCLIM map, the margin around them and the extra room on the right for their
# names, in Web Mercator metres
TURKU_CENTRE = ["Betel", "Puutori", "Virastotalo"]
INSET_MARGIN = 700
INSET_LABEL_ROOM = 1800

# Natural Earth land borders between countries, drawn on the Belgian maps
BORDERS_URL = "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_boundary_lines_land.zip"

# LCZ groups as (legend label, LCZ classes, colour): the compact built types, the other built types and the land cover
# types (LCZ A to G, numbered 11 to 17 in the map), after the built and land cover types of Stewart and Oke (2012).
# The colours differ in lightness as well as hue, so they stay apart with colour-vision deficiencies
LCZ_GROUPS = [
    ("Compact built (LCZ 1-3)", range(1, 4), "#A50F15"),
    ("Other built (LCZ 4-10)", range(4, 11), "#FD8D3C"),
    ("Land cover (LCZ A-G)", range(11, 18), "#31A354"),
]

# Station marker size, with a white edge so it stands out on the faded basemap
STATION_SIZE = 30

# Opacity of the basemap, more faded than on the weight maps, where the coloured markers already stand out
BASEMAP_ALPHA = 0.55

# Font size of the station names on the TURCLIM map, in points
LABEL_SIZE = 6


def load_station_points(dataset):
    """
    Load the stations of a dataset that enter the model comparison, as points in Web Mercator, the projection of the
    basemap.

    Input
    -----
    dataset: Name of the dataset, as in DATASETS ("KMI", "TURKU" or "SYNTHETIC")

    Output
    ------
    Returns a GeoDataFrame with one row per station, indexed by station name, with the colour of its LCZ group.
    """
    # The data file decides which stations take part: the RMI metadata also lists the two stations that were left
    # out for an incomplete record
    names = sorted(pd.read_csv(DATASETS[dataset]["file"], usecols=["station_name"])["station_name"].unique())
    coords = {name: load_station_coordinates(dataset)[name] for name in names}
    lats, lons = zip(*coords.values())
    lcz = load_lcz_labels(coords)["lcz"]
    colours = {lcz_class: colour for _, classes, colour in LCZ_GROUPS for lcz_class in classes}

    points = gpd.GeoDataFrame(
        {"colour": [colours[lcz[name]] for name in names]},
        index=pd.Index(names, name="station"),
        geometry=gpd.points_from_xy(lons, lats),
        crs="EPSG:4326"
    )
    return points.to_crs(epsg=3857)


def frame(points, margin):
    """
    Frame a set of points with a margin around them, in the ASPECT ratio.

    Input
    -----
    points: GeoDataFrame with the points that have to be on the map
    margin: Margin around the points as a fraction of the span they cover

    Output
    ------
    Returns (xmin, xmax, ymin, ymax) of the map.
    """
    xmin, ymin, xmax, ymax = points.total_bounds
    width, height = (xmax - xmin) * (1 + 2 * margin), (ymax - ymin) * (1 + 2 * margin)

    # Widen the narrower side to the aspect ratio, around the centre of the points
    width, height = max(width, height * ASPECT), max(height, width / ASPECT)
    x, y = (xmin + xmax) / 2, (ymin + ymax) / 2

    return x - width / 2, x + width / 2, y - height / 2, y + height / 2


def draw_stations(ax, points, extent, labelled=(), borders=None):
    """
    Draw the stations on a basemap, as dots in the colour of their LCZ group with a white edge.

    Input
    -----
    ax: Axes to draw on
    points: GeoDataFrame with the stations, as returned by load_station_points
    extent: (xmin, xmax, ymin, ymax) of the map
    labelled: Names of the stations to name on the map
    borders: GeoDataFrame with the national borders to draw, or None to draw none
    """
    xmin, xmax, ymin, ymax = extent
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")

    ax.scatter(points.geometry.x, points.geometry.y, s=STATION_SIZE, color=points["colour"], edgecolor="white",
               linewidth=0.8, zorder=3)

    # Thin but darker than the faded borders of the basemap, so the country stands out
    if borders is not None:
        borders.plot(ax=ax, color="0.35", linewidth=0.5, zorder=2)

    for name in labelled:
        point = points.loc[name].geometry
        ax.annotate(name, (point.x, point.y), xytext=(5, 0), textcoords="offset points", va="center",
                    fontsize=LABEL_SIZE, zorder=5,
                    bbox={"facecolor": "white", "alpha": 0.7, "linewidth": 0, "pad": 0.5})

    ctx.add_basemap(ax, source=BASEMAP, headers=TILE_HEADERS, alpha=BASEMAP_ALPHA, attribution=False)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks([])
    ax.set_yticks([])


def add_scale_bar(ax, extent):
    """
    Add a scale bar in the lower left, just above the OpenStreetMap credit, where none of the maps has stations.
    A metre in Web Mercator is 1 / cos(latitude) metres on the ground, so the scale is corrected for the latitude of
    the map centre.

    Input
    -----
    ax: Axes of the map
    extent: (xmin, xmax, ymin, ymax) of the map, in Web Mercator metres
    """
    centre = gpd.GeoSeries(gpd.points_from_xy([0], [(extent[2] + extent[3]) / 2]), crs="EPSG:3857").to_crs(epsg=4326)
    ax.add_artist(ScaleBar(math.cos(math.radians(centre.y[0])), units="m", location="lower left", length_fraction=0.2,
                           width_fraction=0.008, font_properties={"size": LABEL_SIZE}, frameon=False,
                           border_pad=1.2))


def add_attribution(ax):
    """Credit OpenStreetMap on one line, as on the stage-one weight maps."""
    ax.text(0.005, 0.005, "© OpenStreetMap contributors", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=4, zorder=7, bbox={"facecolor": "white", "alpha": 0.7, "linewidth": 0, "pad": 1})


def save_map(fig, name):
    """Save a map as JPEG, as the weight maps: the basemap texture hardly compresses as PNG."""
    fig.savefig(os.path.join(PLOTS_DIR, f"{name}.jpg"), dpi=MAP_DPI, bbox_inches="tight", pad_inches=0.02,
                pil_kwargs={"quality": JPEG_QUALITY})
    plt.close(fig)


def plot_belgian_maps():
    """Plot the RMI and the synthetic network in one shared frame around all Belgian stations."""
    networks = {dataset: load_station_points(dataset) for dataset in ["KMI", "SYNTHETIC"]}
    extent = frame(pd.concat(networks.values()), MARGIN)
    borders = gpd.read_file(BORDERS_URL).to_crs(epsg=3857)

    for dataset, points in networks.items():
        fig, ax = plt.subplots(figsize=(MAP_WIDTH, MAP_WIDTH / ASPECT))
        draw_stations(ax, points, extent, borders=borders)
        add_scale_bar(ax, extent)
        add_attribution(ax)
        save_map(fig, MAP_NAMES[dataset])


def plot_turku_map():
    """Plot the TURCLIM network, with an inset of the three stations in the centre of Turku."""
    points = load_station_points("TURKU")
    extent = frame(points, TURKU_MARGIN)

    fig, ax = plt.subplots(figsize=(MAP_WIDTH, MAP_WIDTH / ASPECT))
    # The centre stations are named in the inset only, as their names would overlap on the overview
    draw_stations(ax, points, extent, labelled=points.index.drop(TURKU_CENTRE))
    add_scale_bar(ax, extent)
    add_attribution(ax)

    # Inset of the centre in the upper left, where the overview has no stations
    centre = points.loc[TURKU_CENTRE]
    xmin, ymin, xmax, ymax = centre.total_bounds
    corners = gpd.points_from_xy([xmin - INSET_MARGIN, xmax + INSET_MARGIN + INSET_LABEL_ROOM],
                                 [ymin - INSET_MARGIN, ymax + INSET_MARGIN])
    inset_extent = frame(gpd.GeoDataFrame(geometry=corners, crs="EPSG:3857"), 0)
    inset = ax.inset_axes([0.02, 0.56, 0.36, 0.36])
    draw_stations(inset, centre, inset_extent, labelled=TURKU_CENTRE)
    ax.indicate_inset_zoom(inset, edgecolor="black", linewidth=0.6, alpha=1)

    save_map(fig, MAP_NAMES["TURKU"])


def plot_legend():
    """Plot the legend of the LCZ groups on its own, as one row that sits under all the maps."""
    fig = plt.figure(figsize=(2 * MAP_WIDTH, 0.3))
    # Marker size in points, the square root of the scatter size of the maps
    handles = [Line2D([], [], linestyle="none", marker="o", markersize=STATION_SIZE ** 0.5, markerfacecolor=colour,
                      markeredgecolor="white", markeredgewidth=0.8, label=label) for label, _, colour in LCZ_GROUPS]
    fig.legend(handles=handles, loc="center", ncol=len(LCZ_GROUPS), frameon=False, fontsize=LABEL_SIZE,
               handletextpad=0.3, columnspacing=2)
    fig.savefig(os.path.join(PLOTS_DIR, "legend.pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(PLOTS_DIR, exist_ok=True)
    # The legend is a vector PDF, with its font embedded as TrueType instead of matplotlib's default Type 3
    plt.rcParams["pdf.fonttype"] = 42
    plot_belgian_maps()
    plot_turku_map()
    plot_legend()
    print(f"Maps written to {PLOTS_DIR}")
