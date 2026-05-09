import pandas as pd
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
import contextily as ctx
import glob
import os


def plot_map(gdf, column, filename, label, cmap, invert_bar=False, vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(12, 6))

    # Normalize colors
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    # Plot without legend (we add a custom colorbar later)
    gdf.plot(
        ax=ax,
        column=column,
        cmap=cmap,
        markersize=130,
        edgecolor="black",
        linewidth=1,
        legend=False,
        norm=norm
    )

    # Basemap
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, alpha=0.75)

    ax.set_axis_off()

    # Compact colorbar
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm._A = []

    cbar = fig.colorbar(
        sm,
        ax=ax,
        fraction=0.02,
        pad=0.015,
        shrink=0.7
    )

    cbar.ax.tick_params(labelsize=8)
    cbar.set_label(label, fontsize=10)

    # Invert colorbar if needed (e.g., for MAE where lower is better, always "better" on top)
    if invert_bar:
        cbar.ax.invert_yaxis()

    # Include a scale bar (necessary to give a sense of distance between the stations on the map)
    scalebar = ScaleBar(
        dx=1,
        units="m",
        location="lower left",
        length_fraction=0.18,
        height_fraction=0.01,
        pad=1.2,
        font_properties={"size": 8},
        color="black",
        box_alpha=0,
        frameon=False
    )
    ax.add_artist(scalebar)

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


# Guarantee output directory exists
plots_dir = "plots/confidence_analysis"
os.makedirs(plots_dir, exist_ok=True)

# Load station results
files = glob.glob("output/confidence_analysis/confidence_station_*.csv")
df_results = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

# Load mapping
df_mapping = pd.read_csv("data/Synthetic/stations_mapping.csv")

# Load coordinates
df_coords = pd.read_csv("data/Synthetic/extracted_points_lat_lon.csv")

# Merge everything
df = df_results.merge(df_mapping, left_on="target_station", right_on="station_name")
df = df.merge(df_coords, on="point_index")

# Basic check
print(df.head())

# Make a GeoDataFrame
gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df.lon, df.lat),
    crs="EPSG:4326"  # lat/lon
)

# Reproject to Web Mercator for better distance representation
gdf = gdf.to_crs(epsg=3857)

# Determine color scale limits (optional, can be set to None for automatic)
conf_min, conf_max = gdf["confidence_mean"].min(), gdf["confidence_mean"].max()
interval_min, interval_max = gdf["interval_size_mean"].min(), gdf["interval_size_mean"].max()
mae_min, mae_max = gdf["mae_mean"].min(), gdf["mae_mean"].max()

# Plot confidence score
plot_map(
    gdf,
    column="confidence_mean",
    filename=f"{plots_dir}/confidence_score_map.png",
    label="Gemiddelde betrouwbaarheidsscore",
    cmap="coolwarm",
    vmin=conf_min,
    vmax=conf_max
)

# Plot interval size
plot_map(
    gdf,
    column="interval_size_mean",
    filename=f"{plots_dir}/confidence_interval_size_map.png",
    label="Gemiddelde intervalgrootte (°C)",
    cmap="coolwarm_r",
    invert_bar=True,
    vmin=interval_min,
    vmax=interval_max
)

# Plot MAE
plot_map(
    gdf,
    column="mae_mean",
    filename=f"{plots_dir}/mae_map.png",
    label="Gemiddelde MAE (°C)",
    cmap="coolwarm_r",
    invert_bar=True,
    vmin=mae_min,
    vmax=mae_max
)
