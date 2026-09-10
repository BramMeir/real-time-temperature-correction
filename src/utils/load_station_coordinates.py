"""
- load_station_coordinates function to load the station coordinates (latitude, longitude) for one of the supported
  datasets (KMI, TURKU, SYNTHETIC). Used by the IDW baseline model to compute the great-circle distances between
  the target station and its neighbours.
"""
import pandas as pd

SUPPORTED_DATASETS = ["KMI", "TURKU", "SYNTHETIC"]


def load_station_coordinates(dataset_name):
    """
    Load the station coordinates for the given dataset.

    Input
    -----
    dataset_name: Name of the dataset ("KMI", "TURKU" or "SYNTHETIC"), case-insensitive.

    Output
    ------
    coords: Dict mapping station_name to a (latitude, longitude) tuple.
    """
    name = dataset_name.upper()

    if name == "KMI":
        metadata = pd.read_csv("data/Full_AWS/metadata.csv")
        coords = {row.station_name: (row.latitude, row.longitude) for row in metadata.itertuples()}
    elif name == "TURKU":
        metadata = pd.read_csv("data/Turku/Turku_coordinates.csv")
        coords = {row.Site_name: (row.Latitude, row.Longitude) for row in metadata.itertuples()}
    elif name == "SYNTHETIC":
        points = pd.read_csv("data/Synthetic/extracted_points_lat_lon.csv")
        mapping = pd.read_csv("data/Synthetic/stations_mapping.csv")

        merged = points.merge(mapping, on="point_index")
        coords = {row.station_name: (row.lat, row.lon) for row in merged.itertuples()}
    else:
        raise ValueError(f"Unknown dataset_name '{dataset_name}', supported datasets are: {SUPPORTED_DATASETS}")

    return coords
