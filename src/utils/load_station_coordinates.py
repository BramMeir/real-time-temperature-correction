import pandas as pd

SUPPORTED_DATASETS = ["RMI", "TURKU", "SYNTHETIC"]


def load_station_coordinates(dataset_name):
    """
    Load the station coordinates for the given dataset.

    Input
    -----
    dataset_name: Name of the dataset ("RMI", "TURKU" or "SYNTHETIC"), case-insensitive.

    Output
    ------
    coords: Dict mapping station_name to a (latitude, longitude) tuple.
    """
    name = dataset_name.upper()

    if name == "RMI":
        metadata = pd.read_csv("data/RMI/stations.csv")
        coords = {row.station_name: (row.latitude, row.longitude) for row in metadata.itertuples()}
    elif name == "TURKU":
        metadata = pd.read_csv("data/Turku/stations.csv")
        coords = {row.Site_name: (row.Latitude, row.Longitude) for row in metadata.itertuples()}
    elif name == "SYNTHETIC":
        points = pd.read_csv("data/Synthetic/stations.csv")
        mapping = pd.read_csv("data/Synthetic/point_names.csv")

        merged = points.merge(mapping, on="point_index")
        coords = {row.station_name: (row.lat, row.lon) for row in merged.itertuples()}
    else:
        raise ValueError(f"Unknown dataset_name '{dataset_name}', supported datasets are: {SUPPORTED_DATASETS}")

    return coords
