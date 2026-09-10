import numpy as np


def _haversine_distance(lat1, lon1, lat2, lon2):
    """
    Compute the great-circle distance between two points on Earth using the haversine formula.

    Input
    -----
    lat1, lon1: Latitude and longitude of the first point (degrees)
    lat2, lon2: Latitude and longitude of the second point (degrees)

    Output
    ------
    distance: Great-circle distance between the two points, in km
    """
    earth_radius_km = 6371.0

    lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    return earth_radius_km * c


def train_idw(coords, target_station, neighbour_stations, power=2, k=None):
    """
    "Train" an inverse distance weighting (IDW) model. IDW has no parameters fitted on the observations: training
    only means computing the fixed spatial weights of the neighbour stations relative to the target station, based
    on their coordinates. These weights are reused unchanged for every forecast.

    Input
    -----
    coords: Dict mapping station_name to a (latitude, longitude) tuple, as returned by load_station_coordinates
    target_station: Name of the target station
    neighbour_stations: List of candidate neighbour station names
    power: Power parameter of the inverse distance weighting (default is 2)
    k: Number of nearest neighbour stations to use (default is None, meaning all of them)

    Output
    ------
    weights: Dict mapping station_name to its normalised weight (weights sum to 1)
    """
    if target_station not in coords:
        raise ValueError(f"Target station '{target_station}' has no coordinates")

    target_lat, target_lon = coords[target_station]

    # Silently skip neighbours without coordinates, they cannot be weighted
    distances = {}
    for station in neighbour_stations:
        if station not in coords:
            continue

        lat, lon = coords[station]
        distances[station] = _haversine_distance(target_lat, target_lon, lat, lon)

    if not distances:
        raise ValueError(f"None of the neighbour stations {neighbour_stations} have coordinates")

    # Keep only the k nearest neighbours if requested
    if k is not None:
        nearest = sorted(distances, key=distances.get)[:k]
        distances = {station: distances[station] for station in nearest}

    # If a neighbour is coincident with the target station, give it all the weight
    coincident = [station for station, distance in distances.items() if distance == 0]
    if coincident:
        weights = {station: 0.0 for station in distances}
        weights[coincident[0]] = 1.0
        return weights

    inverse_distances = {station: 1 / distance ** power for station, distance in distances.items()}
    total = sum(inverse_distances.values())
    weights = {station: value / total for station, value in inverse_distances.items()}

    return weights
