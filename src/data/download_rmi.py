import argparse
import io
import json
import os
import pandas as pd
import requests

# WFS service of the RMI open data platform (opendata.meteo.be), with the AWS observations every 10 minutes
WFS_URL = "https://opendata.meteo.be/service/aws/ows"

OUTPUT_DIR = "data/RMI/open_data"

# The 12 stations of the RMI network in the paper. Zeebrugge and De Haan, the other two AWS stations, are left out,
# as their record of the original period was incomplete
NETWORK = ["BEITEM", "BUZENOL", "DIEPENBEEK", "DOURBES", "ERNAGE", "HUMAIN", "MELLE", "MONT RIGI", "RETIE",
           "SINT-KATELIJNE-WAVER", "STABROEK", "UCCLE"]


def wfs_csv(typename, **params):
    """
    Send one GetFeature request to the RMI WFS service and read the CSV it returns.

    Input
    -----
    typename: Layer to query, e.g. "aws:aws_10min" or "aws:aws_station"
    params: Further request parameters, e.g. cql_filter or propertyName

    Output
    ------
    Returns the response as a DataFrame.
    """
    query = {"service": "WFS", "version": "2.0.0", "request": "GetFeature", "typenames": typename,
             "outputFormat": "csv", **params}
    response = requests.get(WFS_URL, params=query, timeout=600)
    response.raise_for_status()
    return pd.read_csv(io.StringIO(response.text),
                       converters={"qc_flags": lambda flags: json.loads(flags)["validated"]["TEMP_DRY_SHELTER_AVG"]})


def download(stations, years):
    """
    Download every 10-minute temperature of the stations, validated or not, with its RMI validation flag. Every
    10-minute step of the years gets a row, so a step without an observation has an empty temperature.

    Input
    -----
    stations: DataFrame with the columns code and name of the stations to download
    years: Calendar years to download

    Output
    ------
    Returns a DataFrame with the columns code, datetime, temp_dry_avg_2m and validated.
    """
    grid = pd.date_range(f"{years[0]}-01-01", f"{years[-1] + 1}-01-01", freq="10min", inclusive="left")

    frames = []
    for code, name in stations.itertuples(index=False):
        # One request per year keeps every response at about 30 MB, most of it the JSON of the flags, which the
        # converter in wfs_csv reduces to the flag of the temperature as soon as a year is read
        obs = pd.concat(
            wfs_csv("aws:aws_10min",
                    cql_filter=f"code={code} AND timestamp>='{year}-01-01T00:00:00' "
                               f"AND timestamp<'{year + 1}-01-01T00:00:00'",
                    propertyName="timestamp,temp_dry_shelter_avg,qc_flags")
            for year in years
        )
        obs = obs.set_index(pd.to_datetime(obs["timestamp"])).reindex(grid)

        frames.append(pd.DataFrame({
            "code": code,
            "datetime": grid,
            "temp_dry_avg_2m": obs["temp_dry_shelter_avg"],
            # The flag is true for a validated temperature and false or empty otherwise, and steps without an
            # observation, which the reindex leaves empty, count as not validated
            "validated": obs["qc_flags"].eq(True),
        }))
        print(f"Downloaded {name}")

    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download the temperatures of the RMI network of the paper from the "
                                                 "RMI open data platform, as an RMI export for preprocessing.py.")
    parser.add_argument("--first_year", type=int, required=True, help="First calendar year.")
    parser.add_argument("--last_year", type=int, required=True, help="Last calendar year (included).")
    args = parser.parse_args()

    stations = wfs_csv("aws:aws_station")[["code", "name"]]
    stations = stations[stations["name"].isin(NETWORK)]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stations.to_csv(os.path.join(OUTPUT_DIR, "stations.csv"), index=False)
    download(stations, range(args.first_year, args.last_year + 1)).to_csv(
        os.path.join(OUTPUT_DIR, "raw_observations.csv"), index=False)
    print(f"Saved the observations of {args.first_year}-{args.last_year} to {OUTPUT_DIR}")
