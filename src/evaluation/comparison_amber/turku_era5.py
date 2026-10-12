"""
TURCLIM observations joined with the ERA5 series of the nearest land grid point, in the layout of
Jacobs et al. (2024): one column per station and one per station named <station>_ERA5.

The observations are Turku_1H_LI.csv, the wide-format source of the Turku_preprocessed.csv used by
the model comparison, with identical values.
"""
import pandas as pd

OBSERVATIONS_FILE = "data/Turku/Turku_1H_LI.csv"
ERA5_FILE = "data/Turku/Turku_ERA5.csv"


def load_turku_with_era5():
    """
    Join the observations with ERA5. The first two hours exist only in the observations and the
    last two only in ERA5, so both are cut, as in their Execution_Evaluation_GFtechniques.py.
    """
    obs = pd.read_csv(OBSERVATIONS_FILE, parse_dates=["DateTime"], index_col="DateTime")
    era5 = pd.read_csv(ERA5_FILE, parse_dates=["DateTime"], index_col="DateTime")
    era5.columns = [col + "_ERA5" for col in era5.columns]
    return pd.concat([obs, era5], axis=1).iloc[2:-2].asfreq("h")
