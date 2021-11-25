import os
from urllib.error import HTTPError
from ssl import SSLEOFError

import pandas as pd

from .common import is_notebook

try:
    if not is_notebook():
        from tqdm import tqdm
    else:
        from tqdm.notebook import tqdm
except ImportError:
    def tqdm(arg, *args, **kwargs):
        return arg

_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def get_population_data():
    """Get a series mapping countries to their population"""
    df = pd.read_csv(os.path.join(_data_path, "population", "population.csv")).set_index("Location")
    return df["PopTotal"] * 1000


def get_spain_population_data():
    """Get a series mapping Spanish Autonomous communities to their population"""
    df = pd.read_csv(os.path.join(_data_path, "spain-population", "poblacion-ccaa.csv")).set_index("AC")
    return df["Population"]


def iter_JHU_daily(date_start, date_end, countries=None, columns=None):
    if isinstance(countries, str):
        countries = [countries]
    for date in pd.date_range(date_start, date_end):
        d = date.strftime("%m-%d-%Y")  # American format
        try:
            df = pd.read_csv(
                "http://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/%s.csv"
                % d,
                usecols=columns,
            )
            df["date"] = date
            if countries is not None:
                df = df[df["Country_Region"].apply(lambda x: x in countries)]
            yield df
        except HTTPError:
            print("Ignoring error in %s" % str(date.date()))
        except SSLEOFError:
            print("Ignoring ssl error in %s" % str(date.date()))


def fetch_JHU_daily(date_start, date_end, countries=None, columns=None, progress=True):
    """
    Fetch a Dataframe with data from JHU CSSE COVID-19 Data.

    The full repository name is COVID-19 Data Repository by the Center for Systems Science and Engineering (CSSE) at
    Johns Hopkins University.

    Cf. https://github.com/CSSEGISandData/COVID-19 for license and terms of use.

    Args:
        date_start (str): Start date in ISO format (e.g., 2021-01-31).
        date_end (str): End date in ISO format.
        countries (None, str or list of str): If provided, list of countries to keep.
        columns (None or list of str): If provided, lists of columns to use.
        progress (bool): Whether to diplay a progress bar.

    Returns:
        pd.DataFrame: A dataframe with the information.

    """
    n = len(pd.date_range(date_start, date_end))

    return pd.concat(
        tqdm(
            iter_JHU_daily(date_start, date_end, countries=countries, columns=columns),
            total=n,
            disable=not progress,
        ),
        ignore_index=True
    )
