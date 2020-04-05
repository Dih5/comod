import os

import pandas as pd

_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def get_population_data():
    """Get a series mapping countries to their population"""
    df = pd.read_csv(os.path.join(_data_path, "population", "population.csv")).set_index("Location")
    return df["PopTotal"] * 1000


def get_spain_population_data():
    """Get a series mapping Spanish Autonomous communities to their population"""
    df = pd.read_csv(os.path.join(_data_path, "spain-population", "poblacion-ccaa.csv")).set_index("AC")
    return df["Population"]
