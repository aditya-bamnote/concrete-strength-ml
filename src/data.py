"""Data loading helpers for the UCI Concrete Compressive Strength dataset.

Dataset: Yeh, I-C. (1998). Concrete Compressive Strength. UCI ML Repository.
https://doi.org/10.24432/C5PK67   (CC BY 4.0)
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "Concrete_Data.xls"

# Short column names, in the same order as the original file.
FEATURES = [
    "cement",            # kg per m^3 of mixture
    "slag",              # blast furnace slag, kg/m^3
    "fly_ash",           # kg/m^3
    "water",             # kg/m^3
    "superplasticizer",  # kg/m^3
    "coarse_agg",        # coarse aggregate, kg/m^3
    "fine_agg",          # fine aggregate, kg/m^3
    "age",               # curing age in days (1-365)
]
TARGET = "strength"      # compressive strength in MPa

# Friendly labels and units, used by the app and the plots.
LABELS = {
    "cement": "Cement (kg/m³)",
    "slag": "Blast furnace slag (kg/m³)",
    "fly_ash": "Fly ash (kg/m³)",
    "water": "Water (kg/m³)",
    "superplasticizer": "Superplasticizer (kg/m³)",
    "coarse_agg": "Coarse aggregate (kg/m³)",
    "fine_agg": "Fine aggregate (kg/m³)",
    "age": "Curing age (days)",
    "strength": "Compressive strength (MPa)",
}


def load_raw(path=DATA_PATH) -> pd.DataFrame:
    """Read the original .xls and rename the long column names by position."""
    df = pd.read_excel(path)
    if df.shape[1] != len(FEATURES) + 1:
        raise ValueError(f"Expected {len(FEATURES) + 1} columns, found {df.shape[1]}")
    df.columns = FEATURES + [TARGET]
    return df


def load_clean(path=DATA_PATH, drop_duplicates=True) -> pd.DataFrame:
    """Load the data and (optionally) remove exact duplicate rows."""
    df = load_raw(path)
    if drop_duplicates:
        df = df.drop_duplicates().reset_index(drop=True)
    return df
