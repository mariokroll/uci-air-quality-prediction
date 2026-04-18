import numpy as np
import pandas as pd

from src.config import DATASET_FILENAME, FEATURE_COLS, MISSING_VALUE_SENTINEL, RAW_DATA_DIR


def load_raw() -> pd.DataFrame:
    """Load Air Quality UCI dataset with a sorted DatetimeIndex.

    The raw file uses semicolons as field separators and commas as decimal
    separators (European locale).  Missing values are encoded as -200 in the
    original; they are replaced with NaN here so downstream code never sees
    the sentinel.
    """
    path = RAW_DATA_DIR / DATASET_FILENAME
    df = pd.read_csv(
        path,
        sep=";",
        decimal=",",
        na_values=["", " "],
    )

    # The CSV often has two fully-empty trailing columns — drop them.
    df = df.dropna(axis=1, how="all")
    df = df.dropna(axis=0, how="all")

    # Build a proper DatetimeIndex from the Date / Time columns.
    # Raw time format is HH.MM.SS (dots, not colons).
    df["Datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"],
        format="%d/%m/%Y %H.%M.%S",
    )
    df = df.set_index("Datetime").drop(columns=["Date", "Time"])
    df = df.sort_index()

    # Keep only the 13 feature columns defined in config.
    df = df[FEATURE_COLS]

    # Replace the -200 sentinel with NaN.
    df = df.replace(MISSING_VALUE_SENTINEL, np.nan)

    return df
