import pandas as pd


def get_missing_mask(df: pd.DataFrame) -> pd.DataFrame:
    """Return a boolean DataFrame: True where a value is missing."""
    return df.isna()


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-column missing-value counts and percentages."""
    total = len(df)
    counts = df.isna().sum()
    pct = (counts / total * 100).round(2)
    return pd.DataFrame({"missing_count": counts, "missing_pct": pct})
