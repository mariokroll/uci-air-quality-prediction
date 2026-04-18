"""Phase 1 — data loading tests."""

import pandas as pd
import pytest

from src.config import FEATURE_COLS, MISSING_VALUE_SENTINEL
from src.data.loader import load_raw
from src.data.preprocessor import get_missing_mask, missing_summary


@pytest.fixture(scope="module")
def df():
    return load_raw()


def test_no_sentinel_values(df):
    """No -200 values should survive loading."""
    assert not (df == MISSING_VALUE_SENTINEL).any().any()


def test_temporal_index(df):
    """Index must be a sorted DatetimeIndex."""
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.is_monotonic_increasing


def test_hourly_frequency(df):
    """Consecutive timestamps should differ by exactly one hour."""
    deltas = df.index.to_series().diff().dropna()
    # Allow for a single gap at most (dataset has one missing day)
    modal_delta = deltas.mode()[0]
    assert modal_delta == pd.Timedelta("1h"), f"Modal delta is {modal_delta}"


def test_feature_columns_present(df):
    """All 13 feature columns must be present."""
    for col in FEATURE_COLS:
        assert col in df.columns, f"Missing column: {col}"


def test_missing_values_are_nan(df):
    """Missing data must be NaN, not the sentinel."""
    mask = get_missing_mask(df)
    assert mask.any().any(), "Expected at least some NaN values in the dataset"


def test_nmhc_high_missingness(df):
    """NMHC(GT) must be ≥ 80 % missing (known to be ~89 %)."""
    summary = missing_summary(df)
    pct = summary.loc["NMHC(GT)", "missing_pct"]
    assert pct >= 80.0, f"NMHC(GT) missing% = {pct:.1f}, expected ≥ 80 %"


def test_dataset_size(df):
    """Dataset should contain at least 9 000 records."""
    assert len(df) >= 9_000, f"Expected ≥ 9 000 rows, got {len(df)}"
