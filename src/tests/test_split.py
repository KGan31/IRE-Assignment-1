"""
Unit tests for split.py temporal splitting logic and anti-leakage assertions using Polars.
"""

from datetime import datetime, timedelta
from pathlib import Path
import sys
import polars as pl
import pandas as pd
import pytest

# Ensure src/ directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from split import (
    temporal_split_by_ratio,
    temporal_split_relative,
    temporal_split,
    history_as_of,
)


def test_temporal_split_by_ratio_proportions_polars():
    """Verify 80% / 10% / 10% split proportions on 1,000 synthetic impressions with Polars."""
    n_samples = 1000
    base_time = datetime(2024, 1, 1, 0, 0, 0)
    timestamps = [base_time + timedelta(minutes=i * 5) for i in range(n_samples)]

    impressions = pl.DataFrame({
        "impression_id": [f"imp_{i}" for i in range(n_samples)],
        "user_id": [f"u_{i % 50}" for i in range(n_samples)],
        "timestamp": timestamps,
        "candidate_article_ids": [["art_1", "art_2"] for _ in range(n_samples)],
        "clicked_article_ids": [["art_1"] for _ in range(n_samples)],
    })

    train, val, test, train_end, val_end = temporal_split_by_ratio(
        impressions, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1
    )

    assert len(train) == 800
    assert len(val) == 100
    assert len(test) == 100
    assert len(train) + len(val) + len(test) == n_samples


def test_temporal_split_no_future_leakage_polars():
    """Verify strictly monotonic timestamps across train, val, test splits in Polars."""
    n_samples = 500
    base_time = datetime(2024, 1, 1, 0, 0, 0)
    # Generate scrambled timestamps to test temporal sort
    timestamps = [base_time + timedelta(minutes=((i * 37) % n_samples) * 10) for i in range(n_samples)]

    impressions = pl.DataFrame({
        "impression_id": [f"imp_{i}" for i in range(n_samples)],
        "timestamp": timestamps,
    })

    train, val, test, train_end, val_end = temporal_split_by_ratio(
        impressions, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1
    )

    # 1. Check max(train) <= min(val) <= max(val) <= min(test)
    assert train["timestamp"].max() <= val["timestamp"].min(), "Leakage: Train timestamp exceeds Val min timestamp!"
    assert val["timestamp"].max() <= test["timestamp"].min(), "Leakage: Val timestamp exceeds Test min timestamp!"

    # 2. Check boundary cutoff timestamps
    assert train_end == val["timestamp"].min()
    assert val_end == test["timestamp"].min()

    # 3. Check no impression ID overlap between splits
    train_ids = set(train["impression_id"].to_list())
    val_ids = set(val["impression_id"].to_list())
    test_ids = set(test["impression_id"].to_list())
    assert len(train_ids.intersection(val_ids)) == 0
    assert len(val_ids.intersection(test_ids)) == 0
    assert len(train_ids.intersection(test_ids)) == 0


def test_history_as_of_anti_leakage_polars():
    """Verify click history is strictly filtered before cutoff timestamps in Polars."""
    cutoff_ts = datetime(2024, 1, 5, 12, 0, 0)

    history_df = pl.DataFrame({
        "user_id": ["u1", "u1", "u1", "u2", "u2"],
        "clicked_article_id": ["a1", "a2", "a3", "b1", "b2"],
        "click_time": [
            datetime(2024, 1, 4, 10, 0, 0),   # before cutoff
            datetime(2024, 1, 5, 11, 59, 59),  # before cutoff
            datetime(2024, 1, 5, 12, 0, 0),    # exact cutoff (should be excluded)
            datetime(2024, 1, 5, 15, 0, 0),    # after cutoff
            datetime(2024, 1, 6, 8, 0, 0),     # after cutoff
        ],
    })

    filtered_hist = history_as_of(history_df, cutoff_ts)

    assert len(filtered_hist) == 2
    assert set(filtered_hist["clicked_article_id"].to_list()) == {"a1", "a2"}
    assert (filtered_hist["click_time"] < cutoff_ts).all()


def test_temporal_split_by_ratio_pandas_interop():
    """Verify pandas DataFrame input compatibility."""
    n_samples = 100
    base_time = pd.Timestamp("2024-01-01 00:00:00")
    impressions = pd.DataFrame({
        "impression_id": [f"imp_{i}" for i in range(n_samples)],
        "timestamp": [base_time + pd.Timedelta(hours=i) for i in range(n_samples)],
    })

    train, val, test, train_end, val_end = temporal_split_by_ratio(
        impressions, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1
    )

    assert isinstance(train, pd.DataFrame)
    assert len(train) == 80
    assert len(val) == 10
    assert len(test) == 10


def test_temporal_split_by_ratio_invalid_ratios():
    """Check that invalid split ratios raise a ValueError."""
    impressions = pl.DataFrame({
        "impression_id": ["imp1", "imp2"],
        "timestamp": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
    })
    with pytest.raises(ValueError, match="Split ratios must sum to 1.0"):
        temporal_split_by_ratio(impressions, train_ratio=0.7, val_ratio=0.1, test_ratio=0.1)


def test_temporal_split_empty_dataframe():
    """Check that empty DataFrame raises ValueError."""
    empty_df = pl.DataFrame(schema={"impression_id": pl.Utf8, "timestamp": pl.Datetime})
    with pytest.raises(ValueError, match="Cannot split an empty impressions frame"):
        temporal_split_by_ratio(empty_df)
